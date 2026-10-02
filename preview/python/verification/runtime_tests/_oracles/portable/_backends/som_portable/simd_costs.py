"""Optional strict SOM full-cost SIMD experiment; no fastmath or approximations."""
from contextlib import nullcontext
import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.core import cgutils
from numba.core.errors import TypingError
from numba.extending import intrinsic


def _make_cost_intrinsic(width):
    @intrinsic
    def costs(typing_context, x_type, wt_type, v_type, rt_type, out_type, row_type, col_type, gamma_type):
        arrays=(x_type,wt_type,v_type,rt_type,out_type)
        if not all(isinstance(t,types.Array) and t.ndim==2 and t.dtype==types.float64 and t.layout=='C' for t in arrays):
            raise TypingError('SOM costs require C-contiguous float64 matrices')
        if not out_type.mutable or not isinstance(row_type,types.Integer) or not isinstance(col_type,types.Integer) or gamma_type!=types.float64:
            raise TypingError('SOM costs require mutable output, integer indices, and float64 gamma')
        signature=types.int64(*arrays,row_type,col_type,gamma_type)
        def codegen(context,builder,sig,args):
            x,wt,v,rt,out=[context.make_array(t)(context,builder,value=a) for t,a in zip(sig.args[:5],args[:5])]
            d=cgutils.unpack_tuple(builder,x.shape,count=2)[1]
            q=cgutils.unpack_tuple(builder,v.shape,count=2)[1]
            scalar=ir.DoubleType();i32=ir.IntType(32);vf=ir.VectorType(scalar,width)
            zero=ir.Constant(vf,[0.]*width);undef=ir.Constant(vf,ir.Undefined)
            mask=ir.Constant(ir.VectorType(i32,width),[0]*width)
            feature=[cgutils.alloca_once(builder,vf) for _ in range(4)]
            grid=[cgutils.alloca_once(builder,vf) for _ in range(4)]
            for slot in feature+grid:builder.store(zero,slot)
            for grid_part,(dimension,values,centers,values_type,centers_type,slots) in enumerate([(d,x,wt,sig.args[0],sig.args[1],feature),(q,v,rt,sig.args[2],sig.args[3],grid)]):
                active = (builder.if_then(builder.fcmp_ordered('!=',args[7],ir.Constant(scalar,0.)))
                          if grid_part else nullcontext())
                with active:
                    with cgutils.for_range(builder,dimension) as loop:
                        center_pointer=cgutils.get_item_pointer(context,builder,centers_type,centers,[loop.index,args[6]])
                        cv=builder.load(builder.bitcast(center_pointer,vf.as_pointer()),align=1)
                        for delta,slot in enumerate(slots):
                            row=builder.add(args[5],ir.Constant(args[5].type,delta))
                            pointer=cgutils.get_item_pointer(context,builder,values_type,values,[row,loop.index])
                            value=builder.load(pointer,align=1)
                            broadcast=builder.shuffle_vector(builder.insert_element(undef,value,i32(0)),undef,mask)
                            difference=builder.fsub(broadcast,cv)
                            square=builder.fmul(difference,difference)
                            builder.store(builder.fadd(builder.load(slot),square),slot)
            gamma=builder.shuffle_vector(builder.insert_element(undef,args[7],i32(0)),undef,mask)
            for delta,(fs,gs) in enumerate(zip(feature,grid)):
                value=builder.fadd(builder.load(fs),builder.fmul(gamma,builder.load(gs)))
                row=builder.add(args[5],ir.Constant(args[5].type,delta))
                pointer=cgutils.get_item_pointer(context,builder,sig.args[4],out,[row,args[6]])
                builder.store(value,builder.bitcast(pointer,vf.as_pointer()),align=1)
            return ir.Constant(ir.IntType(64),0)
        return signature,codegen
    return costs

_costs4x4=_make_cost_intrinsic(4)
_costs8x4=_make_cost_intrinsic(8)
_costs16x4=_make_cost_intrinsic(16)


@njit(cache=True)
def costs_into(X, WT, V, RT, gamma, output, width=8):
    n,d=X.shape;m=output.shape[1];q=V.shape[1]
    if width not in (4,8,16):raise ValueError('width must be 4, 8, or 16')
    if WT.shape!=(d,m) or V.shape[0]!=n or RT.shape!=(q,m) or output.shape[0]!=n:
        raise ValueError('inconsistent SOM cost shapes')
    stop_rows=(n//4)*4;stop_cols=(m//width)*width;scalar_cols=(m//4)*4
    for i in range(0,stop_rows,4):
        for j in range(0,stop_cols,width):
            if width==16:_costs16x4(X,WT,V,RT,output,i,j,gamma)
            elif width==8:_costs8x4(X,WT,V,RT,output,i,j,gamma)
            else:_costs4x4(X,WT,V,RT,output,i,j,gamma)
        next_col=stop_cols
        if width==16 and next_col+8<=m:
            _costs8x4(X,WT,V,RT,output,i,next_col,gamma)
            next_col+=8
        if next_col+4<=m:
            _costs4x4(X,WT,V,RT,output,i,next_col,gamma)
        for row in range(i,i+4):
            for j in range(scalar_cols,m):
                a=0.;b=0.
                for f in range(d):
                    difference=X[row,f]-WT[f,j];a+=difference*difference
                if gamma:
                    for h in range(q):
                        difference=V[row,h]-RT[h,j];b+=difference*difference
                output[row,j]=a+gamma*b
    for i in range(stop_rows,n):
        for j in range(m):
            a=0.;b=0.
            for f in range(d):
                difference=X[i,f]-WT[f,j];a+=difference*difference
            if gamma:
                for h in range(q):
                    difference=V[i,h]-RT[h,j];b+=difference*difference
            output[i,j]=a+gamma*b
