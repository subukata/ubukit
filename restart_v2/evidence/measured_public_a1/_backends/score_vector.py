"""Target-independent LLVM-vector selector for guarded BLAS score rows.

SIMD is across independent centers; each score still uses one strict float64
subtraction. No reassociation, fastmath or target-specific machine code is used.
"""
import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic
from numba.core.errors import TypingError


@intrinsic
def _top2_groups8(typing_context, scores_type, row_type, cnorm_type):
    # A malformed direct/private call must fail compilation, not reinterpret
    # float32, strided or differently shaped storage as packed float64 lanes.
    if not (isinstance(scores_type, types.Array) and scores_type.ndim == 2
            and scores_type.dtype == types.float64 and scores_type.layout == 'C'
            and isinstance(cnorm_type, types.Array) and cnorm_type.ndim == 1
            and cnorm_type.dtype == types.float64 and cnorm_type.layout == 'C'
            and isinstance(row_type, types.Integer)):
        raise TypingError('score selector requires contiguous float64 matrix/norms and an integer row')
    if types.intp.bitwidth != 64:
        raise TypingError('score SIMD selector requires a 64-bit platform; use scalar BLAS')
    output_type = types.Tuple((types.float64, types.float64, types.int64))
    signature = output_type(scores_type, row_type, cnorm_type)

    def codegen(context, builder, sig, args):
        scores = context.make_array(sig.args[0])(context, builder, value=args[0])
        norms = context.make_array(sig.args[2])(context, builder, value=args[2])
        k = cgutils.unpack_tuple(builder, scores.shape, count=2)[1]
        scalar = ir.DoubleType()
        i32, i64 = ir.IntType(32), ir.IntType(64)
        vf64, vi64 = ir.VectorType(scalar, 8), ir.VectorType(i64, 8)
        neg_inf = ir.Constant(vf64, [-float('inf')] * 8)
        zeroi = ir.Constant(vi64, [0] * 8)
        undefi = ir.Constant(vi64, ir.Undefined)
        broadcast_mask = ir.Constant(ir.VectorType(i32, 8), [0] * 8)
        lanes = ir.Constant(vi64, list(range(8)))
        best_slot = cgutils.alloca_once(builder, vf64)
        second_slot = cgutils.alloca_once(builder, vf64)
        index_slot = cgutils.alloca_once(builder, vi64)
        builder.store(neg_inf, best_slot)
        builder.store(neg_inf, second_slot)
        builder.store(zeroi, index_slot)
        with cgutils.for_range(builder, builder.sdiv(k, i64(8))) as loop:
            c = builder.mul(loop.index, i64(8))
            sp = cgutils.get_item_pointer(context, builder, sig.args[0], scores, [args[1], c])
            nptr = cgutils.get_item_pointer(context, builder, sig.args[2], norms, [c])
            sv = builder.load(builder.bitcast(sp, vf64.as_pointer()), align=8)
            nv = builder.load(builder.bitcast(nptr, vf64.as_pointer()), align=8)
            v = builder.fsub(sv, nv)
            best = builder.load(best_slot)
            second = builder.load(second_slot)
            better = builder.fcmp_ordered('>', v, best)
            second_better = builder.fcmp_ordered('>', v, second)
            second = builder.select(better, best, builder.select(second_better, v, second))
            best = builder.select(better, v, best)
            base = builder.shuffle_vector(builder.insert_element(undefi, c, i32(0)), undefi, broadcast_mask)
            idx = builder.select(better, builder.add(base, lanes), builder.load(index_slot))
            builder.store(best, best_slot)
            builder.store(second, second_slot)
            builder.store(idx, index_slot)
        best_v = builder.load(best_slot)
        second_v = builder.load(second_slot)
        indices = builder.load(index_slot)
        best = ir.Constant(scalar, -float('inf'))
        second = ir.Constant(scalar, -float('inf'))
        index = i64(0)
        for lane in range(8):
            v = builder.extract_element(best_v, i32(lane))
            local_index = builder.extract_element(indices, i32(lane))
            better = builder.or_(builder.fcmp_ordered('>', v, best),
                                 builder.and_(builder.fcmp_ordered('==', v, best),
                                              builder.icmp_signed('<', local_index, index)))
            second = builder.select(better, best,
                                    builder.select(builder.fcmp_ordered('>', v, second), v, second))
            best = builder.select(better, v, best)
            index = builder.select(better, local_index, index)
            local_second = builder.extract_element(second_v, i32(lane))
            second = builder.select(builder.fcmp_ordered('>', local_second, second), local_second, second)
        return context.make_tuple(builder, sig.return_type, (best, second, index))
    return signature, codegen


@njit(cache=True, inline='always')
def top2(scores, row, cnorm):
    best, second, label = _top2_groups8(scores, row, cnorm)
    for c in range((scores.shape[1] // 8) * 8, scores.shape[1]):
        v = scores[row, c] - cnorm[c]
        if v > best:
            second = best
            best = v
            label = c
        elif v > second:
            second = v
    return best, second, label
