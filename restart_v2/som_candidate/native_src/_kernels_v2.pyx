# cython: language_level=3,boundscheck=False,wraparound=False,cdivision=True
# cython: initializedcheck=False,nonecheck=False
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
from scipy.linalg.cython_blas cimport dgemm
from libc.math cimport fabs
cdef extern from "native_v2.h" nogil:
    void som_cost_aos(const double*,const double*,const double*,const double*,double*,Py_ssize_t,Py_ssize_t,Py_ssize_t,double) noexcept
    void som_cost_soa(const double*,const double*,const double*,const double*,double*,Py_ssize_t,Py_ssize_t,Py_ssize_t,double) noexcept
    void som_cost_gemm_finish(const double*,const double*,const double*,double,double*,Py_ssize_t,Py_ssize_t,double) noexcept
    double som_softmax(double*,double*,Py_ssize_t,double,int) noexcept
    double som_softmax_mvec(double*,double*,Py_ssize_t,double) noexcept
    double som_softmax_next2(double*,double*,Py_ssize_t,double,int,const double*,double*,int) noexcept
    void som_add_row(const double*,double*,Py_ssize_t) noexcept
cdef void initial_den(const double[:,::1] P,double[:,::1] dens,Py_ssize_t t,
                      Py_ssize_t start,Py_ssize_t end) noexcept nogil:
    cdef Py_ssize_t i,j,m=P.shape[1]
    for j in range(m): dens[t,j]=0.
    for i in range(start,end): som_add_row(&P[i,0],&dens[t,0],m)
cdef void rows(const double[:,::1] X,const double[:,::1] R,
               const double[:,::1] W,const double[:,::1] WT,
               const double[:,::1] RT,const double[:,::1] V,
               double[:,::1] P,double[:,::1] dens,double[:,::1] objs,
               const double[::1] xn,const double[::1] wn,
               Py_ssize_t t,Py_ssize_t start,Py_ssize_t end,
               double gamma,double lam,int mode,int vector_exp) noexcept nogil:
    cdef Py_ssize_t i,j,m=W.shape[0],d=X.shape[1],k=R.shape[1]
    cdef double obj=0.
    for j in range(m): dens[t,j]=0.
    for i in range(start,end):
        if mode==0:
            som_cost_soa(&X[i,0],&WT[0,0],&V[i,0],&RT[0,0],&P[i,0],m,d,k,gamma)
        elif mode==1:
            som_cost_aos(&X[i,0],&W[0,0],&V[i,0],&R[0,0],&P[i,0],m,d,k,gamma)
        else:
            som_cost_gemm_finish(&V[i,0],&RT[0,0],&wn[0],xn[i],&P[i,0],m,k,gamma)
        obj+=(som_softmax_mvec(&P[i,0],&dens[t,0],m,lam) if vector_exp else som_softmax(&P[i,0],&dens[t,0],m,lam,0))
    objs[t,0]=obj
cdef void normalize_w(double[:,::1] W,const double[:,::1] numerator,
                      const double[:,::1] dens,double[:,::1] WT,
                      double[::1] wn,Py_ssize_t active,Py_ssize_t j) noexcept nogil:
    cdef Py_ssize_t t,q,d=W.shape[1]
    cdef double den=0.,norm=0.,z
    for t in range(active): den+=dens[t,j]
    if den>0.:
        for q in range(d): W[j,q]=numerator[j,q]/den
    for q in range(d):
        z=W[j,q]
        WT[q,j]=z
        norm+=z*z
    wn[j]=norm

def iterate(const double[:,::1] X,const double[:,::1] R,
             const double[:,::1] W0,const double[:,::1] P0,
             double gamma,double lam,int max_iters=100,double tol=1e-4,
             int threads=1,int mode=0,int vector_exp=0):
    cdef int n=<int>X.shape[0],d=<int>X.shape[1],m=<int>R.shape[0],k=<int>R.shape[1]
    cdef int active=min(threads,n),it,t,i,j,q
    cdef double one=1.,zero=0.,minus2=-2.,value,obj=0.,prev=0.,scale
    cdef char no='N',trans='T'
    cdef cnp.ndarray W_np=np.array(W0,copy=True,order='C')
    cdef cnp.ndarray P_np=np.array(P0,copy=True,order='C')
    cdef cnp.ndarray V_np=np.empty((n,k),dtype=np.float64)
    cdef cnp.ndarray history_np=np.empty(max_iters,dtype=np.float64)
    cdef double[:,::1] W=W_np,P=P_np,V=V_np
    cdef double[::1] history=history_np
    cdef double[:,::1] numerator=np.empty((m,d),dtype=np.float64)
    cdef double[:,::1] WT=np.empty((d,m),dtype=np.float64)
    cdef double[:,::1] RT=np.ascontiguousarray(np.asarray(R).T)
    cdef double[:,::1] dens=np.empty((active,m+8),dtype=np.float64)
    cdef double[:,::1] objs=np.empty((active,8),dtype=np.float64)
    cdef double[::1] xn=np.empty(n,dtype=np.float64)
    cdef double[::1] wn=np.empty(m,dtype=np.float64)
    cdef int actual=0
    with nogil:
        if active==1:
            initial_den(P,dens,0,0,n)
        else:
            for t in prange(active,num_threads=active,schedule='static'):
                initial_den(P,dens,t,n*t//active,n*(t+1)//active)
        if mode==2:
            for i in range(n):
                value=0.
                for q in range(d): value+=X[i,q]*X[i,q]
                xn[i]=value
        for it in range(max_iters):
            # V^T = R^T P^T, represented as row-major V=P@R.
            dgemm(&no,&no,&k,&n,&m,&one,<double*>&R[0,0],&k,
                  &P[0,0],&m,&zero,&V[0,0],&k)
            # numerator^T = X^T P, represented as row-major P.T@X.
            dgemm(&no,&trans,&d,&m,&n,&one,<double*>&X[0,0],&d,
                  &P[0,0],&m,&zero,&numerator[0,0],&d)
            if active==1:
                for j in range(m): normalize_w(W,numerator,dens,WT,wn,active,j)
            else:
                for j in prange(m,num_threads=active,schedule='static'):
                    normalize_w(W,numerator,dens,WT,wn,active,j)
            if mode==2:
                dgemm(&trans,&no,&m,&n,&d,&minus2,&W[0,0],&d,
                      <double*>&X[0,0],&d,&zero,&P[0,0],&m)
            if active==1:
                rows(X,R,W,WT,RT,V,P,dens,objs,xn,wn,0,0,n,gamma,lam,mode,vector_exp)
            else:
                for t in prange(active,num_threads=active,schedule='static'):
                    rows(X,R,W,WT,RT,V,P,dens,objs,xn,wn,t,n*t//active,n*(t+1)//active,gamma,lam,mode,vector_exp)
            obj=0.
            for t in range(active): obj+=objs[t,0]
            history[it]=obj
            actual=it+1
            scale=fabs(prev)
            if scale<1.: scale=1.
            if it>0 and fabs(obj-prev)/scale<=tol: break
            prev=obj
    return {'W':W_np,'P':P_np,'V':V_np,'history':history_np[:actual].copy(),'n_iter':actual}

cdef void initial_state(const double[:,::1] X,const double[:,::1] R,
                        const double[:,::1] P,double[:,::1] V,
                        double[:,:,::1] nums,double[:,::1] dens,
                        int t,int start,int end) noexcept nogil:
    cdef int n=end-start,d=<int>X.shape[1],m=<int>R.shape[0],k=<int>R.shape[1]
    cdef double one=1.,zero=0.
    cdef char no='N',trans='T'
    initial_den(P,dens,t,start,end)
    dgemm(&no,&no,&k,&n,&m,&one,<double*>&R[0,0],&k,
          <double*>&P[start,0],&m,&zero,&V[start,0],&k)
    dgemm(&no,&trans,&d,&m,&n,&one,<double*>&X[start,0],&d,
          <double*>&P[start,0],&m,&zero,&nums[t,0,0],&d)

cdef void normalize_stream_w(double[:,::1] W,const double[:,:,::1] nums,
                            const double[:,::1] dens,double[:,::1] WT,
                            double[::1] wn,int active,int j,int mode,bint skip_prep) noexcept nogil:
    cdef int t,q,d=<int>W.shape[1]
    cdef double den=0.,norm=0.,z,value
    for t in range(active): den+=dens[t,j]
    if den>0.:
        for q in range(d):
            value=0.
            for t in range(active): value+=nums[t,j,q]
            W[j,q]=value/den
    for q in range(d):
        z=W[j,q]
        if not skip_prep or mode!=2: WT[q,j]=z
        if not skip_prep or mode==2: norm+=z*z
    wn[j]=norm

cdef void stream_rows(const double[:,::1] X,const double[:,::1] R,
                     const double[:,::1] W,const double[:,::1] WT,
                     const double[:,::1] RT,const double[:,::1] V,
                     double[:,::1] VN,double[:,::1] P,
                     double[:,:,::1] nums,double[:,::1] dens,
                     double[:,::1] objs,const double[::1] xn,
                     const double[::1] wn,int t,int start,int end,
                     double gamma,double lam,int mode,int block_rows,int vector_exp,bint nextstate,bint fuse_v,bint beta_zero) noexcept nogil:
    cdef int b,rows_n,i,j,q,m=<int>W.shape[0],d=<int>X.shape[1],k=<int>R.shape[1]
    cdef double one=1.,zero=0.,minus2=-2.,obj=0.,beta
    cdef char no='N',trans='T'
    if nextstate:
        for j in range(m):
            dens[t,j]=0.
            if not beta_zero:
                for q in range(d): nums[t,j,q]=0.
    b=start
    while b<end:
        rows_n=min(block_rows,end-b)
        if mode==2:
            dgemm(&trans,&no,&m,&rows_n,&d,&minus2,<double*>&W[0,0],&d,
                  <double*>&X[b,0],&d,&zero,&P[b,0],&m)
        for i in range(b,b+rows_n):
            if mode==0:
                som_cost_soa(&X[i,0],&WT[0,0],&V[i,0],&RT[0,0],&P[i,0],m,d,k,gamma)
            elif mode==1:
                som_cost_aos(&X[i,0],&W[0,0],&V[i,0],&R[0,0],&P[i,0],m,d,k,gamma)
            else:
                som_cost_gemm_finish(&V[i,0],&RT[0,0],&wn[0],xn[i],&P[i,0],m,k,gamma)
            if k==2 and fuse_v:
                obj+=som_softmax_next2(&P[i,0],&dens[t,0],m,lam,vector_exp,&RT[0,0],&VN[i,0],nextstate)
            else:
                obj+=(som_softmax_mvec(&P[i,0],&dens[t,0],m,lam) if vector_exp else som_softmax(&P[i,0],&dens[t,0],m,lam,0))
        # New P is still cache-resident; form sufficient statistics for next W,V.
        if not nextstate:
            b+=rows_n
            continue
        if k!=2 or not fuse_v:
            dgemm(&no,&no,&k,&rows_n,&m,&one,<double*>&R[0,0],&k,
                  &P[b,0],&m,&zero,&VN[b,0],&k)
        beta=zero if b==start and beta_zero else one
        dgemm(&no,&trans,&d,&m,&rows_n,&one,<double*>&X[b,0],&d,
              &P[b,0],&m,&beta,&nums[t,0,0],&d)
        b+=rows_n
    objs[t,0]=obj


def iterate_stream(const double[:,::1] X,const double[:,::1] R,
                    const double[:,::1] W0,const double[:,::1] P0,
                    double gamma,double lam,int max_iters=100,double tol=1e-4,
                    int threads=1,int mode=0,int block_rows=256,int vector_exp=0,
                    int optimizations=15):
    cdef int n=<int>X.shape[0],d=<int>X.shape[1],m=<int>R.shape[0],k=<int>R.shape[1]
    cdef int active=min(threads,n),it,t,i,j,q
    cdef double value,obj=0.,prev=0.,scale
    cdef cnp.ndarray W_np=np.array(W0,copy=True,order='C')
    cdef cnp.ndarray P_np=(np.empty((n,m),dtype=np.float64) if optimizations&8 else np.array(P0,copy=True,order='C'))
    cdef cnp.ndarray V_np=np.empty((n,k),dtype=np.float64)
    cdef cnp.ndarray VN_np=np.empty((n,k),dtype=np.float64)
    cdef cnp.ndarray history_np=np.empty(max_iters,dtype=np.float64)
    cdef double[:,::1] W=W_np,P=P_np,V=V_np,VN=VN_np,swap
    cdef double[::1] history=history_np
    cdef double[:,:,::1] nums=np.empty((active,m,d),dtype=np.float64)
    cdef double[:,::1] WT=np.empty((d,m),dtype=np.float64)
    cdef double[:,::1] RT=np.ascontiguousarray(np.asarray(R).T)
    cdef double[:,::1] dens=np.empty((active,m+8),dtype=np.float64)
    cdef double[:,::1] objs=np.empty((active,8),dtype=np.float64)
    cdef double[::1] xn=np.empty(n,dtype=np.float64)
    cdef double[::1] wn=np.empty(m,dtype=np.float64)
    cdef int actual=0
    with nogil:
        if active==1:
            initial_state(X,R,P0,V,nums,dens,0,0,n)
        else:
            for t in prange(active,num_threads=active,schedule='static'):
                initial_state(X,R,P0,V,nums,dens,t,n*t//active,n*(t+1)//active)
        if mode==2:
            for i in range(n):
                value=0.
                for q in range(d): value+=X[i,q]*X[i,q]
                xn[i]=value
        for it in range(max_iters):
            if active==1:
                for j in range(m): normalize_stream_w(W,nums,dens,WT,wn,active,j,mode,(optimizations&4)!=0)
            else:
                for j in prange(m,num_threads=active,schedule='static'):
                    normalize_stream_w(W,nums,dens,WT,wn,active,j,mode,(optimizations&4)!=0)
            if active==1:
                stream_rows(X,R,W,WT,RT,V,VN,P,nums,dens,objs,xn,wn,0,0,n,gamma,lam,mode,block_rows,vector_exp,it+1<max_iters,(optimizations&2)!=0,(optimizations&1)!=0)
            else:
                for t in prange(active,num_threads=active,schedule='static'):
                    stream_rows(X,R,W,WT,RT,V,VN,P,nums,dens,objs,xn,wn,t,n*t//active,n*(t+1)//active,gamma,lam,mode,block_rows,vector_exp,it+1<max_iters,(optimizations&2)!=0,(optimizations&1)!=0)
            obj=0.
            for t in range(active): obj+=objs[t,0]
            history[it]=obj
            actual=it+1
            scale=fabs(prev)
            if scale<1.: scale=1.
            if (it>0 and fabs(obj-prev)/scale<=tol) or actual==max_iters: break
            prev=obj
            swap=V
            V=VN
            VN=swap
    return {'W':W_np,'P':P_np,'V':np.asarray(V),'history':history_np[:actual].copy(),'n_iter':actual}
