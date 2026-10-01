#ifndef SOM_OLP_NATIVE_V2_H
#define SOM_OLP_NATIVE_V2_H
#include <math.h>
#include <stddef.h>
#include <float.h>
#include <immintrin.h>
#if defined(__AVX512F__)
extern __m512d _ZGVeN8v_exp(__m512d);
#elif defined(__AVX2__)
extern __m256d _ZGVdN4v_exp(__m256d);
#endif
/* No reassociation/fastmath compiler flags. SIMD reductions explicitly permit
   changed summation order, but retain double precision and libm functions. */
static inline void som_cost_aos(const double *restrict x,
  const double *restrict w, const double *restrict v,
  const double *restrict r, double *restrict out,
  ptrdiff_t m, ptrdiff_t d, ptrdiff_t k, double gamma) {
    for (ptrdiff_t j=0;j<m;++j) {
        double a=0.,b=0.;
        #pragma omp simd reduction(+:a)
        for (ptrdiff_t q=0;q<d;++q) {
            double z=x[q]-w[j*d+q]; a+=z*z;
        }
        #pragma omp simd reduction(+:b)
        for (ptrdiff_t q=0;q<k;++q) {
            double z=v[q]-r[j*k+q]; b+=z*z;
        }
        out[j]=a+gamma*b;
    }
}
/* Feature-major weights let independent centers occupy SIMD lanes: no
   reassociation of the feature reduction is needed. */
static inline void som_cost_soa(const double *restrict x,
  const double *restrict wt, const double *restrict v,
  const double *restrict rt, double *restrict out,
  ptrdiff_t m, ptrdiff_t d, ptrdiff_t k, double gamma) {
    #pragma omp simd
    for (ptrdiff_t j=0;j<m;++j) { double z=x[0]-wt[j]; out[j]=z*z; }
    for (ptrdiff_t q=1;q<d;++q) {
        const double xq=x[q], *restrict wq=wt+q*m;
        #pragma omp simd
        for (ptrdiff_t j=0;j<m;++j) { double z=xq-wq[j]; out[j]+=z*z; }
    }
    if (k==2) {
        const double v0=v[0],v1=v[1];
        #pragma omp simd
        for (ptrdiff_t j=0;j<m;++j) {
            double a=v0-rt[j],b=v1-rt[m+j]; out[j]+=gamma*(a*a+b*b);
        }
    } else {
        #pragma omp simd
        for (ptrdiff_t j=0;j<m;++j) {
            double b=0.;
            for (ptrdiff_t q=0;q<k;++q) { double z=v[q]-rt[q*m+j]; b+=z*z; }
            out[j]+=gamma*b;
        }
    }
}
static inline void som_cost_gemm_finish(const double *restrict v,
  const double *restrict rt, const double *restrict wnorm,
  double xnorm, double *restrict out,
  ptrdiff_t m,ptrdiff_t k,double gamma) {
    #pragma omp simd
    for (ptrdiff_t j=0;j<m;++j) {
        double local=out[j]+xnorm+wnorm[j], neighbor=0.;
        if (local<0.) local=0.;
        for (ptrdiff_t q=0;q<k;++q) { double z=v[q]-rt[q*m+j]; neighbor+=z*z; }
        out[j]=local+gamma*neighbor;
    }
}
/* out holds costs on entry, probabilities on exit. den is thread-private. */
static inline double som_softmax(double *restrict out,
  double *restrict den,ptrdiff_t m,double lam,int explicit_objective) {
    double mincost=out[0];
    for(ptrdiff_t j=1;j<m;++j) if(out[j]<mincost) mincost=out[j];
    double sum=0.;
    for(ptrdiff_t j=0;j<m;++j) { out[j]=exp(-(out[j]-mincost)/lam); sum+=out[j]; }
    double inv=1./sum;
    #pragma omp simd
    for(ptrdiff_t j=0;j<m;++j) { out[j]*=inv; den[j]+=out[j]; }
    (void)explicit_objective;
    return mincost-lam*log(sum);
}
/* Explicit glibc libmvec, without any fastmath or polynomial approximation
   enabled by this project. It is a different libm implementation, validated
   separately from scalar exp. SIMD sums change floating reduction order. */
static inline double som_softmax_mvec(double *restrict out,
  double *restrict den,ptrdiff_t m,double lam) {
    double mincost=out[0];
    #pragma omp simd reduction(min:mincost)
    for(ptrdiff_t j=1;j<m;++j) if(out[j]<mincost) mincost=out[j];
    double sum=0.;
    ptrdiff_t j=0;
#if defined(__AVX512F__)
    __m512d sumv=_mm512_setzero_pd(), minv=_mm512_set1_pd(mincost),
            lamv=_mm512_set1_pd(-lam);
    for(;j+8<=m;j+=8) {
        __m512d z=_mm512_div_pd(_mm512_sub_pd(_mm512_loadu_pd(out+j),minv),lamv);
        __m512d e=_ZGVeN8v_exp(z);
        _mm512_storeu_pd(out+j,e);
        sumv=_mm512_add_pd(sumv,e);
    }
    sum=_mm512_reduce_add_pd(sumv);
#elif defined(__AVX2__)
    __m256d sumv=_mm256_setzero_pd(), minv=_mm256_set1_pd(mincost),
            lamv=_mm256_set1_pd(-lam);
    for(;j+4<=m;j+=4) {
        __m256d z=_mm256_div_pd(_mm256_sub_pd(_mm256_loadu_pd(out+j),minv),lamv);
        __m256d e=_ZGVdN4v_exp(z);
        _mm256_storeu_pd(out+j,e);
        sumv=_mm256_add_pd(sumv,e);
    }
    double sums[4];_mm256_storeu_pd(sums,sumv);
    sum=(sums[0]+sums[1])+(sums[2]+sums[3]);
#endif
    for(;j<m;++j) { out[j]=exp(-(out[j]-mincost)/lam); sum+=out[j]; }
    double inv=1./sum;
    #pragma omp simd
    for(j=0;j<m;++j) { out[j]*=inv; den[j]+=out[j]; }
    return mincost-lam*log(sum);
}

/* Versioned fused normalization/denominator/next-V path for a 2D grid. */
static inline double som_softmax_next2(double *restrict out,
  double *restrict den,ptrdiff_t m,double lam,int vector_exp,
  const double *restrict rt,double *restrict nextv,int nextstate) {
    double mincost=out[0];
    #pragma omp simd reduction(min:mincost)
    for(ptrdiff_t j=1;j<m;++j) if(out[j]<mincost) mincost=out[j];
    double sum=0.;ptrdiff_t j=0;
#if defined(__AVX512F__)
    if (vector_exp) {
        __m512d sumv=_mm512_setzero_pd(),minv=_mm512_set1_pd(mincost),lv=_mm512_set1_pd(-lam);
        for(;j+8<=m;j+=8) {
            __m512d e=_ZGVeN8v_exp(_mm512_div_pd(_mm512_sub_pd(_mm512_loadu_pd(out+j),minv),lv));
            _mm512_storeu_pd(out+j,e);sumv=_mm512_add_pd(sumv,e);
        }
        sum=_mm512_reduce_add_pd(sumv);
    }
#elif defined(__AVX2__)
    if (vector_exp) {
        __m256d sumv=_mm256_setzero_pd(),minv=_mm256_set1_pd(mincost),lv=_mm256_set1_pd(-lam);
        for(;j+4<=m;j+=4) {
            __m256d e=_ZGVdN4v_exp(_mm256_div_pd(_mm256_sub_pd(_mm256_loadu_pd(out+j),minv),lv));
            _mm256_storeu_pd(out+j,e);sumv=_mm256_add_pd(sumv,e);
        }
        double sv[4];_mm256_storeu_pd(sv,sumv);sum=(sv[0]+sv[1])+(sv[2]+sv[3]);
    }
#endif
    for(;j<m;++j) {out[j]=exp(-(out[j]-mincost)/lam);sum+=out[j];}
    double inv=1./sum;
    if(nextstate) {
        double v0=0.,v1=0.;
        #pragma omp simd reduction(+:v0,v1)
        for(j=0;j<m;++j) {
            double p=out[j]*inv;out[j]=p;den[j]+=p;
            v0+=p*rt[j];v1+=p*rt[m+j];
        }
        nextv[0]=v0;nextv[1]=v1;
    } else {
        #pragma omp simd
        for(j=0;j<m;++j) out[j]*=inv;
    }
    return mincost-lam*log(sum);
}

static inline void som_add_row(const double *restrict p,double *restrict den,ptrdiff_t m) {
    #pragma omp simd
    for(ptrdiff_t j=0;j<m;++j) den[j]+=p[j];
}
#endif
