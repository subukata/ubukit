import numpy as np
import pytest
import ubukit._impl.rough_cmeans as rough
from scipy.spatial.distance import cdist
import importlib.util,sys
from pathlib import Path
from .._support import rough_oracle as frozen
from numpy.testing import assert_array_equal

@pytest.mark.parametrize('d',[1,2,8,17,64,128,784])
@pytest.mark.parametrize('p',[1.,2.,3.,.001,1000.])
def test_certified_rough_blas_masks_across_dimensions(d,p):
 rng=np.random.default_rng(800+d);x=rng.normal(size=(37,d));c=rng.normal(size=(7,d));x[:2]=c[:2]
 for alpha,beta in [(1.,0.),(1.1,.1),(1.5,10.)]:
  a=rough.assign(x,c,alpha=alpha,beta=beta,p=p,backend='scipy',block_size=13);b=rough.assign(x,c,alpha=alpha,beta=beta,p=p,backend='blas',block_size=13)
  for av,bv in zip(a,b):assert_array_equal(av,bv)

@pytest.mark.parametrize('p',[1.,2.])
@pytest.mark.parametrize('scale,offset',[(1.,0.),(1e-140,0.),(1e140,0.),(1e90,1e100)])
def test_certified_blas_canonical_boundary_and_translation(p,scale,offset):
 rng=np.random.default_rng(801);x=offset+rng.normal(size=(1,64))*scale;c=offset+rng.normal(size=(9,64))*scale;q=cdist(x,c,'sqeuclidean');minimum=q.min();alpha=1.
 for target in q[0]:
  beta=(np.sqrt(target)-np.sqrt(minimum)) if p==1 else np.sqrt(max(0.,target-minimum))
  lo=hi=beta;values=[beta]
  for i in range(16):
   lo=np.nextafter(lo,0.);hi=np.nextafter(hi,np.inf);values.extend([lo,hi])
  for beta in values:
   a=rough.assign(x,c,alpha=alpha,beta=beta,p=p,backend='scipy');b=rough.assign(x,c,alpha=alpha,beta=beta,p=p,backend='blas')
   for av,bv in zip(a,b):assert_array_equal(av,bv)

@pytest.mark.parametrize('d',[8,64,128])
@pytest.mark.parametrize('p',[1.,2.,3.])
def test_certified_blas_full_fit_exact_updates(d,p):
 rng=np.random.default_rng(802+d);x=rng.normal(size=(133,d));kw=dict(init=x[:7].copy(),alpha=1.1,beta=.1,p=p,max_iter=12,block_size=53)
 a=rough.fit(x,7,backend='scipy',**kw);b=rough.fit(x,7,backend='blas',**kw)
 for field in ['centers','memberships','upper_memberships']:assert_array_equal(getattr(a,field),getattr(b,field))
 for field in ['n_iter','converged','stop_reason','cycle_length','empty_cluster_updates','fixed_point']:assert getattr(a,field)==getattr(b,field)

@pytest.mark.parametrize('backend',['scipy','blas'])
@pytest.mark.parametrize('scale',[1e-160,1e200])
def test_certified_blas_preserves_range_errors(backend,scale):
 x=np.array([[0.,0.],[scale,0.]]);c=np.array([[0.,0.]])
 with pytest.raises(FloatingPointError):rough.assign(x,c,alpha=1.1,beta=1.,p=2.,backend=backend)

@pytest.mark.parametrize('alpha,beta',[(1e200,0.),(1.1,1e-200),(1.1,1e200)])
def test_certified_blas_exceptional_thresholds(alpha,beta):
 x=np.array([[0.,0.],[1.,2.],[3.,4.]]);c=np.array([[0.,0.],[2.,2.]])
 for p in [1.,2.]:
  a=rough.assign(x,c,alpha=alpha,beta=beta,p=p,backend='scipy');b=rough.assign(x,c,alpha=alpha,beta=beta,p=p,backend='blas')
  for av,bv in zip(a,b):assert_array_equal(av,bv)

@pytest.mark.parametrize('backend',['scipy','numpy','blas'])
@pytest.mark.parametrize('p',[1.,2.,3.,.001,1000.])
def test_coincident_center_shortcut_exact_full_masks(backend,p):
 rng=np.random.default_rng(870);x=rng.normal(size=(1200,32));c=np.repeat(x[:1],8,axis=0)
 a=rough.assign(x,c,p=p,alpha=1.,beta=0.,backend='naive',block_size=127);b=rough.assign(x,c,p=p,alpha=1.,beta=0.,backend=backend,block_size=127)
 for av,bv in zip(a,b):assert_array_equal(av,bv)
 kw=dict(init=c,p=p,alpha=1.1,beta=.1,max_iter=6,block_size=127)
 a=frozen.fit(x,8,backend='scipy' if backend=='blas' else backend,**kw);b=rough.fit(x,8,backend=backend,**kw)
 assert_array_equal(a.upper_memberships,b.upper_memberships);assert_array_equal(a.memberships,b.memberships);assert a.n_iter==b.n_iter
 assert_array_equal(a.centers,b.centers)

@pytest.mark.parametrize('d',[1,2,17,128])
@pytest.mark.parametrize('scale,offset',[(1.,0.),(1e-140,0.),(1e140,0.),(1e90,1e100)])
def test_blas_distance_interval_covers_decimal_and_direct(d,scale,offset):
 from decimal import Decimal,localcontext
 rng=np.random.default_rng(891+d);x=offset+scale*rng.normal(size=(3,d));c=offset+scale*rng.normal(size=(4,d));low,high=rough._blas_squared_distance_bounds(x,c);direct=cdist(x,c,'sqeuclidean')
 assert np.all(low <= direct);assert np.all(direct <= high)
 with localcontext() as ctx:
  ctx.prec=120
  for i in range(len(x)):
   for j in range(len(c)):
    exact=sum((Decimal.from_float(float(x[i,f]))-Decimal.from_float(float(c[j,f])))**2 for f in range(d))
    assert Decimal.from_float(float(low[i,j])) <= exact <= Decimal.from_float(float(high[i,j]))


def test_certified_blas_audited_shape_and_unknown_provider_fallback(monkeypatch):
 rng=np.random.default_rng(1009);x=rng.normal(size=(512,256));c=rng.normal(size=(8,256));stats={}
 M=rough._certified_blas_mask(x,c,1.1,.1,2.,stats)
 exact=rough.assign(x,c,alpha=1.1,beta=.1,p=2.,backend='scipy')[1].T
 assert_array_equal(M,exact)
 if rough._blas_certificate_supported(x,c):
  assert stats['certificate_enabled'];assert stats['direct_rows'] < len(x)
 monkeypatch.setattr(rough,'_BLAS_CERTIFICATE_BUILD',False);stats={}
 assert_array_equal(rough._certified_blas_mask(x,c,1.1,.1,2.,stats),exact)
 assert not stats['certificate_enabled'];assert stats['direct_rows']==len(x)


def test_certified_blas_rejects_small_and_alias_dispatch():
 x=np.ones((512,256));c=np.ones((8,256))
 assert not rough._blas_certificate_supported(x[:1],c)
 assert not rough._blas_certificate_supported(x,c[:1])
 assert not rough._blas_certificate_supported(x[:32],c)
 assert not rough._blas_certificate_supported(x,x[:8])


def test_certified_blas_large_full_fit_exact():
 rng=np.random.default_rng(1010);x=rng.normal(size=(1200,128));kw=dict(init=x[:16].copy(),alpha=1.1,beta=.1,p=2.,max_iter=6)
 a=frozen.fit(x,16,backend='scipy',**kw);b=rough.fit(x,16,backend='blas',**kw)
 for field in ['centers','memberships','upper_memberships']:assert_array_equal(getattr(a,field),getattr(b,field))
 for field in ['n_iter','converged','stop_reason','cycle_length','empty_cluster_updates','fixed_point']:assert getattr(a,field)==getattr(b,field)

@pytest.mark.parametrize('p',[1.,2.])
def test_active_certificate_boundary_rows_are_recomputed(p):
 rng=np.random.default_rng(1011);x=rng.normal(size=(512,256));c=rng.normal(size=(8,256))
 if not rough._blas_certificate_supported(x,c):pytest.skip('requires audited NumPy/OpenBLAS build')
 q=cdist(x[:1],c,'sqeuclidean')[0];minimum=q.min();target=q.max()
 beta=np.sqrt(target)-np.sqrt(minimum) if p==1 else np.sqrt(target-minimum)
 low=high=beta;values=[beta]
 for _ in range(8):low=np.nextafter(low,0.);high=np.nextafter(high,np.inf);values.extend([low,high])
 for beta in values:
  stats={};actual=rough._certified_blas_mask(x,c,1.,beta,p,stats)
  expected=frozen.assign(x,c,alpha=1.,beta=beta,p=p,backend='scipy')[1].T
  assert_array_equal(actual,expected);assert stats['certificate_enabled']
  assert 0 < stats['direct_rows'] < stats['total_rows']


def test_active_gemm_interval_decimal120_oracle():
 from decimal import Decimal,localcontext
 rng=np.random.default_rng(1012);x=rng.normal(size=(512,256));c=rng.normal(size=(8,256))
 if not rough._blas_certificate_supported(x,c):pytest.skip('requires audited NumPy/OpenBLAS build')
 low,high=rough._blas_squared_distance_bounds(x,c);direct=cdist(x,c,'sqeuclidean')
 assert np.all(low<=direct);assert np.all(direct<=high)
 with localcontext() as ctx:
  ctx.prec=120
  for i in [0,1,511]:
   for j in [0,3,7]:
    exact=sum((Decimal.from_float(float(x[i,f]))-Decimal.from_float(float(c[j,f])))**2 for f in range(256))
    assert Decimal.from_float(float(low[i,j]))<=exact<=Decimal.from_float(float(high[i,j]))
