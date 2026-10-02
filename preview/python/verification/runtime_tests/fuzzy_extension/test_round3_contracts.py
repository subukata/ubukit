"""Independent/additional regressions for acceleration round 3."""
from decimal import Decimal,localcontext
import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose,assert_array_equal
from scipy.spatial.distance import cdist
import ubukit._impl.fcm.core as candidate
import ubukit._impl.rough_cmeans as rough
from .._support import fcm_oracle as reference, rough_oracle as rough_reference
def decimal_membership(q,m):
 with localcontext() as ctx:
  ctx.prec=100;qs=[Decimal.from_float(float(v)) for v in q];dm=Decimal.from_float(float(m));minimum=min(qs)
  if minimum==0:
   zeros=[q==0 for q in qs];return np.array([float(Decimal(int(v))/sum(zeros)) for v in zeros])
  ws=[(-(v/minimum).ln()/(dm-1)).exp() for v in qs];total=sum(ws);return np.array([float(v/total) for v in ws])

@pytest.mark.parametrize('m',[np.nextafter(1.,2.),1+1e-12,1+1e-8,1.000099,1.3,1.5,2.,3.,1000.])
def test_membership_wide_exponents(m):
 rng=np.random.default_rng(71);q=np.exp(rng.uniform(-700,700,(40,9)));q[0,:2]=0
 a=reference.memberships_from_squared_distances(q,m);b=candidate.memberships_from_squared_distances(q,m)
 assert_allclose(b,a,atol=3e-16,rtol=2e-13)
 for row in [0,3,8,17]:assert_allclose(b[row],decimal_membership(q[row],m),atol=2e-14,rtol=3e-12)

@pytest.mark.parametrize('m',[np.nextafter(1.,2.),1+1e-12,1+1e-8,1.000099])
@pytest.mark.parametrize('scale',[1e-300,1.,1e300])
def test_near_one_cutoff_preserves_all_representable_weights(m,scale):
 delta=m-1
 ratios=np.array([0,128,512,700,745,800,1000,1024,1024*(1-1e-12),1024*(1+1e-12),2000])*delta
 q=(scale*(1+ratios))[None,:]
 a=reference.memberships_from_squared_distances(q,m);b=candidate.memberships_from_squared_distances(q,m)
 assert_array_equal(a,b)
 assert_allclose(b[0],decimal_membership(q[0],m),atol=2e-15,rtol=5e-12)

@pytest.mark.parametrize('m',[1.000001,1.3,2.,3.])
def test_empty_and_aliased_membership_output(m):
 q=np.empty((0,4));out=np.empty_like(q);assert candidate.memberships_from_squared_distances(q,m,out=out) is out
 q=np.arange(1.,33.).reshape(8,4)[::2];expected=reference.memberships_from_squared_distances(q,m)
 assert candidate.memberships_from_squared_distances(q,m,out=q) is q
 assert_allclose(q,expected,rtol=2e-15,atol=2e-15)

@pytest.mark.parametrize('scale',[1e-310,np.finfo(float).tiny*4,1.,1e307])
def test_reciprocal_guard_extremes(scale):
 q=np.array([[scale,scale*2,scale*3]])
 minimum=candidate._row_minimum(q);out=np.empty_like(q)
 got=candidate._memberships(q,2.,out,minimum=minimum,bounds=(q.min(),q.max()),row_ones=np.ones(3),denominator=np.empty(1))
 assert_allclose(got,reference.memberships_from_squared_distances(q,2),atol=3e-16,rtol=3e-15)
 assert np.isfinite(got).all()

@pytest.mark.parametrize('backend',['numpy','scipy','blas'])
def test_grouped_history_survives_many_chunks(monkeypatch,backend):
 monkeypatch.setattr(candidate,'_MEMBERSHIP_BLOCK_ROWS',17)
 x=np.tile([[0.],[2.]],(100,1));u=np.full((len(x),2),.5)
 got=candidate.fit_fcm(x,init=u,m=1080,max_iter=1,backend=backend,return_history=True)
 expected=float(np.longdouble(400)*np.longdouble(.5)**1080)
 assert abs(got['objective']-expected)<=np.nextafter(0.,1.)
 assert abs(got['objective_history'][0]-expected)<=np.nextafter(0.,1.)

@pytest.mark.parametrize('backend',['scipy','blas'])
@pytest.mark.parametrize('m',[1.01,1.3,1.5,2.,3.,5.])
def test_multi_seed_convergence_and_returned_objective(backend,m):
 for seed in range(3):
  rng=np.random.default_rng(70+seed);n,d,k=400,4,3;truth=np.arange(n)%k;x=rng.normal(size=(k,d))[truth]*3+rng.normal(size=(n,d))*.5;u=rng.random((n,k));before=x.copy()
  a=reference.fit_fcm(x,init=u,m=m,backend=backend,max_iter=80,tol=1e-6)
  b=candidate.fit_fcm(x,init=u,m=m,backend=backend,max_iter=80,tol=1e-6)
  assert b['n_iter']==a['n_iter'];assert b['converged']==a['converged']
  assert_allclose(a['membership'],b['membership'],atol=1e-10,rtol=1e-8)
  assert_allclose(a['centers'],b['centers'],atol=1e-10,rtol=1e-10)
  assert_allclose(b['objective'],np.sum(b['membership']**m*cdist(x,b['centers'],'sqeuclidean')),rtol=3e-14,atol=1e-13)
  assert_array_equal(x,before)

@pytest.mark.parametrize('parallel',[False,True])
def test_new_numba_fused_near_one_and_nonmutating_previous(parallel):
 pytest.importorskip('numba');from ubukit._impl.fcm import _numba
 x=np.array([[0.]]);centers=np.array([[1e150],[np.nextafter(1e150,np.inf)]]);q=cdist(x,centers,'sqeuclidean');m=np.nextafter(1.,2.);previous=np.full((1,2),.5);before=previous.copy();out=np.empty_like(previous)
 fn=_numba.update_membership_parallel if parallel else _numba.update_membership_serial
 delta=fn(x,centers,previous,out,m)
 assert_allclose(out[0],decimal_membership(q[0],m),rtol=2e-15,atol=2e-15);assert_array_equal(previous,before);assert_allclose(delta,np.sum((out-previous)**2),rtol=2e-15)

@pytest.mark.parametrize('p',[.25,.3,.5,1.,2.,3.,8.,64.,128.,1000.])
@pytest.mark.parametrize('scale',[1e-140,1.,1e140])
def test_rough_boundary_guard_and_scale(p,scale):
 alpha,beta=1.2,.7*scale
 radius=scale*(1.2**p+.7**p)**(1/p) if p<128 else 1.2*scale
 vals=[radius];lo=hi=radius
 for _ in range(50):lo=np.nextafter(lo,0);hi=np.nextafter(hi,np.inf);vals.extend([lo,hi])
 sq=np.array([[scale*scale,v*v] for v in vals])
 got=rough._mask_from_sq(sq,alpha,beta,p);old=rough_reference._mask_from_sq(sq,alpha,beta,p);canonical=rough_reference._mask_from_sq(sq,alpha,beta,p,optimized=False)
 assert_array_equal(got,old);assert_array_equal(got,canonical)

@pytest.mark.parametrize('p',[.3,.5,1.,2.,3.,8.,64.,.001,1000.])
def test_rough_numba_radius_filter_retains_canonical_boundary(p):
 pytest.importorskip('numba')
 radius=(1.2**p+.7**p)**(1/p) if .25<=p<=64 else 1.2
 vals=[1.,radius];lo=hi=radius
 for _ in range(25):lo=np.nextafter(lo,0);hi=np.nextafter(hi,np.inf);vals.extend([lo,hi])
 X=np.array([[0.]]);C=np.array(vals)[:,None]
 a=rough_reference.assign(X,C,alpha=1.2,beta=.7,p=p,backend='scipy');b=rough.assign(X,C,alpha=1.2,beta=.7,p=p,backend='numba')
 assert_array_equal(a[1],b[1]);assert_array_equal(a[0],b[0])

@pytest.mark.parametrize('p',[1e-300,1e-17,.0001,.001,.05,.1,.249999])
@pytest.mark.parametrize('k',[4,8,32])
def test_rough_small_p_extrema_certificate(p,k):
 rng=np.random.default_rng(k);sq=np.exp(rng.uniform(-700,700,(45,k)));sq[:3,0]=0
 for alpha,beta in [(1.,1e-320),(1.1,1.),(1e200,1e-140)]:
  got=rough._mask_from_sq(sq,alpha,beta,p);expected=rough_reference._mask_from_sq(sq,alpha,beta,p,optimized=False)
  assert_array_equal(got,expected)

def test_cached_thread_controller_reads_and_restores_limits(monkeypatch):
 from threadpoolctl import threadpool_info
 rng=np.random.default_rng(17);x=rng.normal(size=(70,3));u=rng.random((70,4))
 def states():return {(p['filepath'],p['user_api']):p['num_threads'] for p in threadpool_info() if p['user_api']=='blas'}
 candidate.fit_fcm(x,init=u,max_iter=1,threads=1)
 before=states();actual=candidate._distance;seen=[]
 def checked(*args,**kwargs):
  seen.append(states());assert all(t==1 for t in states().values());return actual(*args,**kwargs)
 monkeypatch.setattr(candidate,'_distance',checked)
 candidate.fit_fcm(x,init=u,max_iter=2,threads=1)
 assert seen;assert states()==before
 def fail(*args,**kwargs):raise RuntimeError('injected failure')
 monkeypatch.setattr(candidate,'_distance',fail)
 with pytest.raises(RuntimeError,match='injected failure'):candidate.fit_fcm(x,init=u,max_iter=1,threads=1)
 assert states()==before

@pytest.mark.parametrize('scale',[1e-140,1.,1e140])
def test_one_dimensional_direct_kernel_matches_cdist(scale):
 rng=np.random.default_rng(91);x=rng.normal(size=(70,1))*scale;c=x[:6]
 for backend in ['numpy','scipy','blas']:
  got=candidate._distance(x,c,backend,np.einsum('ij,ij->i',x,x),np.empty((70,6)))
  assert_array_equal(got,cdist(x,c,'sqeuclidean'))
 assert_array_equal(rough._distances_numpy(x,c),cdist(x,c,'sqeuclidean'))

@pytest.mark.parametrize('d',[16,17,32,128,784])
@pytest.mark.parametrize('k',[4,5,8,10])
def test_wide_numba_distances_preserve_point_arithmetic(d,k):
 pytest.importorskip('numba');from ubukit._impl.fcm import _numba
 rng=np.random.default_rng(d+k);x=rng.normal(size=(3,d));centers=rng.normal(size=(k,d));centers[0]=x[0];previous=rng.random((3,k));previous/=previous.sum(1)[:,None]
 for m in [1+1e-9,1.3,2.,3.5,1000.]:
  a=np.empty_like(previous);b=np.empty_like(previous)
  for i in range(3):
   da=_numba._point_membership(x,centers,previous,a,m,i);db=_numba._point_membership_wide(x,centers,previous,b,m,i)
   assert da==db
  assert_array_equal(a,b)

@pytest.mark.parametrize('d',[16,17,128,784])
@pytest.mark.parametrize('p',[.3,1.,2.,3.,1000.])
def test_wide_rough_numba_preserves_threshold_boundaries(d,p):
 pytest.importorskip('numba')
 radius=(1.2**p+.7**p)**(1/p) if .25<=p<=64 else 1.2
 values=[1.,radius];lo=hi=radius
 for _ in range(12):lo=np.nextafter(lo,0);hi=np.nextafter(hi,np.inf);values.extend([lo,hi])
 X=np.zeros((1,d));C=np.zeros((len(values),d));C[:,0]=values
 a=rough_reference.assign(X,C,alpha=1.2,beta=.7,p=p,backend='scipy');b=rough.assign(X,C,alpha=1.2,beta=.7,p=p,backend='numba')
 assert_array_equal(a[1],b[1]);assert_array_equal(a[0],b[0])

@pytest.mark.parametrize('backend',['numpy','scipy','numba'])
@pytest.mark.parametrize('d',[1,8,32,128,784])
def test_rough_assignment_only_matches_full_step_outputs(backend,d):
 if backend=='numba':pytest.importorskip('numba')
 rng=np.random.default_rng(783+d);X=rng.normal(size=(23,d));C=X[[0,2,6,11]].copy()
 for p in [1.,2.,3.,.001,1000.]:
  U,M=rough.assign(X,C,alpha=1.1,beta=.1,p=p,backend=backend,block_size=7)
  _,_,_,expectedU,expectedM=rough._step(X,C,1.1,.1,p,backend,7,True)
  assert_array_equal(U,expectedU);assert_array_equal(M,expectedM)

@pytest.mark.parametrize('m',[np.nextafter(1.,2.),1+1e-12,1+1e-8,1.000099])
@pytest.mark.parametrize('parallel',[False,True])
def test_numba_near_one_cutoff_matches_original_kernel(m,parallel):
 pytest.importorskip('numba');from ubukit._impl.fcm import _numba
 q=1+np.array([0.,128,512,700,745,800,1000,1024,2000])*(m-1);centers=np.sqrt(q)[:,None];x=np.zeros((3,1));u=np.full((3,len(q)),1/len(q));a=np.empty_like(u);b=np.empty_like(u)
 old=_numba.update_parallel if parallel else _numba.update_serial
 new=_numba.update_membership_parallel if parallel else _numba.update_membership_serial
 old(x,centers,u,a,m);new(x,centers,u,b,m);assert_array_equal(a,b)

def test_empty_read_only_membership_output_keeps_existing_error():
 q=np.empty((0,3));out=np.empty_like(q);out.flags.writeable=False
 with pytest.raises(ValueError,match='read-only'):reference.memberships_from_squared_distances(q,2.,out)
 with pytest.raises(ValueError,match='read-only'):candidate.memberships_from_squared_distances(q,2.,out)

@pytest.mark.parametrize('m',[1.0001,1.01,1.3,1.5,3.,5.,1000.,1e100])
@pytest.mark.parametrize('scale',[1e-300,1e-140,1.,1e140,1e300])
def test_extension_bounded_raw_power_matches_decimal(m,scale):
 q=scale*np.array([[1.,1.25,2.,3.],[1.,np.nextafter(1.,2.),2.,4.]])
 got=candidate._memberships(q,m,bounds=(q.min(),q.max()))
 for i in range(len(q)):
  assert_allclose(got[i],decimal_membership(q[i],m),atol=3e-14,rtol=3e-12)

@pytest.mark.parametrize('m',[3.1,5.,16.,64.,65.,1000.,1e4,1e8,1e16,1e100,1e308])
def test_extension_huge_fuzzifier_retains_scaled_update_bits(m):
 rng=np.random.default_rng(977);q=np.exp(rng.uniform(-30,30,(27,8)))
 got=candidate._memberships(q,m,bounds=(q.min(),q.max()))
 assert_array_equal(got,reference.memberships_from_squared_distances(q,m))

@pytest.mark.parametrize('m',[5.,16.,64.,1000.])
@pytest.mark.parametrize('backend',['scipy','blas'])
def test_full_fit_retains_ordinary_bits_and_robust_pair_contract(m,backend):
 rng=np.random.default_rng(994);x=rng.normal(size=(113,17));u=rng.random((113,7));kw=dict(init=u,m=m,backend=backend,max_iter=6,tol=0)
 try:a=reference.fit_fcm(x,**kw)
 except (ValueError,FloatingPointError) as exc:
  with pytest.raises(type(exc),match=str(exc)):candidate.fit_fcm(x,**kw)
  return
 b=candidate.fit_fcm(x,**kw)
 if m > 32:
  # The finite-m log-weight path intentionally replaces high-m bit-checkpoint
  # preservation. Near-coincident centers can change memberships materially.
  # Verify the returned pair against an independent Decimal membership oracle;
  # full-trajectory accuracy is covered/reported separately, not assumed here.
  assert b['numerical_diagnostics']['arithmetic']=='scaled_log'
  assert np.isfinite(b['centers']).all() and np.isfinite(b['membership']).all()
  q=cdist(x,b['centers'],'sqeuclidean')
  expected=np.asarray([decimal_membership(row,m) for row in q])
  assert_allclose(b['membership'],expected,rtol=3e-13,atol=3e-15)
  assert_array_equal(b['labels'],b['membership'].argmax(axis=1))
  assert b['n_iter']==6 and not b['converged']
  previous=candidate.fit_fcm(x,**dict(kw,max_iter=5))
  assert_allclose(b['delta'],np.linalg.norm(b['membership']-previous['membership']),rtol=3e-14,atol=5e-16)
  with localcontext() as context:
   context.prec=100
   exact_objective=sum(Decimal(float(v))**Decimal(m)*Decimal(float(distance)) for v,distance in zip(b['membership'].ravel(),q.ravel()))
   assert_allclose(b['objective'],float(exact_objective),rtol=3e-13,atol=8*np.nextafter(0.,1.))
   if exact_objective:
    assert_allclose(b['numerical_diagnostics']['log_objective'],float(exact_objective.ln()),rtol=3e-14,atol=1e-12)
  return
 else:
  for key in ['centers','membership','labels']:assert_array_equal(a[key],b[key])
 assert a['delta']==b['delta'];assert a['n_iter']==b['n_iter']
 assert_allclose(a['objective'],b['objective'],rtol=3e-13,atol=8*np.nextafter(0.,1.))
