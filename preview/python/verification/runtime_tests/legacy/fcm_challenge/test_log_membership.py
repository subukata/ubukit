from decimal import Decimal, localcontext
import numpy as np
import pytest
from numpy.testing import assert_allclose
from ubukit_fcm import memberships_from_squared_distances
from ubukit_fcm.log_membership import (membership_logsoftmax, membership_logsumexp,
    membership_scipy_softmax, membership_naive_logsoftmax)
METHODS=[membership_logsoftmax,membership_logsumexp,membership_scipy_softmax]

@pytest.mark.parametrize('method',METHODS)
@pytest.mark.parametrize('m',[1.000001,1.3,2.,3.,1000.])
def test_log_versions_match_inverse(method,m):
    rng=np.random.default_rng(38)
    q=rng.random((100,8))*10
    assert_allclose(method(q,m),memberships_from_squared_distances(q,m),atol=2e-15,rtol=1e-10)
    q=np.array([[0.,0.,9.],[1e-300,1e300,1e-300]])
    assert_allclose(method(q,m),memberships_from_squared_distances(q,m),atol=3e-14,rtol=1e-10)


def decimal_oracle(q,m):
    with localcontext() as ctx:
        ctx.prec=90
        vals=[(-(Decimal.from_float(float(v)).ln())/(Decimal.from_float(float(m))-1)).exp() for v in q]
        total=sum(vals)
        return np.array([float(v/total) for v in vals])


def test_near_one_near_tie_requires_log1p():
    q=np.array([[1e300,np.nextafter(1e300,np.inf)]])
    m=np.nextafter(1.,2.)
    # Centering first also prevents Decimal exp underflow at huge logits.
    with localcontext() as ctx:
        ctx.prec=90
        ratio=Decimal.from_float(float(q[0,1]))/Decimal.from_float(float(q[0,0]))
        weight=(-ratio.ln()/(Decimal.from_float(float(m))-1)).exp()
        expected=np.array([[float(1/(1+weight)),float(weight/(1+weight))]])
    for method in METHODS+[memberships_from_squared_distances]:
        assert_allclose(method(q,m),expected,atol=2e-15,rtol=2e-15)
    assert np.max(abs(membership_naive_logsoftmax(q,m)-expected)) > .1


@pytest.mark.parametrize('parallel',[False,True])
def test_numba_near_one_near_tie(parallel):
    pytest.importorskip('numba')
    from ubukit_fcm import _numba
    x=np.array([[0.]])
    centers=np.array([[1e150],[np.nextafter(1e150,np.inf)]])
    q=(x[:,None,:]-centers[None,:,:])[:,:,0]**2
    m=np.nextafter(1.,2.)
    expected=membership_logsoftmax(q,m)
    out=np.empty((1,2))
    previous=np.full((1,2),.5)
    old_threads=_numba.set_threads(1)
    try:
        fn=_numba.update_parallel if parallel else _numba.update_serial
        fn(x,centers,previous,out,m)
    finally:
        _numba.set_threads(old_threads)
    assert_allclose(out,expected,atol=2e-15,rtol=2e-15)
