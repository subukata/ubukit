"""New lazy facade aliases and explicitly observed compatibility routes."""
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import ubukit._impl.external_metrics as em
import ubukit

AVERAGES=('arithmetic','geometric','min','max')
NAMES=('adjusted_rand_score','adjusted_mutual_info_score','adjusted_scores')
HAS_NUMBA=importlib.util.find_spec('numba') is not None
BACKENDS=['numpy']+(['numba'] if HAS_NUMBA else [])


def isolated(code):
    guard=Path(__file__).with_name('_installed_guard.py')
    pre=("import importlib.util,os;"
         f"_s=importlib.util.spec_from_file_location('_installed_guard',{str(guard)!r});"
         "_g=importlib.util.module_from_spec(_s);_s.loader.exec_module(_g);"
         "_g.check_installed(os.environ['UBUKIT_DISTRIBUTION'],True)\n")
    post="\n_g.check_installed(os.environ['UBUKIT_DISTRIBUTION'],True)\n"
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,'-I','-B','-c',pre+code+post],
                          env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    return result.stdout


def test_lazy_facade_and_direct_alias_identity():
    isolated(r'''
import inspect,sys
import ubukit
assert not any(n in sys.modules for n in ('numpy','scipy','sklearn','numba','llvmlite','ubukit._impl.external_metrics','ubukit._impl._external_metrics_numba'))
names=('adjusted_rand_score','adjusted_mutual_info_score','adjusted_scores')
assert all(name in ubukit.__all__ and name in dir(ubukit) for name in names)
for name in names:
    assert ubukit._EXPORTS[name]==('._impl.external_metrics',name)
    assert name not in ubukit._ADAPTED_EXPORTS
    value=getattr(ubukit,name)
    import ubukit._impl.external_metrics
    assert value is getattr(ubukit._impl.external_metrics,name)
    assert inspect.signature(value)==inspect.signature(getattr(ubukit._impl.external_metrics,name))
assert not any(n.split('.')[0] in ('scipy','sklearn','numba','llvmlite') for n in sys.modules)
assert 'ubukit._impl._external_metrics_numba' not in sys.modules
''')


@pytest.mark.parametrize('average',AVERAGES)
@pytest.mark.parametrize('backend',BACKENDS)
def test_facade_all_averages_backend_identity_and_input_ownership(average,backend):
    rng=np.random.default_rng(12062026)
    x=rng.integers(5,size=74)[::2];y=rng.integers(7,size=74)[::-2]
    before_x=x.copy();before_y=y.copy()
    for name in NAMES:assert getattr(ubukit,name) is getattr(em,name)
    both=ubukit.adjusted_scores(x,y,average_method=average,backend=backend)
    assert both==em.adjusted_scores(x,y,average_method=average,backend=backend)
    assert both['ari']==ubukit.adjusted_rand_score(x,y)
    assert both['ami']==ubukit.adjusted_mutual_info_score(x,y,average_method=average,backend=backend)
    np.testing.assert_array_equal(x,before_x);np.testing.assert_array_equal(y,before_y)


def test_joint_shares_only_within_one_call_and_recomputes_after_mutation():
    x=np.array([0,0,1,1,2,2,3,3]);y=np.array([0,1,0,1,0,1,0,1])
    with patch.object(em,'_contingency',wraps=em._contingency) as contingency:
        before=ubukit.adjusted_scores(x,y)
        assert contingency.call_count==1
        y[:]=x
        after=ubukit.adjusted_scores(x,y)
        assert contingency.call_count==2
    assert before!=after
    assert after=={'ari':1.,'ami':1.}


@pytest.mark.parametrize('average',AVERAGES)
@pytest.mark.parametrize('case',['singular_singleton','near_singleton','high_cluster_density'])
def test_real_compatibility_fallback_calls_sklearn_exactly(average,case):
    from sklearn.metrics import adjusted_mutual_info_score as sklearn_ami
    if case=='singular_singleton':
        x=np.arange(8);y=np.repeat(np.arange(4),2)
    elif case=='near_singleton':
        x=np.arange(1000);y=np.arange(1000);x[1]=0;y[3]=2
    else:
        x=np.arange(512)//2;y=np.roll(x,1)
    assert em._ami(*em._contingency(x,y),average,'numpy') is None
    reference=sklearn_ami(x,y,average_method=average)
    with patch('sklearn.metrics.adjusted_mutual_info_score',wraps=sklearn_ami) as fallback:
        score=ubukit.adjusted_mutual_info_score(x,y,average_method=average)
        assert score==reference
        fallback.assert_called_once()
        args,kwargs=fallback.call_args
        assert args[0] is x and args[1] is y
        assert kwargs=={'average_method':average}


@pytest.mark.parametrize('average',AVERAGES)
def test_forced_cancellation_guard_keeps_compatibility_definition(average):
    from sklearn.metrics import adjusted_mutual_info_score as sklearn_ami
    x=np.arange(64)%4;y=(np.arange(64)//4)%4
    expected=sklearn_ami(x,y,average_method=average)
    # A controlled expectation equal to entropy forces the actual near-zero
    # denominator branch; this is route testing, not a numerical oracle.
    entropy=em._entropy(np.array([16,16,16,16]),len(x))
    with patch.object(em,'_emi',return_value=entropy),patch('sklearn.metrics.adjusted_mutual_info_score',wraps=sklearn_ami) as fallback:
        assert ubukit.adjusted_mutual_info_score(x,y,average_method=average)==expected
        fallback.assert_called_once()


@pytest.mark.parametrize('average',AVERAGES)
def test_ordinary_path_does_not_call_compatibility_fallback(average):
    rng=np.random.default_rng(62026);x=rng.integers(6,size=137);y=rng.integers(8,size=137)
    with patch('sklearn.metrics.adjusted_mutual_info_score',side_effect=AssertionError('unexpected fallback')):
        got=ubukit.adjusted_scores(x,y,average_method=average)
    assert np.isfinite(got['ari']) and np.isfinite(got['ami'])


def test_explicit_numba_module_load_is_lazy_and_dependency_failure_is_explicit():
    isolated(r'''
import importlib.abc,sys
class BlockNumba(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('numba','llvmlite'):
            raise ImportError('optional Numba intentionally unavailable')
sys.meta_path.insert(0,BlockNumba())
import ubukit
x=[0,0,1,1,2,2,3,3];y=[0,1,0,1,0,1,0,1]
assert ubukit.adjusted_scores(x,y)['ari']==ubukit.adjusted_rand_score(x,y)
assert 'ubukit._impl._external_metrics_numba' not in sys.modules
for function in (ubukit.adjusted_mutual_info_score,ubukit.adjusted_scores):
    try:function(x,y,backend='numba')
    except ImportError:pass
    else:raise AssertionError('nontrivial explicit Numba request must report missing dependency')
assert not any(n.split('.')[0] in ('numba','llvmlite') for n in sys.modules)
''')
