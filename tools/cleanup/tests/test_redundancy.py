"""Bitwise comparisons against immutable installed-alpha3 source; no benchmarks."""
import unittest
raise unittest.SkipTest("Historical exact-EFCM cleanup qualification is superseded by the float64-only policy; use the focused EFCM tests instead.")
import ast
import importlib.util
import importlib
import inspect
import math
from pathlib import Path
import struct
import sys
import unittest
import weakref
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load_package(name, directory):
    spec = importlib.util.spec_from_file_location(name, directory/'ubukit/__init__.py',
                                                submodule_search_locations=[str(directory/'ubukit')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return importlib.import_module(name+'._impl.entropy_fcm')

baseline = load_package('_efcm_baseline', ROOT/'baseline')
candidate = load_package('_efcm_candidate', ROOT/'candidate')
EVIDENCE = {}

class RedundancyRegressionTests(unittest.TestCase):
    def assert_bits(self, actual, expected, path='result'):
        self.assertEqual(type(actual), type(expected), path)
        if isinstance(actual, dict):
            self.assertEqual(list(actual), list(expected), path)
            for key in actual:
                self.assert_bits(actual[key], expected[key], path+'.'+key)
        elif isinstance(actual, np.ndarray):
            self.assertEqual(actual.shape, expected.shape, path)
            self.assertEqual(actual.dtype, expected.dtype, path)
            self.assertEqual(actual.tobytes(), expected.tobytes(), path)
        elif isinstance(actual, float):
            self.assertEqual(struct.pack('>d', actual), struct.pack('>d', expected), path)
        else:
            self.assertEqual(actual, expected, path)

    def compare_fit(self, x, **kwargs):
        expected = baseline.fit_entropy_fcm(x, **kwargs)
        actual = candidate.fit_entropy_fcm(x, **kwargs)
        self.assert_bits(actual, expected)
        return actual

    def test_public_signature_and_result_contract_unchanged(self):
        self.assertEqual(inspect.signature(candidate.fit_entropy_fcm), inspect.signature(baseline.fit_entropy_fcm))
        for history in (False, True):
            self.compare_fit([[0.], [1.]],  n_clusters=2, random_state=1, return_history=history)

    def test_ordinary_multiblock_results_and_history_bitwise(self):
        rng = np.random.default_rng(4401)
        for n, d, k in ((9, 3, 4), (257, 2, 3)):
            x, init = rng.normal(size=(n,d)), rng.random((n,k))
            for backend in ('numpy', 'reference'):
                for tau in (0.7, 10.):
                    with self.subTest(shape=(n,d,k), backend=backend, tau=tau):
                        self.compare_fit(x, init=init, tau=tau, backend=backend,
                                         max_iter=3, tol=0, return_history=True)

    def test_extreme_and_subnormal_results_history_bitwise(self):
        tiny = np.nextafter(0.,1.)
        cases = [([[0.],[2.]],np.eye(2),tiny),
                 ([[0.],[1e-160]],np.eye(2),1e-320),
                 ([[-1e308],[1e308]],np.eye(2),1e308),
                 ([[0.],[0.]],np.full((2,8),1/8),1e308),
                 ([[0.],[tiny]],np.ones((2,1)),1.)]
        for x, init, tau in cases:
            for backend in ('numpy','reference'):
                with self.subTest(x=x, tau=tau, backend=backend):
                    self.compare_fit(x, init=init, tau=tau, backend=backend,
                                     max_iter=3, tol=0, return_history=True)

    def test_large_common_cost_preserves_small_gap_and_fallback(self):
        x = [[-1e8,0.],[1e8,0.],[-1e8,1.],[1e8,1.]]
        init = [[1.,0.],[1.,0.],[0.,1.],[0.,1.]]
        for backend in ('numpy','reference'):
            result = self.compare_fit(x,init=init,tau=1.,backend=backend,max_iter=3,tol=0,return_history=True)
            self.assertTrue(result['numerical_diagnostics']['used_reference_fallback'])
        first = candidate.fit_entropy_fcm(x,init=init,tau=1.,max_iter=1)
        p = 1/(1+math.exp(-1))
        np.testing.assert_allclose(first['membership'],[[p,1-p],[p,1-p],[1-p,p],[1-p,p]],rtol=0,atol=2e-16)

    def test_cancelling_centers_tiny_weights_and_empty_clusters(self):
        cases=[([[-1e100],[1e100],[2.]],[[1.,1e-310],[1.,1e-310],[0.,1e-310]]),
               ([[3.,2.]]*3,[[1.,0.,0.]]*3),
               ([[-1e16,4.],[1e16,4.],[1.,4.]],[[.5,.5],[.5,.5],[.5,.5]])]
        for x,init in cases:
            for backend in ('numpy','reference'):
                self.compare_fit(x,init=init,backend=backend,max_iter=3,tol=0,return_history=True)

    def test_exact_cost_reuse_preserves_objective_pair_representation(self):
        cases=[(np.array([[-1e8,0.],[1e8,1.]]),np.array([[0.,0.],[0.,1.]]),1.),
               (np.array([[0.],[1e-160]]),np.array([[0.],[1e-160]]),1e-320),
               (np.array([[-1e308],[1e308]]),np.array([[-1e308],[1e308]]),1e308),
               (np.zeros((3,2)),np.zeros((4,2)),.7),
               (np.array([[-1.],[1.]]),np.zeros((2,1)),1/math.log(2))]
        for x,centers,tau in cases:
            old_u=baseline._reference_membership(x,centers,tau)
            old_pair=baseline._objective_pair(x,centers,old_u,tau)
            new_u,new_pair=candidate._reference_memberships_objective(x,centers,tau)
            self.assert_bits(new_u,old_u)
            self.assertEqual(new_pair,old_pair)
            self.assert_bits(candidate._objective_value(new_pair),baseline._objective_value(old_pair))

    def test_reference_cost_count_reduced_without_full_cost_cache(self):
        x=np.array([[-1.,0.],[0.,1.],[1.,0.]])
        centers=np.array([[0.,0.],[0.,.5]])
        with patch.object(baseline,'_exact_cost',wraps=baseline._exact_cost) as old:
            u=baseline._reference_membership(x,centers,2.)
            pair=baseline._objective_pair(x,centers,u,2.)
        with patch.object(candidate,'_exact_cost',wraps=candidate._exact_cost) as new:
            new_u,new_pair=candidate._reference_memberships_objective(x,centers,2.)
        self.assertEqual(old.call_count,12)
        self.assertEqual(new.call_count,6)
        self.assert_bits(new_u,u)
        self.assertEqual(new_pair,pair)
        EVIDENCE['reference_exact_cost_calls']={'baseline':old.call_count,'candidate':new.call_count,'n':3,'k':2}

    def test_numpy_guarded_block_reuses_exact_costs(self):
        x=np.array([[-1e8,0.],[1e8,1.]])
        centers=np.array([[0.,0.],[0.,1.]])
        with patch.object(baseline,'_exact_cost',wraps=baseline._exact_cost) as old:
            expected=baseline._numpy_memberships_objective(x,centers,1.)
        with patch.object(candidate,'_exact_cost',wraps=candidate._exact_cost) as new:
            actual=candidate._numpy_memberships_objective(x,centers,1.)
        self.assert_bits(actual[0],expected[0]); self.assertEqual(actual[1:],expected[1:])
        self.assertEqual(old.call_count,8); self.assertEqual(new.call_count,4)
        EVIDENCE['guarded_block_exact_cost_calls']={'baseline':old.call_count,'candidate':new.call_count,'n':2,'k':2}

    def test_feature_bounds_reused_for_every_iteration(self):
        x=np.array([[-2.,4.],[-1.,4.],[1.,4.],[2.,4.]])
        snapshots=[]
        original=candidate._numpy_centers
        def wrapped(data,u,old,feature_bounds=None):
            self.assertIsNotNone(feature_bounds)
            snapshots.append((id(feature_bounds),tuple(id(a) for a in feature_bounds),tuple(a.copy() for a in feature_bounds)))
            np.testing.assert_array_equal(feature_bounds[0],data.min(axis=0))
            np.testing.assert_array_equal(feature_bounds[1],data.max(axis=0))
            return original(data,u,old,feature_bounds)
        with patch.object(candidate,'_numpy_centers',wrapped):
            self.compare_fit(x,n_clusters=2,random_state=8,max_iter=4,tol=0,return_history=True)
        self.assertEqual(len(snapshots),4)
        self.assertTrue(all(s[:2]==snapshots[0][:2] for s in snapshots))
        for snap in snapshots[1:]:
            for actual,expected in zip(snap[2],snapshots[0][2]): self.assert_bits(actual,expected)
        EVIDENCE['feature_bounds_identity_reused_iterations']=len(snapshots)

    def test_absolute_input_not_retained_after_center_update(self):
        x=np.array([[-2.,4.],[-1.,4.],[1.,4.],[2.,4.]])
        current_data=[None]
        absolute_refs=[]
        original_centers=candidate._numpy_centers
        original_abs=np.abs
        def track_abs(value,*args,**kwargs):
            result=original_abs(value,*args,**kwargs)
            if value is current_data[0]: absolute_refs.append(weakref.ref(result))
            return result
        def centers(data,u,old,feature_bounds=None):
            current_data[0]=data
            result=original_centers(data,u,old,feature_bounds)
            self.assertTrue(absolute_refs)
            self.assertIsNone(absolute_refs[-1]())
            return result
        with patch.object(candidate,'_numpy_centers',centers), patch.object(np,'abs',track_abs):
            candidate.fit_entropy_fcm(x,2,random_state=8,max_iter=4,tol=0,return_history=True)
        self.assertEqual(len(absolute_refs),4)
        self.assertTrue(all(ref() is None for ref in absolute_refs))
        EVIDENCE['transient_absolute_arrays_released']=len(absolute_refs)

    def test_cancellation_objective_fallback_remains_exact(self):
        # One cluster's zero objective exercises both block and total fallback.
        x=np.array([[4.,-5.]])
        centers=x.copy()
        with patch.object(candidate,'_objective_pair',wraps=candidate._objective_pair) as exact:
            actual=candidate._numpy_memberships_objective(x,centers,1.)
        expected=baseline._numpy_memberships_objective(x,centers,1.)
        self.assert_bits(actual[0],expected[0]);self.assertEqual(actual[1:],expected[1:])
        self.assertEqual(exact.call_count,2)
        self.assertTrue(actual[2])
        EVIDENCE['cancellation_objective_pair_calls_retained']=exact.call_count

    def test_guards_and_objective_arithmetic_unchanged(self):
        # AST equality makes threshold weakening fail independently of outputs.
        def nodes(module):
            return {n.name:n for n in ast.parse(inspect.getsource(module)).body if isinstance(n,ast.FunctionDef)}
        old,new=nodes(baseline),nodes(candidate)
        for name in ('_temperature','_ordinary_range','_exact_cost','_objective_pair','_objective_value','_dyadic','_add','_subtract','_multiply','_sum_exact','_ratio'):
            self.assertEqual(ast.dump(old[name],include_attributes=False),ast.dump(new[name],include_attributes=False),name)
        for module in (baseline,candidate):
            source=inspect.getsource(module._numpy_memberships_objective)
            self.assertIn('8*d*np.spacing(costs.max(axis=1)) > tau*1e-12',source)
            self.assertIn('absolute < 1e-140',source)
            self.assertIn('32*np.finfo(float).eps*(distortion+abs(entropy))',source)
            self.assertIn('64*np.finfo(float).eps*magnitude',source)
        for value in (0.,np.nextafter(1e-140,0.),1e-140,1.,1e140,np.nextafter(1e140,math.inf)):
            for tau in (np.nextafter(1e-140,0.),1e-140,1.,1e140,np.nextafter(1e140,math.inf)):
                x=np.array([[value]])
                self.assertEqual(candidate._ordinary_range(x,tau),baseline._ordinary_range(x,tau))

    def test_validation_error_type_and_text_unchanged(self):
        cases=[({'X':[[0.]],'n_clusters':1,'tau':tau}) for tau in (0,-1,math.nan,math.inf,True,1j,'1',[1.])]
        cases += [{'X':x,'n_clusters':1} for x in ([],[[math.nan]],[[math.inf]],[[1j]])]
        cases += [dict(X=[[0.]],**kw) for kw in ({'n_clusters':0},{'max_iter':0},{'max_iter':True},{'tol':-1},
                   {'init':[[0.,0.]]},{'init':[[-1.,2.]]},{'init':[[math.inf]]},
                   {'init':[[1.,0.]],'n_clusters':3},{'backend':'gpu'})]
        for kwargs in cases:
            errors=[]
            for module in (baseline,candidate):
                try: module.fit_entropy_fcm(**kwargs)
                except Exception as error: errors.append((type(error).__name__,str(error)))
                else: self.fail('invalid input accepted')
            self.assertEqual(errors[0],errors[1],str(kwargs))
        EVIDENCE['validation_error_cases']=len(cases)

    def test_input_copy_and_output_ownership_with_cached_summary(self):
        x=np.arange(24.,dtype=np.float64).reshape(6,4)[:,::2]
        init=np.full((6,3),1/3)
        old_x=x.copy(); old_init=init.copy()
        x.flags.writeable=False;init.flags.writeable=False
        actual=self.compare_fit(x,init=init,max_iter=3,tol=0,return_history=True)
        for value in actual.values():
            if isinstance(value,np.ndarray):
                self.assertFalse(np.shares_memory(value,x) or np.shares_memory(value,init))
        self.assert_bits(x,old_x);self.assert_bits(init,old_init)

if __name__=='__main__': unittest.main(verbosity=2)
