import json
import math
import pathlib
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
from ubukit.optimization import (TPEOptimizer, optimize, float_range, int_range,
    categorical, SearchSpaceExhausted, ProposalError, _Domain, _KDE, _RNG,
    _normal_cdf, _normal_interval, _normal_ppf, _normal_small_logmass)


class DomainTests(unittest.TestCase):
    def test_invalid_spaces(self):
        bad=[{}, {'x':{}}, {'x': {'type':'float','low':2,'high':1}},
             {'x': {'type':'float','low':0,'high':1,'log':True}},
             {'x': {'type':'int','low':0.2,'high':2}},
             {'x': {'type':'float','low':0,'high':math.inf}},
             {'x': {'type':'categorical','choices':[]}},
             {'x': {'type':'categorical','choices':[0,-0.0]}},
             {'x': {'type':'categorical','choices':[{}]}},
             {'x': {'type':'categorical','choices':[math.nan]}},
             {'x': {'type':'float','low':0,'high':1,'step':.1}},
             {'x': {'type':'int','low':-9007199254740991,'high':9007199254740991}}]
        for s in bad:
            with self.subTest(space=s), self.assertRaises(ValueError): TPEOptimizer(s)

    def test_bad_options(self):
        s={'x':float_range(0,1)}
        for opts in [{'seed':-1},{'seed':1.5},{'seed':True},{'seed':2**32},
                     {'n_startup_trials':1},{'n_candidates':0},{'gamma':0},
                     {'gamma':1},{'min_bandwidth':0},{'multivariate':1},
                     {'direction':'foo'},{'weights':'foo'},{'sampler':'foo'}]:
            with self.subTest(opts=opts), self.assertRaises(ValueError): TPEOptimizer(s,**opts)

    def test_scalar_categories(self):
        space={'x':categorical([None,True,False,1,0,'1'])}
        r=optimize(lambda p:0,space,20)
        self.assertEqual(len(r.history),6)
        self.assertEqual(r.stop_reason,'space_exhausted')
        self.assertEqual({(type(t.params['x']).__name__,str(t.params['x'])) for t in r.history},
                         {('NoneType','None'),('bool','True'),('bool','False'),('int','1'),('int','0'),('str','1')})

    def test_unsafe_integer_categories_rejected_without_aliasing(self):
        limit = 2**53-1
        for value in [limit+1, limit+2, -(limit+1), -(limit+2),
                      float(limit+1), -float(limit+1), 1e100]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "safe-integer"):
                    categorical([value, 'x'])
                optimizer = TPEOptimizer({'c': categorical([limit, -limit, 'x'])})
                with self.assertRaisesRegex(ValueError, "safe-integer"):
                    optimizer.add_trial({'c': value}, 0)
                self.assertEqual(optimizer.result().n_attempted, 0)
        with self.assertRaisesRegex(ValueError, "safe-integer"):
            categorical([2**53, 2**53+1])
        optimizer = TPEOptimizer({'c': categorical([limit, -limit, 1.5])})
        self.assertEqual(optimizer.add_trial({'c': limit}, 0).params, {'c': limit})
        self.assertEqual(optimizer.add_trial({'c': -limit}, 1).params, {'c': -limit})
        self.assertEqual(optimizer.add_trial({'c': 1.5}, 2).params, {'c': 1.5})

    def test_large_float_parameter_keys_remain_supported(self):
        # The categorical restriction must not narrow continuous numeric domains.
        result = optimize(lambda p: 0, {'x': float_range(1e100, 2e100)}, 3)
        self.assertEqual(result.n_completed, 3)
        self.assertTrue(all(1e100 <= t.params['x'] <= 2e100 for t in result.history))

    def test_roundtrip_and_prior_mass(self):
        for log in [False,True]:
            for low,high in [(1,1),(1,4),(3,199)]:
                d=_Domain('x',int_range(low,high,log=log))
                total=0
                for k in range(low,high+1):
                    self.assertEqual(d.decode(d.encode(k)),k)
                    total+=math.exp(d.prior_log(k))
                self.assertAlmostEqual(total,1.0,places=13)
        for lo,hi,log in [(0,1,False),(-1.7e308,1.7e308,False),(5e-324,1e308,True),(1e308,1.0000000000000002e308,True)]:
            d=_Domain('x',float_range(lo,hi,log=log))
            for t in [0,.1,.5,.9,1]:
                x=d.decode(t)
                self.assertTrue(math.isfinite(x) and lo<=x<=hi)
            self.assertEqual(d.decode(0),lo)
            self.assertEqual(d.decode(1),hi)

    def test_log_integer_extreme_bin(self):
        d=_Domain('x',int_range(1,9007199254740991,log=True))
        self.assertGreater(d.bin_width(d.high),0)
        expected=math.log1p(1/(d.high-.5))/math.log1p(d.size/.5)
        self.assertAlmostEqual(d.prior_log(d.high),math.log(expected),places=12)
        kde=_KDE([d],[{'x':d.high}], [1],True,.03)
        self.assertTrue(math.isfinite(kde.logpdf({'x':d.high})))

    def test_unicode_sort(self):
        opt=TPEOptimizer({'\U00010000':float_range(0,1),'\ue000':float_range(0,1),'a':float_range(0,1)})
        self.assertEqual(list(opt.ask().params), ['a','\U00010000','\ue000'])


class DensityTests(unittest.TestCase):
    def test_high_precision_narrow_interval_fixtures(self):
        fixtures = json.loads((ROOT/'fixtures'/'normal_interval_reference.json').read_text())
        for case in fixtures['cases']:
            with self.subTest(center=case['center'], width=case['width']):
                log_mass = _normal_small_logmass(case['center'], case['width'])
                self.assertIsNotNone(log_mass)
                self.assertLess(abs(math.expm1(log_mass-case['logMass'])), 4e-12)

    def test_narrow_integer_center_pmf_regression(self):
        d = _Domain('x', int_range(1, 199999999))
        kde = _KDE([d], [{'x': 100000000}], [1.0], True, .03)
        low, high = d.bins(100000000)
        width = d.bin_width(100000000)
        sigma = kde.sigmas[0][0]
        self.assertEqual(sigma, .5)
        norm = kde.norms[0][0]
        expected = width/(sigma*math.sqrt(2*math.pi)*norm)
        actual = math.exp(kde._axis_logs(0, 100000000)[0])
        self.assertLess(abs(actual/expected - 1), 1e-12)
        self.assertLess(abs(_normal_interval(-width, width)/(2*width/math.sqrt(2*math.pi))-1), 1e-12)

    def test_centered_expansion_remainder_bound(self):
        # Worst conservative bound over the allowed rectangle in width and |m*w|.
        w, q = .01, .1
        mw = q+w*w/2
        polynomial = mw**6+15*mw**4*w*w+45*mw*mw*w**4+15*w**6
        relative_bound = polynomial/322560 * math.exp(q+w*w/8)
        self.assertLess(relative_bound, 4e-12)
        self.assertIsNotNone(_normal_small_logmass(10, .01))
        self.assertIsNone(_normal_small_logmass(11, .01))
        self.assertIsNone(_normal_small_logmass(0, .010001))

    def test_cdf_accuracy_and_inverse(self):
        for i in range(-100,101):
            x=i/10
            ref=.5*math.erfc(-x/math.sqrt(2))
            self.assertLess(abs(_normal_cdf(x)-ref),8e-8)
        for p in [1e-10,1e-7,.0001,.01,.1,.5,.9,.99,.9999,1-1e-10]:
            self.assertLess(abs(_normal_cdf(_normal_ppf(p))-p),8e-8)
        self.assertGreater(_normal_interval(9,9.1),0)
        self.assertEqual(_normal_interval(1,0),0)
        self.assertAlmostEqual(_normal_interval(-1,1),.682689492137,places=6)

    def test_numeric_integrates(self):
        d=_Domain('x',float_range(0,1))
        for joint in [False,True]:
            kde=_KDE([d],[{'x':0},{'x':.01},{'x':1}], [1,2,1],joint,.03)
            n=10000
            total=sum(math.exp(kde.logpdf({'x':(i+.5)/n})) for i in range(n))/n
            self.assertAlmostEqual(total,1,places=6)

    def test_discrete_integrates(self):
        for log in [False,True]:
            d=_Domain('x',int_range(1,31,log=log))
            kde=_KDE([d],[{'x':1},{'x':2},{'x':31}], [1,2,1],True,.03)
            self.assertAlmostEqual(sum(math.exp(kde.logpdf({'x':i})) for i in range(1,32)),1,places=12)
        d=_Domain('x',categorical(['a','b','c']))
        kde=_KDE([d],[{'x':'a'},{'x':'a'},{'x':'b'}], [1,2,1],True,.03)
        self.assertAlmostEqual(sum(math.exp(kde.logpdf({'x':v})) for v in d.choices),1,places=14)
        self.assertGreater(math.exp(kde.logpdf({'x':'c'})),0)

    def test_joint_represents_interactions(self):
        ds=[_Domain('x',float_range(0,1)),_Domain('y',float_range(0,1))]
        rows=[{'x':.2,'y':.2},{'x':.8,'y':.8}]*4
        joint=_KDE(ds,rows,[1]*8,True,.03)
        independent=_KDE(ds,rows,[1]*8,False,.03)
        self.assertGreater(joint.logpdf({'x':.2,'y':.2}),joint.logpdf({'x':.2,'y':.8}))
        self.assertAlmostEqual(independent.logpdf({'x':.2,'y':.2}),independent.logpdf({'x':.2,'y':.8}),places=12)

    def test_hundreds_of_dimensions_finite(self):
        ds=[_Domain(str(i),float_range(0,1)) for i in range(300)]
        rows=[{d.name:x for d in ds} for x in [0,.001,1]]
        kde=_KDE(ds,rows,[1,1,1],True,.03)
        self.assertTrue(math.isfinite(kde.logpdf({d.name:.5 for d in ds})))


class LifecycleTests(unittest.TestCase):
    def test_random_seed(self):
        self.assertEqual([_RNG(42).uint32() for _ in range(2)],[2581720956,2581720956])
        a=_RNG(42);b=_RNG(42)
        self.assertEqual([a.uint32() for _ in range(1000)],[b.uint32() for _ in range(1000)])

    def test_min_max_seed_reproducibility(self):
        space={'x':float_range(-2,2),'k':int_range(0,3),'c':categorical(['a','b'])}
        f=lambda p:(p['x']-.37)**2+(p['k']-2)**2+(p['c']!='b')
        a=optimize(f,space,60,seed=3)
        b=optimize(lambda p:-f(p),space,budget=60,seed=3,direction='maximize')
        self.assertEqual([t.params for t in a.history],[t.params for t in b.history])
        self.assertEqual(a.best_params,b.best_params)
        self.assertAlmostEqual(a.best_value,-b.best_value)
        c=optimize(f,space,60,seed=3)
        self.assertEqual(a.to_dict(),c.to_dict())

    def test_issued_ids_ownership_and_tell(self):
        o=TPEOptimizer({'x':float_range(0,1)})
        a=o.ask(); original=a.params['x'];a.params['x']=99
        self.assertEqual(o.result().history[0].params['x'],original)
        with self.assertRaises(ValueError):o.tell(99,0)
        with self.assertRaises(ValueError):o.tell(a.id,math.nan)
        self.assertEqual(o.result().history[0].state,'running')
        o.tell(a.id,0)
        with self.assertRaises(ValueError):o.tell(a.id,0)
        r=o.result();r.history[0].params['x']=88;r.best_params['x']=77
        self.assertEqual(o.result().best_params['x'],original)

    def test_add_trial_atomic_and_duplicates(self):
        o=TPEOptimizer({'x':int_range(1,2)})
        for params,value in [({'x':3},0), ({},0), ({'x':1},math.inf)]:
            with self.assertRaises(ValueError):o.add_trial(params,value)
            self.assertEqual(len(o.result().history),0)
        o.add_trial({'x':1},0);o.add_trial({'x':1},1)
        self.assertEqual(o.ask().params,{'x':2})
        with self.assertRaises(SearchSpaceExhausted):o.ask()

    def test_fail_and_cancel(self):
        o=TPEOptimizer({'x':int_range(1,3)})
        a=o.ask();o.tell(a.id,state='fail',error='expected')
        b=o.ask();o.tell(b.id,state='cancelled')
        self.assertNotEqual(a.params,b.params)
        self.assertIsNone(o.result().best_value)
        with self.assertRaises(ValueError):o.tell(o.ask().id,2,state='fail')

    def test_budget_failure_and_callback(self):
        space={'x':float_range(0,1)}
        r=optimize(lambda p:0,space,0)
        self.assertEqual(r.n_attempted,0);self.assertIsNone(r.best_params)
        r=optimize(lambda p:math.nan,space,20,continue_on_error=True)
        self.assertEqual(r.n_attempted,20);self.assertEqual(r.n_completed,0)
        self.assertTrue(all(t.state=='fail' for t in r.history))
        with self.assertRaises(ValueError):optimize(lambda p:math.inf,space,3)
        with self.assertRaises(RuntimeError):optimize(lambda p:(_ for _ in ()).throw(RuntimeError('bug')),space,3)
        r=optimize(lambda p:p['x'],space,20,callback=lambda t:t.id<2)
        self.assertEqual(r.n_attempted,3);self.assertEqual(r.stop_reason,'callback_stopped')

    def test_finite_exhaustion_fixed_and_allow_repeats(self):
        space={'x':int_range(1,2),'c':categorical(['a','b']),'fixed':float_range(3,3)}
        r=optimize(lambda p:p['x'],space,200)
        self.assertEqual(r.n_attempted,4);self.assertEqual(r.stop_reason,'space_exhausted')
        r=optimize(lambda p:0,space,20,avoid_duplicates=False)
        self.assertEqual(r.n_attempted,20)

    def test_pending_reservation_and_all_tied(self):
        o=TPEOptimizer({'x':int_range(0,40)},n_startup_trials=2)
        trials=[o.ask() for _ in range(10)]
        self.assertEqual(len({t.params['x'] for t in trials}),10)
        for t in reversed(trials):o.tell(t.id,1)
        self.assertEqual(o._models,None)
        o.ask()
        self.assertEqual(o._models,None)

    def test_ei_large_finite_losses(self):
        o=TPEOptimizer({'x':float_range(-1,1)},n_startup_trials=2)
        for x,y in [(-1,-1.7e308),(0,0),(1,1.7e308)]:o.add_trial({'x':x},y)
        t=o.ask();self.assertTrue(math.isfinite(t.params['x']))
        self.assertTrue(all(math.isfinite(w) for w in o._fit()[0].weights))

    def test_no_third_party_import(self):
        import subprocess
        p=subprocess.run([sys.executable,'-I','-S','-c',f"import sys;sys.path.insert(0,{str(pathlib.Path(__import__('importlib').util.find_spec('ubukit').origin).parent.parent)!r});from ubukit import optimize,float_range;assert optimize(lambda p:p['x']**2,{{'x':float_range(-1,1)}},5).n_completed==5"],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
