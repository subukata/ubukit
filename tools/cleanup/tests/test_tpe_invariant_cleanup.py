"""Deterministic baseline/candidate equivalence and eliminated-work regressions.

This suite deliberately records no elapsed time, throughput, or speedup.
The baseline is loaded by path under a separate module identity.
"""
import importlib.util
import inspect
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASELINE = load("tpe_cleanup_baseline", "baseline/ubukit/optimization.py")
CANDIDATE = load("tpe_cleanup_candidate", "candidate/ubukit/optimization.py")
EVIDENCE = {"trajectory_cases": [], "density_cases": 0, "evaluation_counts": [],
            "slice_counts": [], "edge_cases": []}


def scalar_bits(value):
    if isinstance(value, float):
        return ("float", struct.pack("!d", value).hex())
    if isinstance(value, dict):
        return [(key, scalar_bits(item)) for key, item in value.items()]
    if isinstance(value, (list, tuple)):
        return [scalar_bits(item) for item in value]
    return (type(value).__name__, value)


class CountingMath:
    def __init__(self):
        self.logs = []
    def log(self, value):
        self.logs.append(value)
        return math.log(value)
    def __getattr__(self, name):
        return getattr(math, name)


class SliceCountingList(list):
    def __init__(self, values):
        super().__init__(values)
        self.slices = 0
    def __getitem__(self, key):
        if isinstance(key, slice):
            self.slices += 1
        return super().__getitem__(key)


class InvariantCleanupTests(unittest.TestCase):
    def assertExact(self, actual, expected):
        self.assertEqual(scalar_bits(actual), scalar_bits(expected))

    def test_public_exports_and_signatures_unchanged(self):
        self.assertEqual(CANDIDATE.__all__, BASELINE.__all__)
        self.assertEqual(CANDIDATE._VERSION, BASELINE._VERSION)
        for name in BASELINE.__all__:
            if name not in {"SearchSpaceExhausted", "ProposalError"}:
                self.assertEqual(str(inspect.signature(getattr(CANDIDATE, name))),
                                 str(inspect.signature(getattr(BASELINE, name))))
        for name in ("ask", "tell", "add_trial", "result"):
            self.assertEqual(str(inspect.signature(getattr(CANDIDATE.TPEOptimizer, name))),
                             str(inspect.signature(getattr(BASELINE.TPEOptimizer, name))))

    def test_exact_density_edge_cases_and_sampling(self):
        cases = [
            ({"type": "float", "low": 0, "high": 1}, [0, .5, 1]),
            ({"type": "float", "low": -1.7e308, "high": 1.7e308}, [-1.7e308, 0, 1.7e308]),
            ({"type": "float", "low": 5e-324, "high": 1e308, "log": True}, [5e-324, 1, 1e308]),
            ({"type": "float", "low": 1e308, "high": math.nextafter(1e308, math.inf), "log": True},
             [1e308, 1e308, math.nextafter(1e308, math.inf)]),
            ({"type": "int", "low": -3, "high": 4}, [-3, 0, 4]),
            ({"type": "int", "low": 1, "high": 31, "log": True}, [1, 15, 31]),
            ({"type": "int", "low": 1, "high": 199999999}, [1, 100000000, 199999999]),
            ({"type": "int", "low": 1, "high": 9007199254740991, "log": True}, [1, 2**51, 9007199254740991]),
            ({"type": "categorical", "choices": [None, True, 1, "a"]}, [None, True, 1, "a"]),
            ({"type": "float", "low": -0.0, "high": 0.0}, [-0.0, 0.0]),
            ({"type": "int", "low": 3, "high": 3}, [3]),
            ({"type": "categorical", "choices": [False]}, [False]),
        ]
        for case_id, (spec, points) in enumerate(cases):
            for count, joint, bandwidth in itertools.product((0, 1, 4), (False, True), (.03, 1.0)):
                with self.subTest(case=case_id, count=count, joint=joint, bandwidth=bandwidth):
                    rows = [{"x": points[i % len(points)]} for i in range(count)]
                    weights = [0.0 if i == 1 else float(i + 1) for i in range(count)]
                    models = [m._KDE([m._Domain("x", spec)], rows, weights, joint, bandwidth)
                              for m in (BASELINE, CANDIDATE)]
                    for point in points:
                        self.assertExact(models[1]._axis_logs(0, point), models[0]._axis_logs(0, point))
                        self.assertExact(models[1].logpdf({"x": point}), models[0].logpdf({"x": point}))
                    rngs = [m._RNG(0xFFFFFFFF) for m in (BASELINE, CANDIDATE)]
                    for _ in range(5):
                        self.assertExact(models[1].sample(rngs[1]), models[0].sample(rngs[0]))
                        self.assertEqual(rngs[1].state, rngs[0].state)
                    EVIDENCE["density_cases"] += 1

    def test_cached_values_keep_exact_normalizer_expression_and_axis_alignment(self):
        specs = [CANDIDATE.categorical(["a", "b"]), CANDIDATE.float_range(0, 1),
                 CANDIDATE.int_range(1, 100000), CANDIDATE.float_range(2, 2)]
        domains = [CANDIDATE._Domain(str(i), spec) for i, spec in enumerate(specs)]
        rows = [{"0": "a", "1": .01, "2": 7, "3": 2},
                {"0": "b", "1": 1.0, "2": 99999, "3": 2}]
        kde = CANDIDATE._KDE(domains, rows, [1, 2], True, .03)
        self.assertIsNone(kde.log_normalizers[0])
        self.assertIsNone(kde.log_normalizers[3])
        self.assertExact(kde.log_normalizers[1], [math.log(s*CANDIDATE._SQRT2PI*n)
                                                for s, n in zip(kde.sigmas[1], kde.norms[1])])
        self.assertIsNone(kde.log_normalizers[2])

    def test_float_normalizer_evaluations_are_construction_only(self):
        self.check_counts("float", {"type": "float", "low": 0, "high": 1}, [0, .1, .8, 1], .37,
                          baseline_scoring_per_call=4, candidate_scoring_per_call=0)

    def test_narrow_integer_normalizer_evaluations_are_unchanged(self):
        self.check_counts("narrow_integer", {"type": "int", "low": 1, "high": 199999999},
                          [1, 20000000, 160000000, 199999999], 100000000,
                          baseline_scoring_per_call=13, candidate_scoring_per_call=13)

    def test_broad_integer_mass_expression_is_unchanged(self):
        self.check_counts("broad_integer", {"type": "int", "low": 1, "high": 7}, [1, 2, 6, 7], 4,
                          baseline_scoring_per_call=5, candidate_scoring_per_call=5)

    def check_counts(self, label, spec, points, query, *, baseline_scoring_per_call, candidate_scoring_per_call):
        records = []
        outputs = []
        for module, expected in ((BASELINE, baseline_scoring_per_call), (CANDIDATE, candidate_scoring_per_call)):
            domain = module._Domain("x", spec)
            proxy = CountingMath()
            original = module.math
            module.math = proxy
            try:
                kde = module._KDE([domain], [{"x": x} for x in points], [1]*4, True, .03)
                constructor_count = len(proxy.logs)
                self.assertEqual(constructor_count, 9 if module is CANDIDATE and spec["type"] == "float" else 5)
                per_query_counts, result = [], []
                for _ in range(7):
                    before = len(proxy.logs)
                    result.append(kde._axis_logs(0, query))
                    per_query_counts.append(len(proxy.logs) - before)
                self.assertEqual(per_query_counts, [expected]*7)
                records.append({"implementation": "baseline" if module is BASELINE else "candidate",
                                "construction_log_calls": constructor_count,
                                "per_query_log_calls": per_query_counts,
                                "total_log_calls": len(proxy.logs)})
                outputs.append(result)
            finally:
                module.math = original
        self.assertExact(outputs[1], outputs[0])
        EVIDENCE["evaluation_counts"].append({"case": label, "rows": 4, "queries": 7, "records": records})

    def test_tied_history_has_no_per_ask_slice(self):
        self.check_slices("tied", [0.0, -0.0, 0.0, -0.0, 0.0, 0.0])

    def test_early_signal_history_has_no_per_ask_slice(self):
        self.check_slices("early_signal", [0, 1, 0, 0, 0, 0])

    def test_late_signal_history_has_no_per_ask_slice(self):
        self.check_slices("late_signal", [0, 0, 0, 0, 0, 1])

    def check_slices(self, label, values):
        searches = []
        for module in (BASELINE, CANDIDATE):
            search = module.TPEOptimizer({"x": module.float_range(0, 1)}, seed=41,
                                         n_startup_trials=2, n_candidates=4, avoid_duplicates=False)
            for i, value in enumerate(values):
                search.add_trial({"x": i/len(values)}, value)
            search._completed = SliceCountingList(search._completed)
            searches.append(search)
        for _ in range(4):
            self.assertExact(searches[1].ask().params, searches[0].ask().params)
            self.assertEqual(searches[1]._rng.state, searches[0]._rng.state)
        self.assertEqual([s._completed.slices for s in searches], [4, 0])
        self.assertEqual([s._models is None for s in searches], [label == "tied"]*2)
        EVIDENCE["slice_counts"].append({"history": label, "asks": 4, "baseline": 4, "candidate": 0})

    def test_fixed_finite_space_exhaustion(self):
        results = [m.optimize(lambda p: p["k"], {"k": m.int_range(1, 3),
                   "c": m.categorical([None, "x"]), "fixed": m.float_range(2, 2)},
                   20, seed=99, n_startup_trials=2, n_candidates=4).to_dict()
                   for m in (BASELINE, CANDIDATE)]
        self.assertExact(results[1], results[0])
        self.assertEqual(results[1]["stop_reason"], "space_exhausted")
        self.assertEqual(results[1]["n_completed"], 6)
        EVIDENCE["edge_cases"].append("fixed_and_finite_exhaustion")

    def test_pending_reverse_completion_failure_and_imported_ties(self):
        searches = [m.TPEOptimizer({"x": m.float_range(-1, 1), "k": m.int_range(1, 7)},
                                  seed=78, n_startup_trials=2, n_candidates=6)
                    for m in (BASELINE, CANDIDATE)]
        pending = [[s.ask() for _ in range(6)] for s in searches]
        for i in reversed(range(6)):
            for s, trials in zip(searches, pending):
                if i == 2:
                    s.tell(trials[i].id, state="fail", error="expected")
                elif i == 3:
                    s.tell(trials[i].id, state="cancelled")
                else:
                    s.tell(trials[i].id, -0.0 if i % 2 else 0.0)
        for s in searches:
            s.add_trial({"x": 0.0, "k": 4}, 0.0)
        self.assertExact(searches[1].ask().params, searches[0].ask().params)
        self.assertIsNone(searches[1]._models)
        for s in searches:
            s.add_trial({"x": 0.0, "k": 4}, -1.7e308)
            s.add_trial({"x": .25, "k": 5}, 1.7e308)
        for _ in range(6):
            self.assertExact(searches[1].ask().params, searches[0].ask().params)
            self.assertEqual(searches[1]._rng.state, searches[0]._rng.state)
        self.assertExact(searches[1].result().to_dict(), searches[0].result().to_dict())
        EVIDENCE["edge_cases"].append("pending_reverse_completion_fail_cancel_imports_ties_extreme_losses")

    def test_same_models_reused_until_successful_completion(self):
        for module in (BASELINE, CANDIDATE):
            search = module.TPEOptimizer({"x": module.float_range(0, 1)}, seed=41,
                                         n_startup_trials=2, n_candidates=4)
            search.add_trial({"x": 0}, 0)
            search.add_trial({"x": 1}, 1)
            first = search.ask()
            models = search._models
            second = search.ask()
            self.assertIs(search._models, models)
            search.tell(first.id, state="fail")
            search.ask()
            self.assertIs(search._models, models)
            search.tell(second.id, .2)
            self.assertIsNone(search._models)
            search.ask()
            self.assertIsNot(search._models, models)
        EVIDENCE["edge_cases"].append("model_cache_lifecycle")


def trajectory_case(self, *, seed, joint, weighting, direction, sampler):
    searches = []
    for module in (BASELINE, CANDIDATE):
        space = {"linear": module.float_range(-2, 2), "log": module.float_range(1e-7, 2, log=True),
                 "int": module.int_range(-3, 7), "logint": module.int_range(1, 9007199254740991, log=True),
                 "cat": module.categorical([None, True, 1, "x"]), "fixed": module.float_range(3, 3)}
        searches.append(module.TPEOptimizer(space, seed=seed, n_startup_trials=4, n_candidates=8,
                                           multivariate=joint, weights=weighting, direction=direction,
                                           sampler=sampler))
    for i in range(26):
        trials = [s.ask() for s in searches]
        self.assertExact(trials[1].params, trials[0].params)
        self.assertEqual(searches[1]._rng.state, searches[0]._rng.state)
        for search, trial in zip(searches, trials):
            p = trial.params
            if i == 5:
                search.tell(trial.id, state="fail", error="expected")
            elif i == 9:
                search.tell(trial.id, state="cancelled")
            else:
                value = ((p["linear"]-.37)**2 + math.log1p(p["log"]) + (p["int"]-2)**2
                         + math.log1p(p["logint"])/40 + (p["cat"] is not None))
                search.tell(trial.id, value if direction == "minimize" else -value)
    self.assertExact(searches[1].result().to_dict(), searches[0].result().to_dict())
    EVIDENCE["trajectory_cases"].append({"seed": seed, "multivariate": joint, "weights": weighting,
                                        "direction": direction, "sampler": sampler, "asks": 26})


for _seed, _joint, _weight, _direction in itertools.product((0, 42, 0xFFFFFFFF), (False, True),
                                                          ("ei", "uniform"), ("minimize", "maximize")):
    def test(self, seed=_seed, joint=_joint, weighting=_weight, direction=_direction):
        trajectory_case(self, seed=seed, joint=joint, weighting=weighting, direction=direction, sampler="tpe")
    setattr(InvariantCleanupTests, f"test_trajectory_tpe_{_seed}_{_joint}_{_weight}_{_direction}", test)

for _seed in (0, 42, 0xFFFFFFFF):
    def test(self, seed=_seed):
        trajectory_case(self, seed=seed, joint=True, weighting="ei", direction="minimize", sampler="random")
    setattr(InvariantCleanupTests, f"test_trajectory_random_{_seed}", test)


def write_evidence():
    (ROOT / "evidence" / "deterministic_details.json").write_text(json.dumps(EVIDENCE, indent=2) + "\n")


if __name__ == "__main__":
    import atexit
    atexit.register(write_evidence)
    unittest.main(verbosity=2)
