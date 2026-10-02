"""Public imported-observation capture and rejection-atomicity regressions."""
from collections import defaultdict, UserDict
from collections.abc import Mapping
import math
import struct
from types import MappingProxyType
import unittest

from ubukit.optimization import TPEOptimizer, categorical, float_range, int_range


class ChangingMapping(Mapping):
    def __init__(self, first, second):
        self.first, self.second, self.reads = first, second, 0
    def __iter__(self):
        return iter(("x",))
    def __len__(self):
        return 1
    def __getitem__(self, key):
        if key != "x":
            raise KeyError(key)
        self.reads += 1
        if self.reads == 1:
            return self.first
        if isinstance(self.second, Exception):
            raise self.second
        return self.second


DOMAINS = [
    ("int", int_range(1, 8), 2, 3, 9),
    ("log_int", int_range(1, 16, log=True), 4, 8, 17),
    ("float", float_range(-2, 2), .375, .5, 3.0),
    ("log_float", float_range(1e-5, 1, log=True), .013141592653589793, .25, 2.0),
]


class ImportCaptureTests(unittest.TestCase):
    def test_numeric_changing_mapping_is_captured_once(self):
        for label, domain, first, valid, outside in DOMAINS:
            seconds = [valid, outside, math.nan, math.inf, -math.inf,
                       AssertionError("second lookup must never occur")]
            for index, second in enumerate(seconds):
                with self.subTest(domain=label, second_case=index):
                    params = ChangingMapping(first, second)
                    search = TPEOptimizer({"x": domain}, seed=42)
                    trial = search.add_trial(params, 0)
                    self.assertEqual(params.reads, 1)
                    self.assertEqual(trial.params, {"x": first})
                    self.assertEqual(search.result().best_params, {"x": first})
                    self.assertEqual(search.result().n_attempted, 1)
                    self.assertEqual(search.result().n_completed, 1)
                    self.assertEqual(search.result().stop_reason, "not_started")
                    if isinstance(first, float):
                        self.assertEqual(struct.pack("d", trial.params["x"]), struct.pack("d", first))

    def test_stable_mapping_controls_are_preserved(self):
        for label, domain, first, _, _ in DOMAINS:
            calls = []
            params_list = [dict(x=first), UserDict(x=first), MappingProxyType(dict(x=first)),
                           defaultdict(lambda: calls.append(True) or 99, x=first)]
            for params in params_list:
                with self.subTest(domain=label, mapping=type(params).__name__):
                    before = dict(params)
                    trial = TPEOptimizer({"x": domain}).add_trial(params, 0)
                    self.assertEqual(trial.params, {"x": first})
                    self.assertEqual(dict(params), before)
            self.assertEqual(calls, [])

    def test_rejected_first_value_preserves_public_state_and_next_ask(self):
        for label, domain, valid, _, outside in DOMAINS:
            bad_values = [outside, math.nan, math.inf, -math.inf, None, True]
            if domain["type"] == "int":
                bad_values.append(1.5)
            for bad in bad_values:
                with self.subTest(domain=label, bad=repr(bad)):
                    search = TPEOptimizer({"x": domain}, seed=42)
                    control = TPEOptimizer({"x": domain}, seed=42)
                    search.add_trial({"x": valid}, 1); control.add_trial({"x": valid}, 1)
                    before = search.result().to_dict()
                    params = ChangingMapping(bad, valid)
                    with self.assertRaises(ValueError):
                        search.add_trial(params, 0)
                    self.assertEqual(params.reads, 1)
                    self.assertEqual(search.result().to_dict(), before)
                    self.assertEqual(search.ask(), control.ask())

    def test_invalid_observation_metadata_is_atomic(self):
        cases = [dict(value=math.nan), dict(value=math.inf), dict(state="unknown"),
                 dict(state="fail", value=1), dict(state="cancelled", value=1),
                 dict(value=1, error="not allowed"), dict(state="fail", error=object())]
        for options in cases:
            with self.subTest(options=options):
                search = TPEOptimizer({"x": int_range(1, 8)}, seed=42)
                control = TPEOptimizer({"x": int_range(1, 8)}, seed=42)
                before = search.result().to_dict()
                params = ChangingMapping(2, math.nan)
                with self.assertRaises(ValueError):
                    search.add_trial(params, **options)
                self.assertEqual(params.reads, 1)
                self.assertEqual(search.result().to_dict(), before)
                self.assertEqual(search.ask(), control.ask())

    def test_categorical_mapping_capture_and_imported_repeats(self):
        search = TPEOptimizer({"x": categorical(["a", "b"])})
        params = ChangingMapping("a", "not-in-space")
        first = search.add_trial(params, 1)
        self.assertEqual(params.reads, 1)
        second = search.add_trial({"x": "a"}, 0)
        self.assertEqual((first.id, second.id), (0, 1))
        self.assertEqual(search.result().best_params, {"x": "a"})
        self.assertEqual(search.ask().params, {"x": "b"})

    def test_numeric_scalar_conversion_is_captured_once(self):
        class IndexOnce:
            def __init__(self): self.calls = 0
            def __index__(self):
                self.calls += 1
                if self.calls > 1: raise AssertionError("repeated integer conversion")
                return 2
        class FloatOnce(float):
            calls = 0
            def __float__(self):
                self.calls += 1
                if self.calls > 1: raise AssertionError("repeated float conversion")
                return .5
        integer, floating = IndexOnce(), FloatOnce(.5)
        self.assertEqual(TPEOptimizer({"x": int_range(1, 8)}).add_trial({"x": integer}, 0).params, {"x": 2})
        self.assertEqual(TPEOptimizer({"x": float_range(0, 1)}).add_trial({"x": floating}, 0).params, {"x": .5})
        self.assertEqual((integer.calls, floating.calls), (1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
