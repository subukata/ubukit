"""The committed cross-language fixtures must match the current Python results."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

GENERATOR = Path(__file__).parents[2] / "fixtures" / "generate.py"

pytestmark = pytest.mark.skipif(
    not GENERATOR.exists(), reason="fixtures live in the repository only"
)


def test_fixtures_are_current():
    spec = importlib.util.spec_from_file_location("generate", GENERATOR)
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)
    stored = json.loads(generate.PATH.read_text(encoding="utf-8"))
    fresh = generate.build()
    assert len(stored["cases"]) == len(fresh["cases"])
    for old, new in zip(stored["cases"], fresh["cases"], strict=True):
        assert old["method"] == new["method"]
        if isinstance(new["expect"], dict):
            for key, value in new["expect"].items():
                np.testing.assert_allclose(old["expect"][key], value, atol=1e-10, err_msg=key)
        else:
            assert old["expect"] == pytest.approx(new["expect"], abs=1e-12)
