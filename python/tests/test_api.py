"""The public API is __all__ (DESIGN.md, "Compatibility")."""

import inspect

import pytest

import ubukit as ub


@pytest.mark.parametrize("module", [ub, ub.steps])
def test_every_other_name_has_an_underscore(module):
    # Modules are left out: all but steps are internal, and get an
    # underscore with #89.
    names = {
        n for n, v in vars(module).items() if not n.startswith("_") and not inspect.ismodule(v)
    }
    assert names == set(module.__all__) - {"steps"}
