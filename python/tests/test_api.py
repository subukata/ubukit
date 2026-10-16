"""The public API is __all__ (DESIGN.md, "Compatibility")."""

import pytest

import ubukit as ub


@pytest.mark.parametrize("module", [ub, ub.steps, ub.apply, ub.steps.apply])
def test_every_other_name_has_an_underscore(module):
    # Modules included: steps and apply are public, every other one internal.
    names = {n for n in vars(module) if not n.startswith("_")}
    assert names == set(module.__all__)
