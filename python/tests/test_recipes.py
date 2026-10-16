"""docs/recipes.md runs as written: its Python blocks, in order, in one namespace."""

import re
from pathlib import Path

import numpy as np
import pytest

RECIPES = Path(__file__).parents[2] / "docs" / "recipes.md"

pytestmark = pytest.mark.skipif(not RECIPES.exists(), reason="the docs live in the repository only")


def test_the_recipes_run():
    blocks = re.findall(r"```python\n(.*?)```", RECIPES.read_text(encoding="utf-8"), flags=re.S)
    assert len(blocks) >= 8
    namespace: dict = {}
    for n, block in enumerate(blocks, 1):
        exec(compile(block, f"{RECIPES.name}, block {n}", "exec"), namespace)
    # What the text claims: the federated run follows the centralized one.
    np.testing.assert_allclose(namespace["centers"], namespace["central"].centers, atol=1e-12)
