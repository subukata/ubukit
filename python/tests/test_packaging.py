"""The packages carry the repository's license files unchanged."""

from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]

pytestmark = pytest.mark.skipif(
    not (ROOT / "js").is_dir(), reason="the copies are compared in the repository only"
)


@pytest.mark.parametrize("name", ["LICENSE", "THIRD_PARTY_NOTICES.txt"])
@pytest.mark.parametrize("package", ["python", "js"])
def test_license_files_are_copies(package, name):
    assert (ROOT / package / name).read_bytes() == (ROOT / name).read_bytes()
