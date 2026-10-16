"""The packages carry the repository's license files unchanged, and one version."""

import json
import re
from pathlib import Path

import pytest

import ubukit as ub

ROOT = Path(__file__).parents[2]

pytestmark = pytest.mark.skipif(
    not (ROOT / "js").is_dir(), reason="the copies are compared in the repository only"
)


@pytest.mark.parametrize("name", ["LICENSE", "THIRD_PARTY_NOTICES.txt"])
@pytest.mark.parametrize("package", ["python", "js"])
def test_license_files_are_copies(package, name):
    assert (ROOT / package / name).read_bytes() == (ROOT / name).read_bytes()


def test_one_version_everywhere():
    # A release sets the version in these places by hand; release.yml compares
    # two of them with the tag, after it is pushed, when a mistake costs the
    # tag (v* tags can be neither moved nor deleted). Between releases the
    # newest dated heading of the CHANGELOG is the version the packages carry.
    package = json.loads((ROOT / "js" / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "js" / "package-lock.json").read_text(encoding="utf-8"))
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.search(r"^## (\S+) \(\d{4}-\d{2}-\d{2}\)$", changelog, re.MULTILINE)
    versions = {
        "python/src/ubukit/__init__.py": ub.__version__,
        "js/package.json": package["version"],
        "js/package-lock.json": lock["version"],
        'js/package-lock.json, packages[""]': lock["packages"][""]["version"],
        "CHANGELOG.md, newest dated heading": heading and heading.group(1),
    }
    assert len(set(versions.values())) == 1, versions
