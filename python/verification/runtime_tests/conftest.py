import json
from pathlib import Path
import pytest
from ._installed_guard import check_installed


def pytest_addoption(parser):
    parser.addoption('--ubukit-distribution', default='ubukit-bundled-local-preview')
    parser.addoption('--ubukit-provenance-out', default=None)
    parser.addoption('--include-large-sparse', action='store_true', default=False,
                     help='Include the original 8000-point sparse graph regression (not needed for default wheel smoke/regression)')


def pytest_sessionstart(session):
    check_installed(session.config.getoption('--ubukit-distribution'), verify_hashes=True)


def pytest_collection_modifyitems(config, items):
    if config.getoption('--include-large-sparse'):
        return
    skipped = pytest.mark.skip(reason='Default bounded installed-wheel run excludes original 8000-point sparse graph exercise; --include-large-sparse restores it unchanged')
    for item in items:
        if 'test_memory_limit_and_sparse_large_n' in item.nodeid:
            item.add_marker(skipped)


def pytest_sessionfinish(session, exitstatus):
    result = check_installed(session.config.getoption('--ubukit-distribution'), verify_hashes=True)
    result['pytest_exitstatus'] = int(exitstatus)
    out = session.config.getoption('--ubukit-provenance-out')
    if out:
        Path(out).write_text(json.dumps(result, indent=2)+'\n')
