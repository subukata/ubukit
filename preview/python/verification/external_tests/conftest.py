"""The copied validation scripts must each run in a fresh isolated interpreter."""
import json
import os
from pathlib import Path
from ._installed_guard import check_installed

collect_ignore=['test_external_metrics.py','test_portability.py']


def pytest_sessionstart(session):
    check_installed(os.environ.get('UBUKIT_DISTRIBUTION','ubukit-bundled-local-preview'),True)


def pytest_sessionfinish(session,exitstatus):
    result=check_installed(os.environ.get('UBUKIT_DISTRIBUTION','ubukit-bundled-local-preview'),True)
    result['pytest_exitstatus']=int(exitstatus)
    output=Path(os.environ['UBUKIT_EXTERNAL_RESULTS'])/'facade_import_provenance.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
