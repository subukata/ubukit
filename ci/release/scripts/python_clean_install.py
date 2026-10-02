"""Clean-install candidate driver, designed for Windows/macOS/Linux.

Running this script installs test dependencies into new isolated environments.
Targets the reviewed PR8 dev5 tree; cross-OS execution still needs approval.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

p = argparse.ArgumentParser()
p.add_argument('--python-root', type=Path, required=True)
p.add_argument('--artifacts', type=Path, required=True)
p.add_argument('--constraints', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--with-numba', action='store_true')
p.add_argument('--build-constraints', type=Path, required=True)
p.add_argument('--full', action='store_true')
p.add_argument('--include-large-sparse', action='store_true')
a = p.parse_args()
root, out = a.python_root.resolve(), a.output.resolve()
assert not out.exists(), f'Refuse to reuse installation directory: {out}'
required = ['tests/test_som.py', 'tests/test_localized_tails.py',
            'tests/test_backend_contracts.py', 'tools/check_high_precision.py',
            'verification/python_api.py', 'examples/all_methods.py']
for relative in required:
    assert (root/relative).is_file(), f'Missing current candidate input: {relative}'
if a.full:
    assert (root/'verification/runtime_tests/python_api.py').read_bytes() == (root/'verification/python_api.py').read_bytes(), (
        'Full gate blocked: candidate does not contain the reviewed PR8 facade repair '
        'and its VERIFICATION_SNAPSHOT.json hash before selecting a new reviewed candidate commit. '
        'Do not skip the inherited facade stage.')
artifacts = a.artifacts.resolve()
expected = json.loads((artifacts/'SHA256SUMS.json').read_text())
for name, digest in expected.items():
    assert Path(name).name == name
    assert hashlib.sha256((artifacts/name).read_bytes()).hexdigest() == digest, name
wheel = list(artifacts.glob('*.whl'))
sdist = list(artifacts.glob('*.tar.gz'))
assert len(wheel) == len(sdist) == 1
assert set(expected) == {wheel[0].name, sdist[0].name}, 'Checksums must bind exactly both candidate artifacts'
out.mkdir(parents=True)
empty = out/'empty'; empty.mkdir()
env = dict(os.environ)
for k in ('PYTHONPATH', 'PYTHONHOME', 'NUMBA_DISABLE_JIT'):
    env.pop(k, None)
env.update({k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
                            'NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTEST_DISABLE_PLUGIN_AUTOLOAD',
                            'PYTHONDONTWRITEBYTECODE')})
env.update(PYTHONHASHSEED='0', PIP_DISABLE_PIP_VERSION_CHECK='1',
           PIP_CONSTRAINT=str(a.build_constraints.resolve()))
stages = []
def run(label, argv, extra_env=None):
    with (out/(label+'.log')).open('w', encoding='utf-8') as f:
        result = subprocess.run([str(x) for x in argv], cwd=empty,
                                env={**env, **(extra_env or {})}, text=True,
                                stdout=f, stderr=subprocess.STDOUT)
    stages.append({'stage': label, 'exit_code': result.returncode,
                   'command': [str(x) for x in argv]})
    (out/'stages.json').write_text(json.dumps(stages, indent=2))
    if result.returncode:
        print((out/(label+'.log')).read_text(encoding='utf-8', errors='replace'))
        raise SystemExit(result.returncode)
for kind, artifact in [('wheel', wheel[0]), ('sdist', sdist[0])]:
    vp = out/f'venv-{kind}'
    venv.EnvBuilder(with_pip=True, clear=False).create(vp)
    python = vp/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    env['NUMBA_CACHE_DIR'] = str(out/f'numba-cache-{kind}')
    deps = ['numpy','scipy','scikit-learn','threadpoolctl','pytest==8.4.2']
    if a.with_numba: deps.append('numba')
    # Binary-only dependencies make missing platform wheels a clear failure, not an accidental compiler test.
    run(kind+'-dependencies', [python,'-m','pip','install','--no-cache-dir','--only-binary=:all:',
                             '-c',a.constraints.resolve(),*deps])
    # An explicit local sdist exercises PEP 517 isolated build with its declared backend.
    # --no-deps cannot silently alter the verified runtime dependency set.
    run(kind+'-install', [python,'-m','pip','install','--no-cache-dir','--no-deps','-v',artifact])
    if a.with_numba:
        run(kind+'-extra-resolution', [python,'-m','pip','install','--no-index','--only-binary=:all:',
                                       'ubukit-bundled-local-preview[numba]'])
    run(kind+'-pip-check', [python,'-m','pip','check'])
    run(kind+'-freeze', [python,'-m','pip','freeze','--all'])
    run(kind+'-origin', [python,'-I','-B',Path(__file__).with_name('installed_python_gate.py'),
                        root/'staging','numba' if a.with_numba else 'base'])
    site_packages = subprocess.check_output(
        [str(python), '-I', '-B', '-c', 'import sysconfig; print(sysconfig.get_paths()["purelib"])'],
        cwd=empty, env=env, text=True).strip()
    # The backend contract's default is repository source. Select the installed
    # copy explicitly; its deliberate drift tests still use disposable fixtures.
    focused = [root/'tests',root/'verification/test_membership_axis.py',root/'verification/test_preflight.py']
    run(kind+'-focused', [python,'-I','-B','-m','pytest','--import-mode=importlib','-p','no:cacheprovider',
                         '-q','-ra',*focused,'--junitxml='+str(out/(kind+'-focused.xml'))],
        {'UBUKIT_SOURCE_ROOT': site_packages})
    run(kind+'-high-precision', [python,'-I','-B',root/'tools/check_high_precision.py'])
    run(kind+'-example', [python,'-I','-B',root/'examples/som_variants.py'])
    run(kind+'-api', [python,'-I','-B',root/'verification/python_api.py',
                     *(['--with-numba'] if a.with_numba else [])])
    run(kind+'-all-methods', [python,'-I','-B',root/'examples/all_methods.py',
                             *(['--with-numba'] if a.with_numba else [])])
    # Full final regression replay on the wheel. sdist has its own fresh install + focused/API gate.
    if a.full and kind == 'wheel':
        for suite in ('runtime_tests','external_tests'):
            argv = [python,root/'verification'/suite/'run_installed.py','--python',python,
                    '--label','ci-final-'+('numba' if a.with_numba else 'base')]
            if a.with_numba: argv.append('--with-numba')
            if a.include_large_sparse and suite == 'runtime_tests': argv.append('--include-large-sparse')
            run(kind+'-'+suite, argv)
        run('new-tests', [python,'-I','-B','-m','pytest','--import-mode=importlib','-p','no:cacheprovider',
                         '-q','-ra',root/'verification/new_tests','--junitxml='+str(out/'new-tests.xml')])
summary={'status':'passed','full_wheel_regression':a.full,'numba':a.with_numba,
                  'sdist_scope':'fresh install, ownership/hash, current focused suite, 113 oracle cases, facade API, examples',
                  'artifact_sha256':expected,'output':str(out)}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
