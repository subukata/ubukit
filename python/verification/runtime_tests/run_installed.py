#!/usr/bin/env python3
"""Run regression families against a supplied wheel-installed virtualenv."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--python', required=True, help='Absolute virtualenv python executable')
    parser.add_argument('--label', required=True)
    parser.add_argument('--with-numba', action='store_true')
    parser.add_argument('--distribution', default='ubukit')
    parser.add_argument('--include-large-sparse', action='store_true')
    args=parser.parse_args()
    executable=Path(args.python).absolute()
    if not executable.is_file():
        parser.error('virtualenv Python does not exist')
    result_dir=HERE/'results'/args.label
    result_dir.mkdir(parents=True,exist_ok=True)
    empty_cwd=result_dir/'empty_cwd';empty_cwd.mkdir(exist_ok=True)
    env=dict(os.environ)
    for key in ('PYTHONPATH','PYTHONHOME','NUMBA_DISABLE_JIT'):
        env.pop(key,None)
    env.update({key:'1' for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
                                   'NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE',
                                   'PYTEST_DISABLE_PLUGIN_AUTOLOAD')})
    env['NUMBA_CACHE_DIR']=str(result_dir/'numba_cache')
    env['PYTHONHASHSEED']='0'
    stages=[]
    def run(name,command):
        started=time.monotonic()
        output=result_dir/(name+'.log')
        with output.open('w') as f:
            proc=subprocess.run(command,cwd=empty_cwd,env=env,text=True,stdout=f,stderr=subprocess.STDOUT)
        entry={'name':name,'exit_code':proc.returncode,'passed':proc.returncode==0,
               'elapsed_seconds':round(time.monotonic()-started,3),'command':command,
               'log':str(output.relative_to(HERE))}
        stages.append(entry)
        print(json.dumps(entry),flush=True)
        return proc.returncode
    api_argv=[str(HERE/'python_api.py')]+(['--with-numba'] if args.with_numba else [])
    code=("import importlib.util,json,runpy,sys;"
          f"s=importlib.util.spec_from_file_location('_installed_guard',{str(HERE/'_installed_guard.py')!r});"
          "g=importlib.util.module_from_spec(s);s.loader.exec_module(g);"
          f"g.check_installed({args.distribution!r},True);"
          f"sys.argv={api_argv!r};runpy.run_path(sys.argv[0],run_name='__main__');"
          f"print(json.dumps(g.check_installed({args.distribution!r},True)))")
    run('facade_api',[str(executable),'-I','-B','-c',code])
    command=[str(executable),'-I','-B','-m','pytest','--import-mode=importlib',
             '-p','no:cacheprovider','-q','-ra',str(HERE),
             '--ignore='+str(HERE/'_oracles'),'--ignore='+str(HERE/'results'),
             '--ubukit-distribution='+args.distribution,
             '--ubukit-provenance-out='+str(result_dir/'import_provenance.json'),
             '--junitxml='+str(result_dir/'junit.xml')]
    if args.include_large_sparse:command.append('--include-large-sparse')
    run('regressions',command)
    summary={'label':args.label,'python':str(executable),'with_numba':args.with_numba,
             'utc_finished':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'passed':all(s['passed'] for s in stages),'stages':stages,
             'controls':{'isolated_python':True,'bytecode_disabled':True,'pythonpath_unset':True,
                         'third_party_pytest_autoload_disabled':True,'threads':1,
                         'dedicated_numba_cache':env['NUMBA_CACHE_DIR'],
                         'empty_working_directory':str(empty_cwd),
                         'frozen_oracles_are_private_aliases':True,
                         'runtime_record_ownership_and_hash_guard':True},
             'limits':{'large_sparse_8000_point_cases_included':args.include_large_sparse,
                       'excluded_reference_only_fuzzy_tests':13,
                       'external_metrics_covered_in_separate_suite':True,
                       'benchmark_or_external_dataset_runs':False}}
    (result_dir/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    return 0 if summary['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
