#!/usr/bin/env python3
"""Run frozen external-metrics validation and new facade/route checks installed."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--python',required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--with-numba',action='store_true')
    parser.add_argument('--distribution',default='ubukit-bundled-local-preview')
    args=parser.parse_args()
    executable=Path(args.python).absolute()
    if not executable.is_file():parser.error('supplied virtualenv Python does not exist')
    result=HERE/'results'/args.label;result.mkdir(parents=True,exist_ok=True)
    cwd=result/'empty_cwd';cwd.mkdir(exist_ok=True)
    env=dict(os.environ)
    for k in ('PYTHONPATH','PYTHONHOME','NUMBA_DISABLE_JIT'):env.pop(k,None)
    env.update({k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTEST_DISABLE_PLUGIN_AUTOLOAD')})
    env.update(NUMBA_CACHE_DIR=str(result/'numba_cache'),PYTHONHASHSEED='0',UBUKIT_EXTERNAL_RESULTS=str(result),UBUKIT_DISTRIBUTION=args.distribution)
    stages=[]
    def run(name,command):
        start=time.monotonic()
        with (result/(name+'.log')).open('w') as f:
            completed=subprocess.run(command,cwd=cwd,env=env,text=True,stdout=f,stderr=subprocess.STDOUT)
        entry={'name':name,'exit_code':completed.returncode,'passed':completed.returncode==0,
               'elapsed_seconds':round(time.monotonic()-start,3),'command':command,'log':str((result/(name+'.log')).relative_to(HERE))}
        stages.append(entry);print(json.dumps(entry),flush=True)
        return completed.returncode
    def script(name):
        guard=str(HERE/'_installed_guard.py')
        code=("import importlib.util,importlib.metadata,json,os,runpy,sys;"
              f"s=importlib.util.spec_from_file_location('_installed_guard',{guard!r});"
              "g=importlib.util.module_from_spec(s);s.loader.exec_module(g);"
              f"g.check_installed({args.distribution!r},True);"
              f"sys.argv=[{str(HERE/(name+'.py'))!r}];runpy.run_path(sys.argv[0],run_name='__main__');"
              f"r=g.check_installed({args.distribution!r},True);"
              f"open({str(result/(name+'_import_provenance.json'))!r},'w').write(json.dumps(r,indent=2)+chr(10))")
        return [str(executable),'-I','-B','-c',code]
    preflight=("import importlib.util,importlib.metadata,json,sys;"
               "names=['numpy','scipy','scikit-learn','numba'];"
               "versions={n:importlib.metadata.version(n) for n in names if importlib.util.find_spec('sklearn' if n=='scikit-learn' else n)};"
               f"assert ('numba' in versions)=={args.with_numba!r},versions;"
               f"r={{'python':sys.version,'versions':versions,'numba_expected':{args.with_numba!r}}};"
               f"open({str(result/'environment.json')!r},'w').write(json.dumps(r,indent=2)+chr(10));print(json.dumps(r))")
    if run('environment',[str(executable),'-I','-B','-c',preflight]):return 1
    run('test_portability',script('test_portability'))
    command=[str(executable),'-I','-B','-m','pytest','--import-mode=importlib','-p','no:cacheprovider','-q','-ra',str(HERE/'test_facade_external_metrics.py'),'--junitxml='+str(result/'facade_junit.xml')]
    run('facade_and_routes',command)
    run('test_external_metrics',script('test_external_metrics'))
    summary={'label':args.label,'passed':all(s['passed'] for s in stages),'python':str(executable),'with_numba':args.with_numba,'utc_finished':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stages':stages,'controls':{'isolated_python':True,'pythonpath_unset':True,'single_thread':True,'numba_cache':env['NUMBA_CACHE_DIR'],'installed_record_and_hash_guard':True,'original_validation_fixture_sizes_and_tolerances_unchanged':True,'rebenchmark':False}}
    for source,key in [('validation.json','numerical_validation'),('portability_validation.json','portability_validation'),('environment.json','environment')]:
        if (result/source).exists():summary[key]=json.loads((result/source).read_text())
    if (result/'facade_junit.xml').exists():
        suite=ET.parse(result/'facade_junit.xml').getroot().find('testsuite');summary['facade_pytest']=dict(suite.attrib)
    (result/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'label':args.label,'passed':summary['passed'],'summary':str(result/'SUMMARY.json')}),flush=True)
    return 0 if summary['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
