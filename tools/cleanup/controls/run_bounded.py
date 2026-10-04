"""Correctness/build control. No performance measurements or claims."""
from pathlib import Path
import argparse, datetime, json, os, resource, signal, subprocess, sys
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--cwd',default='.');p.add_argument('--cpu',type=int,default=60);p.add_argument('--wall',type=int,default=90);p.add_argument('--memory-mib',type=int,default=1024);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
root=Path(__file__).resolve().parents[1];cmd=a.command
if cmd and cmd[0]=='--':cmd=cmd[1:]
assert cmd
record={'label':a.label,'purpose':'bounded build/install/correctness; no performance benchmark','argv':cmd,'cwd':str(Path(a.cwd).resolve()),'cpu_seconds_per_process':a.cpu,'wall_seconds_process_group':a.wall,'memory_mib_per_process':a.memory_mib or None,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',SOURCE_DATE_EPOCH='1791108000',npm_config_offline='true',npm_config_audit='false',npm_config_fund='false');env.pop('PYTHONPATH',None);env['npm_config_cache']=str(root/'execution_cache/npm')
def limits():
 os.setsid();resource.setrlimit(resource.RLIMIT_CPU,(a.cpu,a.cpu))
 if a.memory_mib:resource.setrlimit(resource.RLIMIT_AS,(a.memory_mib*1024**2,)*2)
with (root/'evidence'/f'{a.label}.log').open('w') as out:
 proc=subprocess.Popen(cmd,cwd=a.cwd,env=env,preexec_fn=limits,stdout=out,stderr=subprocess.STDOUT)
 try:code=proc.wait(timeout=a.wall)
 except subprocess.TimeoutExpired:
  os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124;record['timed_out']=True
record.update(exit_code=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(root/'evidence'/f'{a.label}.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));print('\n'.join((root/'evidence'/f'{a.label}.log').read_text().splitlines()[-12:]));sys.exit(code)
