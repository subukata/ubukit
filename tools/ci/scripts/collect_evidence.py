"""Copy bounded reports only, never environments/caches/source, to one-day artifacts."""
import argparse,json,os,shutil,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--temp',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
items=[]
for folder in ['python-result','javascript-result','browser-results']:
    base=a.temp/folder
    if base.exists():
        items += [(f,Path(folder)/f.name) for f in base.iterdir() if f.is_file() and f.suffix in {'.log','.json','.xml','.png'}]
for suite in ['runtime_tests','external_tests']:
    base=a.source/'python/verification'/suite/'results/ci-final-base'
    if base.exists():items += [(f,Path(suite)/f.relative_to(base)) for f in base.rglob('*') if f.is_file() and f.suffix in {'.log','.json','.xml'}]
size=sum(f.stat().st_size for f,_ in items)
meta={'candidate_sha':os.environ.get('CANDIDATE_SHA'),'harness_sha':os.environ.get('GITHUB_SHA'),
      'run_id':os.environ.get('GITHUB_RUN_ID'),'runner_os':os.environ.get('RUNNER_OS'),
      'runner_arch':os.environ.get('RUNNER_ARCH'),'python':sys.version,'evidence_bytes':size,'files':len(items)}
(out/'environment.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
# No partial silent truncation: fail the evidence stage if unexpectedly oversized.
if size>10*1024*1024:raise SystemExit('Reports exceed 10 MiB; review live job logs before changing this cap')
for source,relative in items:
    target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
print(json.dumps(meta,indent=2))
