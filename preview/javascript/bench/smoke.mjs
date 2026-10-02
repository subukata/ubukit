import {performance} from 'node:perf_hooks';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {gzipSync} from 'node:zlib';
import {cpus} from 'node:os';
import {TPEOptimizer,optimize} from '../consumer/node_modules/ubukit-js/src/optimization.js';
import {problems} from './problems.mjs';
const median=x=>{const a=x.slice().sort((a,b)=>a-b);return a.length%2?a[a.length>>1]:(a[a.length/2-1]+a[a.length/2])/2;};
const quantile=(x,q)=>x.slice().sort((a,b)=>a-b)[Math.min(x.length-1,Math.floor(q*x.length))];
const selected=problems.filter(p=>['sphere_4d','rotated_2d','mixed','boundary_4d'].includes(p.name));
const seeds=[0,1,2,3,4],budget=50,raw=[],summary=[];
for(const p of selected){
 const rows=[];
 for(const seed of seeds){
  const values={};
  for(const sampler of ['random','joint','independent']){
   const start=performance.now();const r=optimize(p.objective,p.space,{seed,nTrials:budget,sampler:sampler==='random'?'random':'tpe',multivariate:sampler!=='independent'});
   values[sampler]=r.bestValue;raw.push({problem:p.name,seed,sampler,budget,attempted:r.nAttempted,completed:r.nCompleted,best:r.bestValue,elapsedMs:performance.now()-start});
  }
  rows.push(values);
 }
 summary.push({problem:p.name,medianRandom:median(rows.map(x=>x.random)),medianJoint:median(rows.map(x=>x.joint)),medianIndependent:median(rows.map(x=>x.independent)),jointWins:rows.filter(x=>x.joint<x.random).length,independentWins:rows.filter(x=>x.independent<x.random).length,seeds:seeds.length});
}
const timing=[];
for(const dimensions of [2,8])for(const observations of [32,128]){
 const space=Object.fromEntries(Array.from({length:dimensions},(_,i)=>['x'+i,{type:'float',low:0,high:1}]));
 const make=()=>{const o=new TPEOptimizer(space,{seed:42});for(let i=0;i<observations;i++)o.addTrial(Object.fromEntries(Object.keys(space).map((k,j)=>[k,((i*31+j*17)%997)/997])),(i*13)%101);return o;};
 for(let warm=0;warm<5;warm++){const o=make();o.ask();}
 const rebuild=[],cached=[];
 for(let repeat=0;repeat<25;repeat++){const o=make();let start=performance.now();o.ask();rebuild.push(performance.now()-start);start=performance.now();o.ask();cached.push(performance.now()-start);}
 timing.push({dimensions,observations,nCandidates:24,rebuildMedianMs:median(rebuild),rebuildP95Ms:quantile(rebuild,.95),cachedMedianMs:median(cached),cachedP95Ms:quantile(cached,.95)});
}
// Coarse retained-heap estimate over 30 independent optimizers; not a peak bound.
const heapSpace=Object.fromEntries(Array.from({length:8},(_,i)=>['x'+i,{type:'float',low:0,high:1}]));
global.gc?.();const before=process.memoryUsage().heapUsed;const retained=[];
for(let k=0;k<30;k++){const o=new TPEOptimizer(heapSpace,{seed:k});for(let i=0;i<128;i++)o.addTrial(Object.fromEntries(Object.keys(heapSpace).map((n,j)=>[n,((i*31+j*17)%997)/997])),(i*13)%101);o.ask();retained.push(o);}
global.gc?.();const heapDelta=process.memoryUsage().heapUsed-before;
const source=readFileSync(new URL('../consumer/node_modules/ubukit-js/src/optimization.js',import.meta.url));
const report={environment:{node:process.version,platform:process.platform,arch:process.arch,cpu:cpus()[0]?.model,gcExposed:typeof global.gc==='function'},scope:'Four synthetic problems, five seeds, 50 objective attempts; lightweight smoke only. No Optuna comparison or superiority claim.',defaults:{nStartupTrials:12,nCandidates:24,gamma:.15,minBandwidth:.03,weights:'ei'},sourceBytes:source.length,gzipBytes:gzipSync(source).length,dependencyCount:0,workerCount:0,summary,timing,heap:{optimizers:30,observations:128,dimensions:8,retainedHeapDeltaBytes:heapDelta,estimatedBytesPerOptimizer:heapDelta/30,note:'Coarse V8 retained-heap delta after GC, not maximum RSS or an allocation bound; environment may have contention.'},raw};
mkdirSync(new URL('../benchmark_replay_reports/',import.meta.url),{recursive:true});
writeFileSync(new URL('../benchmark_replay_reports/smoke.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({summary,timing,heap:report.heap,sourceBytes:report.sourceBytes,gzipBytes:report.gzipBytes},null,2));
