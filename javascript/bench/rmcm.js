// Bounded, matched, single-thread Node CPU observations. No browser claims.
import { performance } from 'node:perf_hooks';
import os from 'node:os';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { dirname } from 'node:path';
import { prepareRMCM, rmcm, seededRandom } from '../src/index.js';
const trials=5, maxIterations=12, seed=82317;
const median=a=>[...a].sort((a,b)=>a-b)[Math.floor(a.length/2)];
const time=fn=>{const t=performance.now(),value=fn();return {ms:performance.now()-t,value};};
function data(n,d,k){const random=seededRandom(seed),x={data:Float64Array.from({length:n*d},()=>10*random()-5),nSamples:n,nFeatures:d};const init=Float64Array.from({length:k*d},(_,z)=>x.data[(Math.floor(z/d)*137%n)*d+z%d]);return {x,init};}
function compare(a,b){let maxCenterAbsError=0,maxMembershipAbsError=0,labelDifferences=0;for(let i=0;i<a.centers.length;++i)maxCenterAbsError=Math.max(maxCenterAbsError,Math.abs(a.centers[i]-b.centers[i]));if(a.membership&&b.membership)for(let i=0;i<a.membership.length;++i)maxMembershipAbsError=Math.max(maxMembershipAbsError,Math.abs(a.membership[i]-b.membership[i]));for(let i=0;i<a.labels.length;++i)labelDifferences+=Number(a.labels[i]!==b.labels[i]);return{maxCenterAbsError,maxMembershipAbsError,labelDifferences,sameIterations:a.iterations===b.iterations,sameStopReason:a.stopReason===b.stopReason};}
const report={runtime:{node:process.version,platform:process.platform,arch:process.arch,cpu:os.cpus()[0]?.model},mode:'warm Node CPU, single-thread JavaScript, five rotating-order repetitions; not a browser benchmark',trials,maxIterations,seed,definitions:{reference:'Explicit R=PH every iteration, then R^T X / R^T 1; same hard assignment and graph builder as adjoint',adjoint:'Prepare Y=P^T X and s=P^T1 once; per iteration aggregate Y and s by original-X hard labels',prepare:'Snapshot, direct O(N²D) two-pass CSR construction, and backend precomputation',fullFit:'One-shot rmcm: preparation plus fit plus requested final R',reusedFit:'Prepared graph/backend arrays reused; fit plus requested final R',ratio:'reference median / adjoint median; above 1 favors adjoint'},sourceSha256:crypto.createHash('sha256').update(fs.readFileSync(new URL('../src/rmcm.js',import.meta.url))).digest('hex'),cases:[]};
for(const shape of [{name:'sparse',n:1000,d:4,k:8,delta:.7},{name:'moderate',n:1000,d:4,k:8,delta:4},{name:'complete',n:400,d:4,k:8,delta:100}]){
 const {x,init}=data(shape.n,shape.d,shape.k),options={delta:shape.delta,initCenters:init,maxIterations,cycleWindow:32};
 const prepared=Object.fromEntries(['reference','adjoint'].map(backend=>[backend,prepareRMCM(x,{...options,backend})]));
 const result={...shape,nEdges:prepared.adjoint.nEdges,directedDensity:prepared.adjoint.nEdges/(shape.n*shape.n),prepare:{},membershipModes:[]};
 // Warm both drivers and both output modes before taking steady-state observations.
 for(const backend of ['reference','adjoint'])for(const returnMembership of [true,false])prepared[backend].fit({...options,returnMembership});
 const prepSamples={reference:[],adjoint:[]};
 for(let trial=0;trial<trials;++trial)for(const backend of trial%2?['adjoint','reference']:['reference','adjoint'])prepSamples[backend].push(time(()=>prepareRMCM(x,{...options,backend})).ms);
 for(const backend of ['reference','adjoint'])result.prepare[backend]={samplesMs:prepSamples[backend],medianMs:median(prepSamples[backend])};
 result.prepare.referenceOverAdjoint=result.prepare.reference.medianMs/result.prepare.adjoint.medianMs;
 for(const returnMembership of [true,false]){
  const mode={returnMembership,backends:{},parity:null};const values={};
  for(const backend of ['reference','adjoint']){mode.backends[backend]={fullFitSamplesMs:[],reusedFitSamplesMs:[]};values[backend]=prepared[backend].fit({...options,returnMembership});}
  mode.parity=compare(values.reference,values.adjoint);
  if(mode.parity.labelDifferences||!mode.parity.sameIterations||!mode.parity.sameStopReason||mode.parity.maxCenterAbsError>1e-10||mode.parity.maxMembershipAbsError>1e-12)throw new Error(`unmatched arithmetic trajectory in ${shape.name}: ${JSON.stringify(mode.parity)}`);
  for(let trial=0;trial<trials;++trial)for(const backend of trial%2?['adjoint','reference']:['reference','adjoint']){
   mode.backends[backend].fullFitSamplesMs.push(time(()=>rmcm(x,{...options,backend,returnMembership})).ms);
   mode.backends[backend].reusedFitSamplesMs.push(time(()=>prepared[backend].fit({...options,returnMembership})).ms);
  }
  for(const backend of ['reference','adjoint']){const b=mode.backends[backend],v=values[backend];b.fullFitMedianMs=median(b.fullFitSamplesMs);b.reusedFitMedianMs=median(b.reusedFitSamplesMs);b.iterations=v.iterations;b.stopReason=v.stopReason;b.cycleLength=v.cycleLength;b.emptyClusterUpdates=v.emptyClusterUpdates;b.estimatedPrimaryBytes=v.estimatedPrimaryBytes;}
  mode.fullFitReferenceOverAdjoint=mode.backends.reference.fullFitMedianMs/mode.backends.adjoint.fullFitMedianMs;
  mode.reusedFitReferenceOverAdjoint=mode.backends.reference.reusedFitMedianMs/mode.backends.adjoint.reusedFitMedianMs;
  result.membershipModes.push(mode);
 }
 report.cases.push(result);
}
report.notes=['Inputs and explicit center initialization are shared. Identical seed numbers are not NumPy RNG parity.','Graph construction is repeated in both full-fit measurements; reused-fit timing excludes it.','The final dense N×K R is included when requested. Reference needs dense R internally even when returnMembership=false.','Complete-graph common-mean shortcut applies equally to both backends.','Ratios are local observations, not a universal speedup or a browser/Python comparison.','Memory estimates cover primary owned numeric arrays, not VM/GC/TypedArray object overhead.'];
const output=process.argv[2]??new URL('../results/benchmark-node.json',import.meta.url);fs.mkdirSync(typeof output === 'string' ? dirname(output) : new URL('.', output), { recursive: true });fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');
for(const c of report.cases)for(const m of c.membershipModes)console.log(`${c.name}, R=${m.returnMembership}, E=${c.nEdges}, full ${m.backends.reference.fullFitMedianMs.toFixed(3)}/${m.backends.adjoint.fullFitMedianMs.toFixed(3)} ms (${m.fullFitReferenceOverAdjoint.toFixed(2)}×), reused ${m.backends.reference.reusedFitMedianMs.toFixed(3)}/${m.backends.adjoint.reusedFitMedianMs.toFixed(3)} ms (${m.reusedFitReferenceOverAdjoint.toFixed(2)}×), iter=${m.backends.adjoint.iterations}`);
