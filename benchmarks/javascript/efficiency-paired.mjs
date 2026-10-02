import {mkdirSync as ensureResultDirectory} from 'node:fs';
ensureResultDirectory(new URL('./results/',import.meta.url),{recursive:true});
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import os from 'node:os';
import { run as oldRun } from '../../javascript/validation/efficiency_baseline_package/src/index.js';
import { run as newRun } from '../../javascript/consumer/node_modules/ubukit-js/src/index.js';
function random(seed) {return ()=> {seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};}
function dataset(n,d,k) {const rng=random(938),data=new Float64Array(n*d);for(let i=0;i<n;i++)for(let f=0;f<d;f++)data[i*d+f]=(i%k)*.05+2*rng()-1;return {data,nSamples:n,nFeatures:d};}
const median=v=>[...v].sort((a,b)=>a-b)[Math.floor(v.length/2)];
const cases=[];
for(const [n,d,k,blockRows] of [[64,2,4,512],[4000,2,64,16],[4000,16,64,16],[3000,64,128,32],[1000,128,256,64],[4000,16,64,512],[1000,3,17,1]]){
 const x=dataset(n,d,k),init=Float64Array.from({length:k*d},(_,j)=>x.data[(j*31)%x.data.length]);
 cases.push({algorithm:'kmeans',n,d,k,blockRows,x,o:{nClusters:k,initCenters:init,kernelBackend:'wasm',blockRows,maxIterations:4},fallback:{wasmCenterCache:false}});
}
for(const algorithm of ['som_batch','som'])for(const [n,d,k] of [[64,2,4],[1000,2,256],[1000,7,64],[600,32,256],[256,128,64]]) {
 const x=dataset(n,d,k),width=Math.sqrt(k),init=Float64Array.from({length:k*d},(_,j)=>x.data[(j*31)%x.data.length]);
 cases.push({algorithm,n,d,k,blockRows:128,x,o:{gridShape:[width,width],initialPrototypes:init,maxIterations:algorithm==='som'?64:3,sigma:2,sigmaEnd:.5,blockRows:128},fallback:{bmuBackend:'scalar'}});
}
// Pure projection is an end-to-end public SOM call with validation, ownership,
// memory accounting, BMU projection and embedding, not a synthetic kernel loop.
for(const [n,d,k] of [[4000,2,256],[2000,16,256],[1000,128,256]]){const x=dataset(n,d,k);cases.push({algorithm:'som_batch',purpose:'projection-only',n,d,k,blockRows:128,x,o:{gridShape:[16,16],maxIterations:0,initialPrototypes:x.data.slice(0,k*d)},fallback:{bmuBackend:'scalar'}});}
const records=[];
for(const c of cases){
 const old=()=>oldRun(c.algorithm,c.x,c.o),candidate=()=>newRun(c.algorithm,c.x,c.o),retained=()=>newRun(c.algorithm,c.x,{...c.o,...c.fallback});
 assert.deepEqual(candidate(),old());assert.deepEqual(retained(),old());
 for(let i=0;i<3;i++){old();candidate();retained();}
 const timings={old:[],candidate:[],retained:[]};const methods={old,candidate,retained};
 if(c.algorithm!=='kmeans'){methods.grouped=()=>newRun(c.algorithm,c.x,{...c.o,bmuBackend:'grouped'});timings.grouped=[];assert.deepEqual(methods.grouped(),old());for(let i=0;i<3;i++)methods.grouped();}
 let result;
 const repetitions=Number(process.env.REPS??9);
 for(let trial=0;trial<repetitions;trial++){
  const order=Object.keys(methods);if(trial%2)order.reverse();
  for(const name of order){const start=performance.now();result=methods[name]();timings[name].push(performance.now()-start);}
 }
 const medians=Object.fromEntries(Object.entries(timings).map(([key,value])=>[key,median(value)]));
 const row={algorithm:c.algorithm,purpose:c.purpose??'training-and-final-projection',n:c.n,d:c.d,k:c.k,blockRows:c.blockRows,iterations:result.iterations,requestedIterations:c.o.maxIterations,medianMs:medians,speedupOldOverCandidate:medians.old/medians.candidate,speedupRetainedOverCandidate:medians.retained/medians.candidate,speedupOldOverGrouped:medians.grouped?medians.old/medians.grouped:null,rawMs:timings,estimatedPrimaryBytes:result.stats?.estimatedPrimaryBytes??result.kernel?.estimatedPeakPrimaryBytes,workspaceBytes:result.kernel?.allocatedWorkspaceBytes??0};records.push(row);console.log(JSON.stringify(row));
}
const sourceManifest=JSON.parse(fs.readFileSync(new URL('../../javascript/SOURCE_MANIFEST.json',import.meta.url)));
const sourceManifestSHA256=createHash('sha256').update(fs.readFileSync(new URL('../../javascript/SOURCE_MANIFEST.json',import.meta.url))).digest('hex');
const report={sourceManifestSHA256,runtimeHashes:Object.fromEntries(Object.entries(sourceManifest.runtime_files).map(([path,v])=>[path,v.sha256])),date:new Date().toISOString(),node:process.version,v8:process.versions.v8,platform:process.platform,cpu:os.cpus()[0].model,cpus:os.cpus().length,method:'one process, three untimed warmups; nine alternating old/new-default/retained-scalar-or-uncached/explicit-grouped whole-call rounds, median; original package and candidate imported separately; correctness deep-equality before timing; advisory lock excludes other coordinated timing jobs, background load not guaranteed absent; no forced GC',records};fs.writeFileSync(new URL('./results/efficiency-paired.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
