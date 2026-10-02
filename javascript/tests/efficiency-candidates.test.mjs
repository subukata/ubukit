import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { run, createSession, createWorkerClient, createRealtimeWorkerClient } from '../consumer/node_modules/ubukit-js/src/index.js';
import { somBmuScalar } from '../consumer/node_modules/ubukit-js/src/som.js';
import { nearestCenter2d, nearestCenterGrouped } from '../consumer/node_modules/ubukit-js/src/iteration-kernels.js';
import { kmeansWasmWorkspace } from '../consumer/node_modules/ubukit-js/src/kmeans-wasm.js';
import { run as oldRun } from '../validation/efficiency_baseline_package/src/index.js';
function random(seed) {return ()=> {seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};}
function input(n,d,seed=7) {const rng=random(seed);return {data:Float64Array.from({length:n*d},()=>4*rng()-2),nSamples:n,nFeatures:d};}
function capture(fn) {try{return {value:fn()};}catch(e){return {error:e.name,message:e.message};}}
function finish(s) {let count=0;while(!s.status.done){s.step(1,{timeBudgetMs:0,maxChunks:1});assert.ok(++count<100000);}return s.snapshot().result;}
for(const d of [1,2,3,4,7,8,16,33,129]) for(const k of [1,2,3,4,5,8,17]) {
 test(`BMU scalar identity D${d} K${k}`,()=>{const x=input(20,d),w=input(k,d,17).data;for(let i=0;i<20;i++){const expected=somBmuScalar(x.data,i*d,w,d,k);assert.equal(nearestCenterGrouped(x.data,i*d,w,d,k),expected);if(d===2)assert.equal(nearestCenter2d(x.data,i*d,w,d,k),expected);}});
}
for(const d of [1,2,3,4,5,8]) test(`BMU extreme exception and tie identity D${d}`,()=>{
 const values=[0,-0,Number.MIN_VALUE,1e-200,1e-160,1e-154,1e-100,1,1e100,1e154,1e155,1e308,-1e308],rng=random(78+d);
 for(let t=0;t<1000;t++){const k=1+t%13,x=Float64Array.from({length:d},()=>values[Math.floor(rng()*values.length)]),w=Float64Array.from({length:k*d},()=>values[Math.floor(rng()*values.length)]);
  const expected=capture(()=>somBmuScalar(x,0,w,d,k));assert.deepEqual(capture(()=>nearestCenterGrouped(x,0,w,d,k)),expected);if(d===2)assert.deepEqual(capture(()=>nearestCenter2d(x,0,w,d,k)),expected);
 }
 const zero=new Float64Array(d),ties=new Float64Array(d*9);assert.equal(nearestCenterGrouped(zero,0,ties,d,9),0);
});
for(const algorithm of ['som','som_batch']) for(const d of [2,7,16]) test(`${algorithm} exact trajectory + memory D${d}`,()=>{
 const x=input(73,d),o={gridShape:[3,3],initialPrototypes:input(9,d,3).data,maxIterations:algorithm==='som'?29:3,blockRows:7,blockUnits:4};
 const expected=oldRun(algorithm,x,o);assert.deepEqual(run(algorithm,x,o),expected);assert.deepEqual(run(algorithm,x,{...o,bmuBackend:'scalar'}),expected);assert.deepEqual(run(algorithm,x,{...o,bmuBackend:'grouped'}),expected);
 assert.deepEqual(run(algorithm,x,{...o,maxScratchBytes:Math.max(1,expected.stats.primaryTrainingScratchBytes??1024*1024)}),expected);
 const s=createSession(algorithm,x,o);assert.deepEqual(finish(s),expected);s.updateParameters({bmuBackend:'scalar'});assert.ok(finish(s).centers.every(Number.isFinite));
 const before=s.snapshot();assert.throws(()=>s.updateParameters({bmuBackend:'invalid'}));assert.deepEqual(s.snapshot(),before);
});
for(const d of [2,7,16,33]) for(const blockRows of [1,17,513]) test(`cached WASM exact fit D${d} B${blockRows}`,()=>{
 const x=input(133,d),o={nClusters:9,initCenters:input(9,d,14).data,maxIterations:5,blockRows,kernelBackend:'wasm'};
 const old=oldRun('kmeans',x,o);assert.deepEqual(run('kmeans',x,o),old);assert.deepEqual(run('kmeans',x,{...o,wasmCenterCache:false}),old);
 assert.equal(run('kmeans',x,o).kernel.allocatedWorkspaceBytes,old.kernel.allocatedWorkspaceBytes);
});
test('WASM cache invalidates mutations of the same center buffer',()=>{
 const w=kmeansWasmWorkspace(1,2,2,1,65536);assert.ok(w);const c=Float64Array.of(0,0,10,10),x=Float64Array.of(9,9);
 w.centers(c,true);w.nearest(x,0,1);assert.equal(w.labels[0],1);c.set([10,10,0,0]);w.invalidateCenters();w.centers(c,true);w.nearest(x,0,1);assert.equal(w.labels[0],0);
 c.set([0,0,10,10]);w.centers(c,false);w.nearest(x,0,1);assert.equal(w.labels[0],1);
});
test('WASM memory-denied fallback remains identical',()=>{
 const x=input(45,7),o={nClusters:5,maxIterations:3,kernelBackend:'wasm',maxMemoryBytes:16*45*7+8*45+24*5*7+32*5};assert.deepEqual(run('kmeans',x,o),oldRun('kmeans',x,o));assert.equal(run('kmeans',x,o).kernel.actual,'javascript');
});
test('cached WASM sessions restart safely and reject invalid updates transactionally',()=>{
 const x=input(73,7),o={nClusters:8,maxIterations:4,blockRows:9,kernelBackend:'wasm'};const s=createSession('kmeans',x,o);assert.deepEqual(finish(s),run('kmeans',x,o));s.updateParameters({wasmCenterCache:false});assert.ok(finish(s).centers.every(Number.isFinite));s.updateData(input(61,7,12));assert.ok(finish(s).centers.every(Number.isFinite));s.reset();assert.ok(finish(s).centers.every(Number.isFinite));const before=s.snapshot();assert.throws(()=>s.updateParameters({wasmCenterCache:'yes'}));assert.deepEqual(s.snapshot(),before);
});
for(const algorithm of ['kmeans','som','som_batch']) test(`${algorithm} candidate module and realtime workers`,async()=>{
 const x=input(53,2),o=algorithm==='kmeans'?{nClusters:8,maxIterations:3,blockRows:5,kernelBackend:'wasm'}:{gridShape:[4,4],maxIterations:3,blockRows:5};const expected=run(algorithm,x,o);
 const a=createWorkerClient({workerFactory:()=>new Worker(new URL('../consumer/node_modules/ubukit-js/src/worker.js',import.meta.url),{type:'module'})}),b=createRealtimeWorkerClient({workerFactory:()=>new Worker(new URL('../consumer/node_modules/ubukit-js/src/realtime-worker.js',import.meta.url),{type:'module'})});
 try {assert.deepEqual(await a.run(algorithm,x,o),expected);assert.deepEqual((await b.run(algorithm,x,o,{timeBudgetMs:0,maxChunks:1})).result,expected);} finally {a.dispose();b.dispose();}
});

// Independent review found a hook added after the first yield can expose live
// centers. Cache invalidation must latch off even if that hook removes itself.
import { kmeansSteps as newKmeansSteps } from '../consumer/node_modules/ubukit-js/src/clustering.js';
import { kmeansSteps as oldKmeansSteps } from '../validation/efficiency_baseline_package/src/clustering.js';
import { SESSION_CHECKPOINT as newCheckpoint } from '../consumer/node_modules/ubukit-js/src/session-hooks.js';
import { SESSION_CHECKPOINT as oldCheckpoint } from '../validation/efficiency_baseline_package/src/session-hooks.js';
for(const removeAfterFirst of [false,true])test(`WASM late checkpoint exposure stays live; remove=${removeAfterFirst}`,()=>{
 function trace(steps,hook){const x=input(17,3,12),o={nClusters:4,initCenters:input(4,3,32).data,maxIterations:3,blockRows:2,kernelBackend:'wasm'},events=[];let live=null,index=0;
  const callback=s=>{live=s.centers;if(removeAfterFirst)delete o[hook];};const it=steps(x,o);
  for(;;){const v=it.next();if(v.done)return {events,result:v.value};events.push(structuredClone(v.value));index++;if(index===1)o[hook]=callback;if(live)live[index%live.length]+=.75;}
 }
 assert.deepEqual(trace(newKmeansSteps,newCheckpoint),trace(oldKmeansSteps,oldCheckpoint));
});

test('elite auto dispatch preserves malformed shape validation',()=>{
 const x=input(3,2);
 for(const gridShape of [[],[1],[1,2,3],[0,2],[-1,2],[1n,2],new BigInt64Array([1n,2n]),new DataView(new ArrayBuffer(8)),[NaN,2],['2',2]]){
  const o={gridShape,maxIterations:0};assert.deepEqual(capture(()=>run('som',x,o)),capture(()=>oldRun('som',x,o)));
 }
});
