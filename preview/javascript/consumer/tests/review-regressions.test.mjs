import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { Worker } from 'node:worker_threads';
import { createSession, createRealtimeWorkerClient, createMetricScheduler, run } from 'ubukit-js';
const X = {data:Float64Array.of(0,1,2,4,5,6), nSamples:6,nFeatures:1};
const grid = {data:Float64Array.of(0,1),nSamples:2,nFeatures:1};
const options = algorithm => ({nClusters:2,initCenters:Uint8Array.of(0,6),maxIterations:3,blockRows:1,tolerance:0,...algorithm==='rmcm'?{delta:1.1,graphBatchPairs:1}:{},...algorithm==='som-olp'?{grid,lambda:.4}:{}});
function finish(s) { for(let count=0;!s.status.done;count++) {assert.ok(count<10000,'session terminates');s.step(100,{timeBudgetMs:1000})}return s.snapshot().result; }
for (const [algorithm,patch] of [['kmeans',{kernelBackend:'typo'}],['som-olp',{kernelBackend:'typo'}],['rmcm',{graphBackend:'typo'}],['som-olp',{maxMemoryBytes:1000000.5}]]) {
  test(`${algorithm}: ${JSON.stringify(patch)} is rejected without changing ready, partial or completed state`,()=>{
    const o=options(algorithm);
    assert.throws(()=>createSession(algorithm,X,{...o,...patch}),{name:'RangeError'});
    for (const phase of ['ready','partial','done']) {
      const s=createSession(algorithm,X,o);
      if(phase==='partial')s.step(1,{maxChunks:1,timeBudgetMs:0});
      if(phase==='done')finish(s);
      const before=s.snapshot();
      assert.throws(()=>s.updateParameters(patch),{name:'RangeError'});assert.deepEqual(s.snapshot(),before);
      assert.throws(()=>s.configure(X,{...o,...patch}),{name:'RangeError'});assert.deepEqual(s.snapshot(),before);
      assert.deepEqual(finish(s),run(algorithm,X,o));s.dispose();
    }
  });
}
for (const algorithm of ['kmeans','fcm','rcm','exrcm','rmcm','som-olp']) {
  test(`${algorithm}: Buffer initialization is copied before caller mutation`,()=>{
    const initCenters=Buffer.from([0,6]),o={...options(algorithm),initCenters};
    const expected=run(algorithm,X,{...o,initCenters:Uint8Array.of(0,6)});
    const s=createSession(algorithm,X,o);initCenters.fill(99);
    assert.deepEqual(finish(s),expected);s.dispose();
  });
}
test('FCM Buffer memberships remain owned across constructor and parameter update',()=>{
  const original=Uint8Array.of(1,0,1,0,1,0,0,1,0,1,0,1);
  for(const method of ['constructor','update']){
    const membership=Buffer.from(original),o={nClusters:2,initMembership:membership,maxIterations:1,tolerance:0};
    const s=method==='constructor'?createSession('fcm',X,o):createSession('fcm',X,{nClusters:2,maxIterations:1,tolerance:0});
    if(method==='update')s.updateParameters({initMembership:membership},{warmStart:false});
    membership.fill(1);
    assert.deepEqual(finish(s),run('fcm',X,{...o,initMembership:original}));s.dispose();
  }
});
test('SOM snapshot preserves aliases while owning numerical storage',()=>{
  const s=createSession('som-olp',X,options('som-olp'));finish(s);
  const a=s.snapshot().result,b=s.snapshot().result;
  assert.equal(a.W,a.centers);assert.equal(a.prototypes,a.centers);assert.equal(a.P,a.membership);
  assert.notEqual(a.centers.buffer,b.centers.buffer);a.centers.fill(999);assert.deepEqual(s.snapshot().result.centers,b.centers);s.dispose();
});
test('coalesced realtime pending Buffer input is snapshotted before dispatch',async()=>{
  class Fake extends EventEmitter {sent=[];postMessage(m){this.sent.push(structuredClone(m))}terminate(){}}
  const worker=new Fake(),client=createRealtimeWorkerClient({workerFactory:()=>worker});
  const old=client.run('kmeans',X,options('kmeans')).catch(e=>e);
  const data=Buffer.from([0,1,2,4,5,6]),latest=client.run('kmeans',{...X,data},options('kmeans')).catch(e=>e);
  data.fill(99);worker.emit('message',{id:1,type:'cancelled'});
  const dispatched=worker.sent.find(m=>m.type==='run'&&m.id===2);
  assert.deepEqual([...dispatched.input.data],[0,1,2,4,5,6]);
  worker.emit('message',{id:2,type:'result',snapshot:{requestId:2}});
  assert.equal((await old).name,'AbortError');assert.equal((await latest).requestId,2);assert.equal(client.busy,false);client.dispose();
});
test('real Node realtime Worker handles owned pending Buffer input and validation rejection reuse',async()=>{
  const workerURL=new URL('./realtime-worker.js',import.meta.resolve('ubukit-js/realtime-worker'));
  const client=createRealtimeWorkerClient({workerFactory:()=>new Worker(workerURL,{type:'module'})});
  try {
    const o=options('kmeans'),old=client.run('kmeans',X,o).catch(e=>e);
    const data=Buffer.from([0,1,2,4,5,6]);
    const latest=client.run('kmeans',{...X,data},o,{warmStart:false});data.fill(99);
    assert.equal((await old).name,'AbortError');assert.deepEqual((await latest).result,run('kmeans',X,o));
    await assert.rejects(client.run('kmeans',X,{...o,kernelBackend:'typo'}),{name:'RangeError'});
    assert.deepEqual((await client.run('kmeans',X,o,{warmStart:false})).result,run('kmeans',X,o));
  }finally{client.dispose()}
});
test('metric scheduler owns Buffer embedding before debounce and caches original contents',async()=>{
  const scheduler=createMetricScheduler({debounceMs:1});
  try {
    const embedding={...X,data:Buffer.from([0,1,2,4,5,6])};
    const pending=scheduler.run(X,{embedding,k:1});embedding.data.set([0,6,1,5,2,4]);
    const expected=run('neighborhood',X,{embedding:X,k:1});
    assert.deepEqual((await pending).result,expected);
    const hit=await scheduler.run(X,{embedding:{...X,data:Buffer.from([0,1,2,4,5,6])},k:1});
    assert.equal(hit.cacheHit,true);assert.deepEqual(hit.result,expected);
  } finally {scheduler.dispose()}
});
test('valid acceleration selections remain accepted through configuration',()=>{
  for(const algorithm of ['kmeans','som-olp','rmcm']) {
    const o=options(algorithm),s=createSession(algorithm,X,o);finish(s);
    const patch=algorithm==='rmcm'?{graphBackend:'wasm-simd'}:{kernelBackend:'wasm'};
    assert.doesNotThrow(()=>s.updateParameters(patch));assert.ok(finish(s));
    assert.doesNotThrow(()=>s.configure(X,{...o,...patch},{warmStart:false}));assert.ok(finish(s));s.dispose();
  }
});
for(const m of [2,2.5,3])test(`FCM m=${m}: tiny weights with normal mass preserve representable weighted centers`,()=>{
  const data={data:Float64Array.of(0,1e-20),nSamples:2,nFeatures:1};
  const small=(4e-308)**(1/m),initMembership=Float64Array.of(1,small,1,small);
  const o={nClusters:2,initMembership,m,maxIterations:1,tolerance:0,blockRows:1};
  const actual=run('fcm',data,o),expected=5e-21;
  assert.ok(Math.abs(actual.centers[0]/expected-1)<1e-14);
  assert.ok(Math.abs(actual.centers[1]/expected-1)<1e-14);
  assert.deepEqual([...actual.membership],[.5,.5,.5,.5]);
  const s=createSession('fcm',data,o);assert.deepEqual(finish(s),actual);s.dispose();
});
for(const kernelBackend of ['javascript','wasm'])test(`SOM ${kernelBackend}: tiny lambda remains finite without changing already-safe JS formulas`,()=>{
  const r=run('som-olp',{data:Float64Array.of(0,2),nSamples:2,nFeatures:1},{grid,initialPrototypes:Float64Array.of(0,2),initialMemberships:Float64Array.of(.5,.5,.5,.5),gamma:0,lambda:1e-310,maxIterations:1,kernelBackend});
  assert.deepEqual([...r.centers],[1,1]);assert.deepEqual([...r.membership],[.5,.5,.5,.5]);assert.ok(r.history.every(Number.isFinite));
});
for(const variant of ['product-underflow','weight-underflow','mixed-sign'])test(`FCM heterogeneous tiny weights: ${variant}`,()=>{
  const mixed=variant==='mixed-sign',scale=variant==='weight-underflow'?1:1e-20;
  const data={data:mixed?Float64Array.of(0,scale,-scale):Float64Array.of(0,scale),nSamples:mixed?3:2,nFeatures:1};
  const small=variant==='weight-underflow'?1e-170:1e-160;
  const initMembership=mixed?Float64Array.of(1,1e-100,1,small,1,2*small):Float64Array.of(1,1e-100,1,small);
  const expected=mixed?-3e-140:1e-140;
  const o={nClusters:2,initMembership,m:2,maxIterations:1,tolerance:0,blockRows:1};
  const actual=run('fcm',data,o);
  assert.ok(Math.abs(actual.centers[1]/expected-1)<1e-13,`${actual.centers[1]} vs ${expected}`);
  const s=createSession('fcm',data,o);assert.deepEqual(finish(s),actual);s.dispose();
});
test('FCM extreme-m displacement survives both original and translated input',()=>{
  let state=(190+128)>>>0;
  const random=()=>{state=(state+0x6D2B79F5)|0;let t=Math.imul(state^(state>>>15),1|state);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296};
  const n=31,d=128,k=5,data=Float64Array.from({length:n*d},()=>random()*2-1),initMembership=Float64Array.from({length:n*k},()=>random());
  const translated=Float64Array.from(data,(v,i)=>v-data[i%d]);
  const o={nClusters:k,initMembership,m:1000,maxIterations:3,tolerance:0,returnHistory:true,blockRows:7};
  for(const values of [data,translated]) {
    const x={data:values,nSamples:n,nFeatures:d};
    const result=run('fcm',x,o);assert.ok(result.centers.every(Number.isFinite));assert.ok(result.membership.every(Number.isFinite));
    const s=createSession('fcm',x,o);assert.deepEqual(finish(s),result);s.dispose();
  }
});
