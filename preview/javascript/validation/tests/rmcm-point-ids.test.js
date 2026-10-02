import test from 'node:test';import assert from 'node:assert/strict';
import {RMCMGraphCache,prepareRMCM} from '../../consumer/node_modules/ubukit-js/src/rmcm.js';
import {prepareRMCM as original} from '../references/round3/rmcm.js';
const rng=seed=>()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
const input=rows=>({data:Float64Array.from(rows.flatMap(r=>r.x)),nSamples:rows.length,nFeatures:rows[0].x.length});
function equal(a,b){assert.deepEqual(a.neighborhoodMatrix(),b.neighborhoodMatrix());assert.deepEqual(a.degrees,b.degrees);}
for(const backend of ['scalar','wasm-simd'])test(`${backend}: stable point IDs retain exact birth/death/reorder graphs and fits`,()=>{
 for(const d of [1,3,8,32,128])for(let seed=0;seed<3;seed++){
  const random=rng(seed+d);let nextId=1000,rows=Array.from({length:67},(_,i)=>({id:i-100,x:Array.from({length:d},()=>random()*2-1)}));
  const cache=new RMCMGraphCache({skin:.2*Math.sqrt(d)});
  for(let frame=0;frame<8;frame++){
   if(frame){rows.splice((frame*3)%rows.length,3);for(const r of rows)r.x=r.x.map(v=>v+(random()-.5)*.0002);for(let j=0;j<1+frame%4;j++)rows.push({id:nextId++,x:Array.from({length:d},()=>random()*2-1)});rows=rows.slice().reverse();}
   const x=input(rows),delta=Math.sqrt(d)*(frame%3?1.01:1),pointIds=rows.map(r=>r.id);
   const a=cache.prepare(x,{delta,pointIds,graphBackend:backend}),b=original(x,{delta});equal(a,b);
   const initCenters=Float64Array.from(rows.slice(0,3).flatMap(r=>r.x)),options={nClusters:3,initCenters,maxIterations:4,cycleWindow:0};
   const aa=a.fit(options),bb=b.fit(options);delete aa.estimatedPrimaryBytes;delete bb.estimatedPrimaryBytes;assert.deepEqual(aa,bb);
  }
  assert.equal(cache.stats.rebuilds,1);assert.equal(cache.stats.incrementalUpdates,7);assert.equal(cache.stats.removedPoints,21);
 }
});
test('Point ID validation, caller mutation and mode changes stay isolated',()=>{
 const x={data:Float64Array.from({length:20},(_,i)=>i),nSamples:20,nFeatures:1},cache=new RMCMGraphCache({skin:1});
 for(const pointIds of [Array(19).fill(1),Array(20).fill(1),Array(20).fill(true),Array.from({length:20},(_,i)=>i+.5),Array.from({length:20},(_,i)=>i?i:2**53)])assert.throws(()=>cache.prepare(x,{delta:1,pointIds}));
 const ids=Float64Array.from({length:20},(_,i)=>i);const a=cache.prepare(x,{delta:1,pointIds:ids});ids.fill(999);
 equal(a,cache.prepare(x,{delta:1,pointIds:Array.from({length:20},(_,i)=>i)}));assert.equal(cache.stats.rebuilds,1);
 equal(a,cache.prepare(x,{delta:1}));assert.equal(cache.stats.rebuilds,2);
});
test('ID birth staging preserves the active edge-budget versus later numeric error order',()=>{
 for(const backend of ['scalar','wasm-simd'])for(const maxEdges of [5,6,7,8,9,25]){
  const cache=new RMCMGraphCache({skin:1});const x={data:Float64Array.of(2,2,5,0,2e-150),nSamples:5,nFeatures:1};
  cache.prepare(x,{delta:1,pointIds:[0,1,2,3,4],graphBackend:backend});x.data[4]=1e-160;let a,b;
  try{original(x,{delta:1,maxEdges});}catch(e){a=e;}
  try{cache.prepare(x,{delta:1,maxEdges,pointIds:[0,1,2,3,5],graphBackend:backend});}catch(e){b=e;}
  assert.ok(a instanceof RangeError);assert.ok(b instanceof RangeError);assert.equal(/maxEdges/.test(a.message),/maxEdges/.test(b.message));
 }
});
test('ID updates and compact-cache preparation retain ordinary fallback at tight budgets',()=>{
 const random=rng(71),rows=Array.from({length:101},(_,i)=>({id:i,x:Array.from({length:8},()=>random())}));
 for(const limit of [18000,25000,40000,80000,200000]){
  const cache=new RMCMGraphCache({skin:.2});const x=input(rows);cache.prepare(x,{delta:1,pointIds:rows.map(r=>r.id)});
  const reordered=[...rows.slice(1),{id:200,x:rows[0].x.slice()}];const y=input(reordered);
  let a,b;try{a=cache.prepare(y,{delta:1,maxMemoryBytes:limit,pointIds:reordered.map(r=>r.id)});}catch(e){a=e;}
  try{b=prepareRMCM(y,{delta:1,maxMemoryBytes:limit});}catch(e){b=e;}
  if(b instanceof Error){assert.ok(a instanceof b.constructor);continue;}assert.ok(!(a instanceof Error),String(a));assert.equal(a.nEdges,b.nEdges);assert.deepEqual(a.degrees,b.degrees);if(limit>=100000)equal(a,b);
 }
});

test('Point-ID births exceeding the cache self-edge limit fall back even with zero candidate pairs',()=>{
 for(const graphBackend of ['scalar','wasm-simd']){
  const cache=new RMCMGraphCache({skin:.1,maxCacheEdges:5});
  cache.prepare({data:Float64Array.of(0,10,20,30),nSamples:4,nFeatures:1},{delta:.1,pointIds:[0,1,2,3],graphBackend});
  const x={data:Float64Array.of(0,10,20,30,40,50),nSamples:6,nFeatures:1};
  const p=cache.prepare(x,{delta:.1,pointIds:[0,1,2,3,4,5],graphBackend});
  equal(p,original(x,{delta:.1}));assert.equal(cache.stats.candidateEdges,0);assert.ok(cache.stats.fallbacks>0);
 }
});

test('Interleaved point-ID generators keep each reference paired with its proof radius',()=>{
 for(const intervening of ['prepare','clear'])for(const graphBackend of ['scalar','wasm-simd']){
  const cache=new RMCMGraphCache({skin:.2}),x={data:Float64Array.of(0,10,20),nSamples:3,nFeatures:1};
  cache.prepare(x,{delta:.1,pointIds:[0,1,2],graphBackend});const originalCover=cache.stats.coverRadius;
  const reversed={data:Float64Array.of(20,10,0),nSamples:3,nFeatures:1};
  const pending=cache.prepareSteps(reversed,{delta:.1,pointIds:[2,1,0],graphBackend,blockRows:1});
  const first=pending.next();assert.equal(first.value.phase,'graph-cache-remap');
  if(intervening==='prepare')cache.prepare(x,{delta:100,pointIds:[0,1,2],graphBackend});else cache.clear();
  let step;do{step=pending.next();}while(!step.done);
  equal(step.value,original(reversed,{delta:.1}));assert.equal(cache.stats.coverRadius,originalCover);
  const next=cache.prepare(reversed,{delta:11,pointIds:[2,1,0],graphBackend});
  equal(next,original(reversed,{delta:11}));assert.equal(next.nEdges,7);
 }
});
