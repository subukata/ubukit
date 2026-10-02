import test from 'node:test';import assert from 'node:assert/strict';
import {RMCMGraphCache,prepareRMCM} from '../../consumer/node_modules/ubukit-js/src/rmcm.js';
import {prepareRMCM as baseline} from '../references/baseline/rmcm.js';
const random=seed=>()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/2**32;};
const make=(n,d,rng)=>({data:Float64Array.from({length:n*d},()=>2*rng()-1),nSamples:n,nFeatures:d});
function check(cache,x,delta,options={}){let a,b;try{a=cache.prepare(x,{delta,...options});}catch(e){a=e;}try{b=baseline(x,{delta,...options});}catch(e){b=e;}
 if(b instanceof Error){assert.ok(a instanceof b.constructor,`${a} != ${b}`);return;}
 assert.ok(!(a instanceof Error),String(a));assert.equal(a.nEdges,b.nEdges);assert.deepEqual(a.degrees,b.degrees);
 if(!options.maxMemoryBytes)assert.deepEqual(a.neighborhoodMatrix(),b.neighborhoodMatrix());return [a,b];
}
test('Moving frames, delta changes, rebuilt CSR and full fit retain baseline state',()=>{
 for(const d of [1,2,3,8,16,32])for(let seed=0;seed<8;seed++){
  const rng=random(seed),x=make(41,d,rng),cache=new RMCMGraphCache({skin:.2});
  const init=Float64Array.from([...x.data.slice(0,d),...x.data.slice(10*d,11*d),...x.data.slice(20*d,21*d)]);
  for(const move of [0,.001,.002,.004,.008,.01,.2,2]){
   const current={...x,data:Float64Array.from(x.data,v=>v+(rng()-.5)*move)},delta=Math.sqrt(d)*(move===.004?1.01:1);
   const [a,b]=check(cache,current,delta,{graphBackend:'wasm-simd'}),o={initCenters:init,maxIterations:7,cycleWindow:0};
   const ar=a.fit(o),br=b.fit(o);delete ar.estimatedPrimaryBytes;delete br.estimatedPrimaryBytes;assert.deepEqual(ar,br);
  }
  assert.ok(cache.stats.rebuilds>=2);assert.ok(cache.stats.reuses>=1);
 }
});
test('Changing N/D, point reordering, delta and copied input safely refresh caches',()=>{
 const rng=random(8),cache=new RMCMGraphCache({skin:.3});
 for(const d of [1,8,3,32])for(const n of [1,3,9,9,17,4,31,31]){
  const x=make(n,d,rng),snapshot=x.data.slice();
  for(const delta of [0,.5,1,10,.1])check(cache,x,delta);
  const p=cache.prepare(x,{delta:1}),b=baseline(x,{delta:1});x.data.fill(99);
  const a=p.fit({nClusters:1}),bb=b.fit({nClusters:1});delete a.estimatedPrimaryBytes;delete bb.estimatedPrimaryBytes;assert.deepEqual(a,bb);
 }
 const stats=cache.stats;cache.clear();assert.equal(cache.stats.candidateEdges,0);assert.equal(cache.stats.rebuilds,stats.rebuilds);
});
test('Oversized skin and tight memory fall back; current edge guard remains exact',()=>{
 const x={data:Float64Array.from({length:20},(_,i)=>i),nSamples:20,nFeatures:1};
 for(const skin of [0,.1,1,100]){
  const cache=new RMCMGraphCache({skin,maxCacheEdges:50});
  for(const delta of [0,.1,1,3])for(const maxEdges of [20,40,58,60,100,400])check(cache,x,delta,{maxEdges});
 }
 for(const maxMemoryBytes of [1,100,1000,1500,3000,10000,100000]){
  const cache=new RMCMGraphCache({skin:.2});for(const delta of [0,1,10])check(cache,x,delta,{maxMemoryBytes});
 }
});
test('Cache preserves zero-radius and subnormal error behavior',()=>{
 for(const value of [0,1e-200,1e-160,1e-155,1e-153])for(const delta of [0,1e-300,1e-154,1]){
  check(new RMCMGraphCache({skin:.2}),{data:Float64Array.of(0,value,2*value),nSamples:3,nFeatures:1},delta);
 }
});
test('Cache preparation cancels and remains reusable',()=>{
 const x=make(101,8,random(8)),cache=new RMCMGraphCache({skin:.2});
 let visits=0;assert.throws(()=>cache.prepare(x,{delta:1,graphBatchPairs:1,shouldCancel:()=>++visits>3}),{name:'AbortError'});
 check(cache,x,1);
 let sawFilter=false;assert.throws(()=>cache.prepare(x,{delta:1,graphBatchPairs:1,onProgress:e=>{if(e.phase==='graph-cache-filter')sawFilter=true;},shouldCancel:()=>sawFilter}),{name:'AbortError'});
 check(cache,x,1);
});
test('Cache snapshots before progress callbacks mutate caller arrays',()=>{
 const x=make(20,3,random(19)),snapshot={...x,data:x.data.slice()},cache=new RMCMGraphCache({skin:100,maxCacheEdges:21});
 const p=cache.prepare(x,{delta:.4,graphBatchPairs:1,onProgress:()=>x.data.fill(99)});
 assert.deepEqual(p.neighborhoodMatrix(),baseline(snapshot,{delta:.4}).neighborhoodMatrix());
});
test('Fallback releases an existing cache before strict ordinary preparation',()=>{
 const x=make(101,8,random(14)),cache=new RMCMGraphCache({skin:.2});cache.prepare(x,{delta:3});assert.ok(cache.stats.cacheBytes>0);
 check(cache,x,1e-200);assert.equal(cache.stats.cacheBytes,0);
 cache.prepare(x,{delta:3});assert.ok(cache.stats.cacheBytes>0);
 check(cache,x,3,{maxMemoryBytes:18000});assert.equal(cache.stats.cacheBytes,0);
});

test('Cached WASM enforces each accepted edge guard before a later subnormal pair',()=>{
  for(const graphBatchPairs of [1,2,3,8,512])for(const blockRows of [1,2,8])for(const maxEdges of [5,6,7,8,9,25]) {
    const cache=new RMCMGraphCache({skin:1});
    const x={data:Float64Array.of(2,2,5,0,2e-150),nSamples:5,nFeatures:1};
    cache.prepare(x,{delta:1,graphBackend:'wasm-simd',graphBatchPairs,blockRows});
    x.data[4]=1e-160;
    const options={delta:1,maxEdges,graphBackend:'wasm-simd',graphBatchPairs,blockRows};
    let expected,actual;
    try { prepareRMCM(x,options); } catch(e) { expected=e; }
    try { cache.prepare(x,options); } catch(e) { actual=e; }
    assert.ok(expected instanceof RangeError);
    assert.ok(actual instanceof RangeError);
    assert.equal(/maxEdges/.test(actual.message),/maxEdges/.test(expected.message));
  }
});
