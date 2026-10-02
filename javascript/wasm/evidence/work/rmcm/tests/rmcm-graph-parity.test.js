import test from 'node:test';
import assert from 'node:assert/strict';
import { prepareRMCM as candidate } from '../javascript/rmcm.js';
import { prepareRMCM as baseline } from '../../../baseline/javascript/rmcm.js';
const next = (x, up) => { if (x===0) return up?Number.MIN_VALUE:-Number.MIN_VALUE; const a=new Float64Array([x]), b=new BigUint64Array(a.buffer);b[0] += (x>0)===up?1n:-1n;return a[0]; };
const random = seed => () => {seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/2**32;};
const input=(n,d,rng,scale=1,offset=0)=>({data:Float64Array.from({length:n*d},()=>((rng()-.5)*2+offset)*scale),nSamples:n,nFeatures:d});
const distance=(x,i,j)=>{let s=0;for(let f=0;f<x.nFeatures;f++){const v=x.data[i*x.nFeatures+f]-x.data[j*x.nFeatures+f];s+=v*v;}return Math.sqrt(s);};
function check(x,delta,graphBackend='scalar',extra={}){
 const o={delta,graphBackend,...extra};let a,b;
 try{a=baseline(x,o)}catch(e){a=e}
 try{b=candidate(x,o)}catch(e){b=e}
 if(a instanceof Error){assert.ok(b instanceof a.constructor,`${a} != ${b}`);return;}
 assert.ok(!(b instanceof Error),String(b));
 assert.equal(a.nEdges,b.nEdges);assert.deepEqual(a.degrees,b.degrees);
 if(!extra.maxMemoryBytes)assert.deepEqual(a.neighborhoodMatrix(),b.neighborhoodMatrix());
}
for(const graphBackend of ['scalar','wasm-simd','grid']){
 test(`${graphBackend} randomized exact adjacent-float boundaries and uneven batches`,()=>{
  for(const d of [1,2,3,4,5,7,8,9,16,31,32,33,64,127])for(let seed=0;seed<20;seed++){
   const x=input(21,d,random(seed));const delta=distance(x,0,1);
   for(const r of [next(delta,false),delta,next(delta,true)])check(x,r,graphBackend,{graphBatchPairs:(seed%7)+1,blockRows:3});
  }
 });
 test(`${graphBackend} scales and large offsets preserve exact graph`,()=>{
  for(const d of [1,3,8,32,65])for(const exponent of [-150,-100,-50,0,50,100,140])for(const p of [0,10,30,50]){
   const x=input(17,d,random(d+500+exponent),10**exponent,2**p);
   if(x.data.some(v=>Math.abs(v)>Math.sqrt(Number.MAX_VALUE/d)/4))continue;
   const delta=distance(x,0,1);
   for(const r of [next(delta,false),delta,next(delta,true)])check(x,r,graphBackend);
  }
 });
 test(`${graphBackend} extreme error and zero radius parity`,()=>{
  for(const value of [0,1e-300,1e-200,1e-162,1e-160,1e-155,1e-154,1e-153])for(const delta of [0,1e-300,1e-162,1e-160,1e-154,1]){
   check({data:Float64Array.of(0,value,2*value),nSamples:3,nFeatures:1},delta,graphBackend);
  }
 });
 test(`${graphBackend} memory guard fallback and edge limit parity`,()=>{
  const x=input(101,3,random(27));
  for(const maxMemoryBytes of [1,100,9000,10000,11000,12000,18000,25000,40000,70000,100000,140000,500000])for(const delta of [0,.1,1,4])check(x,delta,graphBackend,{maxMemoryBytes});
  const dense={data:new Float64Array(50),nSamples:50,nFeatures:1};
  for(const maxEdges of [1,49,50,2499,2500,2501])check(dense,1,graphBackend,{maxEdges});
 });
}
test('Optional WASM feature detection reports scalar fallback at tight memory budget',()=>{
 const x={data:Float64Array.of(0,1,3,9),nSamples:4,nFeatures:1};
 assert.equal(candidate(x,{delta:1,graphBackend:'wasm-simd',maxMemoryBytes:1024}).graphBackendUsed,'scalar');
 assert.equal(candidate(x,{delta:1,graphBackend:'wasm-simd'}).graphBackendUsed,'wasm-simd');
 assert.equal(candidate(x,{delta:1}).graphBackendUsed,'scalar');
 assert.throws(()=>candidate(x,{delta:1,graphBackend:'wrong'}),/graphBackend/);
});
test('Grid cell-boundary rounding and coordinate-range fallback preserve exact edges',()=>{
 for(const d of [1,2,3])for(const offset of [0,1e6,1e12])for(const scale of [1e-100,1,1e100]){
  const delta=scale,width=delta*1.01,values=[0,1,2,17,2**20,2**28];
  const rows=[];
  for(const v of values){const p=offset*scale+v*width;for(const a of [next(p,false),p,next(p,true),p+delta])rows.push(Array(d).fill(a));}
  const x={data:Float64Array.from(rows.flat()),nSamples:rows.length,nFeatures:d};check(x,delta,'grid');
 }
});
test('Exact WASM preserves maxEdges versus later numeric-error ordering within SIMD batches',()=>{
 for(const d of [1,3,8,32,128])for(const badAt of [1,2,7,8,9,13])for(const value of [1e-200,1e-155]){
  const n=15,data=new Float64Array(n*d);data[badAt*d]=value;
  const x={data,nSamples:n,nFeatures:d};
  for(const maxEdges of [n,n+2*Math.max(0,badAt-2),n+2*(badAt-1),n*n])for(const graphBatchPairs of [1,3,8,16,4096]){
   let a,b;try{baseline(x,{delta:1,maxEdges,graphBatchPairs})}catch(e){a=e.message}
   try{candidate(x,{delta:1,maxEdges,graphBatchPairs,graphBackend:'wasm-simd'})}catch(e){b=e.message}
   assert.equal(b,a,JSON.stringify({d,badAt,value,maxEdges,graphBatchPairs}));
  }
 }
});
