import test from 'node:test';
import assert from 'node:assert/strict';
import {createRadiusScanner} from '../javascript/rmcm-wasm.js';
import {prepareRMCM,RMCMGraphCache} from '../javascript/rmcm.js';
import {prepareRMCM as checkpoint} from '../../../baseline/javascript/rmcm.js';
const rng=seed=>()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
function distance(data,i,j,d){let s=0;for(let f=0;f<d;f++){const q=data[i*d+f]-data[j*d+f];s+=q*q;}return Math.sqrt(s);}
test('Float32 screening retains every true/ambiguity-band pair with ordered uneven batches',()=>{
 for(const d of [8,9,16,31,32,64,128])for(const scale of [1e-15,1e-7,1,1e7,1e15])for(const batch of [1,3,4,7,16,19,100]) {
  const n=23,random=rng(d),data=Float64Array.from({length:n*d},()=>scale*(random()*2-1));
  const radius=distance(data,0,1,d)*(1+64*Number.EPSILON*d);
  const scanner=createRadiusScanner(data,n,d,radius,batch,100_000_000);
  assert.equal(scanner?.screening,'float32');
  for(let i=0;i<n;i++)for(let start=i+1;start<n;start+=scanner.capacity){
   const end=Math.min(n,start+scanner.capacity),count=scanner.scan(i,start,end),ids=Array.from(scanner.ids.subarray(0,count));
   assert.deepEqual(ids,[...new Set(ids)].sort((a,b)=>a-b));
   for(let j=start;j<end;j++)if(distance(data,i,j,d)<=radius)assert.ok(ids.includes(j),`${d}/${scale}/${batch}: missed ${i},${j}`);
  }
 }
});
test('Float32 selection falls back for translations, unsafe scales and high dimensions',()=>{
 for(const [d,scale,offset] of [[8,1,1e10],[32,1e-150,0],[128,1e100,0],[1024,1,0]]){
  const n=17,random=rng(d),data=Float64Array.from({length:n*d},()=>scale*(offset+random())),delta=scale*Math.sqrt(d);
  const scanner=createRadiusScanner(data,n,d,delta,32,100_000_000);
  assert.equal(scanner?.exact,true);
  const a=prepareRMCM({data,nSamples:n,nFeatures:d},{delta,graphBackend:'wasm-simd'}),b=checkpoint({data,nSamples:n,nFeatures:d},{delta,graphBackend:'wasm-simd'});
  assert.deepEqual(a.neighborhoodMatrix(),b.neighborhoodMatrix());
 }
});
test('Float32 candidate refinement retains original edge-budget and numeric error ordering',()=>{
 for(const batch of [1,2,4,8,32])for(const maxEdges of [5,6,7,8,9,25]){
  const d=8,data=new Float64Array(5*d);[2,2,5,0,1e-160].forEach((v,i)=>data[i*d]=v);
  const x={data,nSamples:5,nFeatures:d},options={delta:1,maxEdges,graphBatchPairs:batch};let a,b;
  try{checkpoint(x,{...options,graphBackend:'scalar'});}catch(e){a=e;}
  try{prepareRMCM(x,{...options,graphBackend:'wasm-simd'});}catch(e){b=e;}
  assert.ok(a instanceof RangeError);assert.ok(b instanceof RangeError);assert.equal(/maxEdges/.test(a.message),/maxEdges/.test(b.message));
 }
});

test('Higher-dimensional tiny radii select the exact scalar fallback after Float32 refusal',()=>{
 const n=1200,d=32,random=rng(331),data=Float64Array.from({length:n*d},()=>random()*2-1),x={data,nSamples:n,nFeatures:d};
 const p=prepareRMCM(x,{delta:.01,graphBackend:'wasm-simd'}),b=checkpoint(x,{delta:.01,graphBackend:'scalar'});
 assert.equal(p.graphBackendUsed,'scalar');assert.deepEqual(p.neighborhoodMatrix(),b.neighborhoodMatrix());
});
