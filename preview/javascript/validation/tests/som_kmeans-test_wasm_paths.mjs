import test from'node:test';import assert from'node:assert/strict';
import{kmeans as baseKM,kmeansSteps as baseKMSteps}from'../references/baseline/clustering.js';
import{kmeans as actualKM,kmeansSteps as actualKMSteps}from'../../consumer/node_modules/ubukit-js/src/clustering.js';
import{somOlp as baseSOM,somOlpSteps as baseSOMSteps}from'../references/baseline/som-olp.js';
import{somOlp as actualSOM,somOlpSteps as actualSOMSteps}from'../../consumer/node_modules/ubukit-js/src/som-olp.js';
import{seededRandom}from'../../consumer/node_modules/ubukit-js/src/core.js';import{kmeansWasmWorkspace}from'../../consumer/node_modules/ubukit-js/src/kmeans-wasm.js';import{pcaRotationWorkspace}from'../../consumer/node_modules/ubukit-js/src/som-pca-wasm.js';
import{SESSION_CHECKPOINT as baseCheckpoint}from'../references/baseline/session-hooks.js';
import{SESSION_CHECKPOINT as candidateCheckpoint}from'../../consumer/node_modules/ubukit-js/src/session-hooks.js';
function legacyResult(result){const {kernel,...rest}=result;return rest;}
function* legacySteps(it){for(;;){const next=it.next();if(next.done)return legacyResult(next.value);yield next.value;}}
const newKM=(...args)=>legacyResult(actualKM(...args)),newKMSteps=(...args)=>legacySteps(actualKMSteps(...args));
const newSOM=(...args)=>legacyResult(actualSOM(...args)),newSOMSteps=(...args)=>legacySteps(actualSOMSteps(...args));
const rnd=seededRandom(211859),matrix=(n,d,scale=1,offset=0)=>({data:Float64Array.from({length:n*d},()=>offset+scale*(rnd()*2-1)),nSamples:n,nFeatures:d});
function outcome(fn){try{return{value:fn()}}catch(e){return{error:e.name,message:e.message}}}
function walk(iterator){const events=[];let r;while(!(r=iterator.next()).done)events.push(r.value);return{events,value:r.value};}
function alternate(a,b){const av=[],bv=[];let ar,br;do{ar=a.next();br=b.next();av.push(ar);bv.push(br);}while(!ar.done||!br.done);assert.deepEqual(bv,av);}
test('Selected SIMD k-means kernel preserves all states and events over tails, ties and dimensions',()=>{
 for(let trial=0;trial<140;trial++){
  const n=1+Math.floor(rnd()*60),d=[1,2,3,8,17,32,64,128][trial%8],k=1+Math.floor(rnd()*31),x=matrix(n,d),centers=matrix(k,d).data;
  if(trial%7===0)x.data.fill(trial%2?0:-0);if(trial%11===0)centers.fill(0);
  const options={initCenters:centers,nClusters:k,maxIterations:5,blockRows:1+trial%17,kernelBackend:'wasm'};
  assert.deepEqual(outcome(()=>newKM(x,options)),outcome(()=>baseKM(x,options)));
  assert.deepEqual(outcome(()=>walk(newKMSteps(x,options))),outcome(()=>walk(baseKMSteps(x,options))));
 }
});
test('SIMD numerical hazards retain existing underflow, overflow and cutoff errors',()=>{
 for(const scale of[0,1e-200,1e-160,1e-155,1e-154,1e-150,1e150,1e153,1e154])for(const d of[1,2,3,8,17])for(const k of[1,3,8,17]){
  const x=matrix(7,d,scale),centers=matrix(k,d,scale).data;if(k>1)centers.set(x.data.subarray(0,d));
  const options={initCenters:centers,nClusters:k,maxIterations:3,blockRows:3,kernelBackend:'wasm'};
  assert.deepEqual(outcome(()=>newKM(x,options)),outcome(()=>baseKM(x,options)),String([scale,d,k]));
 }
});
test('WASM finalization reads live caller data after its yield and remains independently owned',()=>{
 const a=matrix(129,17),b={...a,data:a.data.slice()},centers=a.data.slice(0,19*17),options={initCenters:centers,nClusters:19,maxIterations:4,blockRows:13,kernelBackend:'wasm'};
 const ia=baseKMSteps(a,options),ib=newKMSteps(b,options);let ar,br;
 do{ar=ia.next();br=ib.next();assert.deepEqual(br,ar);if(!ar.done&&ar.value.phase==='finalize'){const i=ar.value.completedRows;a.data[i*17]+=.01;b.data[i*17]+=.01;}}while(!ar.done);
 const cloned=structuredClone(br.value,{transfer:[br.value.centers.buffer,br.value.labels.buffer,br.value.coreLabels.buffer]});assert.equal(cloned.centers.length,19*17);
});
test('WASM workspaces do not share state across interleaved generators and cancellation',()=>{
 for(const tile of['r2w8','r4w4']){
  const x=matrix(151,32),y=matrix(179,17),o={nClusters:16,seed:73,maxIterations:5,blockRows:9,kernelBackend:'wasm'};
  const a=newKMSteps(x,o),b=newKMSteps(y,o);let ra,rb;do{if(!ra?.done)ra=a.next();if(!rb?.done)rb=b.next();}while(!ra.done||!rb.done);assert.deepEqual(ra.value,baseKM(x,o));assert.deepEqual(rb.value,baseKM(y,o));
  let count=0;assert.throws(()=>newKM(x,{...o,shouldCancel:()=>++count>5}),e=>e.name==='AbortError');assert.deepEqual(newKM(x,o),baseKM(x,o));
 }
});
test('PCA SIMD retains original initialization, training and event sequence',()=>{
 for(const[n,d,q]of[[33,32,1],[48,32,2],[65,64,3],[35,128,2],[64,784,2],[17,784,2],[2,3,2],[128,8,3]]){
  for(const offset of[0,1e10]){
   const x=matrix(n,d,1,offset),grid=matrix(7,q),o={kernelBackend:'wasm',grid,gamma:.3,lambda:2,maxIterations:3,tolerance:0,blockRows:11};
   assert.deepEqual(newSOM(x,o),baseSOM(x,o));alternate(baseSOMSteps(x,o),newSOMSteps(x,o));
  }
 }
});
test('Budget edges and unavailable WebAssembly use the existing JavaScript path',()=>{
 const x=matrix(48,32),grid=matrix(7,2),dim=32,training=8*(7*32+2*7),pca=16*dim*dim+8*(2*32+2*2)+4*dim;
 const estimate=8*(48*32+7*2+7*32+48*7+48*2+3)+4*48+Math.max(training,pca),o={kernelBackend:'wasm',grid,maxIterations:3,gamma:.3,lambda:2,maxMemoryBytes:estimate,maxScratchBytes:pca};
 assert.deepEqual(newSOM(x,o),baseSOM(x,o));assert.equal(pcaRotationWorkspace(32,1e6,0),null);assert.equal(kmeansWasmWorkspace(48,32,16,128,0,'r2w8'),null);
 const saved=globalThis.WebAssembly;try{globalThis.WebAssembly=undefined;assert.deepEqual(newSOM(x,{...o,maxMemoryBytes:1e8,maxScratchBytes:1e8}),baseSOM(x,{...o,maxMemoryBytes:1e8,maxScratchBytes:1e8}));assert.deepEqual(newKM(x,{nClusters:7,kernelBackend:'wasm'}),baseKM(x,{nClusters:7}));}finally{globalThis.WebAssembly=saved;}
});

test('Opt-in execution diagnostics distinguish real kernels, workspace and fallback rows',()=>{
 const x=matrix(64,32),o={kernelBackend:'wasm',nClusters:8,maxIterations:4,blockRows:13},r=actualKM(x,o);
 assert.equal(r.kernel.actual,'wasm');assert.ok(r.kernel.allocatedWorkspaceBytes>0);assert.equal(r.kernel.wasmAssignmentRows,64*r.iterations);assert.equal(r.kernel.wasmFinalizationRows,64);
 const tiny={data:new Float64Array([1e154,1e154]),nSamples:1,nFeatures:2},extreme={kernelBackend:'wasm',initCenters:new Float64Array([1e154,1e154,-1e154,-1e154]),nClusters:2,maxIterations:2};
 const fallback=actualKM(tiny,extreme);assert.equal(fallback.kernel.actual,'mixed');assert.ok(fallback.kernel.javascriptFallbackAssignmentRows>0);assert.deepEqual(legacyResult(fallback),baseKM(tiny,extreme));
 const grid=matrix(8,2),som=actualSOM(x,{kernelBackend:'wasm',grid,maxIterations:3,gamma:.3,lambda:2});assert.equal(som.kernel.pca,'wasm');assert.equal(som.kernel.training,'wasm');assert.equal(som.kernel.wasmPrototypeRows,64*3);assert.equal(som.kernel.wasmMembershipRows,64*3);assert.ok(som.kernel.estimatedPeakPrimaryBytes<=512*1024*1024);
 assert.throws(()=>actualKM(x,{kernelBackend:'bad',nClusters:3}),/kernelBackend/);assert.throws(()=>actualSOM(x,{kernelBackend:'bad',grid}),/kernelBackend/);
});

test('SOM training SIMD handles row/feature/grid tails, zero gamma and live input changes',()=>{
 for(let trial=0;trial<60;trial++){
  const n=3+Math.floor(rnd()*35),d=[1,2,3,8,17,32,128][trial%7],q=[1,2,3][trial%3],m=1+trial%19,x=matrix(n,d),grid=matrix(m,q);
  const o={kernelBackend:'wasm',initializer:'sample',seed:11,grid,gamma:trial%3===0?0:.37,lambda:1.7,maxIterations:4,tolerance:0,blockRows:1+trial%13};
  assert.deepEqual(newSOM(x,o),baseSOM(x,o));alternate(baseSOMSteps(x,o),newSOMSteps(x,o));
 }
 const a=matrix(67,17),b={...a,data:a.data.slice()},ga=matrix(9,3),gb={...ga,data:ga.data.slice()},o={kernelBackend:'wasm',initializer:'sample',seed:71,maxIterations:4,tolerance:0,blockRows:7,gamma:.4,lambda:1.4};
 const ia=baseSOMSteps(a,{...o,grid:ga}),ib=newSOMSteps(b,{...o,grid:gb});let ar,br;
 do{ar=ia.next();br=ib.next();assert.deepEqual(br,ar);if(!ar.done&&ar.value.phase==='update-prototypes'){a.data[0]+=.01;b.data[0]+=.01;ga.data[0]+=.01;gb.data[0]+=.01;}}while(!ar.done);
 const source=br.value;structuredClone(source,{transfer:[source.W.buffer,source.P.buffer,source.V.buffer,source.labels.buffer,source.history.buffer]});
});

test('WASM k-means reads checkpoint-held centers again after every yielded block',()=>{
 const x=matrix(71,17),o={nClusters:9,seed:117,maxIterations:4,blockRows:7,kernelBackend:'wasm'};
 let heldA,heldB;
 const a=baseKMSteps(x,{...o,[baseCheckpoint]:s=>{heldA=s.centers;}});
 const b=newKMSteps(x,{...o,[candidateCheckpoint]:s=>{heldB=s.centers;}});
 let ar,br;
 do{
  ar=a.next();br=b.next();assert.deepEqual(br,ar);
  if(!ar.done&&heldA&&(ar.value.phase==='assignment'||ar.value.phase==='finalize')){
   heldA[11]+=.0003;heldB[11]+=.0003;
  }
 }while(!ar.done);
});

test('WASM SOM preserves empty units, zero iterations, high grid dimensions and overflow errors',()=>{
 for(const d of[1,17,128])for(const q of[1,3,17])for(const maxIterations of[0,1,3]){
  const x=matrix(11,d),grid=matrix(7,q),initialPrototypes=matrix(7,d).data;
  const initialMemberships=new Float64Array(11*7);for(let i=0;i<11;i++)initialMemberships[i*7+i%3]=1;
  const o={grid,initialPrototypes,initialMemberships,maxIterations,gamma:.2,lambda:.7,blockRows:3,kernelBackend:'wasm'};
  assert.deepEqual(newSOM(x,o),baseSOM(x,o));
 }
 for(const scale of[1e-160,1e153,1e154])for(const gamma of[0,.4]){
  const x=matrix(7,17,scale),grid=matrix(5,3,scale),o={grid,initializer:'sample',seed:18,maxIterations:2,gamma,lambda:.7,blockRows:3,kernelBackend:'wasm'};
  // Exceptional arithmetic is revised; compare actual JS/WASM fallback routes.
  assert.deepEqual(outcome(()=>newSOM(x,o)),outcome(()=>newSOM(x,{...o,kernelBackend:'javascript'})));
 }
});
