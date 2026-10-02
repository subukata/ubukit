import test from 'node:test';import assert from 'node:assert/strict';
import {prepareRMCM,prepareRMCMSteps} from '../../consumer/node_modules/ubukit-js/src/rmcm.js';import {prepareRMCM as original} from '../references/round3/rmcm.js';
function result(f){try{return f()}catch(e){return e;}}
test('Certified complete graph preserves original exact outputs',()=>{
 for(const d of [1,3,8,32,128])for(const kind of ['normal','zero','tiny_equal','spacing_boundary']){
  const n=29,data=Float64Array.from({length:n*d},(_,i)=>Math.sin(i+3)),delta=kind==='zero'?0:100;
  if(kind==='zero')data.fill(0);if(kind==='tiny_equal')data.fill(1e-300);
  if(kind==='spacing_boundary'){data.fill(0);const a=2**-458;for(let i=0;i<n;i++)data[i*d]=[0,a,a*(1+Number.EPSILON),-a][i%4];}
  const x={data,nSamples:n,nFeatures:d},a=original(x,{delta}),b=prepareRMCM(x,{delta});assert.deepEqual(a.neighborhoodMatrix(),b.neighborhoodMatrix());
  const options={nClusters:3,initCenters:data.slice(0,3*d),maxIterations:6},aa=result(()=>a.fit(options)),bb=result(()=>b.fit(options));if(aa instanceof Error){assert.ok(bb instanceof aa.constructor);assert.equal(bb.message,aa.message);}else assert.deepEqual(aa,bb);
 }
});
test('Complete graph shortcuts retain the first edge or memory rejection exactly',()=>{
 for(const backend of ['adjoint','reference']){
  const n=17,d=8,data=Float64Array.from({length:n*d},(_,i)=>Math.sin(i)),x={data,nSamples:n,nFeatures:d},fixed=(backend==='adjoint'?16*n*d+20*n:8*n*d+12*n)+4+8*d;
  for(const maxEdges of [n,n+1,n+2,n*n-1,n*n,n*n+1])for(const maxMemoryBytes of [fixed+4*n-1,fixed+4*n,fixed+4*(n+2),fixed+4*n*n-1,fixed+4*n*n,1e6]){
   const o={delta:100,backend,maxEdges,maxMemoryBytes};const a=result(()=>original(x,o)),b=result(()=>prepareRMCM(x,o));
   if(a instanceof Error){assert.ok(b instanceof a.constructor);assert.equal(b.message,a.message);}else{assert.ok(!(b instanceof Error),String(b));assert.equal(a.nEdges,b.nEdges);assert.deepEqual(a.degrees,b.degrees);}
  }
 }
});
test('Tiny nonidentical coordinates retain numeric validation rather than the complete shortcut',()=>{
 for(const value of [1e-300,1e-160])for(const maxEdges of [5,6,7,25]){
  const x={data:Float64Array.of(2,2,5,0,value),nSamples:5,nFeatures:1},o={delta:100,maxEdges};const a=result(()=>original(x,o)),b=result(()=>prepareRMCM(x,o));assert.ok(a instanceof Error);assert.ok(b instanceof a.constructor);assert.equal(b.message,a.message);
 }
});

test('Certified complete-graph progress finishes counting before CSR fill',()=>{
 const x={data:Float64Array.from({length:20},(_,i)=>i),nSamples:20,nFeatures:1},phases=[];
 for(const event of prepareRMCMSteps(x,{delta:100,blockRows:1}))phases.push([event.phase,event.completed,event.total]);
 const completed=phases.findIndex(([phase,done,total])=>phase==='graph-count'&&done===total&&total>0);
 const fill=phases.findIndex(([phase])=>phase==='graph-fill');
 assert.ok(completed>=0&&fill>completed);
});
