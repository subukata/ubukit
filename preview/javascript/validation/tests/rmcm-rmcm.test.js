import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { rmcm, rmcmReference, prepareRMCM, prepareRMCMSteps, run, runAsync, steps, algorithms } from '../../consumer/node_modules/ubukit-js/src/index.js';
const input = rows => ({ data: Float64Array.from(rows.flat()), nSamples: rows.length, nFeatures: rows[0].length });
const flat = x => x.flat?.(Infinity) ?? x;
function close(actual, expected, atol = 5e-13) {
  expected = flat(expected); assert.equal(actual.length, expected.length);
  for (let i=0;i<actual.length;++i) assert.ok(Math.abs(actual[i]-expected[i]) <= atol*Math.max(1,Math.abs(expected[i])), `index ${i}: ${actual[i]} != ${expected[i]}`);
}
const fixture = JSON.parse(fs.readFileSync(new URL('../fixtures/rmcm-python.json', import.meta.url)));
for (const item of fixture.cases) for (const [backend, pythonBackend] of [['adjoint','adjoint'],['reference','numpy']]) {
  test(`Python RMCM parity ${backend}: ${item.name}`, () => {
    const expected=item.backends[pythonBackend], prepared=prepareRMCM(input(item.X),{delta:item.delta,backend});
    const actual=prepared.fit({initCenters:Float64Array.from(item.init.flat()),maxIterations:item.maxIterations,cycleWindow:item.cycleWindow});
    close(actual.centers,expected.centers); close(actual.membership,expected.membership);
    assert.deepEqual([...actual.labels],expected.labels); assert.deepEqual([...prepared.degrees],expected.degrees);
    for(const key of ['iterations','converged','stopReason','cycleLength','nEdges','emptyClusterUpdates']) assert.equal(actual[key],expected[key],key);
    for(let i=0;i<item.X.length;++i) close([actual.membership.subarray(i*item.init.length,(i+1)*item.init.length).reduce((a,b)=>a+b,0)],[1]);
  });
}
test('registry exposes independent RMCM and existing rough algorithms',()=>{assert.ok(algorithms.includes('rmcm'));assert.ok(algorithms.includes('rcm'));assert.ok(algorithms.includes('exrcm'));const x=input([[0],[1],[3]]),o={delta:2,nClusters:2,seed:7};assert.deepEqual(run('rmcm',x,o),rmcm(x,o));});
test('CSR is symmetric, sorted, self including, row normalized and copied',()=>{
  const p=prepareRMCM(input([[0],[1],[3],[10]]),{delta:2}), csr=p.neighborhoodMatrix();
  const sets=[];
  for(let i=0;i<4;++i){const ids=[...csr.indices.slice(csr.indptr[i],csr.indptr[i+1])];assert.deepEqual(ids,[...ids].sort((a,b)=>a-b));assert.ok(ids.includes(i));sets.push(new Set(ids));close([csr.data.subarray(csr.indptr[i],csr.indptr[i+1]).reduce((a,b)=>a+b,0)],[1]);}
  for(let i=0;i<4;++i)for(let j=0;j<4;++j)assert.equal(sets[i].has(j),sets[j].has(i));
  const degrees=p.degrees;degrees.fill(0);csr.indices.fill(0);csr.data.fill(0);assert.deepEqual([...p.degrees],[2,3,2,1]);assert.notDeepEqual(p.neighborhoodMatrix().data,csr.data);
});
test('prepared snapshot and all returned arrays do not alias caller data',()=>{
  const x=input([[0],[1],[9]]),p=prepareRMCM(x,{delta:1});x.data.fill(100);
  const init=Float64Array.of(0,9),copy=init.slice();const a=p.fit({initCenters:init});close(a.centers,[.5,9]);assert.deepEqual(init,copy);a.centers.fill(99);a.labels.fill(1);a.membership.fill(1);close(p.fit({initCenters:init}).centers,[.5,9]);
});
test('seed initialization samples distinct indices reproducibly; explicit inputs control cross-language comparison',()=>{
  const x=input(Array.from({length:12},(_,i)=>[i,i+1])),options={delta:0,nClusters:4,seed:42};
  const a=rmcm(x,options),b=rmcm(x,options);assert.deepEqual(a,b);assert.equal(new Set(a.initIndices).size,4);assert.equal(a.initIndices.length,4);
});
test('omitted memberships keep exact centers and labels',()=>{
  const x=input([[0],[1],[3],[9]]),o={delta:2,initCenters:Float64Array.of(0,9)};
  for(const backend of ['reference','adjoint']){const a=rmcm(x,{...o,backend}),b=rmcm(x,{...o,backend,returnMembership:false});assert.equal(b.membership,null);assert.deepEqual(a.centers,b.centers);assert.deepEqual(a.labels,b.labels);}
});
test('maxIterations output is update-producing rather than nearest-final-center labels',()=>{
  const x=input([[0],[2],[3],[7],[8]]),r=rmcm(x,{delta:2.1,initCenters:Float64Array.of(0,3),maxIterations:1});
  const nearest=[...x.data].map(value=>Math.abs(value-r.centers[0])<=Math.abs(value-r.centers[1])?0:1);assert.notDeepEqual([...r.labels],nearest);
  for(let c=0;c<2;++c){let mass=0,sum=0;for(let i=0;i<5;++i){mass+=r.membership[i*2+c];sum+=r.membership[i*2+c]*x.data[i];}close([r.centers[c]],[sum/mass]);}
});
test('exact two cycle is not convergence; window length limits detection',()=>{
  const x=input([[-1,-3],[5,-3],[3,4],[-1,-2],[1,-3],[2,2]]),o={delta:7.3,initCenters:Float64Array.of(1,-3,2,2)};
  for(const backend of ['reference','adjoint']){const r=rmcm(x,{...o,backend});assert.equal(r.stopReason,'cycle');assert.equal(r.cycleLength,2);assert.equal(r.iterations,3);assert.equal(r.emptyClusterUpdates,0);assert.equal(r.converged,false);const short=rmcm(x,{...o,backend,cycleWindow:1,maxIterations:6});assert.equal(short.stopReason,'max_iter');assert.equal(short.converged,false);}
});
test('full graph canonical occupied means are bitwise equal and empty center is retained',()=>{
  const x=input([[1,-1],[4,5],[-3,-4],[1,3],[-5,3],[2,2],[-2,-2]]);
  const r=rmcm(x,{delta:100,initCenters:Float64Array.of(1,-1,4,5,4,5),maxIterations:1});assert.deepEqual(r.centers.slice(0,2),r.centers.slice(2,4));assert.deepEqual([...r.centers.slice(4)],[4,5]);assert.equal(r.emptyClusterUpdates,1);
});
test('direct graph boundary includes exact sqrt distance without epsilon broadening',()=>{
  const x=input([[0,0],[3,4]]);assert.equal(prepareRMCM(x,{delta:5}).nEdges,4);assert.equal(prepareRMCM(x,{delta:5-Number.EPSILON*4}).nEdges,2);
});
test('preparation and fit generator progress, synchronous cancellation, and clean reuse',()=>{
  let n=0;assert.throws(()=>rmcm(input([[0],[1],[3]]),{delta:2,nClusters:2,shouldCancel:()=>++n>2,graphBatchPairs:1}),{name:'AbortError'});
  assert.throws(()=>rmcm(input([[1]]),{delta:0,nClusters:1,signal:AbortSignal.abort()}),{name:'AbortError'});
  const p=prepareRMCM(input([[0],[1],[3],[9]]),{delta:2});assert.throws(()=>p.fit({nClusters:2,blockRows:1,shouldCancel:()=>true}),{name:'AbortError'});assert.equal(p.fit({nClusters:2}).centers.length,2);
  const phases=[];const r=rmcm(input([[0],[1],[3]]),{delta:2,nClusters:2,graphBatchPairs:1,onProgress:e=>phases.push(e.phase??'iteration')});assert.ok(phases.includes('graph-count'));assert.ok(phases.includes('graph-fill'));assert.ok(phases.includes('adjoint-precompute'));assert.ok(phases.includes('iteration'));assert.ok(r.iterations>0);
  const it=prepareRMCMSteps(input([[0],[1]]),{delta:1});assert.equal(it.next().value.phase,'graph-count');it.return();
});
test('runAsync yields during graph construction, matches sync and can abort before fit',async()=>{
  const x=input(Array.from({length:12},(_,i)=>[i,i%3])),o={delta:2,nClusters:3,seed:42,graphBatchPairs:5,blockRows:2,timeBudgetMs:0};let ticks=0;const timer=setInterval(()=>++ticks,0);let a;try{a=await runAsync('rmcm',x,o);}finally{clearInterval(timer);}assert.ok(ticks>0);assert.deepEqual(a,run('rmcm',x,o));
  const controller=new AbortController();await assert.rejects(runAsync('rmcm',x,{...o,signal:controller.signal,onProgress:e=>{if(e.phase==='graph-count'&&e.completed>0)controller.abort();}}),{name:'AbortError'});
});
test('memory and edge preflight reject rather than silently allocating dense graph',()=>{
  assert.throws(()=>prepareRMCM(input([[0],[0],[0]]),{delta:1,maxEdges:2}),/mandatory self-edge/);
  assert.throws(()=>prepareRMCM(input([[0],[0],[0]]),{delta:1,maxEdges:7}),/maxEdges/);
  assert.throws(()=>prepareRMCM(input([[0],[1]]),{delta:1,maxMemoryBytes:1}),/maxMemoryBytes/);
  const p=prepareRMCM(input([[0],[1],[2]]),{delta:0});assert.throws(()=>p.fit({nClusters:3,maxMemoryBytes:200,cycleWindow:100}),/maxMemoryBytes/);
  assert.throws(()=>prepareRMCM(input([[0]]),{delta:0,maxEdges:2**32}),/CSR index range/);
});
test('malformed data, parameters and unsupported model changes fail explicitly',()=>{
  for(const delta of [-1,NaN,Infinity,true,'1'])assert.throws(()=>prepareRMCM(input([[0]]),{delta}));
  for(const data of [[1],new BigInt64Array([1n]),new DataView(new ArrayBuffer(8)),Float64Array.of(Infinity)])assert.throws(()=>prepareRMCM({data,nSamples:1,nFeatures:1},{delta:0}));
  assert.throws(()=>prepareRMCM({data:Float64Array.of(1),nSamples:1,nFeatures:2},{delta:0}));
  for(const option of [{nClusters:0},{nClusters:4},{nClusters:true},{nClusters:1,maxIterations:0},{nClusters:1,cycleWindow:-1},{nClusters:1,returnMembership:1},{nClusters:1,initCenters:Float64Array.of(NaN)}])assert.throws(()=>rmcm(input([[0],[1],[2]]),{delta:1,...option}));
  for(const option of [{m:2},{excludeSelf:true},{initMembership:Float64Array.of(1)},{tolerance:.1},{backend:'csr'}])assert.throws(()=>rmcm(input([[0]]),{delta:0,nClusters:1,...option}));
  assert.throws(()=>rmcmReference(input([[0]]),{delta:0,nClusters:1,backend:'adjoint'}));
  const p=prepareRMCM(input([[0],[1]]),{delta:1});assert.throws(()=>p.fit({nClusters:1,delta:2}),/fixed/);assert.throws(()=>p.fit({nClusters:1,backend:'reference'}),/fixed/);
});
test('unsupported numeric extremes fail without inventing zero distances or false ties',()=>{
  assert.throws(()=>prepareRMCM(input([[0],[1e308]]),{delta:1}),/safe float64/);
  assert.throws(()=>prepareRMCM(input([[0],[1e-200]]),{delta:1}),/underflow/);
  assert.throws(()=>rmcm(input([[0],[1e-200]]),{delta:0,nClusters:1}),/underflow/);
  assert.throws(()=>rmcm(input([[0],[1]]),{delta:0,nClusters:1,initCenters:Float64Array.of(1e308)}),/safe float64/);
  assert.throws(()=>prepareRMCM(input([[0],[1e-155]]),{delta:1}),/subnormal/);
});
