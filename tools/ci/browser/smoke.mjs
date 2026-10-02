// Exercise the packed consumer over HTTP, not source aliases or Node VM.
import * as api from '/consumer/node_modules/ubukit-js/src/index.js';
const records=[];
const wasmMode=new URL(location.href).searchParams.get('wasm')??'enabled';
const assert=(ok,label)=>{if(!ok)throw Error(label);};
const close=(actual,expected,tol=2e-12)=>{
 assert(actual.length===expected.length,'length');
 actual.forEach((v,i)=>assert(Math.abs(v-expected[i])<=tol*Math.max(1,Math.abs(expected[i])),`value ${i}`));
};
const input=rows=>({data:Float64Array.from(rows.flat()),nSamples:rows.length,nFeatures:rows[0].length});
const opts=c=>({...c.options,gridShape:c.options.gridShape.slice(),initialPrototypes:Float64Array.from(c.initialPrototypes.flat())});
async function check(name,fn){await fn();records.push({name,status:'passed'});}
try {
 const fixture=await (await fetch('/fixtures/som-shared-reference.json')).json();
 for(const c of fixture.cases) await check('fixture '+c.name,async()=>{
  const x=input(c.data),o=opts(c),expected=c.expected.centers.flat(),r=api.run(c.algorithm,x,o);
  close(r.centers,expected);assert(JSON.stringify([...r.labels])===JSON.stringify(c.expected.labels),'labels');
  const s=api.createSession(c.algorithm,x,o);let previous=0,steps=0;
  while(!s.status.done){assert(++steps<100000,'bounded session');s.step(1,{timeBudgetMs:1e6});if(s.status.iteration>previous){close(s.snapshot().result.centers,c.expected.history[s.status.iteration-1].flat());previous=s.status.iteration;}}
  close(s.snapshot().result.centers,expected);s.dispose();
  close((await api.runAsync(c.algorithm,x,{...o,timeBudgetMs:0})).centers,expected);
 });
 for(const algorithm of ['som','som_batch']) {
  const c=fixture.cases.find(x=>x.algorithm===algorithm),x=input(c.data),o=opts(c);
  await check(algorithm+' default module worker and cancel/reuse',async()=>{
   const client=api.createWorkerClient();
   try{
    const pending=client.run(algorithm,x,{...o,maxIterations:100000});
    const caught=pending.then(()=>{throw Error('cancel resolved');},e=>assert(e.name==='AbortError','cancel type'));
    client.cancel();await caught;
    close((await client.run(algorithm,x,o)).centers,c.expected.centers.flat());
   }finally{client.dispose();}
  });
  await check(algorithm+' realtime worker and replacement',async()=>{
   const client=api.createRealtimeWorkerClient();
   try{
    const first=client.run(algorithm,x,{...o,maxIterations:100000},{timeBudgetMs:0,maxChunks:1});
    const caught=first.then(()=>{throw Error('superseded resolved');},e=>assert(e.name==='AbortError','superseded type'));
    const latest=client.run(algorithm,x,o,{timeBudgetMs:0,maxChunks:1});
    await caught;close((await latest).result.centers,c.expected.centers.flat());
   }finally{client.dispose();}
  });
 }
 // PR7 route coverage: inherited fixtures use small grids and do not trigger
 // the new K>=16 low-dimensional auto-selection rule.
 const routeInput={nSamples:23,nFeatures:3,
  data:Float64Array.from({length:69},(_,i)=>Math.sin(i*.31)+Math.cos(i*.17))};
 const same=(a,b,label)=>assert(a.length===b.length&&a.every((v,i)=>Object.is(v,b[i])),label);
 for(const algorithm of ['som','som_batch']) {
  const options={gridShape:[4,4],maxIterations:3,blockRows:3,blockUnits:3,
   initialPrototypes:Float64Array.from({length:48},(_,i)=>Math.sin(i*.23))};
  const elite=api.run(algorithm,routeInput,{...options,bmuBackend:'scalar'});
  for(const bmuBackend of ['grouped','auto']) await check(algorithm+' '+bmuBackend+' versus retained scalar',async()=>{
   const selected={...options,bmuBackend},result=api.run(algorithm,routeInput,selected);
   same(result.centers,elite.centers,'route centers');same(result.labels,elite.labels,'route labels');
   const session=api.createSession(algorithm,routeInput,selected);let count=0;
   try{
    while(!session.status.done){assert(++count<100000,'bounded route session');session.step(1,{timeBudgetMs:0,maxChunks:1});}
    same(session.snapshot().result.centers,elite.centers,'route session');
   }finally{session.dispose();}
   const client=api.createWorkerClient();
   try{same((await client.run(algorithm,routeInput,selected)).centers,elite.centers,'route worker');}
   finally{client.dispose();}
  });
 }
 await check('k-means WASM center-cache preservation',()=>{
  const options={nClusters:4,initCenters:routeInput.data.slice(0,12),maxIterations:4,
   blockRows:3,kernelBackend:'wasm'};
  const original=api.run('kmeans',routeInput,{...options,wasmCenterCache:false});
  const cached=api.run('kmeans',routeInput,{...options,wasmCenterCache:true});
  same(cached.centers,original.centers,'cache centers');same(cached.labels,original.labels,'cache labels');
  assert(Object.is(cached.inertia,original.inertia),'cache inertia');
  const rows=cached.kernel.wasmAssignmentRows+cached.kernel.wasmFinalizationRows;
  if(wasmMode==='enabled'){
   assert(cached.kernel.actual==='wasm','WASM must execute without fallback');
   assert(cached.kernel.wasmAssignmentRows>0&&cached.kernel.wasmFinalizationRows>0,'both WASM phases must execute');
   assert(cached.kernel.javascriptFallbackAssignmentRows+cached.kernel.javascriptFallbackFinalizationRows===0,'unexpected fallback');
  }else{
   assert(cached.kernel.actual==='javascript'&&rows===0,'intentional WASM fallback must execute JavaScript');
   const js=api.run('kmeans',routeInput,{...options,kernelBackend:'javascript'});
   same(cached.centers,js.centers,'fallback centers');same(cached.labels,js.labels,'fallback labels');
   assert(Object.is(cached.inertia,js.inertia),'fallback inertia');
  }
  records.push({name:'k-means observed kernel',wasmMode,actual:cached.kernel.actual,
   wasmRows:cached.kernel.wasmAssignmentRows+cached.kernel.wasmFinalizationRows});
 });
 const efcmFixture=await (await fetch('/fixtures/entropy-fcm-reference.json')).json();
 for(const c of efcmFixture.cases) await check('entropy FCM '+c.name,async()=>{
  const x=input(c.X),before=x.data.slice();
  const options={initMembership:Float64Array.from(c.init.flat()),tau:c.tau,maxIterations:c.iterations,tolerance:0,returnHistory:true};
  const r=api.entropyFcm(x,options);
  close(r.centers,c.centers.flat());close(r.membership,c.membership.flat());
  close(r.objectiveHistory,c.objective_history);close([r.objective],[c.objective]);
  const asyncResult=await api.runAsync('entropy-fcm',x,{...options,timeBudgetMs:0});
  close(asyncResult.membership,r.membership);close(asyncResult.centers,r.centers);close(x.data,before,0);
 });
 await check('entropy FCM one-shot Worker',async()=>{
  const c=efcmFixture.cases[0],x=input(c.X),before=x.data.slice();
  const options={initMembership:Float64Array.from(c.init.flat()),tau:c.tau,maxIterations:c.iterations,tolerance:0};
  const worker=api.createWorkerClient();
  try{const r=await worker.run('entropy-fcm',x,options);close(r.centers,c.centers.flat());close(r.membership,c.membership.flat());close([r.objective],[c.objective]);close(x.data,before,0);}
  finally{worker.dispose();}
 });
 await check('external metric public exports',()=>{
  const r=api.adjustedScores([0,0,1,1],[0,1,0,1]);
  assert(r.ari===-.5 && Math.abs(r.ami+.5)<1e-15,'ARI/AMI');
 });
 window.browserResult={status:'passed',actualBrowserExecution:true,userAgent:navigator.userAgent,records};
}catch(error){window.browserResult={status:'failed',actualBrowserExecution:true,userAgent:navigator.userAgent,records,error:String(error),stack:error.stack};}
document.querySelector('#result').textContent=JSON.stringify(window.browserResult,null,2);
