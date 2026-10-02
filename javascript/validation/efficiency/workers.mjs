import assert from 'node:assert/strict';
import fs from 'node:fs';
import {Worker} from 'node:worker_threads';
import {createWorkerClient,createRealtimeWorkerClient} from '../../consumer/node_modules/ubukit-js/src/index.js';
import {run as frozen} from '../efficiency_baseline_package/src/index.js';
const reports=[];let checks=0;
function input(n,d,seed=1){return{nSamples:n,nFeatures:d,data:Float64Array.from({length:n*d},(_,i)=>Math.sin(i*.73+seed)+Math.cos(i*.09-seed))};}
for(const algorithm of ['som','som_batch','kmeans']){
 const x=input(29,3),o=algorithm==='kmeans'?{nClusters:6,kernelBackend:'wasm',maxIterations:4,blockRows:2}:{gridShape:[4,4],maxIterations:4,blockRows:2,blockUnits:2,bmuBackend:'grouped'};
 const expected=frozen(algorithm,x,o);
 const simple=createWorkerClient({workerFactory:()=>new Worker(new URL('../../consumer/node_modules/ubukit-js/src/worker.js',import.meta.url),{type:'module'})});
 const realtime=createRealtimeWorkerClient({workerFactory:()=>new Worker(new URL('../../consumer/node_modules/ubukit-js/src/realtime-worker.js',import.meta.url),{type:'module'})});
 try{
  assert.deepEqual(await simple.run(algorithm,x,o),expected);checks++;
  assert.deepEqual((await realtime.run(algorithm,x,o,{timeBudgetMs:0,maxChunks:1,warmStart:false})).result,expected);checks++;
  let obsoleteProgress=0;
  const old=realtime.run(algorithm,input(100,3,77),{...o,maxIterations:40,onProgress:()=>obsoleteProgress++},{timeBudgetMs:0,maxChunks:1,warmStart:false}).catch(e=>e);
  const pending=realtime.run(algorithm,input(33,3,88),o,{timeBudgetMs:0,maxChunks:1,warmStart:false}).catch(e=>e);
  const latestInput=input(21,3,99),snapshot=structuredClone(latestInput);
  const latest=realtime.run(algorithm,latestInput,o,{timeBudgetMs:0,maxChunks:1,warmStart:false});latestInput.data.fill(999);
  assert.equal((await old).name,'AbortError');assert.equal((await pending).name,'AbortError');assert.equal(obsoleteProgress,0);checks+=3;
  assert.deepEqual((await latest).result,frozen(algorithm,snapshot,o));checks++;
  reports.push({algorithm,status:'passed'});
 }finally{simple.dispose();realtime.dispose();}
}
const result={status:'passed',checks,groups:reports};console.log(JSON.stringify(result,null,2));
