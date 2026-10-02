import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { createWorkerClient, run } from '../../consumer/node_modules/ubukit-js/src/index.js';
const factory=()=>new Worker(new URL('../../consumer/node_modules/ubukit-js/src/worker.js',import.meta.url),{type:'module'});
const input=()=>({data:Float64Array.of(0,1,3,9),nSamples:4,nFeatures:1});
test('RMCM Worker parity, graph/iteration progress, input preservation and transfer',async()=>{
  const client=createWorkerClient({workerFactory:factory});
  try{
    for(const backend of ['adjoint','reference']){
      const x=input(),options={backend,delta:2,initCenters:Float64Array.of(0,9),graphBatchPairs:1};let progress=0;
      const actual=await client.run('rmcm',x,{...options,onProgress:()=>++progress});assert.deepEqual(actual,run('rmcm',x,options));assert.ok(progress>0);assert.equal(x.data.byteLength,32);
    }
    const x=input();await client.run('rmcm',x,{delta:2,nClusters:2,transferInput:true});assert.equal(x.data.byteLength,0);
    await assert.rejects(client.run('rmcm',input(),{delta:-1,nClusters:2}),/delta/);
    assert.equal((await client.run('rmcm',input(),{delta:2,nClusters:2})).labels.length,4);
  }finally{client.dispose();}
});
test('RMCM Worker can abort graph construction after actual progress and be reused',async()=>{
  const client=createWorkerClient({workerFactory:factory});
  try{
    const controller=new AbortController();let sawGraph=false;
    const n=300,x={data:Float64Array.from({length:n},(_,i)=>i),nSamples:n,nFeatures:1};
    await assert.rejects(client.run('rmcm',x,{delta:1,nClusters:3,graphBatchPairs:16,signal:controller.signal,onProgress:e=>{if(e.phase==='graph-count'){sawGraph=true;controller.abort();}}}),{name:'AbortError'});
    assert.equal(sawGraph,true);assert.equal((await client.run('rmcm',input(),{delta:2,nClusters:2})).labels.length,4);
  }finally{client.dispose();}
});
