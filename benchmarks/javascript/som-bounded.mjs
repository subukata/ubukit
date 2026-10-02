import {mkdirSync as ensureResultDirectory} from 'node:fs';
ensureResultDirectory(new URL('./results/',import.meta.url),{recursive:true});
import fs from 'node:fs';
import {performance} from 'node:perf_hooks';
import {createSession} from '../../javascript/consumer/node_modules/ubukit-js/src/index.js';
const rows=[];
for(const [n,d] of [[1000,3],[1000,20],[256,784]]){
 const data=Float64Array.from({length:n*d},(_,i)=>Math.sin(i*.019)+Math.cos(i*.0037)),x={data,nSamples:n,nFeatures:d},w=new Float64Array(256*d);
 for(let j=0;j<256;j++)w.set(data.subarray((j%n)*d,(j%n+1)*d),j*d);
 for(const algorithm of ['som','som_batch']){
  const options={gridShape:[16,16],initialPrototypes:w,maxIterations:algorithm==='som'?n:2,sigma:8,sigmaEnd:1};
  const times=[];let firstSampleMs,firstEpochMs,primaryBytes;
  for(let trial=0;trial<4;trial++){
   const s=createSession(algorithm,x,options),start=performance.now();let first=null;
   while(!s.status.done){s.step(1,{timeBudgetMs:1e6,maxChunks:100000});if(first===null&&s.status.iteration>0)first=performance.now()-start;}
   const ms=performance.now()-start,r=s.snapshot().result;primaryBytes=r.stats.estimatedPrimaryBytes;s.dispose();
   if(trial){times.push(ms);if(algorithm==='som')firstSampleMs=first;else firstEpochMs=first;}
  }
  times.sort((a,b)=>a-b);rows.push({algorithm,n,d,grid:[16,16],updates:options.maxIterations,medianMs:times[1],rangeMs:[times[0],times[2]],firstSampleMs,firstEpochMs,estimatedPrimaryBytes:primaryBytes,includesInitializationAndFinalProjection:true,steadyStateProcess:true,samples:3});
 }
}
const report={node:process.version,platform:process.platform,date:new Date().toISOString(),note:'Bounded local CPU timings only, not cross-language speed claims. Concurrent machine load is uncontrolled.',rows};fs.writeFileSync(new URL('./results/som-bounded-benchmark.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
