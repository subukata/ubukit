import { performance } from 'node:perf_hooks';
import { readFileSync,writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import os from 'node:os';
import { adjustedRandScore,adjustedMutualInfoScore,adjustedScores } from '../consumer/node_modules/ubukit-js/src/external-metrics.js';
import { baselineARI,baselineAMI,baselineJoint } from './reference-baseline.mjs';
const seed=20261002,n=10000,k=50;let state=seed;
function random(){state^=state<<13;state^=state>>>17;state^=state<<5;return(state>>>0)/2**32;}
const a=Int32Array.from({length:n},(_,i)=>i%k),b=Int32Array.from(a,x=>random()<0.8?x:Math.floor(random()*k));
const fixture={name:'matched-current-10000-k50',seed,generator:'xorshift32; true[i]=i%50; predicted=true with p=0.8, otherwise uniform integer [0,50)',n,k,a:Array.from(a),b:Array.from(b)};
const bytes=JSON.stringify(fixture)+'\n';writeFileSync(new URL('../fixtures/matched-10000-k50.json',import.meta.url),bytes);
const sha256=createHash('sha256').update(bytes).digest('hex'),warmups=20,repeats=21,batchSize=5;
function measure(fn){for(let i=0;i<warmups;i++)fn(a,b);const rawMs=[];let value;for(let i=0;i<repeats;i++){const start=performance.now();for(let j=0;j<batchSize;j++)value=fn(a,b);rawMs.push((performance.now()-start)/batchSize);}const sorted=rawMs.toSorted((x,y)=>x-y);return{medianMs:sorted[Math.floor(repeats/2)],minMs:sorted[0],maxMs:sorted.at(-1),rawMs,value};}
const methods={javascript_dev4:{ari:adjustedRandScore,ami:adjustedMutualInfoScore,joint:adjustedScores,separate:(a,b)=>({ari:adjustedRandScore(a,b),ami:adjustedMutualInfoScore(a,b)})},javascript_ungrouped_baseline:{ari:baselineARI,ami:baselineAMI,joint:baselineJoint}};
const results={};for(const[implementation,fns]of Object.entries(methods)){results[implementation]={};for(const[name,fn]of Object.entries(fns))results[implementation][name]=measure(fn);}
const result={name:fixture.name,n,kTrue:new Set(a).size,kPred:new Set(b).size,seed,fixtureSha256:sha256,packageVersion:JSON.parse(readFileSync(new URL('../consumer/node_modules/ubukit-js/package.json',import.meta.url))).version,node:process.version,cpu:os.cpus()[0]?.model,logicalCPUs:os.cpus().length,platform:process.platform,arch:process.arch,threadSettings:{javascript:'single synchronous JS thread; no workers used'},warmups,repeats,batchSize,timing:'serial median wall milliseconds per invocation from batches of five; array construction excluded; validation/encoding/contingency included; arithmetic AMI; no process-global data cache',results};
writeFileSync(new URL('../reports/matched-node.json',import.meta.url),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
