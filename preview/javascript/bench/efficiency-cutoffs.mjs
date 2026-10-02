import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {performance} from 'node:perf_hooks';
import {run as baseline} from '../validation/efficiency_baseline_package/src/index.js';
import {run} from '../consumer/node_modules/ubukit-js/src/index.js';
const records=[],median=a=>a.sort((x,y)=>x-y)[a.length>>1];
for(const d of [2,7,16,32,128,784])for(const k of [4,16,64,256]){
 const n=500,x={data:new Float64Array(n*d),nSamples:n,nFeatures:d},w=new Float64Array(k*d);
 // Unit zero exact-matches all rows; every other unit exceeds the zero cutoff
 // in the first feature. This deliberately favors the old scalar early exit.
 for(let j=1;j<k;j++)w[j*d]=j;
 const o={gridShape:[k,1],initialPrototypes:w,maxIterations:0};
 const calls={old:()=>baseline('som',x,o),auto:()=>run('som',x,o),scalar:()=>run('som',x,{...o,bmuBackend:'scalar'}),grouped:()=>run('som',x,{...o,bmuBackend:'grouped'})};
 assert.deepEqual(calls.auto(),calls.old());assert.deepEqual(calls.grouped(),calls.old());
 for(let t=0;t<10;t++)for(const f of Object.values(calls))f();
 const times=Object.fromEntries(Object.keys(calls).map(k=>[k,[]]));
 for(let t=0;t<11;t++){const order=Object.keys(calls);if(t%2)order.reverse();for(const name of order){const a=performance.now();calls[name]();times[name].push(performance.now()-a);}}
 const medians=Object.fromEntries(Object.entries(times).map(([k,v])=>[k,median(v)]));const row={d,k,n,medians,speedupOldOverGrouped:medians.old/medians.grouped,speedupScalarOverGrouped:medians.scalar/medians.grouped};records.push(row);console.log(JSON.stringify(row));
}
fs.writeFileSync(new URL('../reports/efficiency-cutoffs.json',import.meta.url),JSON.stringify({sourceManifestSHA256:createHash('sha256').update(fs.readFileSync(new URL('../package/SOURCE_MANIFEST.json',import.meta.url))).digest('hex'),date:new Date().toISOString(),node:process.version,method:'public som zero-training projection; exact initial match, other units cut off after first feature; 10 warmups/11 alternating rounds; median milliseconds',records},null,2)+'\n');
