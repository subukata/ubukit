/** Correct only FCM after its benchmark baseline was replaced. Other algorithm
 * records retain their original measurement window; never imply a full rerun.
 */
import fs from 'node:fs';
import os from 'node:os';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {runClusteringBenchmarks} from './clustering.js';
const measurementStartedAt=new Date().toISOString();
const warm=runClusteringBenchmarks({algorithms:['fcm'],warmups:3,repeats:7})[0];
const firstCalls=[];
for(const implementation of ['fast','scratch']){
 const code=`import {runClusteringFirstCall} from './bench/clustering.js'; console.log(JSON.stringify(runClusteringFirstCall(${JSON.stringify({algorithm:'fcm',implementation})})));`;
 firstCalls.push(JSON.parse(execFileSync(process.execPath,['--input-type=module','-e',code],{encoding:'utf8'})));
}
const sourcePaths=['src/core.js','src/clustering.js','bench/clustering.js','bench/fcm-linear-reference.js'];
const report={measurementStartedAt,measurementFinishedAt:new Date().toISOString(),runtime:process.version,v8:process.versions.v8,cpu:os.cpus()[0]?.model,singleThread:true,browserMeasured:false,baseline:'allocation-reusing Float64Array FCM; per-row inverse-power normalization O(NK), m=2 specialization in both implementations',scope:'Only FCM remeasured; identical N=2000 D=8 K=8 input and explicit U, 12 fixed iterations. Includes full fit plus final labels/objective/FPC/delta. 3 extra warm-ups, 7 alternating repetitions. Fresh process per first-call implementation.',sourceSha256:Object.fromEntries(sourcePaths.map(path=>[path,createHash('sha256').update(fs.readFileSync(path)).digest('hex')])),warm,firstCalls};
fs.mkdirSync('results', { recursive: true });
fs.writeFileSync('results/fcm-linear-baseline-remeasurement.json',JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));
