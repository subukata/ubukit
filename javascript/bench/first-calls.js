import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
const results=[];
for(const algorithm of ['kmeans','fcm','rcm','exrcm'])for(const implementation of ['fast','scratch']){
  const code=`import {runClusteringFirstCall} from './bench/clustering.js'; console.log(JSON.stringify(runClusteringFirstCall(${JSON.stringify({algorithm,implementation})})));`;
  results.push(JSON.parse(execFileSync(process.execPath,['--input-type=module','-e',code],{encoding:'utf8'})));
}
for(const algorithm of ['som-prepared','som-init-plus-kernel','neighborhood','neighborhood-ties'])for(const implementation of ['optimized','baseline'])results.push(JSON.parse(execFileSync(process.execPath,['bench/som-neighborhood.js','--mode','first-call','--size','demo','--algorithm',algorithm,'--implementation',implementation],{encoding:'utf8'})));
fs.writeFileSync('results/node-first-calls.json',JSON.stringify({runtime:process.version,generatedAt:new Date().toISOString(),notes:'Independent fresh process for each implementation; execution timing excludes process startup/import/data creation. One observation per case, not a statistical throughput estimate.',results},null,2));
console.log(results.map(x=>`${x.algorithm}/${x.implementation}: ${x.elapsedMs.toFixed(2)}ms`).join('\n'));
