import {spawnSync} from 'node:child_process';
import fs from 'node:fs';
const records=[];
for(const [d,k] of [[128,16],[784,16],[784,64]])for(let trial=0;trial<3;trial++)for(const variant of trial%2?['scalar','auto','old']:['old','auto','scalar']){
 const r=spawnSync(process.execPath,[new URL('./efficiency-isolated-case.mjs',import.meta.url).pathname,variant,String(d),String(k)],{encoding:'utf8'});if(r.status!==0)throw Error(r.stderr);records.push({...JSON.parse(r.stdout),trial});
}
fs.writeFileSync(new URL('../reports/efficiency-isolated.json',import.meta.url),JSON.stringify({method:'Independent fresh Node process for each variant/shape/trial. 100 warmups, 31 timed whole public calls; exact-first-prototype cutoff-friendly input. Three trials per variant/shape, alternating order. Median per process; no forced GC.',records},null,2));console.log(JSON.stringify(records.map(({rawMs,...rest})=>rest)));
