import {performance} from 'node:perf_hooks';
import {readFileSync,writeFileSync} from 'node:fs';
import os from 'node:os';
import {adjustedRandScore,adjustedMutualInfoScore,adjustedScores} from '../consumer/node_modules/ubukit-js/src/external-metrics.js';
import {baselineARI,baselineAMI,baselineJoint} from './reference-baseline.mjs';
let state=20261002;function random(){state^=state<<13;state^=state>>>17;state^=state<<5;return(state>>>0)/2**32;}
const cases=[];
function add(name,a,b){cases.push({name,a:Array.from(a),b:Array.from(b)});}
for(const[n,k]of[[10000,10],[100000,20],[10000,1000]]){
  const a=Int32Array.from({length:n},(_,i)=>i%k),b=Int32Array.from({length:n},(_,i)=>random()<0.8?a[i]:Math.floor(random()*k));
  add(`balanced-correlated-${n}-k${k}`,a,b);
}
{
 const n=30000,k=20,a=Int32Array.from({length:n},()=>Math.floor(random()**3*k)),b=Int32Array.from(a,x=>random()<0.65?x:Math.floor(random()**3*k));add('unequal-margins-30000-k20',a,b);
}
for(const fixture of JSON.parse(readFileSync(new URL('../fixtures/sklearn-1.8.json',import.meta.url))).cases.filter(x=>x.name.startsWith('load_')))add(fixture.name,fixture.a,fixture.b);
writeFileSync(new URL('../fixtures/benchmark-inputs.json',import.meta.url),JSON.stringify({seed:20261002,cases}));
const fns={ari:adjustedRandScore,ami:adjustedMutualInfoScore,joint:adjustedScores,baselineARI,baselineAMI,baselineJoint,separate:(a,b)=>({ari:adjustedRandScore(a,b),ami:adjustedMutualInfoScore(a,b)})};
let checksum=0;
const rows=[];
function time(fn,a,b,repeats){const values=[];for(let i=0;i<3;i++)fn(a,b);for(let i=0;i<repeats;i++){const start=performance.now(),value=fn(a,b);values.push(performance.now()-start);checksum+=typeof value==='number'?value:value.ari+value.ami;}values.sort((x,y)=>x-y);return{medianMs:values[Math.floor(values.length/2)],minMs:values[0],maxMs:values.at(-1),repeats};}
for(const c of cases){const a=Int32Array.from(c.a),b=Int32Array.from(c.b),times={},values={};for(const[name,fn]of Object.entries(fns)){times[name]=time(fn,a,b,9);values[name]=fn(a,b);}rows.push({name:c.name,n:a.length,kTrue:new Set(a).size,kPred:new Set(b).size,times,values});console.log(c.name,JSON.stringify(times));}
const result={node:process.version,platform:process.platform,arch:process.arch,cpu:os.cpus()[0]?.model,logicalCPUs:os.cpus().length,timing:'serial, 3 warmups then 9 measured calls; median wall milliseconds; construction of typed input outside timing; score validation/encoding/contingency included; no data cache',checksum,rows};
writeFileSync(new URL('../reports/benchmark-node.json',import.meta.url),JSON.stringify(result,null,2)+'\n');
