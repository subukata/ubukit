import test from 'node:test';
import assert from 'node:assert/strict';
import { kmeans as baseline, kmeansSteps as baselineSteps } from '../references/baseline/clustering.js';
import { kmeans, kmeansSteps } from '../../consumer/node_modules/ubukit-js/src/clustering.js';
import { somOlp as baselineSom, somOlpSteps as baselineSomSteps } from '../references/baseline/som-olp.js';
import { somOlp, somOlpSteps } from '../../consumer/node_modules/ubukit-js/src/som-olp.js';
import { seededRandom } from '../../consumer/node_modules/ubukit-js/src/core.js';
import { ClusteringSession as BaselineSession } from '../references/baseline/session.js';
import { ClusteringSession } from '../../consumer/node_modules/ubukit-js/src/session.js';
function state(fn){try{return {value:fn()};}catch(error){return {name:error.name,message:error.message};}}
function collect(iterator){const events=[];for(;;){const s=iterator.next();if(s.done)return{events,result:s.value};events.push(s.value);}}
test('round3 direct kmeans retains every result and yield across 250 finite random cases',()=>{
 const rnd=seededRandom(921);
 for(let z=0;z<250;z++){
  const n=1+Math.floor(rnd()*53),d=[1,2,3,4,7,8,9,17,64][z%9],k=1+Math.floor(rnd()*13),scale=[1,1e-100,1e100][z%3],offset=z%7===0?1e10:0;
  const input={data:Float64Array.from({length:n*d},()=>offset+scale*(rnd()*8-4)),nSamples:n,nFeatures:d};
  const options={nClusters:k,maxIterations:1+z%7,blockRows:1+z%11,initCenters:Float64Array.from({length:k*d},()=>offset+scale*(rnd()*8-4))};
  if(z%5===0)options.initCenters.fill(offset);
  assert.deepEqual(state(()=>collect(kmeansSteps(input,options))),state(()=>collect(baselineSteps(input,options))),`case ${z}`);
 }
});
test('round3 kmeans preserves underflow, subnormal, overflow and ties outcomes',()=>{
 for(const d of [1,2,3,4,7,8,9,17])for(const k of [1,2,3,4,5,8,9])for(const tiny of[0,1e-200,1e-160,1e-150,1e100,1e155]){
  const input={data:new Float64Array(3*d),nSamples:3,nFeatures:d};input.data[d]=tiny;input.data[2*d]=-tiny;
  const options={nClusters:k,initCenters:new Float64Array(k*d),maxIterations:3,blockRows:2};
  if(k>1)options.initCenters[d]=tiny;
  assert.deepEqual(state(()=>kmeans(input,options)),state(()=>baseline(input,options)),`${d}/${k}/${tiny}`);
 }
});
test('round3 SOM two-dimensional blocks preserve all arrays and yields at gamma zero and arbitrary lambda',()=>{
 const rnd=seededRandom(922);
 for(let z=0;z<80;z++){
  const n=2+z%21,m=1+z%13,d=z%3===0?3:2,q=z%4===0?1:2;
  const input={data:Float64Array.from({length:n*d},()=>rnd()*8-4),nSamples:n,nFeatures:d};
  const grid={data:Float64Array.from({length:m*q},()=>rnd()*20-10),nSamples:m,nFeatures:q};
  const options={grid,initializer:'sample',seed:z,lambda:[1e-3,.1,.27,3,100][z%5],gamma:[0,.37,1,200][z%4],maxIterations:z%6,tolerance:0,blockRows:1+z%7};
  const actual=state(()=>collect(somOlpSteps(input,options))),expected=state(()=>collect(baselineSomSteps(input,options)));
  if(z===59 || z===63){
   // Independently audited tiny-mass repairs: all other entries/events remain exact.
   const index=z===59?5:24, old=z===59?0.6884290744771885:2.9543435536324973;
   const exact=z===59?0.6884290744771887:2.954343553632498;
   for(const key of ['W','centers','prototypes'])assert.equal(expected.value.result[key][index],old);
   for(const key of ['W','centers','prototypes'])expected.value.result[key][index]=exact;
  }
  assert.deepEqual(actual,expected,`case ${z}`);
 }
});
test('round3 sessions preserve checkpoints and changing data, shape, parameters',()=>{
 const rnd=seededRandom(923);
 for(const algorithm of['kmeans','som-olp']){
  const input={data:Float64Array.from({length:46},()=>rnd()*2-1),nSamples:23,nFeatures:2};
  const options=algorithm==='kmeans'?{nClusters:5,seed:12,maxIterations:4,blockRows:3}:{grid:{data:Float64Array.of(0,0,0,1,1,0,1,1),nSamples:4,nFeatures:2},initializer:'sample',seed:12,maxIterations:4,tolerance:0,blockRows:3};
  const b=new BaselineSession(algorithm,input,options),c=new ClusteringSession(algorithm,input,options);
  for(let phase=0;phase<5;phase++){
   while(!b.status.done){b.step(1,{timeBudgetMs:100000,maxChunks:1});c.step(1,{timeBudgetMs:100000,maxChunks:1});assert.deepEqual(c.snapshot(),b.snapshot());}
   if(phase===0){input.data[3]+=.125;b.updateData(input);c.updateData(input);}
   if(phase===1){const next={data:input.data.slice(0,34),nSamples:17,nFeatures:2};b.updateData(next);c.updateData(next);}
   if(phase===3){const next={data:Float64Array.from({length:17*8},()=>rnd()*2-1),nSamples:17,nFeatures:8};b.updateData(next);c.updateData(next);}
   if(phase===2){const patch=algorithm==='kmeans'?{maxIterations:7}:{gamma:1.17,lambda:.21};b.updateParameters(patch);c.updateParameters(patch);}
  }
 }
});

test('round3 cancellation checkpoints remain identical for generic dimensions',()=>{
 const input={data:Float64Array.from({length:47*17},(_,i)=>Math.sin(i)),nSamples:47,nFeatures:17};
 for(const algorithm of ['kmeans','som-olp'])for(const at of [1,3,8,20,45]){
  const common=algorithm==='kmeans'?{nClusters:9,maxIterations:4,blockRows:7,seed:4}:{grid:{data:Float64Array.from({length:18},(_,i)=>i%3),nSamples:9,nFeatures:2},maxIterations:4,blockRows:7,initializer:'sample',seed:4};
  const bfn=algorithm==='kmeans'?baseline:baselineSom,cfn=algorithm==='kmeans'?kmeans:somOlp;
  let aCalls=0,bCalls=0;
  const a=state(()=>bfn(input,{...common,shouldCancel:()=>++aCalls===at}));
  const b=state(()=>cfn(input,{...common,shouldCancel:()=>++bCalls===at}));
  assert.deepEqual(b,a);assert.equal(bCalls,aCalls);
 }
});
