import test from 'node:test';import assert from 'node:assert/strict';
import * as beforeSteps from '../references/before_initial_weight_cache/clustering.js';import * as afterSteps from '../../consumer/node_modules/ubukit-js/src/clustering.js';
import * as before from '../references/before_initial_weight_cache/index.js';import * as after from '../../consumer/node_modules/ubukit-js/src/index.js';
for(const d of [1,2,3,5,8,17,128,784])for(const m of [1+Number.EPSILON,1.3,3,1000])test((m>32||m-1<1e-4)?`Stable finite-m arithmetic replaces high-m power cache, D=${d}`:`First-block power cache retains exact FCM output, D=${d},m=${m}`,()=>{
 const random=before.seededRandom(190+d),n=31,k=5,data=Float64Array.from({length:n*d},()=>random()*2-1),initMembership=Float64Array.from({length:n*k},()=>random()),input={data,nSamples:n,nFeatures:d},options={initMembership,nClusters:k,m,maxIterations:3,tolerance:0,returnHistory:true,blockRows:7},eventsA=[],eventsB=[];const a=before.run('fcm',input,{...options,onProgress:e=>eventsA.push(e)});
 // The log-domain route supersedes both exact high-m cache arithmetic and
 // the former squared-distance rejection; ordinary m retains exact parity.
 if((m>32||m-1<1e-4)){const b=after.run('fcm',input,options);assert.ok(b.centers.every(Number.isFinite));assert.ok(b.membership.every(Number.isFinite));assert.equal(b.numericalMode,'log-domain');assert.deepEqual(after.run('fcm',input,{...options,blockRows:1}),b);return;}
 const b=after.run('fcm',input,{...options,onProgress:e=>eventsB.push(e)});assert.deepEqual(b,a);assert.deepEqual(eventsB,eventsA);
});

test('First power cache retains every cooperative yield boundary',()=>{
 const rng=before.seededRandom(98),n=41,d=17,k=6,data=Float64Array.from({length:n*d},()=>rng()),initMembership=Float64Array.from({length:n*k},()=>rng()),input={data,nSamples:n,nFeatures:d},options={initMembership,nClusters:k,m:1.3,maxIterations:3,tolerance:0,blockRows:7};
 function collect(mod){const iterator=mod.fcmSteps(input,options),events=[];for(;;){const next=iterator.next();if(next.done)return {events,result:next.value};events.push(next.value);}}
 assert.deepEqual(collect(afterSteps),collect(beforeSteps));
});
