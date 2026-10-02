import test from 'node:test';import assert from 'node:assert/strict';
import * as checkpoint from '../references/baseline/iteration-kernels.js';import * as candidate from '../../consumer/node_modules/ubukit-js/src/iteration-kernels.js';import {seededRandom} from '../references/baseline/core.js';
for(const d of [1,2,3,4,5,7,8,9,15,16,17,31,32,128,784])for(const weighted of [false,true])test(`FCM center scalar accumulator exact D=${d} weighted=${weighted}`,()=>{
 const random=seededRandom(d+7),n=53,k=7,x=Float64Array.from({length:n*d},()=>random()*1e100-5e99),u=Float64Array.from({length:n*k},()=>random()),sa=new Float64Array(k),sb=new Float64Array(k),ca=new Float64Array(k*d),cb=new Float64Array(k*d),fn=weighted?'fcmWeightedCenterBlock':'fcmCenterBlock';
 for(let i=0;i<n;i+=11){checkpoint[fn](x,u,sa,ca,i,Math.min(n,i+11),d,k,2);candidate[fn](x,u,sb,cb,i,Math.min(n,i+11),d,k,2);assert.deepEqual(sb,sa);assert.deepEqual(cb,ca);}
});
for(const d of [1,2,3,4,7,8,9,16,32,128,784])for(const density of [.1,1])test(`Rough center scalar accumulator exact D=${d} density=${density}`,()=>{
 const random=seededRandom(d+17),n=61,k=9,x=Float64Array.from({length:n*d},()=>random()*1e100-5e99),u=Float64Array.from({length:n*k},()=>random()<density?1/3:0),sa=new Float64Array(k),sb=new Float64Array(k),ca=new Float64Array(k*d),cb=new Float64Array(k*d);
 for(let start=0;start<n;start+=13){const end=Math.min(n,start+13);for(let i=start;i<end;i++)for(let c=0;c<k;c++){const weight=u[i*k+c];if(weight===0)continue;sa[c]+=weight;for(let f=0;f<d;f++)ca[c*d+f]+=weight*x[i*d+f];}candidate.roughCenterBlock(x,u,sb,cb,start,end,d,k);assert.deepEqual(sb,sa);assert.deepEqual(cb,ca);}
});
