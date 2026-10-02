import test from 'node:test';import assert from 'node:assert/strict';
import * as old from '../references/baseline/index.js';import * as current from '../../consumer/node_modules/ubukit-js/src/index.js';
for (const p of [.001,1,2,3,1000]) for(const scale of [1,1e100])test(`Rough coincident centers retain full output and event boundaries p=${p} scale=${scale}`,()=>{
 const n=600,d=64,k=8,random=old.seededRandom(799),data=Float64Array.from({length:n*d},()=>scale*(random()-.5)),center=data.slice(0,d),initCenters=new Float64Array(k*d);for(let c=0;c<k;c++)initCenters.set(center,c*d);
 const input={data,nSamples:n,nFeatures:d},eventsA=[],eventsB=[],options={nClusters:k,initCenters,p,alpha:1.1,beta:.1,maxIterations:6,blockRows:83};
 const a=old.run('exrcm',input,{...options,onProgress:e=>eventsA.push(e)}),b=current.run('exrcm',input,{...options,onProgress:e=>eventsB.push(e)});
 assert.deepEqual(b,a);assert.deepEqual(eventsB,eventsA);
});
