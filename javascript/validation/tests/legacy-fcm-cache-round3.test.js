import test from 'node:test';
import assert from 'node:assert/strict';
import { createSession, fcm, seededRandom } from '../../consumer/node_modules/ubukit-js/src/index.js';
import { fcmMembershipBlock } from '../../consumer/node_modules/ubukit-js/src/iteration-kernels.js';
function data() { const rng=seededRandom(321),n=33,d=17,k=5;return {input:{data:Float64Array.from({length:n*d},()=>rng()*10-5),nSamples:n,nFeatures:d},options:{nClusters:k,initMembership:Float64Array.from({length:n*k},()=>rng()),maxIterations:4,blockRows:4,tolerance:0,returnHistory:true}}; }
for(const m of [1.3,3,1000])test(`power reuse preserves generic m=${m} partial-iteration snapshots`,()=>{
 const {input,options}=data();const session=createSession('fcm',input,{...options,m});
 while(session.status.iteration<1)session.step(1,{timeBudgetMs:1000,maxChunks:1});
 const before=session.snapshot();let partial=false;
 while(session.status.iteration<2){session.step(1,{timeBudgetMs:1000,maxChunks:1});if(session.status.iteration===1){const now=session.snapshot();assert.deepEqual(now.result.membership,before.result.membership);assert.deepEqual(now.result.centers,before.result.centers);partial=true;}}
 assert.ok(partial);
 while(!session.status.done)session.step(1,{timeBudgetMs:1000,maxChunks:1000});
 const actual=session.snapshot().result,expected=fcm(input,{...options,m});
 assert.deepEqual(actual.membership,expected.membership);assert.deepEqual(actual.centers,expected.centers);assert.deepEqual(actual.objectiveHistory,expected.objectiveHistory);session.dispose();
});
test('explicit power cache consumes old memberships only after delta is measured',()=>{
 const m=1.3,x=Float64Array.of(0,1,2),centers=Float64Array.of(0,2),old=Float64Array.of(.7,.3,.5,.5,.2,.8),saved=old.slice(),normal=new Float64Array(6),cached=new Float64Array(6),d=new Float64Array(2),a={delta2:0,objective:0},b={delta2:0,objective:0};
 fcmMembershipBlock(x,centers,saved,normal,d,0,3,1,2,m,a);
 fcmMembershipBlock(x,centers,old,cached,d,0,3,1,2,m,b,null,true);
 assert.deepEqual(cached,normal);assert.deepEqual(a,b);
 for(let i=0;i<old.length;i++)assert.equal(old[i],cached[i]**m);
});
