import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fcmMembershipBlock } from '../../consumer/node_modules/ubukit-js/src/iteration-kernels.js';
const fixture=JSON.parse(fs.readFileSync(new URL('../fixtures/fcm-near-one-round3.json',import.meta.url),'utf8'));
import { fcm, membershipsFromSquaredDistances } from '../../consumer/node_modules/ubukit-js/src/clustering.js';
test('near-one membership retains the accurate log1p distance difference',()=>{
 const q=Float64Array.of(1e300,1.0000000000000002e300),m=1+Number.EPSILON;
 const w=Math.exp(-Math.log1p((q[1]-q[0])/q[0])/(m-1));
 const u=membershipsFromSquaredDistances(q,2,m);
 assert.ok(Math.abs(u[0]-1/(1+w))<2e-15);assert.ok(Math.abs(u[1]-w/(1+w))<2e-15);
});
test('FCM aggregates weak objective terms before underflow',()=>{
 const n=200,data=Float64Array.from({length:n},(_,i)=>i%2*2),initMembership=new Float64Array(n*2).fill(.5);
 const result=fcm({data,nSamples:n,nFeatures:1},{nClusters:2,initMembership,m:1080,maxIterations:1,returnHistory:true});
 const expected=400/64*Number.MIN_VALUE;
 assert.ok(expected>0);assert.ok(Math.abs(result.objective-expected)<=Number.MIN_VALUE);
 assert.ok(Math.abs(result.objectiveHistory[0]-expected)<=Number.MIN_VALUE);
});

test('fused FCM and helper agree with independent 100-digit decimal near-one oracle',()=>{
 const helper=fixture.cases[0],fused=fixture.cases[1];
 const got=membershipsFromSquaredDistances(Float64Array.from(helper.squaredDistances),2,helper.m);
 for(let j=0;j<2;j++)assert.ok(Math.abs(got[j]-helper.membership[j])<2e-15);
 const next=new Float64Array(2),dist=new Float64Array(2);
 fcmMembershipBlock(Float64Array.of(0),Float64Array.of(1e150,1.0000000000000002e150),Float64Array.of(.5,.5),next,dist,0,1,1,2,fused.m,{delta2:0,objective:0});
 assert.deepEqual([...dist],fused.squaredDistances);
 for(let j=0;j<2;j++)assert.ok(Math.abs(next[j]-fused.membership[j])<2e-15);
});
test('weak powers with large distances retain an accurate representable objective',()=>{
 const c=fixture.weakObjectiveLargeScale;
 const r=fcm({data:Float64Array.from({length:c.n},(_,i)=>i%2*c.point),nSamples:c.n,nFeatures:1},{nClusters:2,initMembership:new Float64Array(c.n*2).fill(.5),m:c.m,maxIterations:1,returnHistory:true});
 assert.ok(Math.abs(r.objective-c.expected)/c.expected<5e-13);
 assert.ok(Math.abs(r.objectiveHistory[0]-c.expected)/c.expected<5e-13);
});
test('finite extreme m and all -Infinity log terms produce a finite zero objective',()=>{
 const r=fcm({data:Float64Array.of(0,2),nSamples:2,nFeatures:1},{nClusters:8,initMembership:new Float64Array(16).fill(.125),m:1e308,maxIterations:1,returnHistory:true});
 assert.equal(r.objective,0);assert.equal(r.objectiveHistory[0],0);assert.deepEqual(r.membership,new Float64Array(16).fill(.125));
});
