import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { run, runAsync, algorithms, kmeans, fcm, exrcm, rcm, membershipsFromSquaredDistances } from '../../consumer/node_modules/ubukit-js/src/index.js';
const fixture = name => JSON.parse(fs.readFileSync(new URL(`../fixtures/${name}`, import.meta.url)));
const input = rows => ({data: Float64Array.from(rows.flat()), nSamples: rows.length, nFeatures: rows[0].length});
function close(actual, expected, atol = 2e-12) { expected = expected.flat?.(Infinity) ?? expected; assert.equal(actual.length, expected.length); for (let i = 0; i < actual.length; ++i) assert.ok(Math.abs(actual[i] - expected[i]) <= atol * Math.max(1, Math.abs(expected[i])), `${i}: ${actual[i]} != ${expected[i]}`); }
const fp = fixture('fcm-python.json');
for (const c of fp.cases) test(`FCM Python fixed-init parity m=${c.m}`, () => {
  const result = fcm(input(fp.X), {initMembership: Float64Array.from(fp.init.flat()), m:c.m, maxIterations:fp.max_iter, tolerance:fp.tol, returnHistory:true});
  close(result.centers,c.centers); close(result.membership,c.membership); close(result.objectiveHistory,c.objective_history);
  close([result.objective,result.fpc,result.delta],[c.objective,c.fpc,c.delta]); assert.deepEqual([...result.labels],c.labels); assert.equal(result.iterations,c.n_iter);
});
const rp = fixture('exrcm-python.json');
for (const c of rp.cases) test(`ExRCM Python fixed-init parity ${c.name}`, () => {
  const result = exrcm(input(c.X), {initCenters:Float64Array.from(c.init.flat()), alpha:c.alpha,beta:c.beta,p:c.p,maxIterations:c.max_iter});
  close(result.centers,c.centers); const expected=c.X.flatMap((_,i)=>c.memberships.map(row=>row[i])); close(result.membership,expected,0); assert.equal(result.iterations,c.n_iter); assert.equal(result.converged,true);
});
test('FCM exact zero-distance ties',()=>close(membershipsFromSquaredDistances(Float64Array.of(0,0,4,1,4,9),3),[.5,.5,0,36/49,9/49,4/49]));
test('FCM retains x squared contribution',()=>close(membershipsFromSquaredDistances(Float64Array.of(1,4),2),[.8,.2]));
test('FCM tiny and huge m remain normalized',()=>{for(const m of [1+1e-12,10000]){const r=fcm(input([[0],[1],[3],[9]]),{nClusters:3,m,maxIterations:3,tolerance:0});for(let i=0;i<4;i++)close([r.membership.subarray(i*3,(i+1)*3).reduce((a,b)=>a+b,0)],[1]);for(const c of r.centers)assert.ok(Number.isFinite(c));}});
test('Kmeans ties, empty centers, final pass',()=>{const r=kmeans(input([[0],[2],[10],[12]]),{initCenters:Float64Array.of(0,12,100),maxIterations:1});close(r.centers,[1,11,100]);assert.deepEqual([...r.labels],[0,0,1,1]);assert.equal(r.inertia,4);});
test('Kmeans input is not mutated; deterministic seed',()=>{const x=input([[0,1],[2,3],[8,9],[10,11]]),copy=x.data.slice();const a=kmeans(x,{nClusters:2,seed:123});const b=kmeans(x,{nClusters:2,seed:123});assert.deepEqual(a,b);assert.deepEqual(x.data,copy);});
test('RCM p1 alias and invalid p',()=>{const x=input([[0],[2],[9]]),opts={initCenters:Float64Array.of(0,9)};close(rcm(x,opts).centers,exrcm(x,{...opts,p:1}).centers);assert.throws(()=>rcm(x,{...opts,p:2}));});
test('RCM duplicate centers split membership',()=>{const r=rcm(input([[0],[0],[4]]),{initCenters:Float64Array.of(0,0),alpha:1,beta:0});close(r.membership,[.5,.5,.5,.5,.5,.5]);});
test('validation rejects malformed input and parameters',()=>{assert.throws(()=>run('bad',input([[1]]),{}));assert.throws(()=>kmeans({data:[1],nSamples:1,nFeatures:1},{nClusters:1}));assert.throws(()=>fcm(input([[1]]),{nClusters:1,m:1}));assert.throws(()=>exrcm(input([[1]]),{nClusters:1,alpha:.9}));assert.throws(()=>kmeans(input([[NaN]]),{nClusters:1}));assert.throws(()=>fcm(input([[1]]),{initMembership:Float64Array.of(0)}));assert.deepEqual([...algorithms].sort(),["entropy-fcm","exrcm","fcm","kmeans","neighborhood","rcm","rmcm","som","som-olp","som_batch"]);});
test('AbortSignal and callback cancellation',()=>{const signal=AbortSignal.abort();assert.throws(()=>kmeans(input([[1]]),{nClusters:1,signal}),{name:'AbortError'});let count=0;assert.throws(()=>fcm(input([[0],[1],[2]]),{nClusters:2,maxIterations:50,tolerance:0,shouldCancel:()=>count++>3}),{name:'AbortError'});});
test('runAsync same result and yields to event loop',async()=>{const x=input([[0],[1],[9],[10]]);let ticks=0;const timer=setInterval(()=>ticks++,0);const opts={nClusters:2,maxIterations:4,tolerance:0,seed:2,blockRows:1,timeBudgetMs:0};const actual=await runAsync('fcm',x,opts);clearInterval(timer);assert.deepEqual(actual,run('fcm',x,opts));assert.ok(ticks>0);});
test('FCM recovers an underflowed distance ratio before fractional power',()=>{const u=membershipsFromSquaredDistances(Float64Array.of(1e-200,1e200),2,1001);const r=Math.exp((Math.log(1e-200)-Math.log(1e200))/1000);close(u,[1/(1+r),r/(1+r)]);assert.ok(u[1]>.2);});
test('RCM tiny p beta0 still distinguishes distances',()=>{const r=exrcm(input([[0]]),{initCenters:Float64Array.of(1,2),alpha:1,beta:0,p:1e-17,maxIterations:1});close(r.membership,[1,0]);});
test('ExRCM fractional p retains underflowed beta ratio',()=>{const r=exrcm(input([[0]]),{initCenters:Float64Array.of(1e10,1e154),alpha:1,beta:1e-320,p:.001,maxIterations:1});close(r.membership,[.5,.5]);});
test('unsupported numeric range fails explicitly rather than inventing zero distance',()=>assert.throws(()=>kmeans(input([[0],[1e-200]]),{nClusters:1}),/underflow/));
test('memory guard and kmeans convergence contract are explicit',()=>{assert.throws(()=>fcm(input([[0],[1]]),{nClusters:2,maxMemoryBytes:1}),/maxMemoryBytes/);assert.throws(()=>kmeans(input([[0],[1]]),{nClusters:1,tolerance:.1}),/label convergence/);});
test('ExRCM fractional p recovers rounded subnormal ratio',()=>{const r=exrcm(input([[0]]),{initCenters:Float64Array.of(2000,3.0223505125121568e47),alpha:1,beta:1e-320,p:.002,maxIterations:1});close(r.membership,[.5,.5]);});
test('Kmeans Python explicit-init parity including final labels',()=>{const fp=fixture('kmeans-python.json');const r=kmeans(input(fp.X),{initCenters:Float64Array.from(fp.init.flat()),maxIterations:fp.maxIterations});close(r.centers,fp.centers);close([r.inertia],[fp.inertia]);assert.deepEqual([...r.labels],fp.labels);assert.deepEqual([...r.coreLabels],fp.coreLabels);assert.equal(r.iterations,fp.iterations);});
for(const c of fixture('exrcm-python.json').assignment_edge_cases)test(`ExRCM Python boundary assignment: ${c.name}`,async()=>{const {roughAdmissible}=await import('../../consumer/node_modules/ubukit-js/src/index.js');const distances=c.centers.map(center=>Math.sqrt(center.reduce((s,v,j)=>s+(v-c.X[0][j])**2,0)));const minimum=Math.min(...distances);const mask=distances.map(v=>roughAdmissible(v,minimum,c.alpha,c.beta,c.p));assert.deepEqual(mask,c.upper_memberships.map(row=>row[0]));});

