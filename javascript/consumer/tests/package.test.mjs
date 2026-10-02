import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { Worker as NodeWorker } from 'node:worker_threads';
import * as root from 'ubukit-js';
import * as worker from 'ubukit-js/worker';
import * as session from 'ubukit-js/session';
import * as realtime from 'ubukit-js/realtime-worker';
import * as metrics from 'ubukit-js/metrics';
import * as external from 'ubukit-js/external-metrics';
const moduleURL = import.meta.resolve('ubukit-js');
const packageURL = new URL('../',moduleURL);
const pkg=JSON.parse(fs.readFileSync(new URL('package.json',packageURL)));
const X={data:Float64Array.of(0,.1,.2,4,4.1,4.2),nSamples:6,nFeatures:1};
function opts(a){return {nClusters:2,initCenters:Float64Array.of(0,4),maxIterations:4,tolerance:0,blockRows:2,...['som','som_batch'].includes(a)?{gridShape:[2,1]}:{},...a==='rmcm'?{delta:.3}:{},...a==='som-olp'?{grid:{data:Float64Array.of(-1,1),nSamples:2,nFeatures:1},lambda:.4}:{},...a==='neighborhood'?{embedding:X,k:1}:{}};}
function consume(it){for(;;){const next=it.next();if(next.done)return next.value;}}

test('installed package is a real tarball copy with exact private metadata and additive external-metric exports',async()=>{
 assert.match(moduleURL,/consumer\/node_modules\/ubukit-js\/src\/index\.js$/);
 assert.equal(fs.lstatSync(packageURL).isSymbolicLink(),false);
 assert.equal(pkg.name,'ubukit-js');assert.equal(pkg.version,'0.1.0-dev.6');assert.equal(pkg.private,true);
 assert.deepEqual(pkg.exports,{'.':'./src/index.js','./worker':'./src/worker-client.js','./session':'./src/session.js','./realtime-worker':'./src/realtime-worker-client.js','./metrics':'./src/metric-scheduler.js','./optimization':'./src/optimization.js','./external-metrics':'./src/external-metrics.js'});
 assert.equal(pkg.dependencies,undefined);assert.equal(pkg.scripts,undefined);assert.equal(pkg.license,'MIT AND BSD-3-Clause');
 const expected=['adjustedRandScore','adjustedMutualInfoScore','adjustedScores','adjusted_rand_score','adjusted_mutual_info_score','adjusted_scores','ExternalMetricDomainError','TPEOptimizer','SearchSpaceExhaustedError','ProposalError','floatRange','intRange','categorical','optimize','optimizeAsync','ClusteringSession','PreparedRMCM','RMCMGraphCache','algorithms','createMetricScheduler','createRealtimeWorkerClient','createSession','createWorkerClient','exrcm','fcm','kmeans','membershipsFromSquaredDistances','neighborhood','neighborhoodSteps','normalizeInput','prepareRMCM','prepareRMCMSteps','rcm','rmcm','rmcmReference','rmcmSteps','roughAdmissible','run','runAsync','seededRandom','sessionAlgorithms','somOlp','somOlpSteps','steps','som','somBatch','som_batch','somSteps','somBatchSteps','somProject'];
 assert.deepEqual(Object.keys(root).sort(),expected.sort());
 for(const sub of[worker,session,realtime,metrics,external])for(const[name,value]of Object.entries(sub))assert.equal(root[name],value);
 for(const spec of['ubukit-js/worker-entry','ubukit-js/realtime-worker-entry','ubukit-js/src/index.js'])await assert.rejects(import(spec),{code:'ERR_PACKAGE_PATH_NOT_EXPORTED'});
});
test('every installed runtime hash matches the reviewed package manifest',()=>{
 const manifest=JSON.parse(fs.readFileSync(new URL('SOURCE_MANIFEST.json',packageURL)));
 assert.equal(Object.keys(manifest.runtime_files).length,27);
 for(const[path,{sha256}]of Object.entries(manifest.runtime_files))assert.equal(crypto.createHash('sha256').update(fs.readFileSync(new URL(path,packageURL))).digest('hex'),sha256,path);
});
for(const algorithm of root.algorithms)test(`public root ${algorithm}: run/steps/runAsync agree and inputs remain unchanged`,async()=>{
 const input={...X,data:X.data.slice()},before=input.data.slice(),o=opts(algorithm);
 const r=root.run(algorithm,input,o);assert.deepEqual(consume(root.steps(algorithm,input,o)),r);
 assert.deepEqual(await root.runAsync(algorithm,input,{...o,timeBudgetMs:0}),r);
 assert.deepEqual(input.data,before);
 const direct={kmeans:'kmeans',fcm:'fcm',rcm:'rcm',exrcm:'exrcm',rmcm:'rmcm','som-olp':'somOlp',som:'som',som_batch:'somBatch',neighborhood:'neighborhood'}[algorithm];
 assert.deepEqual(root[direct](input,o),r);
});
test('public helpers and prepared/cache entrypoints execute from installed package',()=>{
 assert.deepEqual(root.membershipsFromSquaredDistances(Float64Array.of(0,0,4),3,2),Float64Array.of(.5,.5,0));
 assert.equal(root.normalizeInput(X).nSamples,6);
 assert.equal(root.seededRandom(42)(),root.seededRandom(42)());
 const o=opts('rmcm'),prepared=root.prepareRMCM(X,o);
 assert.ok(prepared instanceof root.PreparedRMCM);assert.deepEqual(prepared.fit(o),root.rmcm(X,o));
 assert.deepEqual(consume(root.prepareRMCMSteps(X,o)).fit(o),root.rmcm(X,o));
 assert.deepEqual(consume(root.rmcmSteps(X,o)),root.rmcm(X,o));
 assert.deepEqual(root.rmcmReference(X,o),root.rmcm(X,{...o,backend:'reference'}));
 const cache=new root.RMCMGraphCache({skin:.2});
 const numerical=r=>{const {estimatedPrimaryBytes,...rest}=r;assert.ok(estimatedPrimaryBytes>0);return rest;};
 assert.deepEqual(numerical(cache.prepare(X,o).fit(o)),numerical(root.rmcm(X,o)));cache.clear();assert.deepEqual(numerical(cache.prepare(X,o).fit(o)),numerical(root.rmcm(X,o)));
});
test('installed FCM release gate: near-one Decimal100 oracle and subnormal objective sum',()=>{
 const actual=root.membershipsFromSquaredDistances(Float64Array.of(1e300,1.0000000000000002e300),2,1+Number.EPSILON);
 assert.ok(Math.abs(actual[0]-.6614343863724315)<=2e-15);
 const n=200,result=root.fcm({data:Float64Array.from({length:n},(_,i)=>i%2*2),nSamples:n,nFeatures:1},{nClusters:2,initMembership:new Float64Array(n*2).fill(.5),m:1080,maxIterations:1,returnHistory:true});
 assert.equal(result.objective,400/64*Number.MIN_VALUE);assert.equal(result.objectiveHistory[0],400/64*Number.MIN_VALUE);
});
test('default Worker URLs resolve inside installed package and real Node workers complete',async()=>{
 const original=globalThis.Worker,seen=[];
 globalThis.Worker=class extends NodeWorker { constructor(url,options){seen.push(url.href);assert.ok(fs.existsSync(url));super(url,options);} };
 const a=worker.createWorkerClient(),b=realtime.createRealtimeWorkerClient();
 try{
  assert.deepEqual(await a.run('fcm',X,opts('fcm')),root.fcm(X,opts('fcm')));
  const r=await b.run('fcm',X,opts('fcm'));assert.deepEqual(r.result,root.fcm(X,opts('fcm')));
  assert.deepEqual(seen,[new URL('worker.js',moduleURL).href,new URL('realtime-worker.js',moduleURL).href]);
 }finally{a.dispose();b.dispose();if(original===undefined)delete globalThis.Worker;else globalThis.Worker=original;}
});
test('documented Node workerFactory resolves via public client entrypoint and permits cancel/reuse',async()=>{
 const url=new URL('./worker.js',import.meta.resolve('ubukit-js/worker'));
 const c=worker.createWorkerClient({workerFactory:()=>new NodeWorker(url,{type:'module'})});
 try{
  const pending=c.run('fcm',X,{...opts('fcm'),maxIterations:10000});const rejected=assert.rejects(pending,{name:'AbortError'});c.cancel();await rejected;
  assert.deepEqual(await c.run('fcm',X,opts('fcm')),root.fcm(X,opts('fcm')));
 }finally{c.dispose();}
});
test('public session subpath supports cancelled update/reset and owned snapshots',()=>{
 const abort=new AbortController(),s=session.createSession('fcm',X,{...opts('fcm'),signal:abort.signal});
 abort.abort();assert.throws(()=>s.step(),{name:'AbortError'});s.updateParameters({signal:undefined});s.reset();
 while(!s.status.done)s.step(1);
 const first=s.snapshot().result;first.centers.fill(999);assert.notDeepEqual(s.snapshot().result.centers,first.centers);
 s.updateData({...X,data:Float64Array.from(X.data,v=>v+.1)});while(!s.status.done)s.step(1);assert.equal(s.snapshot().result.nSamples,6);s.dispose();
});
test('public metric scheduler subpath supports cache ownership, invalidation and reset',async()=>{
 const scheduler=metrics.createMetricScheduler({debounceMs:0,timeBudgetMs:0,maxChunks:1});
 try{const o={embedding:X,k:1};const first=await scheduler.run(X,o);assert.equal(first.cacheHit,false);first.result.qualities[0].trustworthiness=-1;
 const cached=await scheduler.run(X,o);assert.equal(cached.cacheHit,true);assert.equal(cached.result.qualities[0].trustworthiness,1);scheduler.clearCache();assert.equal((await scheduler.run(X,o)).cacheHit,false);
 }finally{scheduler.dispose();}
});
