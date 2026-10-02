import test from 'node:test';
import assert from 'node:assert/strict';
import {
  TPEOptimizer, SearchSpaceExhaustedError, floatRange, intRange,
  categorical, optimize, optimizeAsync
} from '../consumer/node_modules/ubukit-js/src/optimization.js';
const mixed = () => ({x:floatRange(-2,2),n:intRange(1,9),rate:floatRange(1e-5,1,{log:true}),kind:categorical(['a','b','c'])});
const loss = p => p.x*p.x + (p.n-5)**2 + Math.log10(p.rate)**2 + (p.kind==='b'?0:3);

test('startup seeded reproducibility and known Mulberry32 first draw',()=>{
  const o = new TPEOptimizer({x:floatRange(0,1)}, {seed:0});
  assert.equal(o.ask().params.x, .26642920880112797);
  const a=optimize(p=>p.x*p.x,{x:floatRange(-1,1)},{nTrials:10,seed:77});
  assert.deepEqual(a,optimize(p=>p.x*p.x,{x:floatRange(-1,1)},{nTrials:10,seed:77}));
});
test('mixed TPE deterministic, bounded and complete in both modes',()=>{
  for (const multivariate of [true,false]) {
    const options={seed:123,nTrials:40,multivariate};
    const result=optimize(loss,mixed(),options);
    assert.deepEqual(result,optimize(loss,mixed(),options));
    assert.equal(result.history.length,40);
    for(const t of result.history){ assert.equal(t.state,'complete');assert.ok(t.params.x>=-2&&t.params.x<=2);assert.ok(Number.isInteger(t.params.n));assert.ok(t.params.rate>=1e-5&&t.params.rate<=1);assert.ok(Number.isFinite(t.value)); }
  }
});
test('root exports preserve the original module and add optimization',async()=>{
  const root = await import('../consumer/node_modules/ubukit-js/src/index.js');
  assert.equal(root.TPEOptimizer,TPEOptimizer);
  for(const name of ['run','steps','runAsync','createSession','kmeans','fcm','rmcm','somOlp','neighborhood','createMetricScheduler'])assert.equal(typeof root[name],'function');
});
test('all scalar categorical types are distinct; choice inputs are copied',()=>{
  const choices=[null,false,true,0,1,'0','1'];const space={c:categorical(choices)};const o=new TPEOptimizer(space);
  choices[0]='mutated';space.c.choices[1]='mutated';
  const result=[];for(let i=0;i<7;i++){const t=o.ask();result.push(t.params.c);o.tell(t.id,0);}
  assert.deepEqual(new Set(result),new Set([null,false,true,0,1,'0','1']));
  assert.throws(()=>o.ask(),SearchSpaceExhaustedError);
});
test('minimize and maximize choose a completed observation with stable ties',()=>{
  for(const direction of ['minimize','maximize']){
    const o=new TPEOptimizer({x:intRange(0,3)},{direction});
    o.addTrial({x:0},2);o.addTrial({x:1},-1);o.addTrial({x:2},2);o.addTrial({x:3},null,{state:'fail'});
    assert.equal(o.result().bestTrial.id,direction==='minimize'?1:0);
  }
});
test('ask reserves owned params; tell and result snapshots cannot mutate optimizer',()=>{
  const o=new TPEOptimizer({x:intRange(0,2)});const asked=o.ask(),original=asked.params.x;asked.params.x=999;
  const done=o.tell(asked.id,3);assert.equal(done.params.x,original);done.params.x=999;
  const result=o.result();result.history[0].params.x=999;result.bestParams.x=999;
  assert.equal(o.result().bestParams.x,original);
});
test('invalid tell is atomic and can be corrected; completed id cannot repeat',()=>{
  const o=new TPEOptimizer({x:intRange(0,2)});const t=o.ask();
  assert.throws(()=>o.tell(t.id,NaN));assert.equal(o.history[0].state,'running');
  assert.throws(()=>o.tell(t.id,1,{state:'unknown'}));o.tell(t.id,1);assert.throws(()=>o.tell(t.id,2));assert.throws(()=>o.tell(999,2));
});
test('failed and cancelled trials do not count toward TPE startup or best',()=>{
  const o=new TPEOptimizer({x:floatRange(0,1)},{nStartupTrials:2});
  for(let i=0;i<20;i++){const t=o.ask();o.tell(t.id,null,{state:i%2?'fail':'cancelled',error:'expected'});}
  assert.equal(o.result().bestValue,null);assert.equal(o.result().bestParams,null);
  const t=o.ask();o.tell(t.id,4);assert.equal(o.result().bestValue,4);
});
test('finite exhaustion includes pending and failed reservations; allowDuplicates opts in',()=>{
  const o=new TPEOptimizer({a:intRange(0,2),b:categorical(['a','b'])},{maxDuplicateAttempts:1});
  const values=new Set();for(let i=0;i<6;i++){const t=o.ask();values.add(JSON.stringify(t.params));if(i%2)o.tell(t.id,null,{state:'fail'});}
  assert.equal(values.size,6);assert.throws(()=>o.ask(),SearchSpaceExhaustedError);
  const result=optimize(()=>1,{a:intRange(1,1)},{nTrials:10});assert.equal(result.history.length,1);
  assert.equal(optimize(()=>1,{a:intRange(1,1)},{nTrials:10,allowDuplicates:true}).history.length,10);
});
test('external observations validate atomically and permit real repeated evaluations',()=>{
  const o=new TPEOptimizer({x:intRange(1,3)});
  for(const params of [{},{x:1,y:2},{x:1.5},{x:4}])assert.throws(()=>o.addTrial(params,1));
  assert.throws(()=>o.addTrial({x:1},Infinity));assert.equal(o.history.length,0);
  o.addTrial({x:1},2);o.addTrial({x:1},1);assert.equal(o.history.length,2);assert.equal(o.result().bestValue,1);
});
test('zero budget has an empty well-defined result',()=>{
  const r=optimize(()=>{throw Error('not called');},{x:floatRange(0,1)},{nTrials:0});
  assert.deepEqual(r,{bestParams:null,bestValue:null,bestTrial:null,bestTrialId:null,history:[],nAttempted:0,nCompleted:0,stopReason:'budget_exhausted'});
});
test('objective exceptions and nonfinite values require opt-in continuation',()=>{
  assert.throws(()=>optimize(()=>{throw Error('expected');},{x:floatRange(0,1)},{nTrials:3}),/expected/);
  const r=optimize((p,t)=>t.id===0?NaN:t.id===1?Infinity:2,{x:floatRange(0,1)},{nTrials:3,continueOnError:true});
  assert.deepEqual(r.history.map(t=>t.state),['fail','fail','complete']);assert.equal(r.bestValue,2);
});
test('sync optimize rejects promises and avoids unhandled rejections',()=>{
  assert.throws(()=>optimize(async()=>{throw Error('rejected');},{x:floatRange(0,1)},{nTrials:1}),/optimizeAsync/);
});
test('async and sync objectives produce the same seeded successful trials',async()=>{
  const options={seed:8,nTrials:24};const sync=optimize(loss,mixed(),options),asyncResult=await optimizeAsync(async p=>loss(p),mixed(),options);
  assert.deepEqual(asyncResult,sync);
});
test('async failures are recorded with requested continuation and callbacks',async()=>{
  const events=[];const result=await optimizeAsync(async(p,t)=>{if(t.id===1)throw Error('expected');return t.id;},{x:floatRange(0,1)},{nTrials:4,continueOnError:true,onTrial:async t=>events.push(t)});
  assert.deepEqual(events.map(t=>t.state),['complete','fail','complete','complete']);assert.equal(result.bestValue,0);
});
test('abort signals prevent startup and cancel in-flight objective completion',async()=>{
  const before=new AbortController();before.abort();let count=0;
  assert.throws(()=>optimize(()=>count++,{x:floatRange(0,1)},{signal:before.signal}),{name:'AbortError'});assert.equal(count,0);
  const during=new AbortController();
  await assert.rejects(optimizeAsync(async()=>{during.abort();return 1;},{x:floatRange(0,1)},{signal:during.signal}),{name:'AbortError'});
});
test('names that look like object prototype keys remain ordinary params',()=>{
  const space=JSON.parse('{"__proto__":{"type":"int","low":1,"high":2},"constructor":{"type":"float","low":0,"high":1}}');
  const r=optimize(p=>p.__proto__+p.constructor,space,{nTrials:5});assert.ok(Object.hasOwn(r.bestParams,'__proto__'));assert.ok(Number.isFinite(r.bestValue));
});
test('extreme finite float bounds and logarithmic spans do not overflow proposals',()=>{
  for(const spec of [floatRange(-1e308,1e308),floatRange(1e-300,1e300,{log:true}),floatRange(1e308,1.0000000000000002e308,{log:true})]){
    const r=optimize(()=>1,{x:spec},{nTrials:30,allowDuplicates:true});for(const t of r.history){assert.ok(Number.isFinite(t.params.x));assert.ok(t.params.x>=spec.low&&t.params.x<=spec.high);}
  }
});
test('log-int and near-safe-integer-max bounds remain integers in range',()=>{
  for(const spec of [intRange(1,1000,{log:true}),intRange(Number.MAX_SAFE_INTEGER-7,Number.MAX_SAFE_INTEGER),intRange(Number.MAX_SAFE_INTEGER-7,Number.MAX_SAFE_INTEGER,{log:true})]){
    const r=optimize(p=>p.x,{x:spec},{nTrials:32,allowDuplicates:true});for(const t of r.history){assert.ok(Number.isSafeInteger(t.params.x));assert.ok(t.params.x>=spec.low&&t.params.x<=spec.high);}
  }
});
test('malformed spaces, categories and bounds fail early',()=>{
  for(const space of [null,[],{}, {x:{type:'wat'}},{x:{type:'float',low:0,high:Infinity}},{x:{type:'float',low:2,high:1}},{x:{type:'float',low:0,high:1,log:true}},{x:{type:'int',low:0,high:1.2}},{x:{type:'int',low:-Number.MAX_SAFE_INTEGER,high:Number.MAX_SAFE_INTEGER}},{x:{type:'categorical',choices:[]}},{x:{type:'categorical',choices:[0,-0]}},{x:{type:'categorical',choices:[{}]}},{x:{type:'categorical',choices:[NaN]}}])assert.throws(()=>new TPEOptimizer(space));
});
test('malformed options fail early',()=>{
  for(const options of [{seed:NaN},{seed:1.5},{direction:'minimum'},{sampler:'other'},{nStartupTrials:1},{nCandidates:0},{gamma:0},{gamma:1},{minBandwidth:0},{minBandwidth:2},{multivariate:1},{allowDuplicates:'yes'},{weights:'bad'},{maxDuplicateAttempts:0}])assert.throws(()=>new TPEOptimizer({x:floatRange(0,1)},options));
});
test('repeated and tied observations give finite stable mixture proposals',()=>{
  for(const multivariate of [true,false])for(const weights of ['uniform','ei']){
    const o=new TPEOptimizer({x:floatRange(-1,1),n:intRange(1,9),c:categorical(['a','b'])},{multivariate,weights});
    for(let i=0;i<20;i++)o.addTrial({x:.2,n:3,c:'a'},1e308);
    const t=o.ask();assert.ok(Number.isFinite(t.params.x));o.tell(t.id,1e308);assert.ok(Number.isFinite(o.result().bestValue));
  }
});

test('sparse, inherited-index and explicit-undefined categories fail before construction',()=>{
  const sparse=new Array(2);sparse[1]='valid';
  const inherited=new Array(2);inherited[1]='valid';
  Object.setPrototypeOf(inherited,Object.assign(Object.create(Array.prototype),{0:'inherited'}));
  for(const choices of [sparse,inherited,[undefined,'valid']]){
    assert.throws(()=>categorical(choices));
    assert.throws(()=>new TPEOptimizer({x:{type:'categorical',choices}},{seed:0,avoidDuplicates:false}));
  }
  const valid=categorical([null,'valid']);const o=new TPEOptimizer({x:valid},{seed:0,avoidDuplicates:false});
  delete valid.choices[0]; // Input ownership: the constructed optimizer keeps its validated snapshot.
  const trial=o.ask();assert.equal(trial.params.x,null);assert.equal(o.history.length,1);
});
test('categorical getters and imported params are captured once before validation',()=>{
  let reads=0;const choices=['a','b'];Object.defineProperty(choices,0,{get(){return reads++===0?'a':undefined;}});
  const spec=categorical(choices);assert.equal(reads,1);assert.deepEqual(spec.choices,['a','b']);
  const o=new TPEOptimizer({x:intRange(1,2)});let paramsReads=0;
  const params={get x(){return paramsReads++===0?1:undefined;}};
  o.addTrial(params,1);assert.equal(paramsReads,1);assert.equal(o.history[0].params.x,1);
});
test('imported observations preserve the externally managed lifecycle status',()=>{
  const o=new TPEOptimizer({x:intRange(1,4)});o.addTrial({x:1},1);
  assert.equal(o.result().stopReason,'not_started');assert.equal(o.result().history[0].state,'complete');
  const t=o.ask();o.addTrial({x:3},3);assert.equal(o.result().stopReason,'running');o.tell(t.id,2);
});
test('integer-valued categories require safe integers while float domains keep extreme numbers',()=>{
  for(const value of [2**53,-(2**53),1e100,-1e100,Number.MAX_VALUE]){
    assert.throws(()=>categorical([value]));
    assert.throws(()=>new TPEOptimizer({x:{type:'categorical',choices:[value]}}));
  }
  const o=new TPEOptimizer({x:categorical([Number.MAX_SAFE_INTEGER,-Number.MAX_SAFE_INTEGER,.1])});
  assert.throws(()=>o.addTrial({x:2**53},1));assert.equal(o.history.length,0);
  o.addTrial({x:Number.MAX_SAFE_INTEGER},1);o.addTrial({x:.1},0);assert.equal(o.result().bestParams.x,.1);
  const numeric=new TPEOptimizer({x:floatRange(1e100,1e100)});const t=numeric.ask();numeric.tell(t.id,1);assert.equal(numeric.result().bestParams.x,1e100);
});
