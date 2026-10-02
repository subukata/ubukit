import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {TPEOptimizer} from '../consumer/node_modules/ubukit-js/src/optimization.js';
const source=readFileSync(new URL('../consumer/node_modules/ubukit-js/src/optimization.js',import.meta.url),'utf8');
const internals=await import(`data:text/javascript;base64,${Buffer.from(source+'\nexport {normalCdf,normalPpf,normalLogNarrow,normalInterval,makeRandom,buildModel,validateSpace,normalizeOptions,KDE,binWidth};').toString('base64')}`);
const fixture=JSON.parse(readFileSync(new URL('../fixtures/shared_core_parity.json',import.meta.url)));
const close=(a,b,msg)=>assert.ok(Math.abs(a-b)<=1e-11*Math.max(1,Math.abs(b)),`${msg}: ${a} vs ${b}`);
function sameParams(actual,expected){for(const [k,v]of Object.entries(expected)){if(fixture.space[k].type==='float')close(actual[k],v,k);else assert.equal(actual[k],v,k);}}
test('shared Mulberry32 uint streams are exact for four seeds',()=>{for(const f of fixture.rng){const rng=internals.makeRandom(f.seed);assert.deepEqual(f.uint32.map(()=>rng()*2**32),f.uint32);}});
test('shared normal CDF and inverse approximations match Python',()=>{for(const [x,y]of fixture.normal_cdf)close(internals.normalCdf(x),y,'cdf');for(const [x,y]of fixture.normal_ppf)close(internals.normalPpf(x),y,'ppf');});
for(const f of fixture.fixed_history)test(`shared fixed-history log densities and pending asks joint=${f.multivariate} weights=${f.weights}`,()=>{
 const options={seed:f.seed,multivariate:f.multivariate,weights:f.weights},o=new TPEOptimizer(fixture.space,options);
 for(const t of f.observations)o.addTrial(t.params,t.value);
 const [good,bad]=internals.buildModel(internals.validateSpace(fixture.space),o.history,internals.normalizeOptions(options));
 for(const p of f.densities){close(good.logpdf(p.params),p.log_good,'good');close(bad.logpdf(p.params),p.log_bad,'bad');}
 for(const p of f.asked){const t=o.ask();assert.equal(t.id,p.id);sameParams(t.params,p.params);}
});
for(const f of fixture.trajectories)test(`shared end-to-end 40-trial trajectory joint=${f.multivariate}`,()=>{
 const o=new TPEOptimizer(fixture.space,{seed:f.seed,multivariate:f.multivariate});
 for(const expected of f.history){const t=o.ask();assert.equal(t.id,expected.id);sameParams(t.params,expected.params);o.tell(t.id,expected.value);}
});
test('mixture densities normalize on float and integer supports',()=>{
 const options=internals.normalizeOptions({});
 const floatDomain=internals.validateSpace({x:{type:'float',low:0,high:1}});
 const kde=new internals.KDE(floatDomain,[{x:.1},{x:.5},{x:.9}],[1,2,3],options);
 let sum=0;const count=10000;for(let i=0;i<count;i++)sum+=Math.exp(kde.logpdf({x:(i+.5)/count}))/count;
 assert.ok(Math.abs(sum-1)<1e-5,`float integral ${sum}`);
 for(const log of [false,true]){
 const domains=internals.validateSpace({x:{type:'int',low:1,high:99,log}}),k=new internals.KDE(domains,[{x:1},{x:20},{x:98}],[1,2,3],options);let mass=0;
 for(let x=1;x<=99;x++)mass+=Math.exp(k.logpdf({x}));assert.ok(Math.abs(mass-1)<1e-6,`integer mass ${mass}`);
 }
});
test('log-domain products stay finite in 300 dimensions and huge int bins',()=>{
 const space=Object.fromEntries(Array.from({length:300},(_,i)=>['x'+i,{type:'float',low:0,high:1}]));
 const domains=internals.validateSpace(space),rows=[.1,.8].map(x=>Object.fromEntries(Object.keys(space).map(k=>[k,x])));
 const kde=new internals.KDE(domains,rows,[1,1],internals.normalizeOptions({}));
 assert.ok(Number.isFinite(kde.logpdf(Object.fromEntries(Object.keys(space).map(k=>[k,.2])))));
 const d=internals.validateSpace({x:{type:'int',low:1,high:Number.MAX_SAFE_INTEGER,log:true}}),k=new internals.KDE(d,[{x:1},{x:Number.MAX_SAFE_INTEGER}],[1,1],internals.normalizeOptions({}));
 assert.ok(internals.binWidth(d[0],Number.MAX_SAFE_INTEGER)>0);assert.ok(Number.isFinite(k.logpdf({x:Number.MAX_SAFE_INTEGER})));
});
test('narrow Gaussian intervals match independent 100-digit reference at center and tails',()=>{
 const f=JSON.parse(readFileSync(new URL('../fixtures/normal_interval_reference.json',import.meta.url)));
 for(const c of f.cases){
  const actual=internals.normalLogNarrow(c.center,c.width);
  assert.notEqual(actual,null);
  assert.ok(Math.abs(Math.expm1(actual-c.logMass))<4e-12,`center ${c.center} width ${c.width}: relative mass error ${Math.expm1(actual-c.logMass)}`);
 }
});
test('near-former-cutoff central integer PMF has no CDF cancellation jump',()=>{
 for(const count of [199999999,200000001]){
  const value=(count+1)/2,domains=internals.validateSpace({x:{type:'int',low:1,high:count}});
  const kde=new internals.KDE(domains,[{x:value}],[1],internals.normalizeOptions({}));
  const sigma=kde.sigmas[0][0],norm=kde.norms[0][0],actual=kde.axisLogs(0,value)[0];
  // Centered infinitesimal normal-bin mass; omitted quadratic term < 1e-16 here.
  const expected=Math.log(1/count)-Math.log(sigma*2.5066282746310002*norm);
  assert.ok(Math.abs(Math.expm1(actual-expected))<1e-12,`${count}: ${actual} vs ${expected}`);
 }
});
