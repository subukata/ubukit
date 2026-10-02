import test from 'node:test';
import assert from 'node:assert/strict';
import {somOlp} from '../consumer/node_modules/ubukit-js/src/som-olp.js';
import {somOlp as baseline} from '../validation/baseline_package/src/som-olp.js';
import {somStableCost, somStableMean} from '../consumer/node_modules/ubukit-js/src/som-numerics.js';
const input = (a, d=1) => ({data:Float64Array.from(a), nSamples:a.length/d, nFeatures:d});
const near=(a,b,rel=2e-14,abs=0)=>assert.ok(Math.abs(a-b)<=Math.max(abs,rel*Math.max(Math.abs(a),Math.abs(b))),`${a} != ${b}`);
const backendCases = ['javascript','wasm'];

for(const kernelBackend of backendCases) {
 test(`${kernelBackend}: tiny lambda retains true ties and gamma movement penalty`,()=>{
  const symmetric=somOlp(input([0,0]),{grid:input([-1,1]),lambda:1e-310,maxIterations:5,kernelBackend});
  assert.deepEqual([...symmetric.P],[.5,.5,.5,.5]);assert.deepEqual([...symmetric.history],[2,2]);
  const options={grid:input([0,1]),initialPrototypes:Float64Array.of(2,5),initialMemberships:Float64Array.of(1,0,1,0,0,1),lambda:1e-310,maxIterations:10,kernelBackend};
  const trapped=somOlp(input([0,4,5]),{...options,gamma:4});
  assert.deepEqual([...trapped.P],[1,0,1,0,0,1]);assert.deepEqual([...trapped.history],[8,8]);
  const free=somOlp(input([0,4,5]),{...options,gamma:0});assert.deepEqual([...free.W],[0,4.5]);assert.equal(free.history.at(-1),.5);
 });
 test(`${kernelBackend}: finite weighted means at a huge common offset`,()=>{
  const result=somOlp(input(Array(6).fill(1e308)),{grid:input([-1,1]),lambda:1e-310,maxIterations:3,kernelBackend});
  assert.deepEqual([...result.W],[1e308,1e308]);assert.deepEqual([...result.V],Array(6).fill(0));assert.deepEqual([...result.history],[6,6]);
  if(kernelBackend==='wasm'){assert.equal(result.kernel.training,'javascript');assert.ok(result.kernel.javascriptFallbackRows>0);}
 });
 for(const [label,r,gamma,lambda] of [['overflow',1e200,1e-300,1e100],['underflow',1e-200,1e300,1e-100]]) {
  test(`${kernelBackend}: gamma applied before raw-square ${label}`,()=>{
   const result=somOlp(input([0]),{grid:input([0,r]),initialPrototypes:Float64Array.of(0,0),initialMemberships:Float64Array.of(.75,.25),gamma,lambda,maxIterations:1,kernelBackend});
   near(result.P[0],1/(1+Math.exp(-.5)));near(result.V[0],r*.25);
   const expected=lambda*(.0625-Math.log1p(Math.exp(-.5)));near(result.history[0],expected);
  });
 }
 test(`${kernelBackend}: gamma rescues overflow in grid subtraction`,()=>{
  const result=somOlp(input([0]),{grid:input([-1e308,1e308]),initialPrototypes:Float64Array.of(0,0),initialMemberships:Float64Array.of(1,0),gamma:1e-310,lambda:1e306,maxIterations:1,kernelBackend});
  near(result.P[0],1/(1+Math.exp(-4)));near(result.history[0],-1e306*Math.log1p(Math.exp(-4)),2e-12);
 });
 test(`${kernelBackend}: collectively representable initial costs from underflowing squares`,()=>{
  const delta=2**-538;
  const result=somOlp(input([0,0,0,0],4),{grid:input([0,1]),initialPrototypes:Float64Array.from([...Array(4).fill(delta),...Array(4).fill(2*delta)]),lambda:Number.MIN_VALUE,maxIterations:0,kernelBackend});
  near(result.P[0],1/(1+Math.exp(-3)));near(result.P[0]+result.P[1],1);
 });
 test(`${kernelBackend}: PCA covariance accumulation overflow is temporary`,()=>{
  const result=somOlp(input(Array.from({length:10},(_,i)=>(i%2?1:-1)*6e153)),{grid:input([-1,1]),pcaScale:.1,lambda:1e308,maxIterations:0,kernelBackend});
  near(result.W[0],-6e152);near(result.W[1],6e152);near(result.P[0],1/(1+Math.exp(-.144)));
 });
 test(`${kernelBackend}: PCA recovers collectively representable covariance`,()=>{
  const delta=2**-538, x=[...Array(4).fill(-delta),...Array(4).fill(delta)];
  const result=somOlp(input(x,4),{grid:input([-1,1]),pcaScale:2,lambda:Number.MIN_VALUE,maxIterations:0,kernelBackend});
  for(let j=0;j<8;j++)near(result.W[j],j<4?-2*delta:2*delta);
  near(result.P[0],1/(1+Math.exp(-8)));near(result.P[3],result.P[0]);
 });
 test(`${kernelBackend}: huge grid mean and grid normalization stay finite`,()=>{
  const flat=somOlp(input([-1,1]),{grid:input([1e308,1e308]),lambda:1e-310,maxIterations:2,kernelBackend});
  assert.deepEqual([...flat.W],[0,0]);assert.deepEqual([...flat.V],[1e308,1e308]);assert.deepEqual([...flat.history],[2,2]);
  const span=somOlp(input([-1,1]),{grid:input([1.7e308,1.7e308,-1.7e308]),gamma:0,lambda:1,maxIterations:0,kernelBackend});
  near(span.W[0],1);near(span.W[1],1);near(span.W[2],-2);
 });
 test(`${kernelBackend}: tiny cluster mass retains representable weighted mean`,()=>{
  const result=somOlp(input([.001,.002]),{grid:input([0,1]),initialPrototypes:Float64Array.of(0,0),initialMemberships:Float64Array.of(1,Number.MIN_VALUE,1,Number.MIN_VALUE),gamma:0,lambda:.1,maxIterations:1,kernelBackend});
  near(result.W[0],.0015);near(result.W[1],.0015);
 });
 test(`${kernelBackend}: gamma zero ignores grid-distance arithmetic only`,()=>{
  const result=somOlp(input([0]),{grid:input([-1e308,1e308]),initialPrototypes:Float64Array.of(0,0),initialMemberships:Float64Array.of(1,0),gamma:0,lambda:1,maxIterations:1,kernelBackend});
  assert.deepEqual([...result.P],[.5,.5]);assert.equal(result.V[0],-1e308);near(result.history[0],-Math.log(2));
 });
 test(`${kernelBackend}: unrepresentable costs and objectives raise explicitly`,()=>{
  for(const x of [1e200,1e-200]) assert.throws(()=>somOlp(input([x]),{grid:input([0]),initialPrototypes:Float64Array.of(0),maxIterations:0,kernelBackend}),/overflow|underflow/);
  assert.throws(()=>somOlp(input([0]),{grid:input([0,2]),initialPrototypes:Float64Array.of(0,0),initialMemberships:Float64Array.of(1,0),gamma:1e308,maxIterations:1,kernelBackend}),/cost overflow/);
  assert.throws(()=>somOlp(input([0,0,0]),{grid:input([-1e154,1e154]),initialPrototypes:Float64Array.of(0,0),initialMemberships:Float64Array.from(Array(6).fill(.5)),gamma:1,lambda:1,maxIterations:1,kernelBackend}),/objective overflow/);
  assert.throws(()=>somOlp(input([0,0,0]),{grid:input([0,0]),gamma:0,lambda:1.7e308,maxIterations:1,kernelBackend}),/objective overflow/);
 });
 test(`${kernelBackend}: exact ordinary-path parity with prior implementation`,()=>{
  for(const [n,d,m,q] of [[17,1,3,1],[19,2,7,2],[23,8,9,3]]) {
   const data=input(Array.from({length:n*d},(_,i)=>Math.sin(i*.17)*3),d),grid=input(Array.from({length:m*q},(_,i)=>Math.cos(i*.31)),q);
   for(const provided of [false,true]) {
    const options={grid,gamma:.37,lambda:.19,maxIterations:6,tolerance:0,kernelBackend};
    if(provided) {options.initialPrototypes=Float64Array.from({length:m*d},(_,i)=>Math.sin(i*.17)); options.initialMemberships=Float64Array.from({length:n*m},()=>1/m);}
    const actual=somOlp(data,options),expected=baseline(data,options);
    for(const key of ['W','P','V','history','labels'])assert.deepEqual(actual[key],expected[key],`${key}: ${[n,d,m,q,provided]}`);
    assert.equal(actual.iterations,expected.iterations);assert.equal(actual.converged,expected.converged);
   }
  }
 });
}
test('stable weighted mean preserves opposite finite endpoints',()=>{
 assert.equal(somStableMean(Float64Array.of(-1e308,1e308),0,1,2),0);
 assert.equal(somStableMean(Float64Array.of(1e308,1e308),0,1,2),1e308);
});
test('PCA temporary centered copy is counted in memory limits',()=>{
 assert.throws(()=>somOlp(input(Array(100).fill(1e308)),{grid:input([-1,1]),maxIterations:0,maxScratchBytes:800}),/PCA primary scratch/);
});

const oracle=JSON.parse((await import('node:fs')).readFileSync(new URL('./decimal-oracle.json',import.meta.url),'utf8'));
for(const kernelBackend of backendCases)test(`${kernelBackend}: independent 500-digit Decimal composite cost and one-step oracle`,()=>{
 for(const c of oracle.cases){
  const result=somOlp(input([0]),{grid:input(c.grid,c.q),initialPrototypes:Float64Array.of(0,0,0),initialMemberships:Float64Array.from(c.initial),gamma:c.gamma,lambda:c.lambda_,maxIterations:1,kernelBackend});
  for(let h=0;h<c.q;h++)near(result.V[h],c.v[h]);
  for(let j=0;j<3;j++) {
   near(somStableCost(Float64Array.of(0),0,Float64Array.of(0),0,1,Float64Array.from(c.v),0,Float64Array.from(c.grid),j*c.q,c.q,c.gamma),c.costs[j],3e-14);
   near(result.P[j],c.p[j],3e-14);
  }
  near(result.history[0],c.objective,5e-14);
 }
});
test('independent 500-digit Decimal weighted means',()=>{
 for(const c of oracle.means)near(somStableMean(Float64Array.from(c.data),0,1,c.data.length,Float64Array.from(c.weights)),c.result,4e-15);
});

for(const kernelBackend of backendCases)test(`${kernelBackend}: huge offset with representable nonzero scatter`,()=>{
 const offset=2**532,delta=2**490;
 const data=input([offset-delta,offset,offset+delta]);
 const initial=Float64Array.of(.75,.25,.5,.5,.25,.75),copy=initial.slice(),original=data.data.slice();
 const result=somOlp(data,{grid:input([0,1]),initialPrototypes:Float64Array.of(offset,offset),initialMemberships:initial,gamma:0,lambda:delta*delta,maxIterations:1,kernelBackend});
 assert.equal(result.W[0],offset-delta/3);assert.equal(result.W[1],offset+delta/3);
 for(let i=0;i<3;i++){const a=data.data[i]-result.W[0],b=data.data[i]-result.W[1];near(result.P[i*2],1/(1+Math.exp((a*a-b*b)/(delta*delta))));}
 assert.deepEqual(initial,copy);assert.deepEqual(data.data,original);
});
for(const kernelBackend of backendCases)test(`${kernelBackend}: cost-minimum shift precedes finite large-logit division`,()=>{
 const bytes=new ArrayBuffer(8),floats=new Float64Array(bytes),bits=new BigUint64Array(bytes);
 floats[0]=1e150;bits[0]+=1n;const second=floats[0];
 const c0=1e150*1e150,c1=second*second,lambda=.7*(c1-c0);
 const result=somOlp(input([0]),{grid:input([0,1]),initialPrototypes:Float64Array.of(1e150,second),lambda,maxIterations:0,kernelBackend});
 near(result.P[0],1/(1+Math.exp(-(c1-c0)/lambda)));
});

test('mixed-sign finite-span means preserve a small residual after cancellation',()=>{
 for(const magnitude of [8e307,1e200,1e100]) {
  near(somStableMean(Float64Array.of(magnitude,1,-magnitude),0,1,3),1/3,3e-15);
  near(somStableMean(Float64Array.of(-magnitude,-1,magnitude),0,1,3),-1/3,3e-15);
 }
});

test('cold binary mean retains residuals across the entire exponent range',()=>{
 const bytes=new ArrayBuffer(8),floats=new Float64Array(bytes),bits=new BigUint64Array(bytes);
 floats[0]=1e308;bits[0]-=1n;const next= floats[0];
 near(somStableMean(Float64Array.of(1e308,-1e308,1e-200),0,1,3),1e-200/3,2e-15);
 near(somStableMean(Float64Array.of(1e308,1e308,-next,-next,0),0,1,5),(1e308-next)*2/5,2e-15);
 floats[0]=2**499;bits[0]-=1n;const previous=floats[0],gap=2**499-previous;
 assert.equal(somStableMean(Float64Array.of(2**499,-previous,-gap,1),0,1,4),.25);
});

for(const kernelBackend of backendCases)test(`${kernelBackend}: finite objective survives overflowed separate aggregates`,()=>{
 const result=somOlp(input([0,0,0]),{grid:input([-1e154,1e154]),initialPrototypes:Float64Array.of(0,0),initialMemberships:Float64Array.from(Array(6).fill(.5)),gamma:1,lambda:1e308,maxIterations:1,kernelBackend});
 near(result.history[0],9.205584583201643e307,3e-15);
 assert.deepEqual([...result.P],Array(6).fill(.5));
});

for(const kernelBackend of backendCases)test(`${kernelBackend}: free energy retains tail after maximum probability rounds to one`,()=>{
 const radius=Math.sqrt(4e307),lambda=1e306;
 const result=somOlp(input([0]),{grid:input([0,radius]),initialPrototypes:Float64Array.of(0,0),initialMemberships:Float64Array.of(1,0),gamma:1,lambda,maxIterations:1,kernelBackend});
 const expected=-lambda*Math.log1p(Math.exp(-(radius*radius)/lambda));
 assert.equal(result.P[0],1);assert.ok(result.P[1]>0);
 near(result.history[0],expected,3e-14);
});

for(const kernelBackend of backendCases)test(`${kernelBackend}: original-unit stopping survives opposite-sign difference overflow`,()=>{
 const n=360,radius=Math.sqrt(1e307);
 const result=somOlp(input(Array(n).fill(0)),{grid:input([-radius,radius]),initialPrototypes:Float64Array.of(0,0),initialMemberships:Float64Array.from({length:n*2},(_,i)=>i%2?.25:.75),gamma:1,lambda:1.2e307,tolerance:2,maxIterations:3,kernelBackend});
 assert.equal(result.iterations,2);assert.equal(result.converged,true);
 near(result.history[0],1.5260548801317555e308,3e-13);
 near(result.history[1],-5.958286851929646e307,3e-13);
 assert.equal(Math.abs(result.history[1]-result.history[0]),Infinity);
 const ratio=Math.abs(result.history[1]/Math.abs(result.history[0])-1);
 assert.ok(ratio<=2);near(ratio,1+5.958286851929646e307/1.5260548801317555e308,3e-13);
});
