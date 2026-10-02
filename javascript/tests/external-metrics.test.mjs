import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import * as scores from '../consumer/node_modules/ubukit-js/src/external-metrics.js';
import * as root from '../consumer/node_modules/ubukit-js/src/index.js';
const {adjustedRandScore: ari, adjustedMutualInfoScore: ami, adjustedScores: both, ExternalMetricDomainError} = scores;
const methods = ['arithmetic', 'geometric', 'min', 'max'];
const near = (actual, expected, tolerance = 2e-12) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} vs ${expected}, error ${Math.abs(actual-expected)}`);
const panel = JSON.parse(readFileSync(new URL('../fixtures/sklearn-1.8.json', import.meta.url)));
let maxARI = 0, maxAMI = 0, worstAMI;
for (const fixture of panel.cases) test(`sklearn 1.8 parity: ${fixture.name}`, () => {
  const {a,b} = fixture;
  near(ari(a,b),fixture.ari,1e-15);
  maxARI = Math.max(maxARI,Math.abs(ari(a,b)-fixture.ari));
  const ka=new Set(a).size,kb=new Set(b).size;
  for (const method of methods) {
    const options = {averageMethod:method};
    const singular=method==='min' && Math.min(ka,kb)>1 && ka!==kb && (ka===a.length||kb===a.length);
    if(singular) {assert.throws(()=>ami(a,b,options),{code:'AMI_SINGULAR_NORMALIZATION'});continue;}
    const value=ami(a,b,options), error=Math.abs(value-fixture.ami[method]);
    if(error>maxAMI){maxAMI=error;worstAMI={name:fixture.name,method,actual:value,expected:fixture.ami[method]};}
    // High-K sklearn is a reference measurement, not an oracle for its rounding artifacts.
    const tolerance=fixture.name==='balanced-3000-1500'?2e-8:2e-10;
    near(value,fixture.ami[method],tolerance);
    assert.deepEqual(both(a,b,options),{ari:ari(a,b),ami:value});
    near(ami(b,a,options),value,2e-14);
  }
});
test('reference error report',()=>{
  const result={sklearn:panel.sklearn,cases:panel.cases.length,maxARI,maxAMI,worstAMI};
  writeFileSync(new URL('../reports/sklearn-parity.json',import.meta.url),JSON.stringify(result,null,2)+'\n');
  console.log(result);
});
test('all exports and snake-case aliases are installed',()=>{
  for(const name of ['adjustedRandScore','adjustedMutualInfoScore','adjustedScores','ExternalMetricDomainError'])assert.equal(root[name],scores[name]);
  assert.equal(root.adjusted_rand_score,ari);assert.equal(root.adjusted_mutual_info_score,ami);assert.equal(root.adjusted_scores,both);
});
test('empty, singleton, constant and equivalent partitions',()=>{
  for(const [a,b] of [[[],[]],[[0],[9]],[[0,0],[1,1]],[[1,2,3],[8,7,6]],[[1,1,2,2],['b','b','a','a']]])
    for(const averageMethod of methods)assert.deepEqual(both(a,b,{averageMethod}),{ari:1,ami:1});
  for(const averageMethod of methods) {
    assert.deepEqual(both([0,0,0],[0,1,2],{averageMethod}),{ari:0,ami:0});
    assert.deepEqual(both([0,1,2],[0,0,0],{averageMethod}),{ari:0,ami:0});
  }
});
test('singleton-vs-other arithmetic/geometric/max zero; min explicit undefined domain',()=>{
  for(const n of [3,10,3000]){
    const a=Uint32Array.from({length:n},(_,i)=>i),b=a.slice();b[1]=0;
    for(const averageMethod of ['arithmetic','geometric','max']){
      assert.equal(ami(a,b,{averageMethod}),0);assert.equal(ami(b,a,{averageMethod}),0);
    }
    assert.throws(()=>ami(a,b,{averageMethod:'min'}),error=>error instanceof ExternalMetricDomainError && error.code==='AMI_SINGULAR_NORMALIZATION');
    assert.equal(ari(a,b),0);
  }
});
test('analytic high-K doublet oracle through n=100000',()=>{
  for(const n of [4,10,100,1000,3000,100000]){
    const a=Uint32Array.from({length:n},(_,i)=>i),b=a.slice();a[1]=0;b[3]=2;
    const exact=-2/(n*(n-1)-2);
    assert.equal(ari(a,b),exact);
    for(const averageMethod of methods)near(ami(a,b,{averageMethod}),exact,5e-15);
  }
});
test('independent pair-count ARI oracle over every partition up to size five',()=>{
  function partitions(n){const result=[];function go(a,max){if(a.length===n){result.push(a);return;}for(let j=0;j<=max+1;++j)go([...a,j],Math.max(max,j));}if(n===0)return [[]];go([0],0);return result;}
  for(let n=0;n<=5;++n) for(const a of partitions(n))for(const b of partitions(n)){
    let tp=0n,fp=0n,fn=0n,tn=0n;
    for(let i=0;i<n;++i)for(let j=i+1;j<n;++j){const x=a[i]===a[j],y=b[i]===b[j];if(x&&y)++tp;else if(x)++fn;else if(y)++fp;else ++tn;}
    const expected=fp===0n&&fn===0n?1:Number(2n*(tp*tn-fp*fn))/Number((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn));
    assert.equal(ari(a,b),expected);
  }
});
test('negative, large safe integer, Unicode, typed-array labels and renaming invariance',()=>{
  const a=[-3,-3,Number.MAX_SAFE_INTEGER,Number.MAX_SAFE_INTEGER,7,8],b=[2,2,2,1,1,2];
  const reference=both(a,b);
  assert.deepEqual(both(['α','α','🌱','🌱','x','y'],Int32Array.from(b)),reference);
  assert.deepEqual(both(Float64Array.from(a),Uint8Array.from(b)),reference);
  assert.deepEqual(both([0,-0,1,1,2,3],b),reference);
});
test('sparse contingency with many singleton labels and row/column permutations',()=>{
  const a=Array.from({length:2000},(_,i)=>Math.floor(i/2));
  const b=Array.from({length:2000},(_,i)=>Math.floor((i+1)/2));
  const baseline=both(a,b);
  const order=Array.from({length:2000},(_,i)=>(i*997)%2000);
  near(both(order.map(i=>a[i]),order.map(i=>b[i])).ami,baseline.ami,1e-14);
  assert.equal(both(order.map(i=>a[i]),order.map(i=>b[i])).ari,baseline.ari);
});
test('does not mutate, retain or cache input arrays',()=>{
  const a=[0,0,1,1],b=[0,1,0,1],ac=a.slice(),bc=b.slice();
  const before=both(a,b);assert.deepEqual(a,ac);assert.deepEqual(b,bc);
  b[1]=0;b[2]=1;assert.notDeepEqual(both(a,b),before);
});
test('invalid inputs and options are rejected consistently',()=>{
  for(const invalid of [null,undefined,'abc',{},new Set([1]),new DataView(new ArrayBuffer(8))])assert.throws(()=>ari(invalid,[]),TypeError);
  for(const invalid of [[1.1],[NaN],[Infinity],[-Infinity],[Number.MAX_SAFE_INTEGER+1],[true],[null],[undefined],[{}],[[1]],[1n],new BigInt64Array([1n]),[,'x']])assert.throws(()=>ari(invalid,Array(invalid.length).fill(0)),TypeError);
  assert.throws(()=>ari([1,'1'],[0,0]),TypeError);
  assert.throws(()=>ari([0],[]),RangeError);
  for(const options of [null,4,[],{averageMethod:'median'},{maxExpectedTerms:0},{maxExpectedTerms:1.5},{maxExpectedTerms:Infinity}]){
    assert.throws(()=>ami([],[],options));assert.throws(()=>both([],[],options));
  }
  const a=[0,0,1,1,2,2],b=[0,1,0,2,1,2];
  assert.throws(()=>ami(a,b,{maxExpectedTerms:1}),{code:'AMI_WORK_LIMIT'});
  assert.ok(Number.isFinite(ami(a,b,{maxExpectedTerms:100})));
});

test('AMI length limit is checked before encoding/allocation',()=>{
  const oversized = new Proxy([], {get(target,key){if(key==='length')return 2**26+1;throw new Error('must not read labels');}});
  assert.throws(()=>ami(oversized,oversized),{code:'AMI_SAMPLE_LIMIT'});
  assert.throws(()=>both(oversized,oversized),{code:'AMI_SAMPLE_LIMIT'});
});

test('expectation budget decisions are symmetric on entropy ties',()=>{
 const a=[0,0,0,0,1,2,3,4],b=[0,0,1,1,2,2,3,3];
 for(const limit of [1,3,4,10,100]){
  const outcome=(x,y)=>{try{return ami(x,y,{maxExpectedTerms:limit});}catch(e){return e.code;}};
  assert.equal(outcome(a,b),outcome(b,a));
 }
});
