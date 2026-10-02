import test from 'node:test';
import assert from 'node:assert/strict';
import { adjustedRandScore as ari, adjustedMutualInfoScore as ami, adjustedScores as both } from 'ubukit-js/external-metrics';

const methods = ['arithmetic', 'geometric', 'min', 'max'];
const labels = (n, values) => Int32Array.from({length:n}, (_,i) => values[i % values.length]);
function matchesArrayPath(a,b) {
  const plainA=Array.from(a), plainB=Array.from(b);
  assert.equal(ari(a,b),ari(plainA,plainB));
  for (const averageMethod of methods) {
    const options={averageMethod};
    assert.equal(ami(a,b,options),ami(plainA,plainB,options));
    assert.deepEqual(both(a,b,options),both(plainA,plainB,options));
  }
}

test('Int32 encoding agrees exactly at lengths 255 and 256',()=>{
  for (const n of [255,256]) matchesArrayPath(labels(n,[7,7,2,11,2,5,11]),labels(n,[3,1,1,3,8]));
});

test('dense labels preserve first-occurrence order for every AMI normalizer',()=>{
  const a=labels(1024,[47,3,47,19,3,8,19,19]),b=labels(1024,[12,12,2,39,2,12,7]);
  matchesArrayPath(a,b);
  matchesArrayPath(a.toReversed(),b.toReversed());
});

test('negative first labels and late outliers retain Map equality and IDs',()=>{
  const negative=labels(512,[-17,-17,-2,-31]),b=labels(512,[5,2,5,7,2]);
  matchesArrayPath(negative,b);
  for (const outlier of [-1,512,2147483647]) {
    const a=labels(512,[7,3,7,19]);a[509]=outlier;a[510]=7;a[511]=outlier;
    matchesArrayPath(a,b);
  }
});

test('ordinary Arrays and strings retain validation and label equality',()=>{
  const a=Array.from({length:512},(_,i)=>['α','0','🌱','0'][i%4]),b=labels(512,[2,3,2,1]);
  const expected=both(labels(512,[0,1,2,1]),b);
  assert.deepEqual(both(a,b),expected);
  assert.deepEqual(both(Array.from({length:512},(_,i)=>i%4<2?-0:1),b),both(labels(512,[0,0,1,1]),b));
  for (const bad of [NaN,Infinity,0.5,Number.MAX_SAFE_INTEGER+1]) {
    const invalid=Array(512).fill(0);invalid[300]=bad;
    for (const fn of [ari,ami,both]) assert.throws(()=>fn(invalid,b),{name:'TypeError',message:'labelsTrue[300] must be a string or safe integer'});
  }
  const mixed=Array(512).fill(0);mixed[300]='0';
  assert.throws(()=>both(mixed,b),{name:'TypeError',message:'labelsTrue must not mix string and integer labels'});
});

test('encoding does not mutate inputs or retain stale cross-call state',()=>{
  const a=labels(512,[7,3,7,19]),b=a.slice(),copies=[a.slice(),b.slice()];
  const before=both(a,b);
  assert.deepEqual([a,b],copies);
  b[300]=2147483647;
  const after=both(a,b);
  assert.notDeepEqual(after,before);
  assert.deepEqual(after,both(Array.from(a),Array.from(b)));
  assert.equal(b[300],2147483647);
});

test('lookup scratch stays capped at 65536 slots and handles boundary fallback',()=>{
  const n=65537,a=labels(n,[65535,1,8]),b=labels(n,[2,3,4]);
  a[n-2]=65536;a[n-1]=2147483647;b[n-2]=65537;b[n-1]=2147483646;
  const expected=both(Array.from(a),Array.from(b));
  const Original=globalThis.Uint32Array,allocations=[];
  // Probe allocations through the public call, without exposing private helpers.
  // The two N-sized output-code buffers are exempt from the lookup scratch cap.
  globalThis.Uint32Array=new Proxy(Original,{construct(Target,args,newTarget){
    const size=args[0];allocations.push(size);
    assert.ok(Number.isSafeInteger(size)&&(size===n||size<=65536),'oversized scratch allocation');
    return Reflect.construct(Target,args,newTarget);
  }});
  try { assert.deepEqual(both(a,b),expected); }
  finally { globalThis.Uint32Array=Original; }
  assert.equal(allocations.filter(size=>size===n).length,2,'only the two code buffers may exceed the scratch cap');
});

test('ordinary Array proxies do not require prototype inspection at the threshold',()=>{
  for (const n of [255,256]) {
    const a=new Proxy(Array(n).fill(0),{getPrototypeOf(){throw Error('must not inspect Array prototype');}});
    const b=new Int32Array(n);
    assert.equal(ari(a,b),1);
    assert.equal(ami(a,b),1);
    assert.deepEqual(both(a,b),{ari:1,ami:1});
  }
});

test('revoked proxies retain input and option validation order',()=>{
  for (const n of [255,256]) {
    const {proxy,revoke}=Proxy.revocable(Array(n).fill(0),{});revoke();
    for (const fn of [ari,ami,both]) {
      let labelReads=0;
      const a=new Proxy(Array(n).fill(0),{get(target,key,receiver){
        if(typeof key==='string'&&/^\d+$/.test(key))++labelReads;
        return Reflect.get(target,key,receiver);
      }});
      assert.throws(()=>fn(proxy,new Int32Array(n)),TypeError);
      assert.throws(()=>fn(a,proxy),TypeError);
      assert.equal(labelReads,0,'both inputs must be validated before encoding');
      assert.throws(()=>fn(null,proxy),{name:'TypeError',message:'labelsTrue must be an Array or numeric TypedArray'});
    }
    for (const fn of [ami,both]) assert.throws(()=>fn(proxy,[],{averageMethod:'median'}),{name:'RangeError',message:'averageMethod must be arithmetic, geometric, min, or max'});
  }
});

test('genuine TypedArrays with custom Proxy prototypes use the generic path',()=>{
  for (const n of [255,256]) for (const fn of [ari,ami,both]) {
    const a=new Int32Array(n);let visits=0;
    const prototype=new Proxy(Int32Array.prototype,{getPrototypeOf(){
      if(++visits>1)throw Error('extra prototype-chain traversal');
      return null;
    }});
    Object.setPrototypeOf(a,prototype);Object.defineProperty(a,'length',{value:n});
    assert.deepEqual(fn(a,new Int32Array(n)),fn===both?{ari:1,ami:1}:1);
    assert.equal(visits,1,'only existing DataView validation may inspect the prototype chain');
  }
});

test('post-validation object lengths stay on the generic path without extra coercion',()=>{
  const a=new Int32Array(256);let reads=0,coercions=0;
  const length={length:256,[Symbol.toPrimitive](){++coercions;return 256;}};
  Object.defineProperty(a,'length',{get(){return ++reads<4?256:length;}});
  assert.deepEqual(both(a,new Int32Array(256)),{ari:1,ami:1});
  assert.equal(reads,4);assert.equal(coercions,257);
});

test('post-validation fractional lengths retain the existing generic-path outcome',()=>{
  const typed=new Int32Array(257);typed[256]=256;let typedReads=0,plainReads=0;
  Object.defineProperty(typed,'length',{get(){return ++typedReads<4?256:256.5;}});
  const values=Array(257).fill(0);values[256]=256;
  const plain=new Proxy(values,{get(target,key,receiver){
    return key==='length'?(++plainReads<4?256:256.5):Reflect.get(target,key,receiver);
  }});
  assert.deepEqual(both(typed,new Int32Array(256)),both(plain,new Int32Array(256)));
  assert.equal(typedReads,4);assert.equal(plainReads,4);
});
