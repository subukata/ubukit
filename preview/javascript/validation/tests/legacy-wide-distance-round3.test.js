import test from 'node:test';import assert from 'node:assert/strict';
import { fullSquaredDistance, fullSquaredDistancesRow } from '../../consumer/node_modules/ubukit-js/src/iteration-kernels.js';
import {seededRandom} from '../../consumer/node_modules/ubukit-js/src/core.js';
function outcome(fn){try{return{value:fn()};}catch(e){return{name:e.name,message:e.message};}}
test('four-center lanes preserve every direct distance bit and minimum in generic dimensions',()=>{
 const rng=seededRandom(123);
 for(const d of [1,2,3,4,8,16,17,32,128,784])for(const k of [1,2,3,4,5,7,8,10,64]){
  const x=Float64Array.from({length:d},()=>rng()*10-5),centers=Float64Array.from({length:k*d},()=>rng()*10-5),out=new Float64Array(k),ref=new Float64Array(k);centers.set(x,0);
  for(let c=0;c<k;c++)ref[c]=fullSquaredDistance(x,0,centers,c*d,d);
  const min=fullSquaredDistancesRow(x,0,centers,d,k,out);assert.deepEqual(out,ref);assert.equal(min,Math.min(...ref));
 }
});
test('four-center lanes retain direct distance error classes and messages',()=>{
 for(const value of [0,1e-160,1e-200,1e308]){
  const x=Float64Array.of(0),centers=new Float64Array(4).fill(value),out=new Float64Array(4);
  assert.deepEqual(outcome(()=>fullSquaredDistancesRow(x,0,centers,1,4,out)),outcome(()=>fullSquaredDistance(x,0,centers,0,1)));
 }
});
