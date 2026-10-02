import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {TPEOptimizer} from '../consumer/node_modules/ubukit-js/src/optimization.js';
const {fixtures}=JSON.parse(readFileSync(new URL('../fixtures/python_parity.json',import.meta.url)));
let maxError=0;
for(const fixture of fixtures)test(`Python reference parity: ${fixture.name}`,()=>{
  const o=new TPEOptimizer(fixture.space,fixture.options);
  for(const t of fixture.initial)o.addTrial(t.params,t.value,{state:t.state});
  for(const expected of fixture.steps){
    const t=o.ask();assert.equal(t.id,expected.id);
    for(const [name,value] of Object.entries(expected.params)){
      const actual=t.params[name],domain=fixture.space[name];
      if(domain.type==='float'){
        const err=Math.abs(actual-value)/Math.max(1,Math.abs(value));maxError=Math.max(maxError,err);
        assert.ok(err<=1e-12,`${fixture.name} trial ${t.id} ${name}: ${actual} vs ${value}, relative/absolute error ${err}`);
      }else assert.equal(actual,value,`${fixture.name} trial ${t.id} ${name}`);
    }
    o.tell(t.id,expected.value);
  }
});
test('parity numeric tolerance report',()=>{console.log(`Maximum normalized Python/JS proposal difference: ${maxError}`);});
