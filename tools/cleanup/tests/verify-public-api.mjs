import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
const baselineRoot=new URL('../../../baseline/javascript/',import.meta.url);
const installedRoot=new URL('./node_modules/ubukit-js/',import.meta.url);
const original=JSON.parse(readFileSync(new URL('package.json',baselineRoot)));
const installed=JSON.parse(readFileSync(new URL('package.json',installedRoot)));
const results={runtime:process.version,packageVersion:installed.version,imports:[],purpose:'public package subpath and export shape checks, not a browser/full release gate'};
test('npm metadata and complete subpath map unchanged',()=>assert.deepEqual(installed,original));
for(const [subpath,file] of Object.entries(original.exports)) {
 const specifier=subpath==='.'?'ubukit-js':`ubukit-js/${subpath.slice(2)}`;
 const got=await import(specifier),prior=await import(new URL(file,baselineRoot));
 const resolved=import.meta.resolve(specifier);
 results.imports.push({specifier,resolved,baseline:new URL(file,baselineRoot).href,keys:Object.keys(got)});
 test(`${specifier}: public exports, call arities and classes unchanged`,()=>{
  assert.equal(resolved,new URL(file,installedRoot).href);
  assert.deepEqual(Object.keys(got),Object.keys(prior));
  for(const key of Object.keys(prior)) {
   assert.equal(typeof got[key],typeof prior[key]);
   if(typeof got[key]==='function') {
    assert.equal(got[key].name,prior[key].name);assert.equal(got[key].length,prior[key].length);
    if(prior[key].prototype)assert.deepEqual(Object.getOwnPropertyNames(got[key].prototype),Object.getOwnPropertyNames(prior[key].prototype));
   } else assert.deepEqual(got[key],prior[key]);
  }
 });
}
writeFileSync(new URL('../../../evidence/NPM_PUBLIC_API_IMPORTS.json',import.meta.url),JSON.stringify(results,null,2)+'\n');
