// Offline route/import closure check against an existing frozen installed consumer.
// Usage: node tools/tests/check_browser_assets.mjs /path/to/javascript-result/candidate
// Does not install, start a browser, execute package algorithms or modify files.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {resolve,sep,extname} from 'node:path';
import {fileURLToPath} from 'node:url';
import vm from 'node:vm';

assert.ok(process.argv[2], 'Supply an existing javascript_clean_install.py candidate directory');
const root=resolve(process.argv[2]);
const repository=fileURLToPath(new URL('../../',import.meta.url));
const here=resolve(repository,'tools/ci/browser');
const browserFixtures=resolve(repository,'tests/integration/browser');
const source=await readFile(resolve(here,'run.mjs'),'utf8');
assert.ok(source.includes("new URL('../../../tests/integration/browser/',import.meta.url)"),'Browser fixture directory mapping changed; review this closure test');
const start=source.indexOf('const server=createServer(');
const end=source.indexOf('\nawait new Promise(resolve=>server.listen',start);
assert.ok(start>=0&&end>start,'Cannot isolate the actual static request handler');
let handler;
const context={root,here,browserFixtures,resolve,sep,extname,readFile,URL,
 mime:{'.html':'text/html','.mjs':'text/javascript','.js':'text/javascript','.json':'application/json','.wasm':'application/wasm'},
 createServer(fn){handler=fn;return {};}};
vm.runInNewContext(source.slice(start,end),context,{timeout:1000});
assert.equal(typeof handler,'function');

const runtimeBase=resolve(root,'consumer/node_modules/ubukit-js');
const runtimeManifest=JSON.parse(await readFile(resolve(runtimeBase,'SOURCE_MANIFEST.json'),'utf8'));
const expectedManifest=JSON.parse(await readFile(resolve(repository,'javascript/SOURCE_MANIFEST.json'),'utf8'));
assert.deepEqual(runtimeManifest,expectedManifest,'Installed package is not the reviewed candidate');
for(const [name,entry] of Object.entries(runtimeManifest.runtime_files)){
 const actual=createHash('sha256').update(await readFile(resolve(runtimeBase,name))).digest('hex');
 assert.equal(actual,entry.sha256,`Installed runtime hash mismatch: ${name}`);
}

async function request(url){
 const result={status:null,headers:{},body:null};
 const response={
  writeHead(status,headers={}){result.status=status;result.headers=headers;return this;},
  end(body){result.body=body;return this;}
 };
 await handler({url},response);
 return result;
}
const queue=['/','/smoke.mjs','/browser-smoke.html','/browser-worker.mjs'];
const visited=new Set(),edges=[],nodeOnly=[];
while(queue.length){
 const route=queue.shift();
 if(visited.has(route))continue;
 visited.add(route);
 const response=await request(route);
 assert.equal(response.status,200,`Browser asset is not served: ${route}`);
 const text=response.body.toString('utf8');
 if(route.endsWith('.json')){JSON.parse(text);continue;}
 const references=new Set();
 for(const pattern of [
  /\b(?:import|export)\s+[^;]*?\bfrom\s*['"]([^'"]+)['"]/g,
  /\bimport\s*\(\s*['"]([^'"]+)['"]/g,
  /\bfetch\s*\(\s*['"]([^'"]+)['"]/g,
  /\bnew\s+URL\s*\(\s*['"]([^'"]+)['"]\s*,\s*import\.meta\.url/g,
  /<script\b[^>]*\bsrc=['"]([^'"]+)['"]/g
 ])for(const match of text.matchAll(pattern))references.add(match[1]);
 for(const specifier of references){
  if(specifier.startsWith('node:')){
   // These two imports are guarded by the runtime's browser/Node worker split.
   assert.equal(specifier,'node:worker_threads');
   assert.ok(route.endsWith('/worker.js')||route.endsWith('/realtime-worker.js'));
   assert.match(text,/typeof\s+self\s*!==\s*['"]undefined['"]/);
   nodeOnly.push({route,specifier});continue;
  }
  assert.ok(specifier.startsWith('.')||specifier.startsWith('/'),`Unexpected browser import: ${specifier}`);
  const target=new URL(specifier,'http://localhost'+route);
  assert.equal(target.origin,'http://localhost','External browser resource');
  const dependency=target.pathname+target.search;
  edges.push({from:route,to:dependency});queue.push(dependency);
 }
}
for(const expected of ['/fixtures/sklearn-1.8.json','/fixtures/som-shared-reference.json',
 '/consumer/node_modules/ubukit-js/src/worker.js',
 '/consumer/node_modules/ubukit-js/src/realtime-worker.js',
 '/consumer/node_modules/ubukit-js/src/external-metrics.js'])assert.ok(visited.has(expected),`Missing closure branch: ${expected}`);
for(const route of ['/candidate-policy.json','/tests/integration/browser/browser-smoke.html','/browser-fixtures/browser-worker.mjs','/..%2F..%2Ftools/ci/candidate-policy.json']){
 const response=await request(route);assert.ok(response.status===403||response.status===404,`Unexpected harness file exposure: ${route}`);
}
const csp=await request('/?wasm=csp');
assert.equal(csp.status,200);
assert.ok(csp.headers['Content-Security-Policy']);
assert.ok(!csp.headers['Content-Security-Policy'].includes('wasm-unsafe-eval'));
console.log(JSON.stringify({status:'passed',check:'offline_browser_asset_closure',browserExecution:false,
 installedRuntimeFiles:Object.keys(runtimeManifest.runtime_files).length,
 routes:[...visited].sort(),dependencyEdges:edges.length,nodeOnlyConditionalImports:nodeOnly,
 fixtureRoutesWhitelisted:true,harnessTraversalRejected:true,cspCompileDenialRetained:true},null,2));
