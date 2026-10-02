import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const base = new URL('../consumer/node_modules/ubukit-js/src/',import.meta.url);
const context=vm.createContext({performance,setTimeout,clearTimeout,AbortController});
const cache=new Map();
function load(url){if(cache.has(url.href))return cache.get(url.href);const mod=new vm.SourceTextModule(fs.readFileSync(url,'utf8'),{context,identifier:url.href});cache.set(url.href,mod);return mod;}
const root=load(new URL('index.js',base));await root.link((specifier,parent)=>{assert.ok(specifier.startsWith('./'),'external import forbidden: '+specifier);return load(new URL(specifier,parent.identifier));});await root.evaluate();
context.lib=root.namespace;
const result=await vm.runInContext(`(async()=>{
 const x={data:Float64Array.of(0,0,1,1,3,3),nSamples:3,nFeatures:2},o={gridShape:[2,2],seed:42,epochs:2};
 for(const a of ['som','som_batch']){const r=lib.run(a,x,o);if(!r.centers.every(Number.isFinite)||r.labels.length!==3)throw new Error('bad '+a);const s=lib.createSession(a,x,o);while(!s.status.done)s.step();if(s.snapshot().result.labels.length!==3)throw new Error('bad session');await lib.runAsync(a,x,{...o,timeBudgetMs:0});}
 return {noNodeGlobals:typeof process==='undefined'&&typeof Buffer==='undefined'&&typeof require==='undefined',passed:true};
})()`,context);
assert.equal(result.noNodeGlobals,true);assert.equal(result.passed,true);
const report={status:'passed',pureWebLikeVm:true,modules:cache.size,noNodeGlobals:true,actualBrowserExecution:false,node:process.version};fs.writeFileSync(new URL('../reports/som-browser-compatibility.json',import.meta.url),JSON.stringify(report,null,2)+'\n');console.log(report);
