// Run against the frozen installed tarball; never relabel WebKit as native Safari.
import {chromium,firefox,webkit} from '@playwright/test';
import {createServer} from 'node:http';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {resolve,sep,extname} from 'node:path';
import {fileURLToPath} from 'node:url';
const root=resolve(process.argv[2]),output=resolve(process.argv[3]);
const here=fileURLToPath(new URL('.',import.meta.url));
const mime={'.html':'text/html','.mjs':'text/javascript','.js':'text/javascript','.json':'application/json','.wasm':'application/wasm'};
await mkdir(output,{recursive:true});
const server=createServer(async(req,res)=>{
 try{
  const url=new URL(req.url,'http://localhost'),name=decodeURIComponent(url.pathname);
  const special={'/':resolve(here,'smoke.html'),'/smoke.mjs':resolve(here,'smoke.mjs')};
  const path=special[name]??resolve(root,'.'+name);
  if(!special[name]&&!path.startsWith(root+sep)){res.writeHead(403).end();return;}
  const bytes=await readFile(path);
  const headers={'Content-Type':mime[extname(path)]??'application/octet-stream','Cache-Control':'no-store'};
  // No wasm-unsafe-eval: only this scenario's document enforces compile denial.
  if(url.searchParams.get('wasm')==='csp')headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; worker-src 'self'; connect-src 'self'; style-src 'self' 'unsafe-inline'";
  res.writeHead(200,headers);res.end(bytes);
 }catch{res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const url=`http://127.0.0.1:${server.address().port}`;
const reports=[];
try{
 for(const [engineName,engine] of Object.entries({chromium,firefox,webkit})){
  let browser;
  try{
   browser=await engine.launch({headless:true});
   for(const scenario of ['enabled','unavailable','csp','external-metrics']){
    const name=`${engineName}-${scenario}`,errors=[];let page;
    let report={engine:engineName,scenario,version:browser.version(),status:'failed'};
    try{
     page=await browser.newPage();
     if(scenario==='unavailable')await page.addInitScript(()=>{globalThis.WebAssembly=undefined;});
     page.on('pageerror',e=>errors.push(String(e)));
     page.on('requestfailed',r=>errors.push(`requestfailed ${r.url()} ${r.failure()?.errorText}`));
     await page.goto(url+(scenario==='external-metrics'?'/browser-smoke.html':`/?wasm=${scenario}`));
     await page.waitForFunction(()=>window.browserResult!==undefined,{},{timeout:90000});
     const result=await page.evaluate(()=>window.browserResult);
     report={...report,...result,actualBrowserExecution:true};
     await page.screenshot({path:resolve(output,name+'.png'),fullPage:true});
    }catch(error){report={...report,status:'failed',error:String(error)};}
    finally{if(page)try{await page.close();}catch(error){errors.push(`page close: ${error}`);}}
    report.errors=errors;reports.push(report);
    await writeFile(resolve(output,name+'.json'),JSON.stringify(report,null,2));
   }
  }catch(error){reports.push({engine:engineName,status:'failed',error:String(error),phase:'launch-or-close'});}
  finally{if(browser)try{await browser.close();}catch(error){reports.push({engine:engineName,status:'failed',phase:'close',error:String(error)});}}
 }
}finally{
 await writeFile(resolve(output,'summary.json'),JSON.stringify(reports,null,2));
 await new Promise(resolve=>server.close(resolve));
}
if(reports.length!==12||reports.some(r=>r.status!=='passed'||r.errors?.length))process.exitCode=1;
console.log(JSON.stringify(reports,null,2));
