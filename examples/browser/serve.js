import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root=fileURLToPath(new URL('../../',import.meta.url));
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.json':'application/json; charset=utf-8','.css':'text/css; charset=utf-8'};
const port=Number(process.env.PORT ?? 8765);
http.createServer(async(req,res)=>{try{const url=new URL(req.url,'http://localhost');const requested=decodeURIComponent(url.pathname==='/'?'/examples/browser/index.html':url.pathname);const filename=path.resolve(root,'.'+requested);if(!filename.startsWith(root)){res.writeHead(403);res.end('Forbidden');return;}const data=await readFile(filename);res.writeHead(200,{'Content-Type':types[path.extname(filename)]??'application/octet-stream','Cache-Control':'no-store'});res.end(data);}catch{res.writeHead(404);res.end('Not found');}}).listen(port,'127.0.0.1',()=>console.log(`UbuKit demo http://localhost:${port}/ ; browser tests /examples/browser/test.html`));
