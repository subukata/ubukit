import {performance} from 'node:perf_hooks';
const [variant,dArg,kArg]=process.argv.slice(2),d=+dArg,k=+kArg,n=500;
const {run}=await import(variant==='old'?'../../javascript/validation/efficiency_baseline_package/src/index.js':'../../javascript/consumer/node_modules/ubukit-js/src/index.js');
const x={data:new Float64Array(n*d),nSamples:n,nFeatures:d},w=new Float64Array(k*d);for(let j=1;j<k;j++)w[j*d]=j;
const o={gridShape:[k,1],initialPrototypes:w,maxIterations:0,...variant==='scalar'?{bmuBackend:'scalar'}:{}};
for(let i=0;i<100;i++)run('som',x,o);
const times=[];for(let i=0;i<31;i++){const start=performance.now();run('som',x,o);times.push(performance.now()-start);}
times.sort((a,b)=>a-b);console.log(JSON.stringify({variant,n,d,k,medianMs:times[15],rawMs:times,node:process.version}));
