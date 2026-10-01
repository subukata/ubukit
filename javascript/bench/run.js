import fs from 'node:fs';
import os from 'node:os';
import { runClusteringBenchmarks } from './clustering.js';
const result={generatedAt:new Date().toISOString(),runtime:process.version,v8:process.versions.v8,platform:process.platform,arch:process.arch,cpu:os.cpus()[0]?.model,logicalCPUs:os.cpus().length,backend:'single-thread JavaScript CPU',browserMeasured:false,notes:['No GPU or WASM.','First call is first call in this process; imports excluded.','Warm medians from five repetitions with alternating implementation order.','Scratch baselines are readable naive JavaScript, not a best-of-library competitor.'],clustering:runClusteringBenchmarks()};
if(process.argv.includes('--all')){const {runSomNeighborhoodBenchmarks}=await import('./som-neighborhood.js');result.somAndNeighborhood=await runSomNeighborhoodBenchmarks();}
fs.mkdirSync(new URL('../results/',import.meta.url),{recursive:true});fs.writeFileSync(new URL('../results/node-benchmark.json',import.meta.url),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
