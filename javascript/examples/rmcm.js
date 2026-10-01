import { rmcm, rmcmReference, prepareRMCM, runAsync } from '../src/index.js';
const input={ data:Float64Array.of(0,1,3), nSamples:3, nFeatures:1 };
const options={delta:2,initCenters:Float64Array.of(0,3),maxIterations:1};
const fast=rmcm(input,options), reference=rmcmReference(input,options);
console.log('Adjoint centers:',[...fast.centers]);      // [1, 2.2]
console.log('Reference centers:',[...reference.centers]);
console.log('R (samples × clusters):',[...fast.membership]);
console.log('Stop reason:',fast.stopReason);            // max_iter, not converged
// Reuse the input snapshot, fixed CSR and adjoint arrays across restarts.
const prepared=prepareRMCM(input,{delta:2});
console.log('Prepared run:',prepared.fit({nClusters:2,seed:12}));
// Cooperative fallback; use createWorkerClient().run('rmcm',...) for large demos.
const controller=new AbortController();
await runAsync('rmcm',input,{...options,timeBudgetMs:0,signal:controller.signal});
