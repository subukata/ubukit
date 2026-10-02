import test from 'node:test';import assert from 'node:assert/strict';
import {membershipsFromSquaredDistances as before} from '../references/baseline/clustering.js';import {membershipsFromSquaredDistances as after} from '../../consumer/node_modules/ubukit-js/src/clustering.js';
for(const m of [1+Number.EPSILON,1+1e-12,1+1e-8,1.000099])for(const scale of [1e-300,1,1e300])test(`Near-one representable weights retain exact bits m=${m} scale=${scale}`,()=>{
 const delta=m-1,dist=Float64Array.from([0,1,128,512,700,745,800,1000,1024,2048],r=>scale*(1+r*delta));assert.deepEqual(after(dist,dist.length,m),before(dist,dist.length,m));
});
