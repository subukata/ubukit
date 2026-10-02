import test from 'node:test';
import assert from 'node:assert/strict';
import { seededRandom, squaredDistance } from '../../consumer/node_modules/ubukit-js/src/core.js';
import { somPrototypeBlock, somMembershipBlock, somPrototype2dBlock, somMembership2dBlock,
  somPrototypeGroupedBlock, somMembershipGroupedBlock, kmeansAssignment2dBlock,
  kmeansGroupedAssignmentBlock, kmeansFinalize2dBlock, kmeansFinalizeGroupedBlock } from '../../consumer/node_modules/ubukit-js/src/iteration-kernels.js';

test('SOM grouped and specialized kernels preserve exact generic row/block accumulation',()=>{
 const rnd=seededRandom(718),n=19;
 for(const d of[1,2,3,8,17,64])for(const q of[1,2,3])for(const m of[1,3,4,9,17]){
  const x=Float64Array.from({length:n*d},()=>rnd()*6-3),r=Float64Array.from({length:m*q},()=>rnd()*2-1),p=Float64Array.from({length:n*m},()=>rnd());
  const aN=new Float64Array(m*d),bN=new Float64Array(m*d),aD=new Float64Array(m),bD=new Float64Array(m),aV=new Float64Array(n*q),bV=new Float64Array(n*q);
  const proto=d===2&&q===2?somPrototype2dBlock:somPrototypeGroupedBlock;
  for(let start=0;start<n;start+=7){const end=Math.min(n,start+7);somPrototypeBlock(x,p,r,aN,aD,aV,start,end,d,m,q);proto(x,p,r,bN,bD,bV,start,end,d,m,q);}
  assert.deepEqual(bN,aN);assert.deepEqual(bD,aD);assert.deepEqual(bV,aV);
  const w=Float64Array.from(aN,(v,i)=>v/aD[Math.floor(i/d)]),a=p.slice(),b=p.slice(),aCost=new Float64Array(m),bCost=new Float64Array(m),aAcc={distortion:3,entropy:7},bAcc={distortion:3,entropy:7};
  const member=d===2&&q===2?somMembership2dBlock:somMembershipGroupedBlock;
  for(let start=0;start<n;start+=7){const end=Math.min(n,start+7);somMembershipBlock(x,w,aV,r,a,aCost,start,end,d,m,q,.37,1.3,aAcc);member(x,w,bV,r,b,bCost,start,end,d,m,q,.37,1.3,bAcc);}
  assert.deepEqual(b,a);assert.deepEqual(bAcc,aAcc);assert.deepEqual(bCost,aCost);
 }
});

test('Lloyd row blocks preserve scalar direct distances, first ties, sums and final inertia',()=>{
 const rnd=seededRandom(719),n=23;
 for(const d of[1,2,3,8,17,64])for(const k of[1,3,4,5,9,17]){
  const x=Float64Array.from({length:n*d},()=>rnd()*8-4),centers=Float64Array.from({length:k*d},()=>rnd()*8-4);
  if(k>1)centers.set(centers.subarray(0,d),d);
  const labels=new Int32Array(n).fill(-1),expected=new Int32Array(n),sums=new Float64Array(k*d),expectedSums=new Float64Array(k*d),counts=new Float64Array(k),expectedCounts=new Float64Array(k);let inertia=7;
  for(let i=0;i<n;++i){let best=0,dist=Infinity;for(let c=0;c<k;++c){const value=squaredDistance(x,i*d,centers,c*d,d,dist);if(value<dist){dist=value;best=c;}}expected[i]=best;++expectedCounts[best];for(let f=0;f<d;++f)expectedSums[best*d+f]+=x[i*d+f];inertia+=dist;}
  const assign=d===2?kmeansAssignment2dBlock:kmeansGroupedAssignmentBlock,finish=d===2?kmeansFinalize2dBlock:kmeansFinalizeGroupedBlock;let changed=0,actualInertia=7;
  for(let start=0;start<n;start+=7){const end=Math.min(n,start+7);changed+=assign(x,centers,labels,sums,counts,start,end,d,k);actualInertia=finish(x,centers,labels,start,end,d,k,actualInertia);}
  assert.equal(changed,n);assert.deepEqual(labels,expected);assert.deepEqual(sums,expectedSums);assert.deepEqual(counts,expectedCounts);assert.equal(actualInertia,inertia);
 }
});
