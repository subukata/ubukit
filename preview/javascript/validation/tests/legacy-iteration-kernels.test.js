import test from 'node:test';
import assert from 'node:assert/strict';
import { fullSquaredDistance, fcmCenterBlock, fcmMembershipBlock, somPrototypeBlock, somMembershipBlock } from '../../consumer/node_modules/ubukit-js/src/iteration-kernels.js';
import { squaredDistance, seededRandom } from '../../consumer/node_modules/ubukit-js/src/core.js';
import { membershipsFromSquaredDistances } from '../../consumer/node_modules/ubukit-js/src/clustering.js';
const tiny = 2.2250738585072014e-308;
function objectiveTerm(u, distance, m) {
  if (u === 0 || distance === 0) return 0;
  const w = m === 2 ? u * u : u ** m;
  return w < tiny ? Math.exp(m * Math.log(u) + Math.log(distance)) : w * distance;
}
function outcome(fn) { try { return { value: fn() }; } catch (e) { return { name: e.name, message: e.message }; } }
test('full distance preserves sequential sum and exact underflow/overflow errors', () => {
  const rng = seededRandom(101);
  for (const d of [1,2,3,4,5,7,8,9,16,17]) {
    const a = Float64Array.from({length:d + 2},()=>rng()*100-50), b = Float64Array.from({length:d + 4},()=>rng()*100-50);
    assert.equal(fullSquaredDistance(a,1,b,2,d),squaredDistance(a,1,b,2,d));
  }
  for (const [a,b] of [[0,0],[-0,0],[0,1e-160],[0,1e-200],[0,1e308],[-1e308,1e308]]) {
    const x=Float64Array.of(a),y=Float64Array.of(b);
    assert.deepEqual(outcome(()=>fullSquaredDistance(x,0,y,0,1)),outcome(()=>squaredDistance(x,0,y,0,1)));
  }
});
test('FCM center blocks preserve row order for generic dimensions and exponents', () => {
  const rng=seededRandom(102), n=23,k=5;
  for(const d of [1,3,4,8,17]) for(const m of [1.3,2,3.5,10000]) {
    const data=Float64Array.from({length:n*d},()=>rng()*8-4),u=Float64Array.from({length:n*k},()=>rng());
    const expected=new Float64Array(k*d),expectedSums=new Float64Array(k),actual=new Float64Array(k*d),sums=new Float64Array(k);
    for(let i=0;i<n;i++)for(let c=0;c<k;c++){const value=u[i*k+c],w=m===2?value*value:value**m;expectedSums[c]+=w;for(let f=0;f<d;f++)expected[c*d+f]+=w*data[i*d+f];}
    for(let start=0;start<n;start+=7)fcmCenterBlock(data,u,sums,actual,start,Math.min(n,start+7),d,k,m);
    assert.deepEqual(actual,expected);assert.deepEqual(sums,expectedSums);
  }
});
test('FCM fused membership blocks retain distances, zero ties, powers and cumulative objective', () => {
 const rng=seededRandom(103),n=9,k=4;
 for(const d of [1,3,4,8,17])for(const m of [1+1e-9,1.3,2,3.5,10000]){
  const data=Float64Array.from({length:n*d},()=>rng()*8-4),centers=Float64Array.from({length:k*d},()=>rng()*8-4),old=Float64Array.from({length:n*k},()=>rng());
  centers.set(data.subarray(0,d),0);centers.set(data.subarray(0,d),d);
  const distances=new Float64Array(n*k);
  for(let i=0;i<n;i++)for(let c=0;c<k;c++)distances[i*k+c]=squaredDistance(data,i*d,centers,c*d,d);
  const expected=membershipsFromSquaredDistances(distances,k,m),actual=new Float64Array(n*k),dist=new Float64Array(k),acc={delta2:3,objective:7};let delta2=3,objective=7;
  for(let z=0;z<expected.length;z++){const diff=expected[z]-old[z];delta2+=diff*diff;objective+=objectiveTerm(expected[z],distances[z],m);}
  for(let start=0;start<n;start+=4)fcmMembershipBlock(data,centers,old,actual,dist,start,Math.min(n,start+4),d,k,m,acc);
  assert.deepEqual(actual,expected);assert.deepEqual(acc,{delta2,objective});
 }
});
test('SOM fused row blocks preserve old-P prototypes/embedding and softmax/objective order',()=>{
 const rng=seededRandom(104),n=13,m=5;
 for(const d of [1,3,4,8,17])for(const q of [1,2,3]){
  const data=Float64Array.from({length:n*d},()=>rng()*8-4),r=Float64Array.from({length:m*q},()=>rng()),p=Float64Array.from({length:n*m},()=>rng());
  const expectedNum=new Float64Array(m*d),expectedDen=new Float64Array(m),expectedV=new Float64Array(n*q);
  for(let i=0;i<n;i++)for(let j=0;j<m;j++){const value=p[i*m+j];expectedDen[j]+=value;for(let f=0;f<d;f++)expectedNum[j*d+f]+=value*data[i*d+f];for(let h=0;h<q;h++)expectedV[i*q+h]+=value*r[j*q+h];}
  const numerator=new Float64Array(m*d),denominator=new Float64Array(m),v=new Float64Array(n*q),w=new Float64Array(m*d);
  for(let start=0;start<n;start+=4)somPrototypeBlock(data,p,r,numerator,denominator,v,start,Math.min(n,start+4),d,m,q);
  assert.deepEqual(numerator,expectedNum);assert.deepEqual(denominator,expectedDen);assert.deepEqual(v,expectedV);
  for(let j=0;j<m;j++)for(let f=0;f<d;f++)w[j*d+f]=numerator[j*d+f]/denominator[j];
  const cost=new Float64Array(m),expected=new Float64Array(n*m),acc={distortion:3,entropy:7};let distortion=3,entropy=7;
  const squared=(a,ao,b,bo,d)=>{let sum=0;for(let f=0;f<d;f++){const x=a[ao+f]-b[bo+f];sum+=x*x;}return sum;};
  for(let i=0;i<n;i++){let min=Infinity,den=0;for(let j=0;j<m;j++){cost[j]=squared(data,i*d,w,j*d,d)+.5*squared(v,i*q,r,j*q,q);min=Math.min(min,cost[j]);}for(let j=0;j<m;j++){const x=Math.exp(-(cost[j]-min)/1.2);expected[i*m+j]=x;den+=x;}for(let j=0;j<m;j++)expected[i*m+j]/=den;for(let j=0;j<m;j++){const value=expected[i*m+j];distortion+=value*cost[j];if(value>0)entropy+=value*Math.log(value);}}
  for(let start=0;start<n;start+=4)somMembershipBlock(data,w,v,r,p,cost,start,Math.min(n,start+4),d,m,q,.5,1.2,acc);
  assert.deepEqual(p,expected);assert.deepEqual(acc,{distortion,entropy});
 }
});
