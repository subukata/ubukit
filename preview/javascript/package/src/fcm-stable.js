import { trackWeakMembershipDelta, membershipDelta } from './iteration-kernels.js';
/** Guarded finite-m FCM arithmetic. This is not the m=infinity approximation. */
import { assertFinite, checkCancelled, finiteNumber, labelsFromMembership, progress, seededRandom, restoreCenters } from './core.js';
import { checkpoint, SESSION_WARM_CENTERS, SESSION_FCM_STATE } from './session-hooks.js';
const MIN_NORMAL = 2.2250738585072014e-308;

export function needsStableFCM(x, options, m) {
  if (options[SESSION_FCM_STATE] || m > 32 || m - 1 < 1e-4) return true;
  const bound = Math.sqrt(Number.MAX_VALUE / x.nFeatures) / (8 * Math.max(1, x.nSamples));
  for (const a of [x.data, options.initCenters]) if (a) for (let j=0;j<a.length;++j) {
    const v=a[j],av=Math.abs(v),origin=x.data[j%x.nFeatures];
    if (av > bound || av > 0 && av < 1e-140 || v!==0 && (v-origin)+origin===0) return true;
  }
  if (options.initMembership) {
    const members = options.initMembership, k = members.length / x.nSamples, threshold = Math.exp(-600/m)*k;
    for (let i = 0; i < x.nSamples; ++i) {
      let maximum = 0; for (let c = 0; c < k; ++c) maximum = Math.max(maximum, members[i*k+c]);
      for (let c = 0; c < k; ++c) { const v = members[i*k+c]; if (v > 0 && v / maximum < threshold) return true; }
    }
  }
  return false;
}
function logRatio(a, b) {
  const relative = (a - b) / b;
  return Number.isFinite(relative) && relative > -1 ? Math.log1p(relative) : Math.log(a) - Math.log(b);
}
/** Original differences identify exact zero; no squared-distance floor is used. */
export function stableDistance(x, xo, y, yo, d) {
  let q = 0, scale = 0, ssq = 1, half = false;
  for (let f = 0; f < d; ++f) {
    const a = x[xo + f], b = y[yo + f];
    let delta = Math.abs(a - b);
    q += delta * delta;
    if (!Number.isFinite(delta)) { half = true; break; }
    if (delta !== 0) {
      if (scale < delta) { const r = scale / delta; ssq = 1 + ssq * r * r; scale = delta; }
      else { const r = delta / scale; ssq += r * r; }
    }
  }
  if (half) {
    scale = 0; ssq = 1; q = Infinity;
    for (let f = 0; f < d; ++f) {
      const delta = Math.abs(x[xo + f] / 2 - y[yo + f] / 2);
      if (delta !== 0) {
        if (scale < delta) { const r = scale / delta; ssq = 1 + ssq * r * r; scale = delta; }
        else { const r = delta / scale; ssq += r * r; }
      }
    }
  }
  if (scale === 0) return { q: 0, log: -Infinity, scale: 0, ssq: 1, half: false };
  return { q, log: q >= MIN_NORMAL && Number.isFinite(q) ? Math.log(q) : 2 * Math.log(scale) + Math.log(ssq) + (half ? 2 * Math.LN2 : 0), scale, ssq, half };
}
function distanceLogRatio(a, b) {
  if (a.q >= MIN_NORMAL && b.q >= MIN_NORMAL && Number.isFinite(a.q) && Number.isFinite(b.q)) return logRatio(a.q, b.q);
  return 2 * logRatio(a.scale, b.scale) + logRatio(a.ssq, b.ssq) + (Number(a.half) - Number(b.half)) * 2 * Math.LN2;
}
// Near m=1 amplifies norm rounding by 1/(m-1). Recover low squared-norm
// bits with error-free differences/products on an exact power-of-two scale.
function preciseNearOneNorms(data, centers, i, d, k) {
  let maximum=0;
  for(let f=0;f<d;++f)maximum=Math.max(maximum,Math.abs(data[i*d+f]));
  for(const v of centers)maximum=Math.max(maximum,Math.abs(v));
  if(maximum===0)return null;
  const scale=2**Math.max(-1074,Math.min(1023,Math.floor(Math.log2(maximum))));
  const hi=new Float64Array(k),lo=new Float64Array(k);
  for(let c=0;c<k;++c){let sum=0,correction=0;
    for(let f=0;f<d;++f){
      const a=data[i*d+f]/scale,b=centers[c*d+f]/scale,h=a-b,v=a-h;
      const low=(a-(h+v))+(v-b),product=h*h,split=134217729*h,hh=split-(split-h),hl=h-hh;
      const error=((hh*hh-product)+2*hh*hl)+hl*hl+2*h*low+low*low;
      const next=sum+product;correction+=(Math.abs(sum)>=Math.abs(product)?(sum-next)+product:(product-next)+sum)+error;sum=next;
    }
    if(!(sum>=MIN_NORMAL))return null;
    hi[c]=sum;lo[c]=correction;
  }
  return {hi,lo};
}
function preciseNormDifference(norms,c,j){
  const a=norms.hi[c],b=norms.hi[j],h=a-b,v=a-h;
  return h+((a-(h+v))+(v-b)+(norms.lo[c]-norms.lo[j]));
}
function addLog(state, term) {
  state.positive = true;
  if (term === -Infinity) return;
  if (state.sum === 0) { state.max = term; state.sum = 1; }
  else if (term <= state.max) state.sum += Math.exp(term - state.max);
  else { state.sum = state.sum * Math.exp(state.max - term) + 1; state.max = term; }
}
function objectiveResult(state) {
  const log = state.sum === 0 ? -Infinity : state.max + Math.log(state.sum);
  const value = Math.exp(log);
  return { objective: value, objectiveLog: log, objectiveRepresentation: value === Infinity ? 'overflow' : value === 0 ? (log === -Infinity && !state.positive ? 'exact_zero' : 'underflow') : 'finite' };
}
function scaleScore(score, to, from) {
  if (score === 0) return 0;
  const ratio = to / from;
  return Number.isFinite(ratio) && ratio > 0 ? score * ratio : score / from * to;
}
/** Compare weights after removing a column-specific common -m*log(count).
 * Positive rows use count=K and finite residual=m*log(K*u); zero rows use
 * their exact zero count and residual=0. No huge common log is ever formed.
 */
function normalizedLogWeights(state, out, n, k, m) {
  if(state.initial){
    const {rowSums}=state.initial;
    for(let c=0;c<k;++c){
      let reference=-1,referenceInteger=0n,referenceSum=0n;
      for(let i=0;i<n;++i)if(state.scores[i*k+c]>0){
        const candidate=binaryInteger(state.scores[i*k+c]);
        if(reference<0||candidate*referenceSum>referenceInteger*rowSums[i]){reference=i;referenceInteger=candidate;referenceSum=rowSums[i];}
      }
      for(let i=0;i<n;++i)out[i*k+c]=reference<0||state.scores[i*k+c]===0?-Infinity:m*logBinaryRatio(binaryInteger(state.scores[i*k+c])*referenceSum,referenceInteger*rowSums[i]);
    }
    return;
  }
  const difference = (i, j, c, power) => {
    const score = state.scores[i*k+c] - state.scores[j*k+c];
    return state.counts[i] === state.counts[j]
      ? scaleScore(score, power, state.power)
      : power * (score / state.power - logRatio(state.counts[i], state.counts[j]));
  };
  for (let c = 0; c < k; ++c) {
    let reference = -1;
    for (let i = 0; i < n; ++i) if (state.scores[i*k+c] !== -Infinity &&
      (reference < 0 || difference(i, reference, c, 1) > 0)) reference = i;
    for (let i = 0; i < n; ++i) out[i*k+c] = reference < 0 || state.scores[i*k+c] === -Infinity
      ? -Infinity : difference(i, reference, c, m);
  }
}
const bitsView = new DataView(new ArrayBuffer(8));
function binaryParts(value) {
  bitsView.setFloat64(0, value, false);
  const bits=bitsView.getBigUint64(0,false),field=Number((bits>>52n)&2047n),fraction=bits&((1n<<52n)-1n);
  return {negative:(bits>>63n)!==0n,mantissa:field?fraction+(1n<<52n):fraction,exponent:field?field-1075:-1074};
}
function binaryInteger(value){const p=binaryParts(value);return p.mantissa<<BigInt(p.exponent+1074);}
function leadingInteger(value){const shift=Math.max(0,value.toString(2).length-54);return {leading:Number(value>>BigInt(shift)),shift};}
function logBinaryRatio(a,b){
  const delta=a-b;if(delta===0n)return 0;
  const magnitude=delta<0n?-delta:delta;
  if(magnitude*2n<b){
    const x=leadingInteger(magnitude),y=leadingInteger(b);
    return Math.log1p((delta<0n?-1:1)*(x.leading/y.leading)*2**(x.shift-y.shift));
  }
  const x=leadingInteger(a),y=leadingInteger(b);
  return Math.log(x.leading/y.leading)+(x.shift-y.shift)*Math.LN2;
}
/** Exact signed binary product accumulation, followed by correctly rounded
 * division. Used for subnormal products or overflowing compensated sums. */
function divideBinaryInteger(integer, exponentUnit, denominator) {
  if(integer===0n)return 0;
  const negative=integer<0n;if(negative)integer=-integer;
  const den=binaryParts(denominator),offset=exponentUnit-den.exponent;
  let exponent=integer.toString(2).length-den.mantissa.toString(2).length;
  if(exponent>=0 ? integer<(den.mantissa<<BigInt(exponent)) : (integer<<BigInt(-exponent))<den.mantissa)--exponent;
  const unit=Math.max(-1074,exponent+offset-52),shift=offset-unit;
  const numerator=shift>=0?integer<<BigInt(shift):integer,divisor=shift>=0?den.mantissa:den.mantissa<<BigInt(-shift);
  let quotient=numerator/divisor;const remainder=numerator%divisor;
  if(2n*remainder>divisor || 2n*remainder===divisor && (quotient&1n))++quotient;
  const value=Number(quotient)*2**unit;return negative?-value:value;
}
function exactWeightedMean(data,n,d,f,logs,k,c,denominator){
  let integer=0n,weakMax=-Infinity,weakSum=0;
  for(let i=0;i<n;++i){const value=data[i*d+f],logw=logs[i*k+c];if(value===0||logw===-Infinity)continue;
    const weight=Math.exp(logw);
    if(weight>0){const a=binaryParts(value),b=binaryParts(weight),v=(a.mantissa*b.mantissa)<<BigInt(a.exponent+b.exponent+2148);integer+=a.negative?-v:v;}
    else {const l=logw+Math.log(Math.abs(value));if(l>weakMax){weakSum=weakSum*Math.exp(weakMax-l)+Math.sign(value);weakMax=l;}else weakSum+=Math.sign(value)*Math.exp(l-weakMax);}
  }
  const strong=divideBinaryInteger(integer,-2148,denominator);
  const weak=weakSum===0?0:Math.sign(weakSum)*Math.exp(weakMax+Math.log(Math.abs(weakSum))-Math.log(denominator));
  return strong+weak;
}
function stableWeightedMean(data, n, d, f, logs, k, c, denominator) {
  let sum=0,correction=0,overflow=false,weakMax=-Infinity,weakSum=0,weakProducts=false;
  for(let i=0;i<n;++i){
    const value=data[i*d+f],logw=logs[i*k+c];if(value===0||logw===-Infinity)continue;
    const weight=Math.exp(logw);let term=weight*value;
    if(Math.abs(term)<MIN_NORMAL)weakProducts=true;
    const logterm=term===0?logw+Math.log(Math.abs(value)):0;
    if(term===0 && weight===0)term=Math.sign(value)*Math.exp(logterm);
    if(term===0){
      if(logterm>weakMax){weakSum=weakSum*Math.exp(weakMax-logterm)+Math.sign(value);weakMax=logterm;}
      else weakSum+=Math.sign(value)*Math.exp(logterm-weakMax);
      continue;
    }
    const next=sum+term;
    correction+=Math.abs(sum)>=Math.abs(term)?(sum-next)+term:(term-next)+sum;sum=next;
    if(!Number.isFinite(sum)||!Number.isFinite(correction))overflow=true;
  }
  if(overflow||weakProducts)return exactWeightedMean(data,n,d,f,logs,k,c,denominator);
  return (sum+correction)/denominator;
}

function translatedStable(x, init) {
  const {data, nSamples:n, nFeatures:d} = x, origin = data.slice(0,d), out = new Float64Array(data.length);
  for (let f = 0; f < d; ++f) {
    for (let i = 0; i < n; ++i) if (!Number.isFinite(data[i*d+f]-origin[f]) || (data[i*d+f]-origin[f])+origin[f]!==data[i*d+f]) { origin[f]=0; break; }
    if (init) for (let j=f;j<init.length;j+=d) if (!Number.isFinite(init[j]-origin[f]) || (init[j]-origin[f])+origin[f]!==init[j]) { origin[f]=0; break; }
    for (let i=0;i<n;++i) out[i*d+f]=data[i*d+f]-origin[f];
  }
  return {data:out,origin};
}
function initialState(u, n, k) {
  const scores=u.slice(),counts=new Float64Array(n),rowSums=new Array(n);
  for(let i=0;i<n;++i){
    let maximum=0,integer=0n;
    for(let c=0;c<k;++c){const value=u[i*k+c];finiteNumber(value,'initMembership',0);maximum=Math.max(maximum,value);integer+=binaryInteger(value);}
    if(!(maximum>0))throw new RangeError('every initMembership row must have a positive sum');
    rowSums[i]=integer;let sum=0;
    for(let c=0;c<k;++c)sum+=u[i*k+c]/maximum;
    counts[i]=sum;
    for(let c=0;c<k;++c){const j=i*k+c;u[j]=(u[j]/maximum)/sum;}
  }
  return {scores,counts,power:1,initial:{rowSums}};
}
function membershipRow(data, centers, i, d, k, m, u, state, distances) {
  const off=i*k; let zeros=0, minimum=0;
  for(let c=0;c<k;++c){const a=stableDistance(data,i*d,centers,c*d,d);distances[c]=a;if(a.log===-Infinity)++zeros;if(c>0 && (a.log===-Infinity && distances[minimum].log!==-Infinity || a.log!==-Infinity && distances[minimum].log!==-Infinity && distanceLogRatio(a,distances[minimum])<0))minimum=c;}
  if(zeros){
    state.counts[i]=zeros;
    for(let c=0;c<k;++c){const zero=distances[c].log===-Infinity;u[off+c]=zero?1/zeros:0;state.scores[off+c]=zero?0:-Infinity;}
    return;
  }
  state.counts[i]=k;
  const precise=m-1<1e-4?preciseNearOneNorms(data,centers,i,d,k):null;
  if(precise)for(let c=0;c<k;++c)if(preciseNormDifference(precise,c,minimum)<0)minimum=c;
  const beta=1/(m-1), gamma=m/(m-1); let sum=0, em1=0, near=true;
  for(let c=0;c<k;++c){const r=precise?Math.log1p(preciseNormDifference(precise,c,minimum)/(precise.hi[minimum]+precise.lo[minimum])):distanceLogRatio(distances[c],distances[minimum]), a=-r*beta;state.scores[off+c]=r;u[off+c]=Math.exp(a);sum+=u[off+c];em1+=Math.expm1(a);if(a<-.5)near=false;}
  // expm1/log1p retains O(1/m) differences even when public U rounds to 1/K.
  const correction=near?m*Math.log1p(em1/k):m*(Math.log(sum)-Math.log(k));
  for(let c=0;c<k;++c){state.scores[off+c]=-gamma*state.scores[off+c]-correction;u[off+c]/=sum;}
}
function trueMembershipObjective(data, centers, u, n,d,k,m,blockRows,options) {
  const acc={max:-Infinity,sum:0};
  for(let i=0;i<n;++i){if(i%blockRows===0)checkCancelled(options);for(let c=0;c<k;++c){const v=u[i*k+c];if(v>0){const q=stableDistance(data,i*d,centers,c*d,d);if(q.log!==-Infinity)addLog(acc,m*Math.log(v)+q.log);}}}
  return objectiveResult(acc);
}
export function* stableFCMSteps(x, options, cfg) {
  const {n,d,k,maxIterations,blockRows,tolerance}=cfg,m=options.m??2;
  // Checked separately because the guarded path retains an additional log U array.
  const estimated=16*n*d+24*n*k+328*n+32*k*d+48*k+16*d+8*maxIterations;
  if(estimated>(options.maxMemoryBytes??512*1024**2))throw new RangeError(`stable FCM primary arrays ${estimated} bytes exceed maxMemoryBytes`);
  const shifted=translatedStable(x,options.initCenters??options[SESSION_WARM_CENTERS]),data=shifted.data,centers=new Float64Array(k*d);
  let u=new Float64Array(n*k),unew=new Float64Array(n*k),state;
  const warmState=options[SESSION_FCM_STATE],distances=new Array(k),history=[];
  if(options.initMembership!=null){
    if(!ArrayBuffer.isView(options.initMembership)||options.initMembership.length!==u.length)throw new RangeError('initMembership must be a TypedArray of nSamples * nClusters values');
    u.set(options.initMembership);state=initialState(u,n,k);
    if(warmState){state={scores:warmState.scores.slice(),counts:warmState.counts.slice(),power:warmState.power};u.set(options.initMembership);}
  }else if(options.initCenters!=null){
    if(!ArrayBuffer.isView(options.initCenters)||options.initCenters.length!==k*d)throw new RangeError('initCenters must be a TypedArray of nClusters * nFeatures values');
    assertFinite(options.initCenters,'initCenters');
    for(let j=0;j<k*d;++j)centers[j]=options.initCenters[j]-shifted.origin[j%d];
    state={scores:new Float64Array(n*k),counts:new Float64Array(n),power:m};
    for(let i=0;i<n;++i)membershipRow(data,centers,i,d,k,m,u,state,distances);
  }else{
    const rng=seededRandom(options.seed??0);for(let j=0;j<u.length;++j)u[j]=rng()||Number.EPSILON;state=initialState(u,n,k);
  }
  // Empty columns retain warm centers, otherwise the finite data mean.
  const warm=options[SESSION_WARM_CENTERS];
  if(warm)for(let j=0;j<k*d;++j)centers[j]=warm[j]-shifted.origin[j%d];
  else {unew.fill(0);for(let f=0;f<d;++f){const value=stableWeightedMean(data,n,d,f,unew,k,0,n);for(let c=0;c<k;++c)centers[c*d+f]=value;}}
  const previous=centers.slice(),scale=new Float64Array(d);
  for(let i=0;i<n;++i)for(let f=0;f<d;++f)scale[f]=Math.max(scale[f],Math.abs(data[i*d+f]));
  let iterations=0,delta=Infinity,converged=false,centerRelativeDelta=Infinity,membershipResolutionLostRows=0,obj={objective:Infinity,objectiveLog:Infinity,objectiveRepresentation:'overflow'};
  for(iterations=1;iterations<=maxIterations;++iterations){
    previous.set(centers);normalizedLogWeights(state,unew,n,k,m);
    // Column work is chunked so cancellation/session stepping remains cooperative.
    for(let c=0;c<k;++c){
      checkCancelled(options);let denominator=0;for(let i=0;i<n;++i)denominator+=Math.exp(unew[i*k+c]);
      if(denominator>0)for(let f=0;f<d;++f)centers[c*d+f]=stableWeightedMean(data,n,d,f,unew,k,c,denominator);
      yield{phase:'centers',iteration:iterations,completedRows:c+1,totalRows:k};
    }
    assertFinite(centers,'centers');centerRelativeDelta=0;
    for(let j=0;j<centers.length;++j){const s=Math.max(scale[j%d],Math.abs(previous[j]),Math.abs(centers[j]));if(s>0)centerRelativeDelta=Math.max(centerRelativeDelta,Math.abs(centers[j]/s-previous[j]/s));}
    let delta2=0;const deltaState={delta2:0};state.power=m;delete state.initial;
    for(let start=0;start<n;start+=blockRows){
      checkCancelled(options);const end=Math.min(n,start+blockRows);
      for(let i=start;i<end;++i){membershipRow(data,centers,i,d,k,m,unew,state,distances);for(let c=0;c<k;++c){const change=unew[i*k+c]-u[i*k+c];delta2+=change*change;if(delta2<MIN_NORMAL&&change!==0)trackWeakMembershipDelta(deltaState,change);}}
      yield{phase:'membership',iteration:iterations,completedRows:end,totalRows:n};
    }
    [u,unew]=[unew,u];deltaState.delta2=delta2;delta=membershipDelta(deltaState);converged=delta<tolerance;
    membershipResolutionLostRows=0;
    for(let i=0;i<n;++i){let same=true,different=false;for(let c=1;c<k;++c){same&&=u[i*k+c]===u[i*k];different||=state.scores[i*k+c]!==state.scores[i*k];}if(same&&different)++membershipResolutionLostRows;}
    obj=trueMembershipObjective(data,centers,u,n,d,k,m,blockRows,options);if(options.returnHistory)history.push(obj.objective);
    const result={algorithm:'fcm',centers,membership:u,membershipLayout:'samples-clusters',iterations,converged,nSamples:n,nFeatures:d,nClusters:k,m,...obj,delta,centerRelativeDelta,membershipConvergenceOnly:converged&&centerRelativeDelta>tolerance,numericalMode:'log-domain',membershipResolutionLostRows,...(options.returnHistory?{objectiveHistory:Float64Array.from(history)}:{})};
    result[SESSION_FCM_STATE]=state;checkpoint(options,result,shifted.origin);
    const event={algorithm:'fcm',iteration:iterations,maxIterations,...obj,delta,centerRelativeDelta,converged,membershipConvergenceOnly:result.membershipConvergenceOnly};progress(options,event);yield event;
    if(converged)break;
  }
  iterations=Math.min(iterations,maxIterations);restoreCenters(centers,shifted.origin);
  yield{phase:'finalize',completedRows:0,totalRows:n};obj=trueMembershipObjective(x.data,centers,u,n,d,k,m,blockRows,options);
  let fpc=0;for(const v of u)fpc+=v*v;
  const result={algorithm:'fcm',centers,membership:u,membershipLayout:'samples-clusters',labels:labelsFromMembership(u,n,k),...obj,fpc:fpc/n,delta,iterations,converged,nSamples:n,nFeatures:d,nClusters:k,m,backend:'javascript-float64',centerRelativeDelta,membershipConvergenceOnly:converged&&centerRelativeDelta>tolerance,numericalMode:'log-domain',membershipResolutionLostRows,...(options.returnHistory?{objectiveHistory:Float64Array.from(history)}:{})};
  Object.defineProperty(result,SESSION_FCM_STATE,{value:state});return result;
}
