/** Allocation-free row blocks shared by synchronous/cooperative execution.
 * Accumulators enter and leave each block without reordering floating sums.
 * Cancellation and events remain at the caller's existing block boundaries.
 */
const MIN_NORMAL = 2.2250738585072014e-308;

/** Cold stopping-norm recovery. Positive differences must not vanish merely
 * because their squares are subnormal. Normal sums retain their exact path. */
export function trackWeakMembershipDelta(accum, difference) {
  const value=Math.abs(difference),scale=accum.deltaScale??0;
  if(value>scale){const ratio=scale/value;accum.deltaScaledSum=1+(accum.deltaScaledSum??0)*ratio*ratio;accum.deltaScale=value;}
  else {const ratio=value/scale;accum.deltaScaledSum=(accum.deltaScaledSum??0)+ratio*ratio;}
}
export function membershipDelta(accum) {
  return accum.delta2<MIN_NORMAL&&(accum.deltaScale??0)>0
    ? accum.deltaScale*Math.sqrt(accum.deltaScaledSum) : Math.sqrt(accum.delta2);
}

/** Group weak terms before exp so representable totals are not lost. */
export function addLogObjectiveTerm(state, term) {
  // Finite m can still send m*log(u) to -Infinity. Such a term is exactly
  // unrepresentable even after every feasible array element is aggregated.
  if (term === -Infinity) return;
  if (state.weakSum === 0) { state.weakMax = term; state.weakSum = 1; }
  else if (term <= state.weakMax) state.weakSum += Math.exp(term - state.weakMax);
  else {
    state.weakSum = state.weakSum * Math.exp(state.weakMax - term) + 1;
    state.weakMax = term;
  }
}
export function logObjectiveValue(state) {
  return state.weakSum === 0 ? 0 : Math.exp(state.weakMax + Math.log(state.weakSum));
}

/** Full distance: membership algorithms do not benefit from cutoff branches. */
export function fullSquaredDistance(x, xo, y, yo, dimensions) {
  let sum = 0;
  let f = 0;
  for (; f + 3 < dimensions; f += 4) {
    const a = x[xo + f] - y[yo + f], b = x[xo + f + 1] - y[yo + f + 1], c = x[xo + f + 2] - y[yo + f + 2], e = x[xo + f + 3] - y[yo + f + 3];
    sum += a * a; sum += b * b; sum += c * c; sum += e * e;
  }
  for (; f < dimensions; ++f) { const delta = x[xo + f] - y[yo + f]; sum += delta * delta; }
  if (sum > 0 && sum < MIN_NORMAL) throw new RangeError('squared distance is subnormal; rescale input');
  if (sum === 0) for (let f = 0; f < dimensions; ++f) if (x[xo + f] !== y[yo + f]) throw new RangeError('squared distance underflow; rescale input');
  if (!Number.isFinite(sum)) throw new RangeError('squared distance overflow; rescale input');
  return sum;
}

/** Four independent center lanes preserve feature-order summation exactly. */
export function fullSquaredDistancesRow(data, io, centers, d, k, dist) {
  let c = 0;
  for (; c + 3 < k; c += 4) {
    const c0 = c * d, c1 = c0 + d, c2 = c1 + d, c3 = c2 + d;
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (let f = 0; f < d; ++f) {
      const x = data[io + f], a = x - centers[c0 + f], b = x - centers[c1 + f], q = x - centers[c2 + f], e = x - centers[c3 + f];
      s0 += a * a; s1 += b * b; s2 += q * q; s3 += e * e;
    }
    dist[c] = s0; dist[c + 1] = s1; dist[c + 2] = s2; dist[c + 3] = s3;
  }
  for (; c < k; ++c) dist[c] = fullSquaredDistance(data, io, centers, c * d, d);
  let minimum = Infinity;
  for (c = 0; c < k; ++c) {
    const value = dist[c];
    if (value > 0 && value < MIN_NORMAL) throw new RangeError('squared distance is subnormal; rescale input');
    if (value === 0) for (let f = 0; f < d; ++f) if (data[io + f] !== centers[c * d + f]) throw new RangeError('squared distance underflow; rescale input');
    if (!Number.isFinite(value)) throw new RangeError('squared distance overflow; rescale input');
    if (value < minimum) minimum = value;
  }
  return minimum;
}

/** Column-local accumulators remove repeated typed-array read/modify/write.
 * Each scalar still accumulates samples in increasing order, including the
 * incoming block total, so checkpoint boundaries and Float64 sums are exact.
 */
function fcmColumn1(data,u,sums,numerator,start,end,d,k,m,weighted=false) {
 for(let c=0;c<k;c++){
  const co=c*d;let total=sums[c],a0=numerator[co+0];
  for(let i=start;i<end;i++){
   const io=i*d,v=u[i*k+c],w=weighted?v:v*v;
   total+=w;a0+=w*data[io+0];
  }
  sums[c]=total;numerator[co+0]=a0;
 }
}
function fcmColumn3(data,u,sums,numerator,start,end,d,k,m,weighted=false) {
 for(let c=0;c<k;c++){
  const co=c*d;let total=sums[c],a0=numerator[co+0],a1=numerator[co+1],a2=numerator[co+2];
  for(let i=start;i<end;i++){
   const io=i*d,v=u[i*k+c],w=weighted?v:v*v;
   total+=w;a0+=w*data[io+0];a1+=w*data[io+1];a2+=w*data[io+2];
  }
  sums[c]=total;numerator[co+0]=a0;numerator[co+1]=a1;numerator[co+2]=a2;
 }
}
function fcmColumn4(data,u,sums,numerator,start,end,d,k,m,weighted=false) {
 for(let c=0;c<k;c++){
  const co=c*d;let total=sums[c],f=0;
  if(d>=4){
   let a=numerator[co],b=numerator[co+1],e=numerator[co+2],g=numerator[co+3];
   for(let i=start;i<end;i++){
    const io=i*d,v=u[i*k+c],w=weighted?v:v*v;
    total+=w;a+=w*data[io];b+=w*data[io+1];e+=w*data[io+2];g+=w*data[io+3];
   }
   numerator[co]=a;numerator[co+1]=b;numerator[co+2]=e;numerator[co+3]=g;f=4;
  }else{
   for(let i=start;i<end;i++){const v=u[i*k+c];total+=weighted?v:v*v;}
  }
  sums[c]=total;
  for(;f+3<d;f+=4){
   let a=numerator[co+f],b=numerator[co+f+1],e=numerator[co+f+2],g=numerator[co+f+3];
   for(let i=start;i<end;i++){
    const io=i*d+f,v=u[i*k+c],w=weighted?v:v*v;
    a+=w*data[io];b+=w*data[io+1];e+=w*data[io+2];g+=w*data[io+3];
   }
   numerator[co+f]=a;numerator[co+f+1]=b;numerator[co+f+2]=e;numerator[co+f+3]=g;
  }
  for(;f<d;f++){
   let a=numerator[co+f];for(let i=start;i<end;i++){const v=u[i*k+c],w=weighted?v:v*v;a+=w*data[i*d+f];}numerator[co+f]=a;
  }
 }
}
function fcmColumn2(data,u,sums,numerator,start,end,d,k,m,weighted=false) {
 for(let c=0;c<k;c++){
  const co=c*d;let total=sums[c],f=0;
  if(d>=2){
   let a=numerator[co],b=numerator[co+1];
   for(let i=start;i<end;i++){
    const io=i*d,v=u[i*k+c],w=weighted?v:v*v;
    total+=w;a+=w*data[io];b+=w*data[io+1];
   }
   numerator[co]=a;numerator[co+1]=b;f=2;
  }else{
   for(let i=start;i<end;i++){const v=u[i*k+c];total+=weighted?v:v*v;}
  }
  sums[c]=total;
  for(;f+1<d;f+=2){
   let a=numerator[co+f],b=numerator[co+f+1];
   for(let i=start;i<end;i++){
    const io=i*d+f,v=u[i*k+c],w=weighted?v:v*v;
    a+=w*data[io];b+=w*data[io+1];
   }
   numerator[co+f]=a;numerator[co+f+1]=b;
  }
  for(;f<d;f++){
   let a=numerator[co+f];for(let i=start;i<end;i++){const v=u[i*k+c],w=weighted?v:v*v;a+=w*data[i*d+f];}numerator[co+f]=a;
  }
 }
}

// Generated wider blocks
function fcmColumn8(data,u,sums,numerator,start,end,d,k,m,weighted=false) {
 for(let c=0;c<k;c++){
 const co=c*d;let total=sums[c],f=0;
 if(d>=8){
 let a0=numerator[co+0+0],a1=numerator[co+0+1],a2=numerator[co+0+2],a3=numerator[co+0+3],a4=numerator[co+0+4],a5=numerator[co+0+5],a6=numerator[co+0+6],a7=numerator[co+0+7];
for(let i=start;i<end;i++){
const io=i*d+0,v=u[i*k+c],w=weighted?v:v*v;
total+=w;
a0+=w*data[io+0];
a1+=w*data[io+1];
a2+=w*data[io+2];
a3+=w*data[io+3];
a4+=w*data[io+4];
a5+=w*data[io+5];
a6+=w*data[io+6];
a7+=w*data[io+7];
}
numerator[co+0+0]=a0;
numerator[co+0+1]=a1;
numerator[co+0+2]=a2;
numerator[co+0+3]=a3;
numerator[co+0+4]=a4;
numerator[co+0+5]=a5;
numerator[co+0+6]=a6;
numerator[co+0+7]=a7;
 f=8;
 }else{for(let i=start;i<end;i++){const v=u[i*k+c];total+=weighted?v:v*v;}}
 sums[c]=total;
 for(;f+7<d;f+=8){
 let a0=numerator[co+f+0],a1=numerator[co+f+1],a2=numerator[co+f+2],a3=numerator[co+f+3],a4=numerator[co+f+4],a5=numerator[co+f+5],a6=numerator[co+f+6],a7=numerator[co+f+7];
for(let i=start;i<end;i++){
const io=i*d+f,v=u[i*k+c],w=weighted?v:v*v;
a0+=w*data[io+0];
a1+=w*data[io+1];
a2+=w*data[io+2];
a3+=w*data[io+3];
a4+=w*data[io+4];
a5+=w*data[io+5];
a6+=w*data[io+6];
a7+=w*data[io+7];
}
numerator[co+f+0]=a0;
numerator[co+f+1]=a1;
numerator[co+f+2]=a2;
numerator[co+f+3]=a3;
numerator[co+f+4]=a4;
numerator[co+f+5]=a5;
numerator[co+f+6]=a6;
numerator[co+f+7]=a7;
 }
 for(;f<d;f++){let a=numerator[co+f];for(let i=start;i<end;i++){const v=u[i*k+c],w=weighted?v:v*v;a+=w*data[i*d+f];}numerator[co+f]=a;}
 }
}

/** Column sums for dense rough masks, retaining each sample addition order. */
export function roughCenterBlock(data,u,sums,numerator,start,end,d,k) {
 for(let c=0;c<k;c++){
 const co=c*d;let total=sums[c],f=0;
 if(d>=8){
 let a0=numerator[co+0+0],a1=numerator[co+0+1],a2=numerator[co+0+2],a3=numerator[co+0+3],a4=numerator[co+0+4],a5=numerator[co+0+5],a6=numerator[co+0+6],a7=numerator[co+0+7];
for(let i=start;i<end;i++){
const io=i*d+0,v=u[i*k+c],w=v;
    if (v === 0) continue;
total+=w;
a0+=w*data[io+0];
a1+=w*data[io+1];
a2+=w*data[io+2];
a3+=w*data[io+3];
a4+=w*data[io+4];
a5+=w*data[io+5];
a6+=w*data[io+6];
a7+=w*data[io+7];
}
numerator[co+0+0]=a0;
numerator[co+0+1]=a1;
numerator[co+0+2]=a2;
numerator[co+0+3]=a3;
numerator[co+0+4]=a4;
numerator[co+0+5]=a5;
numerator[co+0+6]=a6;
numerator[co+0+7]=a7;
 f=8;
 }else{for(let i=start;i<end;i++){const v=u[i*k+c];total+=v;}}
 sums[c]=total;
 for(;f+7<d;f+=8){
 let a0=numerator[co+f+0],a1=numerator[co+f+1],a2=numerator[co+f+2],a3=numerator[co+f+3],a4=numerator[co+f+4],a5=numerator[co+f+5],a6=numerator[co+f+6],a7=numerator[co+f+7];
for(let i=start;i<end;i++){
const io=i*d+f,v=u[i*k+c],w=v;
    if (v === 0) continue;
a0+=w*data[io+0];
a1+=w*data[io+1];
a2+=w*data[io+2];
a3+=w*data[io+3];
a4+=w*data[io+4];
a5+=w*data[io+5];
a6+=w*data[io+6];
a7+=w*data[io+7];
}
numerator[co+f+0]=a0;
numerator[co+f+1]=a1;
numerator[co+f+2]=a2;
numerator[co+f+3]=a3;
numerator[co+f+4]=a4;
numerator[co+f+5]=a5;
numerator[co+f+6]=a6;
numerator[co+f+7]=a7;
 }
 for(;f<d;f++){let a=numerator[co+f];for(let i=start;i<end;i++){const v=u[i*k+c],w=v;if(v===0)continue;a+=w*data[i*d+f];}numerator[co+f]=a;}
 }
}
export function fcmWeightedCenterBlock(data, u, sums, numerator, start, end, d, k, m) {
  if (d === 1) return fcmColumn1(data, u, sums, numerator, start, end, d, k, m, true);
  if (d === 2) return fcmColumn2(data, u, sums, numerator, start, end, d, k, m, true);
  if (d === 3) return fcmColumn3(data, u, sums, numerator, start, end, d, k, m, true);
  if (d >= 8) return fcmColumn8(data, u, sums, numerator, start, end, d, k, m, true);
  if (d >= 4) return fcmColumn4(data, u, sums, numerator, start, end, d, k, m, true);
  for (let i = start; i < end; ++i) {
    const io = i * d, off = i * k;
    for (let c = 0; c < k; ++c) {
      const value = u[off + c], weight = value, co = c * d;
      sums[c] += weight;
      let f = 0;
      for (; f + 3 < d; f += 4) {
        numerator[co + f] += weight * data[io + f]; numerator[co + f + 1] += weight * data[io + f + 1];
        numerator[co + f + 2] += weight * data[io + f + 2]; numerator[co + f + 3] += weight * data[io + f + 3];
      }
      for (; f < d; ++f) numerator[co + f] += weight * data[io + f];
    }
  }
}

function fcmCenterBlockM2(data, u, sums, numerator, start, end, d, k, m) {
  if (d === 1) return fcmColumn1(data, u, sums, numerator, start, end, d, k, m, false);
  if (d === 2) return fcmColumn2(data, u, sums, numerator, start, end, d, k, m, false);
  if (d === 3) return fcmColumn3(data, u, sums, numerator, start, end, d, k, m, false);
  if (d >= 8) return fcmColumn8(data, u, sums, numerator, start, end, d, k, m, false);
  if (d >= 4) return fcmColumn4(data, u, sums, numerator, start, end, d, k, m, false);
  for (let i = start; i < end; ++i) {
    const io = i * d, off = i * k;
    for (let c = 0; c < k; ++c) {
      const value = u[off + c], weight = value * value, co = c * d;
      sums[c] += weight;
      let f = 0;
      for (; f + 3 < d; f += 4) {
        numerator[co + f] += weight * data[io + f]; numerator[co + f + 1] += weight * data[io + f + 1];
        numerator[co + f + 2] += weight * data[io + f + 2]; numerator[co + f + 3] += weight * data[io + f + 3];
      }
      for (; f < d; ++f) numerator[co + f] += weight * data[io + f];
    }
  }
}

export function fcmCenterBlock(data, u, sums, numerator, start, end, d, k, m) {
  if (m === 2) return fcmCenterBlockM2(data, u, sums, numerator, start, end, d, k, m);
  for (let i = start; i < end; ++i) {
    const io = i * d, off = i * k;
    for (let c = 0; c < k; ++c) {
      const value = u[off + c], weight = value ** m, co = c * d;
      sums[c] += weight;
      let f = 0;
      for (; f + 3 < d; f += 4) {
        numerator[co + f] += weight * data[io + f]; numerator[co + f + 1] += weight * data[io + f + 1];
        numerator[co + f + 2] += weight * data[io + f + 2]; numerator[co + f + 3] += weight * data[io + f + 3];
      }
      for (; f < d; ++f) numerator[co + f] += weight * data[io + f];
    }
  }
}

function fcmMembershipBlockM2(data, centers, old, next, dist, start, end, d, k, m, accum, objectiveState, cacheWeights) {
  let delta2 = accum.delta2, objective = accum.objective;
  for (let i = start; i < end; ++i) {
    const io = i * d, off = i * k; let minimum = Infinity, sum = 0;
    if (d >= 16 && k >= 4) minimum = fullSquaredDistancesRow(data, io, centers, d, k, dist);
    else for (let c = 0; c < k; ++c) {
      const distance = fullSquaredDistance(data, io, centers, c * d, d);
      dist[c] = distance; minimum = Math.min(minimum, distance);
    }
    if (minimum === 0) {
      for (let c = 0; c < k; ++c) { const weight = dist[c] === 0 ? 1 : 0; next[off + c] = weight; sum += weight; }
    } else {
      for (let c = 0; c < k; ++c) { const weight = minimum / dist[c]; next[off + c] = weight; sum += weight; }
    }
    for (let c = 0; c < k; ++c) {
      const value = next[off + c] / sum; next[off + c] = value;
      const diff = value - old[off + c]; delta2 += diff * diff;
      if(delta2<MIN_NORMAL && diff!==0)trackWeakMembershipDelta(accum,diff);
      const distance = dist[c];
      const weight = value * value;
      if (cacheWeights) old[off + c] = weight;
      if (value !== 0 && distance !== 0) {
        if (weight < MIN_NORMAL) {
          const term = 2 * Math.log(value) + Math.log(distance);
          if (objectiveState) addLogObjectiveTerm(objectiveState, term);
          else objective += Math.exp(term);
        } else objective += weight * distance;
      }
    }
  }
  accum.delta2 = delta2; accum.objective = objective;
}

export function fcmMembershipBlock(data, centers, old, next, dist, start, end, d, k, m, accum, objectiveState = null, cacheWeights = false) {
  if (m === 2) return fcmMembershipBlockM2(data, centers, old, next, dist, start, end, d, k, m, accum, objectiveState, cacheWeights);
  let delta2 = accum.delta2, objective = accum.objective;
  const power = 1 / (m - 1), nearOne = m - 1 < 1e-4;
  for (let i = start; i < end; ++i) {
    const io = i * d, off = i * k; let minimum = Infinity, sum = 0;
    if (d >= 16 && k >= 4) minimum = fullSquaredDistancesRow(data, io, centers, d, k, dist);
    else for (let c = 0; c < k; ++c) {
      const distance = fullSquaredDistance(data, io, centers, c * d, d);
      dist[c] = distance; minimum = Math.min(minimum, distance);
    }
    if (minimum === 0) {
      for (let c = 0; c < k; ++c) { const weight = dist[c] === 0 ? 1 : 0; next[off + c] = weight; sum += weight; }
    } else if (m === 2) {
      for (let c = 0; c < k; ++c) { const weight = minimum / dist[c]; next[off + c] = weight; sum += weight; }
    } else {
      for (let c = 0; c < k; ++c) { const ratio = minimum / dist[c]; const weight = nearOne ? (dist[c] - minimum <= minimum * (1024 * (m - 1)) ? Math.exp(-Math.log1p((dist[c] - minimum) / minimum) * power) : 0) : ratio < MIN_NORMAL ? Math.exp((Math.log(minimum) - Math.log(dist[c])) * power) : power === 0.5 ? Math.sqrt(ratio) : power === 2 ? ratio * ratio : ratio ** power; next[off + c] = weight; sum += weight; }
    }
    for (let c = 0; c < k; ++c) {
      const value = next[off + c] / sum; next[off + c] = value;
      const diff = value - old[off + c]; delta2 += diff * diff;
      if(delta2<MIN_NORMAL && diff!==0)trackWeakMembershipDelta(accum,diff);
      const distance = dist[c];
      const weight = m === 2 ? value * value : value ** m;
      if (cacheWeights) old[off + c] = weight;
      if (value !== 0 && distance !== 0) {
        if (weight < MIN_NORMAL) {
          const term = m * Math.log(value) + Math.log(distance);
          if (objectiveState) addLogObjectiveTerm(objectiveState, term);
          else objective += Math.exp(term);
        } else objective += weight * distance;
      }
    }
  }
  accum.delta2 = delta2; accum.objective = objective;
}

export function somPrototypeBlock(data, p, r, numerator, denominator, v, start, end, d, m, q) {
  for (let i = start; i < end; ++i) {
    const io = i * d, po = i * m, vo = i * q;
    for (let j = 0; j < m; ++j) {
      const value = p[po + j], wo = j * d, ro = j * q;
      denominator[j] += value;
      let f = 0;
      for (; f + 3 < d; f += 4) {
        numerator[wo + f] += value * data[io + f]; numerator[wo + f + 1] += value * data[io + f + 1];
        numerator[wo + f + 2] += value * data[io + f + 2]; numerator[wo + f + 3] += value * data[io + f + 3];
      }
      for (; f < d; ++f) numerator[wo + f] += value * data[io + f];
      for (let h = 0; h < q; ++h) v[vo + h] += value * r[ro + h];
    }
  }
}
function somDistance(a, ao, b, bo, d) {
  let sum = 0;
  let f = 0;
  for (; f + 3 < d; f += 4) {
    const x0 = a[ao + f] - b[bo + f], x1 = a[ao + f + 1] - b[bo + f + 1], x2 = a[ao + f + 2] - b[bo + f + 2], x3 = a[ao + f + 3] - b[bo + f + 3];
    sum += x0 * x0; sum += x1 * x1; sum += x2 * x2; sum += x3 * x3;
  }
  for (; f < d; ++f) { const delta = a[ao + f] - b[bo + f]; sum += delta * delta; }
  if (!Number.isFinite(sum)) throw new RangeError('SOM distance overflowed; rescale the data');
  return sum;
}
export function somMembershipBlock(data, w, v, r, p, cost, start, end, d, m, q, gamma, lambda, accum) {
  let distortion = accum.distortion, entropy = accum.entropy;
  for (let i = start; i < end; ++i) {
    const io = i * d, vo = i * q, po = i * m; let minimum = Infinity, den = 0;
    for (let j = 0; j < m; ++j) {
      const value = somDistance(data, io, w, j * d, d) + gamma * somDistance(v, vo, r, j * q, q);
      if (!Number.isFinite(value)) throw new RangeError('SOM local cost overflowed; rescale the data/gamma');
      cost[j] = value; if (value < minimum) minimum = value;
    }
    for (let j = 0; j < m; ++j) { const value = Math.exp(-(cost[j] - minimum) / lambda); p[po + j] = value; den += value; }
    for (let j = 0; j < m; ++j) {
      const value = p[po + j] / den; p[po + j] = value;
      distortion += value * cost[j]; if (value > 0) entropy += value * Math.log(value);
    }
  }
  accum.distortion = distortion; accum.entropy = entropy;
}

export function somMembership2dBlock(data, w, v, r, p, cost, start, end, d, m, q, gamma, lambda, accum) {
  let distortion = accum.distortion, entropy = accum.entropy;
  for (let i = start; i < end; ++i) {
    const io = i * 2, po = i * m, x = data[io], y = data[io + 1], vx = v[io], vy = v[io + 1];
    let minimum = Infinity, den = 0;
    for (let j = 0; j < m; ++j) {
      const off = j * 2, dx = x - w[off], dy = y - w[off + 1], rx = vx - r[off], ry = vy - r[off + 1];
      const feature = dx * dx + dy * dy, grid = rx * rx + ry * ry;
      if (!Number.isFinite(feature) || !Number.isFinite(grid)) throw new RangeError('SOM distance overflowed; rescale the data');
      const value = feature + gamma * grid;
      if (!Number.isFinite(value)) throw new RangeError('SOM local cost overflowed; rescale the data/gamma');
      cost[j] = value; if (value < minimum) minimum = value;
    }
    for (let j = 0; j < m; ++j) { const value = Math.exp(-(cost[j] - minimum) / lambda); p[po + j] = value; den += value; }
    for (let j = 0; j < m; ++j) {
      const value = p[po + j] / den; p[po + j] = value;
      distortion += value * cost[j]; if (value > 0) entropy += value * Math.log(value);
    }
  }
  accum.distortion = distortion; accum.entropy = entropy;
}

export function somPrototype2dBlock(data, p, r, numerator, denominator, v, start, end, d, m, q) {
    for (let i = start; i < end; ++i) {
      const io = i * 2, po = i * m, x = data[io], y = data[io + 1];
      let vx = v[io], vy = v[io + 1];
      for (let j = 0; j < m; ++j) {
        const value = p[po + j], off = j * 2;
        denominator[j] += value;
        numerator[off] += value * x; numerator[off + 1] += value * y;
        vx += value * r[off]; vy += value * r[off + 1];
      }
      v[io] = vx; v[io + 1] = vy;
    }
}

export function kmeansAssignment2dBlock(data, centers, labels, sums, counts, start, end, d, k) {
  let changed = 0;
  for (let i = start; i < end; ++i) {
    const io = i * 2, x = data[io], y = data[io + 1];
    let best = 0, bestDistance = Infinity;
    for (let c = 0; c < k; ++c) {
      const co = c * 2, dx = x - centers[co], dy = y - centers[co + 1];
      const distance = dx * dx + dy * dy;
      if (distance > bestDistance) continue;
      if (distance > 0 && distance < MIN_NORMAL) throw new RangeError('squared distance is subnormal; rescale input');
      if (distance === 0 && (x !== centers[co] || y !== centers[co + 1])) throw new RangeError('squared distance underflow; rescale input');
      if (!Number.isFinite(distance)) throw new RangeError('squared distance overflow; rescale input');
      if (distance < bestDistance) { bestDistance = distance; best = c; }
    }
    if (labels[i] !== best) ++changed;
    labels[i] = best; ++counts[best];
    const offset = best * 2;
    sums[offset] += x; sums[offset + 1] += y;
  }
  return changed;
}

/** Four center distances share each feature load. Every distance retains
 * increasing-feature addition order. A group stops only when all four partial
 * sums exceed the current best, so exact first-index ties remain eligible.
 */
export function kmeansGroupedAssignmentBlock(data, centers, labels, sums, counts, start, end, d, k) {
 let changed=0;
 for(let i=start;i<end;++i) {
  const io=i*d; let best=0,bestDistance=Infinity,c=0;
  for(;c+3<k;c+=4) {
   const o0=c*d,o1=o0+d,o2=o1+d,o3=o2+d;
   let s0=0,s1=0,s2=0,s3=0,f=0;
   for(;f+3<d;f+=4) {
    const x0=data[io+f+0];
    const a00=x0-centers[o0+f+0];s0+=a00*a00;
    const a01=x0-centers[o1+f+0];s1+=a01*a01;
    const a02=x0-centers[o2+f+0];s2+=a02*a02;
    const a03=x0-centers[o3+f+0];s3+=a03*a03;
    const x1=data[io+f+1];
    const a10=x1-centers[o0+f+1];s0+=a10*a10;
    const a11=x1-centers[o1+f+1];s1+=a11*a11;
    const a12=x1-centers[o2+f+1];s2+=a12*a12;
    const a13=x1-centers[o3+f+1];s3+=a13*a13;
    const x2=data[io+f+2];
    const a20=x2-centers[o0+f+2];s0+=a20*a20;
    const a21=x2-centers[o1+f+2];s1+=a21*a21;
    const a22=x2-centers[o2+f+2];s2+=a22*a22;
    const a23=x2-centers[o3+f+2];s3+=a23*a23;
    const x3=data[io+f+3];
    const a30=x3-centers[o0+f+3];s0+=a30*a30;
    const a31=x3-centers[o1+f+3];s1+=a31*a31;
    const a32=x3-centers[o2+f+3];s2+=a32*a32;
    const a33=x3-centers[o3+f+3];s3+=a33*a33;
    if(s0>bestDistance&&s1>bestDistance&&s2>bestDistance&&s3>bestDistance)break;
   }
   if(s0>bestDistance&&s1>bestDistance&&s2>bestDistance&&s3>bestDistance)continue;
   for(;f<d;++f){const x=data[io+f];
    const a0=x-centers[o0+f];s0+=a0*a0;
    const a1=x-centers[o1+f];s1+=a1*a1;
    const a2=x-centers[o2+f];s2+=a2*a2;
    const a3=x-centers[o3+f];s3+=a3*a3;
    if(s0>bestDistance&&s1>bestDistance&&s2>bestDistance&&s3>bestDistance)break;
   }
   if(s0<=bestDistance){
    if(s0>0&&s0<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
    if(s0===0)for(let f=0;f<d;++f)if(data[io+f]!==centers[o0+f])throw new RangeError('squared distance underflow; rescale input');
    if(!Number.isFinite(s0))throw new RangeError('squared distance overflow; rescale input');
    if(s0<bestDistance){best=c+0;bestDistance=s0;}
   }
   if(s1<=bestDistance){
    if(s1>0&&s1<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
    if(s1===0)for(let f=0;f<d;++f)if(data[io+f]!==centers[o1+f])throw new RangeError('squared distance underflow; rescale input');
    if(!Number.isFinite(s1))throw new RangeError('squared distance overflow; rescale input');
    if(s1<bestDistance){best=c+1;bestDistance=s1;}
   }
   if(s2<=bestDistance){
    if(s2>0&&s2<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
    if(s2===0)for(let f=0;f<d;++f)if(data[io+f]!==centers[o2+f])throw new RangeError('squared distance underflow; rescale input');
    if(!Number.isFinite(s2))throw new RangeError('squared distance overflow; rescale input');
    if(s2<bestDistance){best=c+2;bestDistance=s2;}
   }
   if(s3<=bestDistance){
    if(s3>0&&s3<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
    if(s3===0)for(let f=0;f<d;++f)if(data[io+f]!==centers[o3+f])throw new RangeError('squared distance underflow; rescale input');
    if(!Number.isFinite(s3))throw new RangeError('squared distance overflow; rescale input');
    if(s3<bestDistance){best=c+3;bestDistance=s3;}
   }
  }
  for(;c<k;++c){const co=c*d;let sum=0;
   for(let f=0;f<d;++f){const delta=data[io+f]-centers[co+f];sum+=delta*delta;if(sum>bestDistance)break;}
   if(sum>bestDistance)continue;
   if(sum>0&&sum<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
   if(sum===0)for(let f=0;f<d;++f)if(data[io+f]!==centers[co+f])throw new RangeError('squared distance underflow; rescale input');
   if(!Number.isFinite(sum))throw new RangeError('squared distance overflow; rescale input');
   if(sum<bestDistance){best=c;bestDistance=sum;}
  }
  if(labels[i]!==best)++changed;labels[i]=best;++counts[best];
  const off=best*d;for(let f=0;f<d;++f)sums[off+f]+=data[io+f];
 }
 return changed;
}

export function kmeansFinalize2dBlock(data, centers, labels, start, end, d, k, initialInertia) {
  let inertia = initialInertia;
  for (let i = start; i < end; ++i) {
    const io = i * 2, x = data[io], y = data[io + 1];
    let best = 0, bestDistance = Infinity;
    for (let c = 0; c < k; ++c) {
      const co = c * 2, dx = x - centers[co], dy = y - centers[co + 1];
      const distance = dx * dx + dy * dy;
      if (distance > bestDistance) continue;
      if (distance > 0 && distance < MIN_NORMAL) throw new RangeError('squared distance is subnormal; rescale input');
      if (distance === 0 && (x !== centers[co] || y !== centers[co + 1])) throw new RangeError('squared distance underflow; rescale input');
      if (!Number.isFinite(distance)) throw new RangeError('squared distance overflow; rescale input');
      if (distance < bestDistance) { bestDistance = distance; best = c; }
    }
    labels[i] = best;
    inertia += bestDistance;
  }
  return inertia;
}


export function kmeansFinalizeGroupedBlock(data, centers, labels, start, end, d, k, initialInertia) {
 let inertia=initialInertia;
 for(let i=start;i<end;++i) {
  const io=i*d; let best=0,bestDistance=Infinity,c=0;
  for(;c+3<k;c+=4) {
   const o0=c*d,o1=o0+d,o2=o1+d,o3=o2+d;
   let s0=0,s1=0,s2=0,s3=0,f=0;
   for(;f+3<d;f+=4) {
    const x0=data[io+f+0];
    const a00=x0-centers[o0+f+0];s0+=a00*a00;
    const a01=x0-centers[o1+f+0];s1+=a01*a01;
    const a02=x0-centers[o2+f+0];s2+=a02*a02;
    const a03=x0-centers[o3+f+0];s3+=a03*a03;
    const x1=data[io+f+1];
    const a10=x1-centers[o0+f+1];s0+=a10*a10;
    const a11=x1-centers[o1+f+1];s1+=a11*a11;
    const a12=x1-centers[o2+f+1];s2+=a12*a12;
    const a13=x1-centers[o3+f+1];s3+=a13*a13;
    const x2=data[io+f+2];
    const a20=x2-centers[o0+f+2];s0+=a20*a20;
    const a21=x2-centers[o1+f+2];s1+=a21*a21;
    const a22=x2-centers[o2+f+2];s2+=a22*a22;
    const a23=x2-centers[o3+f+2];s3+=a23*a23;
    const x3=data[io+f+3];
    const a30=x3-centers[o0+f+3];s0+=a30*a30;
    const a31=x3-centers[o1+f+3];s1+=a31*a31;
    const a32=x3-centers[o2+f+3];s2+=a32*a32;
    const a33=x3-centers[o3+f+3];s3+=a33*a33;
    if(s0>bestDistance&&s1>bestDistance&&s2>bestDistance&&s3>bestDistance)break;
   }
   if(s0>bestDistance&&s1>bestDistance&&s2>bestDistance&&s3>bestDistance)continue;
   for(;f<d;++f){const x=data[io+f];
    const a0=x-centers[o0+f];s0+=a0*a0;
    const a1=x-centers[o1+f];s1+=a1*a1;
    const a2=x-centers[o2+f];s2+=a2*a2;
    const a3=x-centers[o3+f];s3+=a3*a3;
    if(s0>bestDistance&&s1>bestDistance&&s2>bestDistance&&s3>bestDistance)break;
   }
   if(s0<=bestDistance){
    if(s0>0&&s0<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
    if(s0===0)for(let f=0;f<d;++f)if(data[io+f]!==centers[o0+f])throw new RangeError('squared distance underflow; rescale input');
    if(!Number.isFinite(s0))throw new RangeError('squared distance overflow; rescale input');
    if(s0<bestDistance){best=c+0;bestDistance=s0;}
   }
   if(s1<=bestDistance){
    if(s1>0&&s1<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
    if(s1===0)for(let f=0;f<d;++f)if(data[io+f]!==centers[o1+f])throw new RangeError('squared distance underflow; rescale input');
    if(!Number.isFinite(s1))throw new RangeError('squared distance overflow; rescale input');
    if(s1<bestDistance){best=c+1;bestDistance=s1;}
   }
   if(s2<=bestDistance){
    if(s2>0&&s2<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
    if(s2===0)for(let f=0;f<d;++f)if(data[io+f]!==centers[o2+f])throw new RangeError('squared distance underflow; rescale input');
    if(!Number.isFinite(s2))throw new RangeError('squared distance overflow; rescale input');
    if(s2<bestDistance){best=c+2;bestDistance=s2;}
   }
   if(s3<=bestDistance){
    if(s3>0&&s3<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
    if(s3===0)for(let f=0;f<d;++f)if(data[io+f]!==centers[o3+f])throw new RangeError('squared distance underflow; rescale input');
    if(!Number.isFinite(s3))throw new RangeError('squared distance overflow; rescale input');
    if(s3<bestDistance){best=c+3;bestDistance=s3;}
   }
  }
  for(;c<k;++c){const co=c*d;let sum=0;
   for(let f=0;f<d;++f){const delta=data[io+f]-centers[co+f];sum+=delta*delta;if(sum>bestDistance)break;}
   if(sum>bestDistance)continue;
   if(sum>0&&sum<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
   if(sum===0)for(let f=0;f<d;++f)if(data[io+f]!==centers[co+f])throw new RangeError('squared distance underflow; rescale input');
   if(!Number.isFinite(sum))throw new RangeError('squared distance overflow; rescale input');
   if(sum<bestDistance){best=c;bestDistance=sum;}
  }
  labels[i]=best;inertia+=bestDistance;
 }
 return inertia;
}

/** Four independent prototypes share feature loads without changing any sum. */
export function somPrototypeGroupedBlock(data, p, r, numerator, denominator, v, start, end, d, m, q) {
  for (let i = start; i < end; ++i) {
    const io=i*d, po=i*m, vo=i*q;
    let j=0;
    for (;j+3<m;j+=4) {
      const p0=p[po+j],p1=p[po+j+1],p2=p[po+j+2],p3=p[po+j+3];
      const w0=j*d,w1=w0+d,w2=w1+d,w3=w2+d,r0=j*q,r1=r0+q,r2=r1+q,r3=r2+q;
      denominator[j]+=p0;denominator[j+1]+=p1;denominator[j+2]+=p2;denominator[j+3]+=p3;
      let f=0;
      for(;f+3<d;f+=4) {
        const x0=data[io+f+0];
        numerator[w0+f+0]+=p0*x0;
        numerator[w1+f+0]+=p1*x0;
        numerator[w2+f+0]+=p2*x0;
        numerator[w3+f+0]+=p3*x0;
        const x1=data[io+f+1];
        numerator[w0+f+1]+=p0*x1;
        numerator[w1+f+1]+=p1*x1;
        numerator[w2+f+1]+=p2*x1;
        numerator[w3+f+1]+=p3*x1;
        const x2=data[io+f+2];
        numerator[w0+f+2]+=p0*x2;
        numerator[w1+f+2]+=p1*x2;
        numerator[w2+f+2]+=p2*x2;
        numerator[w3+f+2]+=p3*x2;
        const x3=data[io+f+3];
        numerator[w0+f+3]+=p0*x3;
        numerator[w1+f+3]+=p1*x3;
        numerator[w2+f+3]+=p2*x3;
        numerator[w3+f+3]+=p3*x3;
      }
      for(;f<d;++f) {const x=data[io+f];numerator[w0+f]+=p0*x;numerator[w1+f]+=p1*x;numerator[w2+f]+=p2*x;numerator[w3+f]+=p3*x;}
      for(let h=0;h<q;++h) {let value=v[vo+h];value+=p0*r[r0+h];value+=p1*r[r1+h];value+=p2*r[r2+h];value+=p3*r[r3+h];v[vo+h]=value;}
    }
    for(;j<m;++j) {const value=p[po+j],wo=j*d,ro=j*q;denominator[j]+=value;for(let f=0;f<d;++f)numerator[wo+f]+=value*data[io+f];for(let h=0;h<q;++h)v[vo+h]+=value*r[ro+h];}
  }
}

/** Full direct distances for four units at once; softmax/objective order is unchanged. */
export function somMembershipGroupedBlock(data, w, v, r, p, cost, start, end, d, m, q, gamma, lambda, accum) {
  let distortion=accum.distortion,entropy=accum.entropy;
  for(let i=start;i<end;++i) {
    const io=i*d,vo=i*q,po=i*m;let minimum=Infinity,den=0,j=0;
    for(;j+3<m;j+=4) {
      const w0=j*d,w1=w0+d,w2=w1+d,w3=w2+d,r0=j*q,r1=r0+q,r2=r1+q,r3=r2+q;
      let s0=0,s1=0,s2=0,s3=0,f=0,g0=0,g1=0,g2=0,g3=0;
      for(;f+3<d;f+=4) {
        const x0=data[io+f+0];
        const a00=x0-w[w0+f+0];s0+=a00*a00;
        const a01=x0-w[w1+f+0];s1+=a01*a01;
        const a02=x0-w[w2+f+0];s2+=a02*a02;
        const a03=x0-w[w3+f+0];s3+=a03*a03;
        const x1=data[io+f+1];
        const a10=x1-w[w0+f+1];s0+=a10*a10;
        const a11=x1-w[w1+f+1];s1+=a11*a11;
        const a12=x1-w[w2+f+1];s2+=a12*a12;
        const a13=x1-w[w3+f+1];s3+=a13*a13;
        const x2=data[io+f+2];
        const a20=x2-w[w0+f+2];s0+=a20*a20;
        const a21=x2-w[w1+f+2];s1+=a21*a21;
        const a22=x2-w[w2+f+2];s2+=a22*a22;
        const a23=x2-w[w3+f+2];s3+=a23*a23;
        const x3=data[io+f+3];
        const a30=x3-w[w0+f+3];s0+=a30*a30;
        const a31=x3-w[w1+f+3];s1+=a31*a31;
        const a32=x3-w[w2+f+3];s2+=a32*a32;
        const a33=x3-w[w3+f+3];s3+=a33*a33;
      }
      for(;f<d;++f) {const x=data[io+f],a=x-w[w0+f],b=x-w[w1+f],c=x-w[w2+f],e=x-w[w3+f];s0+=a*a;s1+=b*b;s2+=c*c;s3+=e*e;}
      for(let h=0;h<q;++h) {const x=v[vo+h],a=x-r[r0+h],b=x-r[r1+h],c=x-r[r2+h],e=x-r[r3+h];g0+=a*a;g1+=b*b;g2+=c*c;g3+=e*e;}
      if(!Number.isFinite(s0)||!Number.isFinite(g0))throw new RangeError('SOM distance overflowed; rescale the data');
      const c0=s0+gamma*g0;if(!Number.isFinite(c0))throw new RangeError('SOM local cost overflowed; rescale the data/gamma');
      cost[j+0]=c0;if(c0<minimum)minimum=c0;
      if(!Number.isFinite(s1)||!Number.isFinite(g1))throw new RangeError('SOM distance overflowed; rescale the data');
      const c1=s1+gamma*g1;if(!Number.isFinite(c1))throw new RangeError('SOM local cost overflowed; rescale the data/gamma');
      cost[j+1]=c1;if(c1<minimum)minimum=c1;
      if(!Number.isFinite(s2)||!Number.isFinite(g2))throw new RangeError('SOM distance overflowed; rescale the data');
      const c2=s2+gamma*g2;if(!Number.isFinite(c2))throw new RangeError('SOM local cost overflowed; rescale the data/gamma');
      cost[j+2]=c2;if(c2<minimum)minimum=c2;
      if(!Number.isFinite(s3)||!Number.isFinite(g3))throw new RangeError('SOM distance overflowed; rescale the data');
      const c3=s3+gamma*g3;if(!Number.isFinite(c3))throw new RangeError('SOM local cost overflowed; rescale the data/gamma');
      cost[j+3]=c3;if(c3<minimum)minimum=c3;
    }
    for(;j<m;++j) {const value=somDistance(data,io,w,j*d,d)+gamma*somDistance(v,vo,r,j*q,q);if(!Number.isFinite(value))throw new RangeError('SOM local cost overflowed; rescale the data/gamma');cost[j]=value;if(value<minimum)minimum=value;}
    for(let j=0;j<m;++j) {const value=Math.exp(-(cost[j]-minimum)/lambda);p[po+j]=value;den+=value;}
    for(let j=0;j<m;++j) {const value=p[po+j]/den;p[po+j]=value;distortion+=value*cost[j];if(value>0)entropy+=value*Math.log(value);}
  }
  accum.distortion=distortion;accum.entropy=entropy;
}

/** Allocation-free BMU: original grouped KMeans lane arithmetic, with no
 * changes to the established KMeans hot loops. First-index ties and the
 * cutoff-sensitive underflow/overflow checks match core.squaredDistance. */
export function nearestCenterGrouped(data, io, centers, d, k) {
 let best=0,bestDistance=Infinity,c=0;
  for(;c+3<k;c+=4) {
   const o0=c*d,o1=o0+d,o2=o1+d,o3=o2+d;
   let s0=0,s1=0,s2=0,s3=0,f=0;
   for(;f+3<d;f+=4) {
    const x0=data[io+f+0];
    const a00=x0-centers[o0+f+0];s0+=a00*a00;
    const a01=x0-centers[o1+f+0];s1+=a01*a01;
    const a02=x0-centers[o2+f+0];s2+=a02*a02;
    const a03=x0-centers[o3+f+0];s3+=a03*a03;
    const x1=data[io+f+1];
    const a10=x1-centers[o0+f+1];s0+=a10*a10;
    const a11=x1-centers[o1+f+1];s1+=a11*a11;
    const a12=x1-centers[o2+f+1];s2+=a12*a12;
    const a13=x1-centers[o3+f+1];s3+=a13*a13;
    const x2=data[io+f+2];
    const a20=x2-centers[o0+f+2];s0+=a20*a20;
    const a21=x2-centers[o1+f+2];s1+=a21*a21;
    const a22=x2-centers[o2+f+2];s2+=a22*a22;
    const a23=x2-centers[o3+f+2];s3+=a23*a23;
    const x3=data[io+f+3];
    const a30=x3-centers[o0+f+3];s0+=a30*a30;
    const a31=x3-centers[o1+f+3];s1+=a31*a31;
    const a32=x3-centers[o2+f+3];s2+=a32*a32;
    const a33=x3-centers[o3+f+3];s3+=a33*a33;
    if(s0>bestDistance&&s1>bestDistance&&s2>bestDistance&&s3>bestDistance)break;
   }
   if(s0>bestDistance&&s1>bestDistance&&s2>bestDistance&&s3>bestDistance)continue;
   for(;f<d;++f){const x=data[io+f];
    const a0=x-centers[o0+f];s0+=a0*a0;
    const a1=x-centers[o1+f];s1+=a1*a1;
    const a2=x-centers[o2+f];s2+=a2*a2;
    const a3=x-centers[o3+f];s3+=a3*a3;
    if(s0>bestDistance&&s1>bestDistance&&s2>bestDistance&&s3>bestDistance)break;
   }
   if(s0<=bestDistance){
    if(s0>0&&s0<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
    if(s0===0)for(let f=0;f<d;++f)if(data[io+f]!==centers[o0+f])throw new RangeError('squared distance underflow; rescale input');
    if(!Number.isFinite(s0))throw new RangeError('squared distance overflow; rescale input');
    if(s0<bestDistance){best=c+0;bestDistance=s0;}
   }
   if(s1<=bestDistance){
    if(s1>0&&s1<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
    if(s1===0)for(let f=0;f<d;++f)if(data[io+f]!==centers[o1+f])throw new RangeError('squared distance underflow; rescale input');
    if(!Number.isFinite(s1))throw new RangeError('squared distance overflow; rescale input');
    if(s1<bestDistance){best=c+1;bestDistance=s1;}
   }
   if(s2<=bestDistance){
    if(s2>0&&s2<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
    if(s2===0)for(let f=0;f<d;++f)if(data[io+f]!==centers[o2+f])throw new RangeError('squared distance underflow; rescale input');
    if(!Number.isFinite(s2))throw new RangeError('squared distance overflow; rescale input');
    if(s2<bestDistance){best=c+2;bestDistance=s2;}
   }
   if(s3<=bestDistance){
    if(s3>0&&s3<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
    if(s3===0)for(let f=0;f<d;++f)if(data[io+f]!==centers[o3+f])throw new RangeError('squared distance underflow; rescale input');
    if(!Number.isFinite(s3))throw new RangeError('squared distance overflow; rescale input');
    if(s3<bestDistance){best=c+3;bestDistance=s3;}
   }
  }
  for(;c<k;++c){const co=c*d;let sum=0;
   for(let f=0;f<d;++f){const delta=data[io+f]-centers[co+f];sum+=delta*delta;if(sum>bestDistance)break;}
   if(sum>bestDistance)continue;
   if(sum>0&&sum<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
   if(sum===0)for(let f=0;f<d;++f)if(data[io+f]!==centers[co+f])throw new RangeError('squared distance underflow; rescale input');
   if(!Number.isFinite(sum))throw new RangeError('squared distance overflow; rescale input');
   if(sum<bestDistance){best=c;bestDistance=sum;}
  }
 return best;
}

/** Dedicated 2D BMU; no full row scratch, no distance-vector allocation. */
export function nearestCenter2d(data, io, centers, d, k) {
 const x=data[io],y=data[io+1];let best=0,bestDistance=Infinity;
 for(let c=0;c<k;c++) {
  const co=2*c,dx=x-centers[co],dy=y-centers[co+1],distance=dx*dx+dy*dy;
  if(distance>bestDistance)continue;
  if(distance>0&&distance<MIN_NORMAL)throw new RangeError('squared distance is subnormal; rescale input');
  if(distance===0&&(x!==centers[co]||y!==centers[co+1]))throw new RangeError('squared distance underflow; rescale input');
  if(!Number.isFinite(distance))throw new RangeError('squared distance overflow; rescale input');
  if(distance<bestDistance){bestDistance=distance;best=c;}
 }
 return best;
}
