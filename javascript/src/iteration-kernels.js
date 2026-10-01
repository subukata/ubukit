/** Allocation-free row blocks shared by synchronous/cooperative execution.
 * Accumulators enter and leave each block without reordering floating sums.
 * Cancellation and events remain at the caller's existing block boundaries.
 */
const MIN_NORMAL = 2.2250738585072014e-308;

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

export function fcmCenterBlock(data, u, sums, numerator, start, end, d, k, m) {
  for (let i = start; i < end; ++i) {
    const io = i * d, off = i * k;
    for (let c = 0; c < k; ++c) {
      const value = u[off + c], weight = m === 2 ? value * value : value ** m, co = c * d;
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

export function fcmMembershipBlock(data, centers, old, next, dist, start, end, d, k, m, accum) {
  let delta2 = accum.delta2, objective = accum.objective;
  const power = 1 / (m - 1);
  for (let i = start; i < end; ++i) {
    const io = i * d, off = i * k; let minimum = Infinity, sum = 0;
    for (let c = 0; c < k; ++c) {
      const distance = fullSquaredDistance(data, io, centers, c * d, d);
      dist[c] = distance; minimum = Math.min(minimum, distance);
    }
    if (minimum === 0) {
      for (let c = 0; c < k; ++c) { const weight = dist[c] === 0 ? 1 : 0; next[off + c] = weight; sum += weight; }
    } else if (m === 2) {
      for (let c = 0; c < k; ++c) { const weight = minimum / dist[c]; next[off + c] = weight; sum += weight; }
    } else {
      for (let c = 0; c < k; ++c) { const ratio = minimum / dist[c]; const weight = ratio < MIN_NORMAL ? Math.exp((Math.log(minimum) - Math.log(dist[c])) * power) : ratio ** power; next[off + c] = weight; sum += weight; }
    }
    for (let c = 0; c < k; ++c) {
      const value = next[off + c] / sum; next[off + c] = value;
      const diff = value - old[off + c]; delta2 += diff * diff;
      const distance = dist[c];
      if (value !== 0 && distance !== 0) {
        const weight = m === 2 ? value * value : value ** m;
        objective += weight < MIN_NORMAL ? Math.exp(m * Math.log(value) + Math.log(distance)) : weight * distance;
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
