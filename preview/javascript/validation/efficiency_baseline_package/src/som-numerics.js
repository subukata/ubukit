/** Exceptional SOM arithmetic. Ordinary kernels deliberately keep their order. */
const LOW = 1e-100, HIGH = 1e100;
export function somNeedsStableRange(values) {
  for (let i = 0; i < values.length; i++) {
    const magnitude = Math.abs(values[i]);
    if (magnitude > HIGH || (magnitude !== 0 && magnitude < LOW)) return true;
  }
  return false;
}
export function somNeedsStableGamma(gamma) {
  return gamma !== 0 && (gamma < LOW || gamma > HIGH);
}

/** Weighted squared norm, with gamma applied before any square is formed.
 * A scaled sum of squares also recovers sums of individually underflowing
 * squares. Unrepresentable positive composite costs are rejected, never tied.
 */
export function somStableCost(a, ao, b, bo, d, v = null, vo = 0, r = null, ro = 0, q = 0, gamma = 0) {
  let scale = 0, sum = 1, nonzero = false;
  const root = Math.sqrt(gamma);
  for (let part = 0; part < (gamma === 0 ? 1 : 2); part++) {
    const left = part ? v : a, right = part ? r : b;
    const lo = part ? vo : ao, rr = part ? ro : bo;
    const count = part ? q : d, weight = part ? root : 1;
    for (let f = 0; f < count; f++) {
      const x = left[lo + f], y = right[rr + f];
      if (x === y) continue;
      nonzero = true;
      const delta = x - y;
      // Opposite finite endpoints may overflow subtraction; scaling each
      // endpoint first is safe whenever the weighted difference is finite.
      const value = Math.abs(Number.isFinite(delta) ? delta * weight : x * weight - y * weight);
      if (!Number.isFinite(value)) throw new RangeError('SOM local cost overflowed; rescale the data/gamma');
      if (value === 0) continue;
      if (scale < value) { const ratio = scale / value; sum = 1 + sum * ratio * ratio; scale = value; }
      else { const ratio = value / scale; sum += ratio * ratio; }
    }
  }
  // This order does not form scale squared before multiplication by sum.
  const cost = scale * (scale * sum);
  if (!Number.isFinite(cost)) throw new RangeError('SOM local cost overflowed; rescale the data/gamma');
  if (cost === 0 && nonzero) throw new RangeError('SOM positive local cost underflowed; rescale the data/gamma/lambda');
  return cost;
}

// Exact binary accumulation is confined to the cold mean/embedding path.
// Coordinate scaling or log products can lose tiny residuals before very
// large signed terms cancel, even when the final mean is representable.
const partsView = new DataView(new ArrayBuffer(8));
function binaryParts(value) {
  partsView.setFloat64(0, value, false);
  const high = partsView.getUint32(0, false), low = partsView.getUint32(4, false);
  const exponent = (high >>> 20) & 2047;
  let integer = (BigInt(high & 0xfffff) << 32n) | BigInt(low);
  if (exponent) integer |= 1n << 52n;
  if (high >>> 31) integer = -integer;
  return [integer, exponent ? exponent - 1075 : -1074];
}
function accumulator() { return { integer: 0n, exponent: 0 }; }
function addBinary(sum, integer, exponent) {
  if (integer === 0n) return;
  if (sum.integer === 0n) { sum.integer = integer; sum.exponent = exponent; }
  else if (exponent < sum.exponent) {
    sum.integer = (sum.integer << BigInt(sum.exponent - exponent)) + integer;
    sum.exponent = exponent;
  } else sum.integer += integer << BigInt(exponent - sum.exponent);
}
function addProduct(sum, value, weight = 1) {
  const [integer, exponent] = binaryParts(value);
  if (weight === 1) { addBinary(sum, integer, exponent); return; }
  const [wi, we] = binaryParts(weight);
  addBinary(sum, integer * wi, exponent + we);
}
function binaryPair(sum) {
  if (sum.integer === 0n) return [0, 0];
  const sign = sum.integer < 0n ? -1 : 1;
  const magnitude = sum.integer < 0n ? -sum.integer : sum.integer;
  const bits = magnitude.toString(2).length;
  let mantissa;
  if (bits > 53) {
    const shift = BigInt(bits - 53); let top = magnitude >> shift;
    const remainder = magnitude - (top << shift), half = 1n << (shift - 1n);
    if (remainder > half || (remainder === half && (top & 1n))) top += 1n;
    mantissa = Number(top) * 2 ** -53;
  } else mantissa = Number(magnitude) * 2 ** -bits;
  return [sign * mantissa, sum.exponent + bits];
}
function binaryScale(value, exponent) {
  while (exponent > 1023) { value *= 2 ** 1023; exponent -= 1023; }
  while (exponent < -1022) { value *= 2 ** -1022; exponent += 1022; }
  return value * 2 ** exponent;
}

/** Cold weighted mean. Products and summation retain binary operands before
 * final float64 division; no caller-owned arrays are changed. */
export function somStableMean(data, offset, stride, count, weights = null, weightOffset = 0, weightStride = 1) {
  let minimum = Infinity, maximum = -Infinity;
  const numerator = accumulator(), denominator = accumulator();
  for (let i = 0; i < count; i++) {
    const weight = weights ? weights[weightOffset + i * weightStride] : 1;
    if (weight <= 0) continue;
    const value = data[offset + i * stride];
    minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
    addProduct(numerator, value, weight); addProduct(denominator, weight);
  }
  if (minimum === Infinity) return NaN;
  if (minimum === maximum) return minimum;
  const [nm, ne] = binaryPair(numerator), [dm, de] = binaryPair(denominator);
  const result = binaryScale(nm / dm, ne - de);
  // A convex mean is within its endpoints; clamp only rounding overshoot.
  return Math.min(maximum, Math.max(minimum, result));
}

/** V is a weighted sum, not a renormalized mean: initial P row sums are
 * accepted within 1e-8, so retain their original mathematical semantics. */
function stableWeightedSum(data, offset, stride, count, weights, weightOffset) {
  const sum = accumulator();
  for (let j = 0; j < count; j++) {
    const weight = weights[weightOffset + j];
    if (weight > 0) addProduct(sum, data[offset + j * stride], weight);
  }
  const [mantissa, exponent] = binaryPair(sum);
  const result = binaryScale(mantissa, exponent);
  if (!Number.isFinite(result)) throw new RangeError('SOM embedding overflowed; rescale the grid');
  return result;
}

export function somRepairPrototypes(data, p, r, w, v, denominator, n, d, m, q, stableRange) {
  for (let j = 0; j < m; j++) if (denominator[j] > 0) {
    // Tiny cluster mass can underflow p*x even for otherwise ordinary data.
    let repair = stableRange || denominator[j] < LOW;
    for (let f = 0; !repair && f < d; f++) repair = !Number.isFinite(w[j * d + f]);
    if (repair) for (let f = 0; f < d; f++) w[j * d + f] = somStableMean(data, f, d, n, p, j, m);
  }
  if (stableRange) for (let i = 0; i < n; i++) for (let h = 0; h < q; h++) v[i * q + h] = stableWeightedSum(r, h, q, m, p, i * m);
}

export function somStableMembershipBlock(data, w, v, r, p, cost, start, end, d, m, q, gamma, lambda, accum) {
  const objective = accum.exactObjective ?? (accum.exactObjective = accumulator());
  for (let i = start; i < end; i++) {
    const po = i * m; let minimum = Infinity, den = 0;
    for (let j = 0; j < m; j++) {
      const value = somStableCost(data, i * d, w, j * d, d, v, i * q, r, j * q, q, gamma);
      cost[j] = value; minimum = Math.min(minimum, value);
    }
    let units = 0, tail = 0, tailError = 0;
    for (let j = 0; j < m; j++) {
      const value = Math.exp(-(cost[j] - minimum) / lambda); p[po + j] = value;
      if (value === 1) units++;
      else {
        const next = tail + value;
        tailError += Math.abs(tail) >= Math.abs(value) ? (tail - next) + value : (value - next) + tail;
        tail = next;
      }
    }
    den = units + (tail + tailError);
    for (let j = 0; j < m; j++) p[po + j] /= den;
    // The minimized row objective preserves a small softmax tail even when
    // the published maximum probability rounds to exactly one.
    const logTotal = Math.log(units) + Math.log1p((tail + tailError) / units);
    addProduct(objective, minimum);
    addProduct(objective, lambda, -logTotal);
  }
}

export function somStableObjective(accum) {
  const [mantissa, exponent] = binaryPair(accum.exactObjective);
  return binaryScale(mantissa, exponent);
}

/** Center and scale only PCA's temporary input. This is not model scaling:
 * gamma, lambda, distances and returned history retain the original units. */
export function somStablePcaInput(data, n, d) {
  const mean = new Float64Array(d), centered = new Float64Array(data.length);
  for (let f = 0; f < d; f++) mean[f] = somStableMean(data, f, d, n);
  let scale = 0, finiteDifferences = true;
  for (let i = 0; i < n; i++) for (let f = 0; f < d; f++) {
    const delta = data[i * d + f] - mean[f];
    finiteDifferences &&= Number.isFinite(delta); scale = Math.max(scale, Math.abs(delta));
  }
  if (!finiteDifferences) {
    scale = 0;
    for (let i = 0; i < data.length; i++) scale = Math.max(scale, Math.abs(data[i]));
  }
  if (scale === 0) scale = 1;
  for (let i = 0; i < n; i++) for (let f = 0; f < d; f++) centered[i * d + f] = finiteDifferences
    ? (data[i * d + f] - mean[f]) / scale : data[i * d + f] / scale - mean[f] / scale;
  return { data: centered, nSamples: n, nFeatures: d, mean, scale };
}

export function somGridNormalization(r, m, q, mean, maximum) {
  for (let h = 0; h < q; h++) {
    mean[h] = somStableMean(r, h, q, m);
    for (let j = 0; j < m; j++) maximum[h] = Math.max(maximum[h], Math.abs(r[j * q + h] - mean[h]));
    if (!Number.isFinite(maximum[h])) {
      let half = 0;
      for (let j = 0; j < m; j++) half = Math.max(half, Math.abs(r[j * q + h] / 2 - mean[h] / 2));
      maximum[h] = -half; // Negative marks a half-scale denominator.
    } else maximum[h] = Math.max(maximum[h], 1e-12);
  }
}
export function somGridFactor(value, mean, maximum) {
  if (maximum >= 0) return (value - mean) / maximum;
  // Only opposite extreme endpoints can overflow the centered grid span.
  // A half-scale difference and denominator represent the same ratio.
  return (value / 2 - mean / 2) / -maximum;
}
