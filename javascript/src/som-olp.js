import { somPrototypeBlock, somMembershipBlock } from './iteration-kernels.js';
import { checkpoint } from './session-hooks.js';
/** SOM-OLP, ported from Seiki Ubukata's MIT-licensed implementation.
 * Upstream commit 4361175b776987d65c348d0132b31d43505e1069.
 * Preserves old-P -> V/W -> new-P -> objective -> stopping order.
 * PCA initialization is a JS eigensolver, not NumPy LAPACK SVD bit parity.
 * Supply initialPrototypes AND initialMemberships for cross-language kernel parity.
 */
import { normalizeInput, checkCancelled, progress, seededRandom, assertFinite, labelsFromMembership } from './core.js';

function integer(value, name, min = 1) {
  if (!Number.isSafeInteger(value) || value < min) throw new RangeError(`${name} must be an integer >= ${min}`);
  return value;
}
function finite(value, name, min = 0) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min) throw new RangeError(`${name} must be finite and >= ${min}`);
  return value;
}
function copyArray(value, length, name) {
  if (!ArrayBuffer.isView(value) || value instanceof DataView || typeof value[0] === 'bigint' || value.length !== length) {
    throw new RangeError(`${name} must be a numeric TypedArray with ${length} entries`);
  }
  return assertFinite(Float64Array.from(value), name);
}
function distance(a, ao, b, bo, d) {
  let sum = 0;
  for (let f = 0; f < d; f++) { const delta = a[ao + f] - b[bo + f]; sum += delta * delta; }
  if (!Number.isFinite(sum)) throw new RangeError('SOM distance overflowed; rescale the data');
  return sum;
}
function probabilityRow(cost, p, offset, lambda) {
  let min = Infinity;
  for (let j = 0; j < cost.length; j++) if (cost[j] < min) min = cost[j];
  let den = 0;
  for (let j = 0; j < cost.length; j++) { const v = Math.exp(-(cost[j] - min) / lambda); p[offset + j] = v; den += v; }
  for (let j = 0; j < cost.length; j++) p[offset + j] /= den;
}
function* emit(options, event) { progress(options, event); yield event; }

// Cyclic symmetric Jacobi eigensolver. PCA uses Xc'Xc/N (or its dual), sorted
// descending; each principal loading's largest-magnitude entry is positive.
function* pcaPrototypes(x, grid, w, options, maxScratchBytes) {
  const { data, nSamples: n, nFeatures: d } = x;
  const { data: r, nSamples: m, nFeatures: q } = grid;
  const dim = Math.min(n, d), components = Math.min(q, d);
  if (components > Math.min(n, d)) throw new RangeError('Too few samples for the requested PCA grid rank');
  const dimensionCap = integer(options.pcaMaxDimension ?? 128, 'pcaMaxDimension');
  if (dim > dimensionCap) throw new RangeError(`PCA eigensystem dimension ${dim} exceeds pcaMaxDimension ${dimensionCap}; supply initialPrototypes/initialMemberships, choose initializer:'sample', or explicitly raise the cap`);
  const scratchBytes = 16 * dim * dim + 8 * (2 * d + 2 * q) + 4 * dim;
  if (scratchBytes > maxScratchBytes) throw new RangeError(`PCA primary scratch needs ${scratchBytes} bytes`);
  const scale = finite(options.pcaScale ?? 2, 'pcaScale', -Infinity);
  const mean = new Float64Array(d);
  // Translation before averaging avoids unnecessary cancellation for large offsets.
  for (let f = 0; f < d; f++) {
    const origin = data[f]; let sum = 0;
    for (let i = 0; i < n; i++) sum += data[i * d + f] - origin;
    mean[f] = origin + sum / n;
  }
  const gram = new Float64Array(dim * dim), vectors = new Float64Array(dim * dim);
  const dual = n < d;
  for (let a = 0; a < dim; a++) {
    checkCancelled(options); vectors[a * dim + a] = 1;
    for (let b = a; b < dim; b++) {
      let sum = 0;
      if (dual) for (let f = 0; f < d; f++) sum += (data[a * d + f] - mean[f]) * (data[b * d + f] - mean[f]);
      else for (let i = 0; i < n; i++) sum += (data[i * d + a] - mean[a]) * (data[i * d + b] - mean[b]);
      const value = sum / n;
      if (!Number.isFinite(value)) throw new RangeError('PCA covariance overflowed; rescale the data');
      gram[a * dim + b] = value; gram[b * dim + a] = value;
    }
    if ((a + 1) % 8 === 0 || a + 1 === dim) yield* emit(options, { algorithm: 'som-olp', phase: 'pca-covariance', completed: a + 1, total: dim });
  }
  let diagonalScale = 0;
  for (let a = 0; a < dim; a++) diagonalScale = Math.max(diagonalScale, Math.abs(gram[a * dim + a]));
  const threshold = (options.pcaTolerance ?? 1e-13) * diagonalScale;
  finite(options.pcaTolerance ?? 1e-13, 'pcaTolerance');
  const maxSweeps = integer(options.pcaMaxSweeps ?? 80, 'pcaMaxSweeps');
  let converged = dim === 1 || diagonalScale === 0, sweeps = 0;
  while (!converged && sweeps < maxSweeps) {
    checkCancelled(options);
    for (let a = 0; a < dim - 1; a++) {
      for (let b = a + 1; b < dim; b++) {
        const ab = gram[a * dim + b];
        if (Math.abs(ab) <= threshold) continue;
        const tau = (gram[b * dim + b] - gram[a * dim + a]) / (2 * ab);
        const t = tau === 0 ? 1 : Math.sign(tau) / (Math.abs(tau) + Math.hypot(1, tau));
        const c = 1 / Math.sqrt(1 + t * t), s = t * c;
        gram[a * dim + a] -= t * ab; gram[b * dim + b] += t * ab;
        gram[a * dim + b] = 0; gram[b * dim + a] = 0;
        for (let j = 0; j < dim; j++) {
          if (j !== a && j !== b) {
            const ja = gram[j * dim + a], jb = gram[j * dim + b];
            const nextA = c * ja - s * jb, nextB = s * ja + c * jb;
            gram[j * dim + a] = gram[a * dim + j] = nextA;
            gram[j * dim + b] = gram[b * dim + j] = nextB;
          }
          const va = vectors[j * dim + a], vb = vectors[j * dim + b];
          vectors[j * dim + a] = c * va - s * vb; vectors[j * dim + b] = s * va + c * vb;
        }
      }
    }
    sweeps++;
    let maxOff = 0;
    for (let a = 0; a < dim - 1; a++) for (let b = a + 1; b < dim; b++) maxOff = Math.max(maxOff, Math.abs(gram[a * dim + b]));
    converged = maxOff <= threshold;
    yield* emit(options, { algorithm: 'som-olp', phase: 'pca-eigen', completed: sweeps, total: maxSweeps });
  }
  if (!converged) throw new Error('PCA eigensolver did not converge; increase pcaMaxSweeps or supply initialization');
  const sorted = Uint32Array.from({ length: dim }, (_, i) => i);
  sorted.sort((a, b) => gram[b * dim + b] - gram[a * dim + a] || a - b);
  const rMean = new Float64Array(q), rMax = new Float64Array(q);
  for (let h = 0; h < q; h++) {
    for (let j = 0; j < m; j++) rMean[h] += r[j * q + h];
    rMean[h] /= m;
    for (let j = 0; j < m; j++) rMax[h] = Math.max(rMax[h], Math.abs(r[j * q + h] - rMean[h]));
    rMax[h] = Math.max(rMax[h], 1e-12);
  }
  for (let j = 0; j < m; j++) w.set(mean, j * d);
  const loading = new Float64Array(d);
  for (let h = 0; h < components; h++) {
    const eigenIndex = sorted[h], variance = Math.max(0, gram[eigenIndex * dim + eigenIndex]);
    loading.fill(0);
    if (variance > 0) {
      if (dual) {
        const den = Math.sqrt(n * variance);
        for (let f = 0; f < d; f++) {
          let sum = 0;
          for (let i = 0; i < n; i++) sum += (data[i * d + f] - mean[f]) * vectors[i * dim + eigenIndex];
          loading[f] = sum / den;
        }
      } else for (let f = 0; f < d; f++) loading[f] = vectors[f * dim + eigenIndex];
    }
    let largest = 0;
    for (let f = 1; f < d; f++) if (Math.abs(loading[f]) > Math.abs(loading[largest])) largest = f;
    const signedScale = scale * Math.sqrt(variance) * (loading[largest] < 0 ? -1 : 1);
    for (let j = 0; j < m; j++) {
      const factor = (r[j * q + h] - rMean[h]) / rMax[h] * signedScale;
      for (let f = 0; f < d; f++) w[j * d + f] += factor * loading[f];
    }
  }
  assertFinite(w, 'PCA prototypes');
  return { kind: 'pca-jacobi', numpySvdParity: false, signConvention: 'largest-absolute-loading-positive',
    eigensystemDimension: dim, dual, sweeps, primaryScratchBytes: scratchBytes };
}

export function* somOlpSteps(input, options = {}) {
  const x = normalizeInput(input), grid = normalizeInput(options.grid);
  const { data, nSamples: n, nFeatures: d } = x;
  const { data: r, nSamples: m, nFeatures: q } = grid;
  const gamma = finite(options.gamma ?? 1, 'gamma');
  const lambda = finite(options.lambda ?? options.lam ?? 0.1, 'lambda');
  if (lambda === 0) throw new RangeError('lambda must be greater than zero');
  const tolerance = finite(options.tolerance ?? 1e-4, 'tolerance');
  const maxIterations = integer(options.maxIterations ?? 100, 'maxIterations', 0);
  const blockRows = integer(options.blockRows ?? 128, 'blockRows');
  const maxScratchBytes = integer(options.maxScratchBytes ?? 32 * 1024 * 1024, 'maxScratchBytes');
  if (![n * m, m * d, n * q].every(Number.isSafeInteger)) throw new RangeError('SOM shape exceeds safe integer range');
  const trainingScratchBytes = 8 * (m * d + 2 * m);
  if (trainingScratchBytes > maxScratchBytes) throw new RangeError(`SOM primary training scratch needs ${trainingScratchBytes} bytes`);
  const initialW = options.initialPrototypes ?? options.initCenters;
  const initialP = options.initialMemberships ?? options.initMembership;
  const maxMemoryBytes = integer(options.maxMemoryBytes ?? 512 * 1024 * 1024, 'maxMemoryBytes');
  const pcaDimension = Math.min(n, d);
  const pcaScratchBytes = initialW == null && (options.initializer ?? 'pca') === 'pca'
    ? 16 * pcaDimension * pcaDimension + 8 * (2 * d + 2 * q) + 4 * pcaDimension : 0;
  const estimatedPrimaryBytes = 8 * (n * d + m * q + m * d + n * m + (maxIterations > 0 ? n * q : 0) + maxIterations)
    + 4 * n + Math.max(trainingScratchBytes, pcaScratchBytes);
  if (estimatedPrimaryBytes > maxMemoryBytes) throw new RangeError(`SOM estimated primary arrays need ${estimatedPrimaryBytes} bytes, above maxMemoryBytes=${maxMemoryBytes}`);
  if (initialP != null && initialW == null) throw new RangeError('initialMemberships requires initialPrototypes to preserve empty-unit semantics');
  const w = initialW == null ? new Float64Array(m * d) : copyArray(initialW, m * d, 'initialPrototypes');
  const p = initialP == null ? new Float64Array(n * m) : copyArray(initialP, n * m, 'initialMemberships');
  let initialization;
  checkCancelled(options);
  if (initialW != null) initialization = { kind: initialP == null ? 'provided-prototypes' : 'provided-prototypes-and-memberships', numpySvdParity: 'caller-supplied' };
  else if ((options.initializer ?? 'pca') === 'pca') initialization = yield* pcaPrototypes(x, grid, w, options, maxScratchBytes);
  else if (options.initializer === 'sample') {
    const rng = seededRandom(options.seed ?? 0);
    // Independent samples with replacement, intentionally not NumPy RNG/PCA.
    for (let j = 0; j < m; j++) { const index = Math.floor(rng() * n); w.set(data.subarray(index * d, (index + 1) * d), j * d); }
    initialization = { kind: 'sample-with-replacement', seed: options.seed ?? 0, numpySvdParity: false };
  } else throw new RangeError("initializer must be 'pca' or 'sample'");
  const cost = new Float64Array(m);
  if (initialP == null) {
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      for (let i = start; i < Math.min(n, start + blockRows); i++) {
        for (let j = 0; j < m; j++) cost[j] = distance(data, i * d, w, j * d, d);
        probabilityRow(cost, p, i * m, lambda);
      }
      yield* emit(options, { algorithm: 'som-olp', phase: 'initialize-memberships', completed: Math.min(n, start + blockRows), total: n });
    }
  } else {
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < m; j++) { const value = p[i * m + j]; if (value < 0) throw new RangeError('initialMemberships must be nonnegative'); sum += value; }
      if (Math.abs(sum - 1) > 1e-8) throw new RangeError('Each initialMemberships row must sum to one within 1e-8');
    }
  }
  const v = maxIterations === 0 ? null : new Float64Array(n * q);
  const numerator = new Float64Array(m * d), denominator = new Float64Array(m);
  const history = [];
  const accum = { distortion: 0, entropy: 0 };
  let converged = false;
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    checkCancelled(options); numerator.fill(0); denominator.fill(0); v.fill(0);
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      somPrototypeBlock(data, p, r, numerator, denominator, v, start, Math.min(n, start + blockRows), d, m, q);
      yield* emit(options, { algorithm: 'som-olp', phase: 'update-prototypes', iteration: iteration + 1, completed: Math.min(n, start + blockRows), total: n });
    }
    for (let j = 0; j < m; j++) if (denominator[j] > 0) for (let f = 0; f < d; f++) w[j * d + f] = numerator[j * d + f] / denominator[j];
    assertFinite(w, 'SOM prototypes'); assertFinite(v, 'SOM embedding');
    accum.distortion = 0; accum.entropy = 0;
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      somMembershipBlock(data, w, v, r, p, cost, start, Math.min(n, start + blockRows), d, m, q, gamma, lambda, accum);
      yield* emit(options, { algorithm: 'som-olp', phase: 'update-memberships', iteration: iteration + 1, completed: Math.min(n, start + blockRows), total: n });
    }
    const objective = accum.distortion + lambda * accum.entropy;
    if (!Number.isFinite(objective)) throw new RangeError('SOM objective overflowed; rescale the data');
    history.push(objective);
    converged = iteration > 0 && Math.abs(objective - history[iteration - 1]) / Math.max(1, Math.abs(history[iteration - 1])) <= tolerance;
    checkpoint(options, { algorithm: 'som-olp', prototypes: w, centers: w, W: w, memberships: p, membership: p, P: p, embedding: v, V: v, history: Float64Array.from(history), objective, iterations: iteration + 1, nIter: iteration + 1, converged, nSamples: n, nFeatures: d, nUnits: m, nClusters: m, nComponents: q, initialization });
    yield* emit(options, { algorithm: 'som-olp', phase: 'iteration', iteration: iteration + 1, objective, maxIterations });
    if (converged) break;
  }
  const labels = labelsFromMembership(p, n, m);
  return { algorithm: 'som-olp', prototypes: w, centers: w, W: w,
    memberships: p, membership: p, P: p, embedding: v, V: v, labels,
    history: Float64Array.from(history), iterations: history.length, nIter: history.length,
    converged, nSamples: n, nFeatures: d, nUnits: m, nClusters: m, nComponents: q,
    initialization, stats: { primaryTrainingScratchBytes: trainingScratchBytes, estimatedPrimaryBytes, blockRows,
      membershipBytes: p.byteLength, embeddingBytes: v?.byteLength ?? 0,
      returnOrder: 'W and V from pre-update P; P from final softmax',
      complexity: 'O(iterations * N * M * (D + Q)) time; O(N*M + N*Q + M*D) state' } };
}

export function somOlp(input, options = {}) {
  const steps = somOlpSteps(input, options);
  for (;;) { const step = steps.next(); if (step.done) return step.value; }
}
