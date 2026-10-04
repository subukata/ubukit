import { somNeedsStableRange, somNeedsStableGamma, somStableCost, somRepairPrototypes, somStableMembershipBlock, somStableObjective, somStablePcaInput, somGridNormalization, somGridFactor } from './som-numerics.js';
import { somTrainingWorkspace, normalizeInitialCosts, normalizeMembershipCosts } from './som-training-wasm.js';
import { pcaRotationWorkspace } from './som-pca-wasm.js';
import { somPrototypeBlock, somMembershipBlock, somPrototype2dBlock, somMembership2dBlock, somPrototypeGroupedBlock, somMembershipGroupedBlock } from './iteration-kernels.js';
import { SESSION_CHECKPOINT } from './session-hooks.js';
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
export function* pcaPrototypes(x, grid, w, options, maxScratchBytes, extraMemoryBytes = Infinity, execution = null) {
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
  const rotation = options.kernelBackend === 'wasm' ? pcaRotationWorkspace(dim, maxScratchBytes - (8 * (2 * d + 2 * q) + 4 * dim), extraMemoryBytes) : null;
  if(execution){execution.pca=rotation?'wasm':'javascript';execution.pcaWorkspaceBytes=rotation?.allocatedBytes??0;execution.pcaExtraBytes=rotation?rotation.allocatedBytes-16*dim*dim:0;}
  const gram = rotation?.gram ?? new Float64Array(dim * dim), vectors = rotation?.vectors ?? new Float64Array(dim * dim);
  const dual = n < d;
  for (let a = 0; a < dim; a++) {
    checkCancelled(options); vectors[a * dim + a] = 1;
    let b = a;
    for (; b + 3 < dim; b += 4) {
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      if (dual) {
        const ao = a * d, b0 = b * d, b1 = b0 + d, b2 = b1 + d, b3 = b2 + d;
        for (let f = 0; f < d; f++) {
          const origin = mean[f], av = data[ao + f] - origin;
          s0 += av * (data[b0 + f] - origin); s1 += av * (data[b1 + f] - origin);
          s2 += av * (data[b2 + f] - origin); s3 += av * (data[b3 + f] - origin);
        }
      } else {
        const ma = mean[a], m0 = mean[b], m1 = mean[b + 1], m2 = mean[b + 2], m3 = mean[b + 3];
        for (let i = 0; i < n; i++) {
          const io = i * d, av = data[io + a] - ma;
          s0 += av * (data[io + b] - m0); s1 += av * (data[io + b + 1] - m1);
          s2 += av * (data[io + b + 2] - m2); s3 += av * (data[io + b + 3] - m3);
        }
      }
      const v0 = s0 / n, v1 = s1 / n, v2 = s2 / n, v3 = s3 / n;
      if (!Number.isFinite(v0) || !Number.isFinite(v1) || !Number.isFinite(v2) || !Number.isFinite(v3)) throw new RangeError('PCA covariance overflowed; rescale the data');
      gram[a * dim + b] = gram[b * dim + a] = v0;
      gram[a * dim + b + 1] = gram[(b + 1) * dim + a] = v1;
      gram[a * dim + b + 2] = gram[(b + 2) * dim + a] = v2;
      gram[a * dim + b + 3] = gram[(b + 3) * dim + a] = v3;
    }
    for (; b < dim; b++) {
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
        if (rotation) rotation.rotate(a, b, c, s);
        else for (let j = 0; j < dim; j++) {
          if (j !== a && j !== b) {
            const ja = gram[a * dim + j], jb = gram[b * dim + j];
            const nextA = c * ja - s * jb, nextB = s * ja + c * jb;
            gram[j * dim + a] = gram[a * dim + j] = nextA;
            gram[j * dim + b] = gram[b * dim + j] = nextB;
          }
          const va = vectors[a * dim + j], vb = vectors[b * dim + j];
          vectors[a * dim + j] = c * va - s * vb; vectors[b * dim + j] = s * va + c * vb;
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
  const stableGrid = somNeedsStableRange(r);
  if (stableGrid) somGridNormalization(r, m, q, rMean, rMax);
  else for (let h = 0; h < q; h++) {
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
          for (let i = 0; i < n; i++) sum += (data[i * d + f] - mean[f]) * vectors[eigenIndex * dim + i];
          loading[f] = sum / den;
        }
      } else for (let f = 0; f < d; f++) loading[f] = vectors[eigenIndex * dim + f];
    }
    let largest = 0;
    for (let f = 1; f < d; f++) if (Math.abs(loading[f]) > Math.abs(loading[largest])) largest = f;
    const signedScale = scale * Math.sqrt(variance) * (loading[largest] < 0 ? -1 : 1);
    for (let j = 0; j < m; j++) {
      const factor = (stableGrid ? somGridFactor(r[j * q + h], rMean[h], rMax[h]) : (r[j * q + h] - rMean[h]) / rMax[h]) * signedScale;
      for (let f = 0; f < d; f++) w[j * d + f] += factor * loading[f];
    }
  }
  assertFinite(w, 'PCA prototypes');
  return { kind: 'pca-jacobi', numpySvdParity: false, signConvention: 'largest-absolute-loading-positive',
    eigensystemDimension: dim, dual, sweeps, primaryScratchBytes: scratchBytes };
}

export function* somOlpSteps(input, options = {}) {
  const x = normalizeInput(input), grid = normalizeInput(options.grid);
  const kernelBackend = options.kernelBackend ?? 'javascript';
  if(!['javascript','wasm'].includes(kernelBackend))throw new RangeError("kernelBackend must be 'javascript' or 'wasm'");
  const execution={requested:kernelBackend,pca:'not-used',initialMembership:'provided',training:'not-run',pcaWorkspaceBytes:0,pcaExtraBytes:0,trainingWorkspaceBytes:0,trainingExtraBytes:0,wasmInitializationRows:0,wasmPrototypeRows:0,wasmMembershipRows:0,javascriptFallbackRows:0};
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
  const stableData = somNeedsStableRange(data), stableGrid = somNeedsStableRange(r);
  const usesPca = initialW == null && (options.initializer ?? 'pca') === 'pca';
  const stablePcaBytes = usesPca && stableData ? 8 * (n * d + d) : 0;
  const pcaScratchBytes = initialW == null && (options.initializer ?? 'pca') === 'pca'
    ? 16 * pcaDimension * pcaDimension + 8 * (2 * d + 2 * q) + 4 * pcaDimension + stablePcaBytes : 0;
  const estimatedPrimaryBytes = 8 * (n * d + m * q + m * d + n * m + (maxIterations > 0 ? n * q : 0) + maxIterations)
    + 4 * n + Math.max(trainingScratchBytes, pcaScratchBytes);
  if (estimatedPrimaryBytes > maxMemoryBytes) throw new RangeError(`SOM estimated primary arrays need ${estimatedPrimaryBytes} bytes, above maxMemoryBytes=${maxMemoryBytes}`);
  if (initialP != null && initialW == null) throw new RangeError('initialMemberships requires initialPrototypes to preserve empty-unit semantics');
  const w = initialW == null ? new Float64Array(m * d) : copyArray(initialW, m * d, 'initialPrototypes');
  const p = initialP == null ? new Float64Array(n * m) : copyArray(initialP, n * m, 'initialMemberships');
  let initialization;
  checkCancelled(options);
  if (initialW != null) initialization = { kind: initialP == null ? 'provided-prototypes' : 'provided-prototypes-and-memberships', numpySvdParity: 'caller-supplied' };
  else if ((options.initializer ?? 'pca') === 'pca') {
    if (stablePcaBytes > 0) {
      if (pcaScratchBytes > maxScratchBytes) throw new RangeError(`PCA primary scratch needs ${pcaScratchBytes} bytes`);
      const scaled = somStablePcaInput(data, n, d);
      initialization = yield* pcaPrototypes(scaled, grid, w, options, maxScratchBytes - stablePcaBytes, maxMemoryBytes - estimatedPrimaryBytes, execution);
      for (let j = 0; j < m; j++) for (let f = 0; f < d; f++) w[j * d + f] = scaled.mean[f] + w[j * d + f] * scaled.scale;
      assertFinite(w, 'PCA prototypes');
      initialization.primaryScratchBytes += stablePcaBytes;
    } else initialization = yield* pcaPrototypes(x, grid, w, options, maxScratchBytes, maxMemoryBytes - estimatedPrimaryBytes, execution);
  }
  else if (options.initializer === 'sample') {
    const rng = seededRandom(options.seed ?? 0);
    // Independent samples with replacement, intentionally not NumPy RNG/PCA.
    for (let j = 0; j < m; j++) { const index = Math.floor(rng() * n); w.set(data.subarray(index * d, (index + 1) * d), j * d); }
    initialization = { kind: 'sample-with-replacement', seed: options.seed ?? 0, numpySvdParity: false };
  } else throw new RangeError("initializer must be 'pca' or 'sample'");
  // SIMD kernels intentionally preserve the ordinary arithmetic. Exceptional
  // ranges use the same original-unit model through the JS recovery path.
  const stableRange = stableData || stableGrid || somNeedsStableRange(w);
  const stableCosts = stableRange || somNeedsStableGamma(gamma);
  const initialDistance = stableRange ? somStableCost : distance;
  let trainingWasm = null, workspaceAttempted = false;
  const allocateWorkspace = () => {
    if(kernelBackend === 'wasm' && !stableCosts && !workspaceAttempted){workspaceAttempted=true;trainingWasm=somTrainingWorkspace(n,d,m,q,blockRows,maxMemoryBytes-estimatedPrimaryBytes,maxScratchBytes);}
  };
  const cost = new Float64Array(m);
  if (initialP == null) {
    allocateWorkspace();execution.initialMembership=trainingWasm?'wasm':'javascript';
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      if (!trainingWasm && kernelBackend === 'wasm') execution.javascriptFallbackRows += Math.min(n, start + blockRows) - start;
      if(!trainingWasm)for (let i = start; i < Math.min(n, start + blockRows); i++) {
        for (let j = 0; j < m; j++) cost[j] = initialDistance(data, i * d, w, j * d, d);
        probabilityRow(cost, p, i * m, lambda);
      }
      else for(let from=start;from<Math.min(n,start+blockRows);from+=trainingWasm.rows){
        const end=Math.min(n,start+blockRows,from+trainingWasm.rows);trainingWasm.initialMembership(data,w,from,end);execution.wasmInitializationRows+=end-from;
        normalizeInitialCosts(trainingWasm.costs,trainingWasm.flags,p,from,end,m,lambda);
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
  if(maxIterations>0)allocateWorkspace();
  execution.training=maxIterations===0?'not-run':trainingWasm?'wasm':'javascript';execution.trainingWorkspaceBytes=trainingWasm?.allocatedBytes??0;execution.trainingExtraBytes=trainingWasm?.extraBytes??0;
  execution.primaryTrainingScratchBytes=trainingWasm?trainingWasm.allocatedBytes+8*m:trainingScratchBytes;execution.estimatedPeakPrimaryBytes=estimatedPrimaryBytes+Math.max(execution.pcaExtraBytes,execution.trainingExtraBytes);
  const numerator = trainingWasm?.numerator ?? new Float64Array(m * d), denominator = trainingWasm?.denominator ?? new Float64Array(m);
  const history = [];
  const accum = { distortion: 0, entropy: 0 };
  const prototypeBlock = d === 2 && q === 2 ? somPrototype2dBlock : m < 4 ? somPrototypeBlock : somPrototypeGroupedBlock;
  const membershipBlock = stableCosts ? somStableMembershipBlock : d === 2 && q === 2 ? somMembership2dBlock : m < 4 ? somMembershipBlock : somMembershipGroupedBlock;
  let converged = false;
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    checkCancelled(options); numerator.fill(0); denominator.fill(0); v.fill(0);
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      if(!trainingWasm)prototypeBlock(data, p, r, numerator, denominator, v, start, Math.min(n, start + blockRows), d, m, q);
      else for(let from=start;from<Math.min(n,start+blockRows);from+=trainingWasm.rows){const end=Math.min(n,start+blockRows,from+trainingWasm.rows);trainingWasm.prototype(data,p,r,v,from,end);execution.wasmPrototypeRows+=end-from;}
      yield* emit(options, { algorithm: 'som-olp', phase: 'update-prototypes', iteration: iteration + 1, completed: Math.min(n, start + blockRows), total: n });
    }
    for (let j = 0; j < m; j++) if (denominator[j] > 0) for (let f = 0; f < d; f++) w[j * d + f] = numerator[j * d + f] / denominator[j];
    somRepairPrototypes(data, p, r, w, v, denominator, n, d, m, q, stableRange);
    assertFinite(w, 'SOM prototypes'); assertFinite(v, 'SOM embedding');
    accum.distortion = 0; accum.entropy = 0;
    if (stableCosts) accum.exactObjective = null;
    for (let start = 0; start < n; start += blockRows) {
      checkCancelled(options);
      if (!trainingWasm && kernelBackend === 'wasm') execution.javascriptFallbackRows += Math.min(n, start + blockRows) - start;
      if(!trainingWasm)membershipBlock(data, w, v, r, p, cost, start, Math.min(n, start + blockRows), d, m, q, gamma, lambda, accum);
      else for(let from=start;from<Math.min(n,start+blockRows);from+=trainingWasm.rows){
        const end=Math.min(n,start+blockRows,from+trainingWasm.rows);trainingWasm.membership(data,w,v,r,from,end,gamma);execution.wasmMembershipRows+=end-from;
        normalizeMembershipCosts(trainingWasm.costs,trainingWasm.flags,p,from,end,m,lambda,accum);
      }
      yield* emit(options, { algorithm: 'som-olp', phase: 'update-memberships', iteration: iteration + 1, completed: Math.min(n, start + blockRows), total: n });
    }
    const objective = stableCosts ? somStableObjective(accum) : accum.distortion + lambda * accum.entropy;
    if (!Number.isFinite(objective)) throw new RangeError('SOM objective overflowed; rescale the data');
    history.push(objective);
    converged = false;
    if (iteration > 0) {
      const previous = history[iteration - 1], denominator = Math.max(1, Math.abs(previous));
      const difference = Math.abs(objective - previous);
      // Both objectives are finite, but their opposite-sign difference may
      // overflow. Preserve the original-unit ratio without that intermediate.
      const relative = Number.isFinite(difference) ? difference / denominator
        : Math.abs(objective / denominator - previous / denominator);
      converged = relative <= tolerance;
    }
    // Keep history for convergence/final output, but do not snapshot it unused.
    const checkpointHook = options[SESSION_CHECKPOINT];
    if (checkpointHook != null) Reflect.apply(checkpointHook, options, [{ algorithm: 'som-olp', prototypes: w, centers: w, W: w, memberships: p, membership: p, P: p, embedding: v, V: v, history: Float64Array.from(history), objective, iterations: iteration + 1, nIter: iteration + 1, converged, nSamples: n, nFeatures: d, nUnits: m, nClusters: m, nComponents: q, initialization }, null]);
    yield* emit(options, { algorithm: 'som-olp', phase: 'iteration', iteration: iteration + 1, objective, maxIterations });
    if (converged) break;
  }
  const labels = labelsFromMembership(p, n, m);
  return { algorithm: 'som-olp', prototypes: w, centers: w, W: w,
    memberships: p, membership: p, P: p, embedding: v, V: v, labels,
    history: Float64Array.from(history), iterations: history.length, nIter: history.length,
    converged, nSamples: n, nFeatures: d, nUnits: m, nClusters: m, nComponents: q,
    initialization, ...(options.kernelBackend != null ? {kernel:execution} : {}), stats: { primaryTrainingScratchBytes: trainingScratchBytes, estimatedPrimaryBytes, blockRows,
      membershipBytes: p.byteLength, embeddingBytes: v?.byteLength ?? 0,
      returnOrder: 'W and V from pre-update P; P from final softmax',
      complexity: 'O(iterations * N * M * (D + Q)) time; O(N*M + N*Q + M*D) state' } };
}

export function somOlp(input, options = {}) {
  const steps = somOlpSteps(input, options);
  for (;;) { const step = steps.next(); if (step.done) return step.value; }
}
