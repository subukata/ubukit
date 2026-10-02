/** Traditional online SOM and exact frozen-assignment BatchSOM.
 * Rectangular, non-periodic, x-fast lattice; Gaussian neighborhoods.
 * No sample×unit×feature tensor, no runtime dependencies.
 */
import { normalizeInput, positiveInteger, finiteNumber, seededRandom, assertFinite, squaredDistance, checkCancelled, progress } from './core.js';
import { somSteps as eliteSomSteps, somBatchSteps as eliteSomBatchSteps } from './som-elite.js';
import { nearestCenter2d, nearestCenterGrouped } from './iteration-kernels.js';
import { checkpoint } from './session-hooks.js';
import { pcaPrototypes } from './som-olp.js';
import { somStablePcaInput, somStableMean } from './som-numerics.js';

const int = (v, name, min = 0) => { if (!Number.isSafeInteger(v) || v < min) throw new RangeError(`${name} must be a safe integer >= ${min}`); return v; };
const finite = (v, name, min = 0) => finiteNumber(v, name, min);
const array = (v, length, name) => {
  if (!ArrayBuffer.isView(v) || v instanceof DataView || typeof v[0] === 'bigint' || v.length !== length) throw new RangeError(`${name} must be a numeric TypedArray with ${length} entries`);
  return assertFinite(v, name);
};
export function somSchedule(start, end, t, total, schedule = 'geometric') {
  const a = total <= 1 ? 0 : t / (total - 1);
  if (a === 0) return start;
  if (a === 1) return end;
  return schedule === 'linear' || start === 0 || end === 0 ? (1 - a) * start + a * end : Math.exp((1 - a) * Math.log(start) + a * Math.log(end));
}
/** Validation is shared by one-shot kernels and transactional session updates. */
export function validateSOM(input, options = {}, batch = false) {
  const x = normalizeInput(input), n = x.nSamples, d = x.nFeatures;
  if (!options || typeof options !== 'object' || Array.isArray(options)) throw new TypeError('options must be an object');
  const shape = options.gridShape ?? [16, 16];
  if ((!Array.isArray(shape) && !ArrayBuffer.isView(shape)) || shape.length !== 2) throw new RangeError('gridShape must be [width, height]');
  const width = positiveInteger(shape[0], 'grid width'), height = positiveInteger(shape[1], 'grid height');
  const k = positiveInteger(width * height, 'grid size');
  if (k > 0x7fffffff) throw new RangeError('grid size exceeds Int32 labels');
  const epochs = int(options.epochs ?? 100, 'epochs');
  const total = int(options.maxIterations ?? (batch ? epochs : epochs * n), 'maxIterations');
  const sigma = finite(options.sigma ?? Math.max(width, height) / 2, 'sigma');
  const sigmaEnd = finite(options.sigmaEnd ?? Math.min(.5, sigma), 'sigmaEnd');
  const learningRate = finite(options.learningRate ?? .5, 'learningRate');
  const learningRateEnd = finite(options.learningRateEnd ?? Math.min(.05, learningRate), 'learningRateEnd');
  if (sigmaEnd > sigma || learningRateEnd > learningRate) throw new RangeError('SOM schedules must be nonincreasing');
  if (learningRate > 1 || learningRateEnd > 1) throw new RangeError('learning rates must be in [0, 1]');
  if (batch && (Object.hasOwn(options, 'learningRate') || Object.hasOwn(options, 'learningRateEnd'))) throw new RangeError('BatchSOM uses exact weighted means, not a learning rate');
  if (!['geometric', 'linear'].includes(options.schedule ?? 'geometric')) throw new RangeError("schedule must be 'geometric' or 'linear'");
  if (!['sample', 'pca'].includes(options.initializer ?? 'sample')) throw new RangeError("initializer must be 'sample' or 'pca'");
  if (!['auto', 'scalar', 'grouped'].includes(options.bmuBackend ?? 'auto')) throw new RangeError("bmuBackend must be 'auto', 'scalar', or 'grouped'");
  if (options.kernelBackend != null && options.kernelBackend !== 'javascript') throw new RangeError("Traditional SOM currently uses kernelBackend:'javascript'");
  if (options.shuffle != null && options.shuffle !== false) throw new RangeError('SOM currently uses sequential input order; shuffle must be false');
  if ((options.tolerance ?? 0) !== 0) throw new RangeError('SOM uses a fixed schedule; tolerance must be 0');
  for (const key of ['initialMemberships', 'initMembership', 'grid']) if (Object.hasOwn(options, key)) throw new RangeError(`${key} is not a traditional SOM parameter; use gridShape and initialPrototypes`);
  if (options.nClusters != null && options.nClusters !== k) throw new RangeError('nClusters must equal gridShape width * height');
  const seed = int(options.seed ?? 0, 'seed'); if (seed > 0xffffffff) throw new RangeError('seed must be a uint32'); seededRandom(seed);
  const blockRows = positiveInteger(options.blockRows ?? 128, 'blockRows');
  const blockUnits = positiveInteger(options.blockUnits ?? 32, 'blockUnits');
  const maxMemoryBytes = positiveInteger(options.maxMemoryBytes ?? 512 * 1024 ** 2, 'maxMemoryBytes');
  const maxScratchBytes = positiveInteger(options.maxScratchBytes ?? 32 * 1024 ** 2, 'maxScratchBytes');
  positiveInteger(options.pcaMaxDimension ?? 128, 'pcaMaxDimension');
  positiveInteger(options.pcaMaxSweeps ?? 80, 'pcaMaxSweeps');
  finite(options.pcaTolerance ?? 1e-13, 'pcaTolerance'); finite(options.pcaScale ?? 2, 'pcaScale');
  const initial = options.initialPrototypes ?? options.initCenters;
  if (initial != null) array(initial, k * d, 'initialPrototypes');
  const usesPca = initial == null && options.initializer === 'pca', dim = Math.min(n, d), q = Math.min(2, n, d);
  if (usesPca && dim > (options.pcaMaxDimension ?? 128)) throw new RangeError('PCA dimension exceeds pcaMaxDimension; use sample or explicit prototypes');
  // Includes owned X, two center buffers, final labels and lattice embedding;
  // batch adds grouped sums/counts, one separable intermediate, two 1-D kernels.
  const trainScratch = batch ? 16 * k * (d + 1) + 8 * (width * width + height * height) + 8 * n + 16 * d : 0;
  const pcaScratch = usesPca ? 16 * dim * dim + 8 * (n * d + 3 * d + 2 * q + k * q) + 4 * dim : 0;
  const estimatedPrimaryBytes = 8 * n * d + 16 * k * d + 20 * n + Math.max(trainScratch, pcaScratch);
  if (![k * d, n * d, estimatedPrimaryBytes, trainScratch, pcaScratch].every(Number.isSafeInteger)) throw new RangeError('SOM allocation size exceeds safe integer range');
  if (Math.max(trainScratch, pcaScratch) > maxScratchBytes) throw new RangeError('SOM scratch exceeds maxScratchBytes');
  if (estimatedPrimaryBytes > maxMemoryBytes) throw new RangeError(`SOM primary arrays need ${estimatedPrimaryBytes} bytes, above maxMemoryBytes`);
  return { x, n, d, k, q: 2, width, height, total, epochs, sigma, sigmaEnd, learningRate, learningRateEnd, seed, initial, blockRows, blockUnits, maxMemoryBytes, maxScratchBytes, trainScratch, pcaScratch, estimatedPrimaryBytes, maxIterations: total };
}
function* emit(options, event) { progress(options, event); yield event; }
function bmu(data, offset, w, d, k) {
  let best = 0, distance = Infinity;
  for (let j = 0; j < k; ++j) { const value = squaredDistance(data, offset, w, j * d, d, distance); if (value < distance) { distance = value; best = j; } }
  return best;
}
function gaussian(distance, sigma) {
  if (distance === 0) return 1;
  if (sigma === 0) return 0;
  const ratio = distance / sigma;
  return Math.exp(-.5 * ratio * ratio);
}
function interpolate(a, b, weight) {
  if (weight === 0 || a === b) return a;
  if (weight === 1) return b;
  const oldWeight = 1 - weight, value = oldWeight * a + weight * b;
  if (!Number.isFinite(value) || a * Math.sign(b) < 0 && Math.abs(value) < Math.max(Math.abs(a), Math.abs(b)) * 1e-8) {
    return somStableMean(Float64Array.of(a, b), 0, 1, 2, Float64Array.of(oldWeight, weight));
  }
  return Math.min(Math.max(a, b), Math.max(Math.min(a, b), value));
}
function weightedAverage(a, b, wa, wb, total) {
  if (wa === 0 || a === b) return b;
  const extreme = v => v !== 0 && (Math.abs(v) < 1e-100 || Math.abs(v) > 1e100);
  if (extreme(a) || extreme(b) || extreme(wa) || extreme(wb)) {
    return somStableMean(Float64Array.of(a, b), 0, 1, 2, Float64Array.of(wa, wb));
  }
  // Two independently normalized terms preserve small old-weight contributions
  // even when the new-weight fraction rounds to one.
  return Math.min(Math.max(a, b), Math.max(Math.min(a, b), a * (wa / total) + b * (wb / total)));
}
function* train(input, options, batch) {
  const cfg = validateSOM(input, options, batch);
  const { n, d, k, width, height, total, initial, blockRows, blockUnits } = cfg;
  const algorithm = batch ? 'som_batch' : 'som', unit = batch ? 'epoch' : 'sample';
  // Retain the original cutoff BMU as an explicit elite fallback.
  // Small grids and higher-dimensional cutoff-friendly data retain the
  // established scalar route; grouped is an explicit option for wider data.
  const groupedBMU = options.bmuBackend === 'grouped' || ((options.bmuBackend ?? 'auto') === 'auto' && k >= 16 && d <= 8);
  // Direct iterators, as well as sessions, own their numerical inputs after start.
  const data = Float64Array.from(cfg.x.data), x = { data, nSamples: n, nFeatures: d };
  let w = initial == null ? new Float64Array(k * d) : Float64Array.from(initial), next = new Float64Array(k * d);
  let initialization = { kind: 'provided-prototypes' };
  checkCancelled(options);
  if (initial == null && options.initializer === 'pca') {
    const q = Math.min(2, n, d), gridData = new Float64Array(k * q);
    for (let j = 0; j < k; j++) { gridData[j * q] = j % width; if (q === 2) gridData[j * q + 1] = Math.floor(j / width); }
    const scaled = somStablePcaInput(data, n, d);
    const pcaOptions = { ...options, onProgress: undefined };
    const it = pcaPrototypes(scaled, { data: gridData, nSamples: k, nFeatures: q }, w, pcaOptions, cfg.maxScratchBytes);
    for (;;) { const step = it.next(); if (step.done) { initialization = step.value; break; } yield* emit(options, { ...step.value, algorithm }); }
    for (let j = 0; j < k; j++) for (let f = 0; f < d; f++) w[j * d + f] = scaled.mean[f] + w[j * d + f] * scaled.scale;
    assertFinite(w, 'PCA prototypes');
  } else if (initial == null) {
    const rng = seededRandom(cfg.seed);
    for (let j = 0; j < k; j++) { const i = Math.floor(rng() * n); w.set(data.subarray(i * d, (i + 1) * d), j * d); }
    initialization = { kind: 'sample-with-replacement', seed: cfg.seed, rng: 'mulberry32' };
  }
  // Packed sufficient statistics: each unit has D means followed by its count.
  // Mean/count pairs avoid overflowing raw sums and retain tiny-mass means.
  const labels = new Int32Array(n);
  let stableBatch = false, usedStableRepair = false;
  const featureMagnitude = batch ? new Float64Array(d) : null, signedFeature = batch ? new Float64Array(d) : null;
  if (batch) for (let f = 0; f < d; f++) {
    let largest = 0, smallest = Infinity, low = Infinity, high = -Infinity;
    for (let i = 0; i < n; i++) { const value = Math.abs(data[i * d + f]); largest = Math.max(largest, value); low = Math.min(low, data[i * d + f]); high = Math.max(high, data[i * d + f]); if (value > 0) smallest = Math.min(smallest, value); }
    featureMagnitude[f] = largest; signedFeature[f] = low < 0 && high > 0 ? 1 : 0;
    stableBatch ||= largest >= 1e100 || (smallest < 1e-100) || largest / smallest >= 1e12;
  }
  let repairWeights = null;
  const stride = d + 1, groups = batch ? new Float64Array(k * stride) : null, horizontal = batch ? new Float64Array(k * stride) : null;
  const hx = batch ? new Float64Array(width * width) : null, hy = batch ? new Float64Array(height * height) : null;
  const state = (iterations, extra = {}) => ({ algorithm, centers: w, prototypes: w, W: w, labels: null, embedding: null, projectionStatus: 'not-computed',
    iterations, nIter: iterations, unit, iterationUnit: unit, maxIterations: total, samplesSeen: batch ? iterations * n : iterations,
    epochsCompleted: batch ? iterations : Math.floor(iterations / n), converged: false, nSamples: n, nFeatures: d,
    nUnits: k, nClusters: k, nComponents: 2, gridShape: [width, height], initialization, ...extra });
  let last = { sigma: null, learningRate: null, sampleIndex: null, bmu: null };
  for (let t = 0; t < total; t++) {
    checkCancelled(options);
    const sigma = somSchedule(cfg.sigma, cfg.sigmaEnd, t, total, options.schedule);
    if (!batch) {
      const i = t % n, best = (groupedBMU ? (d === 2 ? nearestCenter2d(data, i * d, w, d, k) : nearestCenterGrouped(data, i * d, w, d, k)) : bmu(data, i * d, w, d, k)), bx = best % width, by = Math.floor(best / width);
      const eta = somSchedule(cfg.learningRate, cfg.learningRateEnd, t, total, options.schedule);
      for (let start = 0; start < k; start += blockUnits) {
        checkCancelled(options);
        for (let j = start; j < Math.min(k, start + blockUnits); j++) {
          const h = gaussian(Math.hypot(j % width - bx, Math.floor(j / width) - by), sigma), weight = eta * h;
          for (let f = 0; f < d; f++) next[j * d + f] = interpolate(w[j * d + f], data[i * d + f], weight);
        }
        if (start + blockUnits < k) yield* emit(options, { algorithm, phase: 'update-prototypes', iteration: t + 1, unit, completed: Math.min(k, start + blockUnits), total: k });
      }
      last = { sigma, learningRate: eta, sampleIndex: i, bmu: best };
    } else {
      groups.fill(0); horizontal.fill(0);
      // All BMUs read frozen w. The next buffer is not visible until complete.
      for (let start = 0; start < n; start += blockRows) {
        checkCancelled(options);
        for (let i = start; i < Math.min(n, start + blockRows); i++) {
          const best = (groupedBMU ? (d === 2 ? nearestCenter2d(data, i * d, w, d, k) : nearestCenterGrouped(data, i * d, w, d, k)) : bmu(data, i * d, w, d, k)), offset = best * stride; labels[i] = best;
          const previous = groups[offset + d], count = previous + 1;
          for (let f = 0; f < d; f++) groups[offset + f] = weightedAverage(groups[offset + f], data[i * d + f], previous, 1, count);
          groups[offset + d] = count;
        }
        yield* emit(options, { algorithm, phase: 'frozen-bmu-assignment', iteration: t + 1, unit, completed: Math.min(n, start + blockRows), total: n });
      }
      assertFinite(groups, 'BatchSOM sufficient statistics');
      for (let a = 0; a < width; a++) for (let b = 0; b < width; b++) hx[a * width + b] = gaussian(Math.abs(a - b), sigma);
      for (let a = 0; a < height; a++) for (let b = 0; b < height; b++) hy[a * height + b] = gaussian(Math.abs(a - b), sigma);
      // H((x,y),(a,b))=Hx(x,a)*Hy(y,b): exact separable Gaussian smoothing.
      for (let start = 0; start < k; start += blockUnits) {
        checkCancelled(options);
        for (let j = start; j < Math.min(k, start + blockUnits); j++) {
          const y = Math.floor(j / width), target = j * stride, row = (j % width) * width;
          for (let a = 0; a < width; a++) {
            const source = (y * width + a) * stride, weight = hx[row + a] * groups[source + d];
            if (weight === 0) continue;
            const previous = horizontal[target + d], totalWeight = previous + weight;
            for (let f = 0; f < d; f++) horizontal[target + f] = weightedAverage(horizontal[target + f], groups[source + f], previous, weight, totalWeight);
            horizontal[target + d] = totalWeight;
          }
        }
        yield* emit(options, { algorithm, phase: 'batch-horizontal', iteration: t + 1, unit, completed: Math.min(k, start + blockUnits), total: k });
      }
      for (let start = 0; start < k; start += blockUnits) {
        checkCancelled(options);
        for (let j = start; j < Math.min(k, start + blockUnits); j++) {
          const xpos = j % width, row = Math.floor(j / width) * height;
          let denominator = 0;
          next.set(w.subarray(j * d, (j + 1) * d), j * d);
          for (let b = 0; b < height; b++) {
            const source = (b * width + xpos) * stride, weight = hy[row + b] * horizontal[source + d];
            if (weight === 0) continue;
            const totalWeight = denominator + weight;
            for (let f = 0; f < d; f++) next[j * d + f] = weightedAverage(next[j * d + f], horizontal[source + f], denominator, weight, totalWeight);
            denominator = totalWeight;
          }
        }
        if (start + blockUnits < k) yield* emit(options, { algorithm, phase: 'batch-vertical', iteration: t + 1, unit, completed: Math.min(k, start + blockUnits), total: k });
      }
      // Original-sample repair is also necessary when signed smoothing nearly
      // cancels: a tiny residual can already be lost inside a rounded group.
      for (let j = 0; j < k; j++) {
        let repair = stableBatch;
        for (let f = 0; !repair && f < d; f++) repair ||= signedFeature[f] !== 0 && Math.abs(next[j * d + f]) < featureMagnitude[f] * 1e-8;
        if (!repair) continue;
        usedStableRepair = true; repairWeights ??= new Float64Array(n);
        checkCancelled(options); let mass = 0;
        for (let i = 0; i < n; i++) { const best = labels[i], h = gaussian(Math.hypot(j % width - best % width, Math.floor(j / width) - Math.floor(best / width)), sigma); repairWeights[i] = h; mass += h; }
        if (mass > 0) for (let f = 0; f < d; f++) {
          if (stableBatch || signedFeature[f] !== 0 && Math.abs(next[j * d + f]) < featureMagnitude[f] * 1e-8) next[j * d + f] = somStableMean(data, f, d, n, repairWeights);
        }
        yield* emit(options, { algorithm, phase: 'batch-stable-mean', iteration: t + 1, unit, completed: j + 1, total: k });
      }
      last = { sigma, learningRate: null, sampleIndex: null, bmu: null };
    }
    assertFinite(next, 'SOM updated prototypes'); checkCancelled(options);
    [w, next] = [next, w];
    checkpoint(options, state(t + 1, last));
    yield* emit(options, { algorithm, phase: 'iteration', iteration: t + 1, unit, maxIterations: total, ...last });
  }
  const embedding = new Float64Array(n * 2);
  for (let start = 0; start < n; start += blockRows) {
    checkCancelled(options);
    for (let i = start; i < Math.min(n, start + blockRows); i++) { const best = (groupedBMU ? (d === 2 ? nearestCenter2d(data, i * d, w, d, k) : nearestCenterGrouped(data, i * d, w, d, k)) : bmu(data, i * d, w, d, k)); labels[i] = best; embedding[2 * i] = best % width; embedding[2 * i + 1] = Math.floor(best / width); }
    yield* emit(options, { algorithm, phase: 'final-labels', unit, completed: Math.min(n, start + blockRows), total: n });
  }
  return state(total, { ...last, labels, embedding, V: embedding, projectionStatus: 'current-prototypes', stats: { stableBatchRepair: usedStableRepair, estimatedPrimaryBytes: cfg.estimatedPrimaryBytes, primaryTrainingScratchBytes: cfg.trainScratch, blockRows, blockUnits,
    complexity: batch ? 'O(epochs*(N*M*D+M*D*(width+height))); O(N*D+M*D+width^2+height^2+N) space' : 'O(updates*M*D+N*M*D); O(N*D+M*D+N) space',
    training: batch ? 'frozen BMUs; grouped sufficient statistics; separable Gaussian weighted means' : 'sequential samples; eta*h*(x-w)' } });
}
/** The unmodified full scalar kernel remains its own JIT compilation unit.
 * Lightweight shape dispatch does not duplicate input scans or allocations;
 * the selected implementation performs full numerical and option validation.
 * Malformed shapes use the original validator. Invalid route names use the
 * candidate validator so the additive option is rejected consistently. */
function useEliteSOM(input, options) {
  const route = options?.bmuBackend ?? 'auto';
  if (route === 'scalar') return true;
  if (route !== 'auto') return false;
  const d = input?.nFeatures, shape = options?.gridShape;
  if (!Number.isSafeInteger(d) || d < 1 || d > 8) return true;
  if (shape == null) return false; // The validated default is 16 by 16.
  if ((!Array.isArray(shape) && !ArrayBuffer.isView(shape)) || shape.length !== 2) return true;
  const width = shape[0], height = shape[1];
  if (!Number.isSafeInteger(width) || width < 1 || !Number.isSafeInteger(height) || height < 1) return true;
  const k = width * height;
  return !Number.isSafeInteger(k) || k < 16 || k > 0x7fffffff;
}
export function* somSteps(input, options = {}) { return yield* (useEliteSOM(input, options) ? eliteSomSteps(input, options) : train(input, options, false)); }
export function* somBatchSteps(input, options = {}) { return yield* (useEliteSOM(input, options) ? eliteSomBatchSteps(input, options) : train(input, options, true)); }
const consume = it => { for (;;) { const step = it.next(); if (step.done) return step.value; } };
export const som = (input, options = {}) => consume(somSteps(input, options));
export const somBatch = (input, options = {}) => consume(somBatchSteps(input, options));
export const som_batch = somBatch;

/** Explicit fresh BMU projection, O(N*M*D); no training or model mutation. */
export function somProject(input, model, options = {}) {
  if (!model || typeof model !== 'object' || (model.centers ?? model.prototypes) == null || model.gridShape == null) throw new TypeError('model must have centers/prototypes and gridShape');
  const x = normalizeInput(input);
  if (model.nFeatures != null && model.nFeatures !== x.nFeatures) throw new RangeError('Projection feature count differs from model');
  const result = som(x, { ...options, gridShape: model.gridShape, initialPrototypes: model.centers ?? model.prototypes, maxIterations: 0 });
  return { labels: result.labels, embedding: result.embedding, projectionStatus: 'current-prototypes' };
}

// Internal differential-test entry point; package root exports stay unchanged.
export { bmu as somBmuScalar };
