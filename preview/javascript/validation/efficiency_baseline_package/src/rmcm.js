import { createRadiusGrid } from './rmcm-grid.js';
import { createRadiusScanner, createCsrFilter } from './rmcm-wasm.js';
import { checkpoint } from './session-hooks.js';
import { positiveInteger, finiteNumber, assertFinite, seededRandom, checkCancelled, progress, squaredDistance } from './core.js';

const DEFAULT_MEMORY = 512 * 1024 ** 2;
const MAX_INDEX = 0xffffffff;
const PREPARED_TOKEN = Symbol();
const OWNED_DATA = Symbol();
const PREPARED_GRAPHS = new WeakMap();
const consume = iterator => { let step; do { step = iterator.next(); } while (!step.done); return step.value; };
function memoryGuard(bytes, limit) {
  if (!Number.isSafeInteger(bytes) || bytes > limit) throw new RangeError(`estimated primary arrays ${bytes} bytes exceed maxMemoryBytes=${limit}`);
}
function nonnegativeInteger(value, name) {
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`${name} must be a nonnegative safe integer`);
  return value;
}
function numericArray(value, length, name) {
  if (!ArrayBuffer.isView(value) || value instanceof DataView || typeof value[0] === 'bigint' || value.length !== length) throw new TypeError(`${name} must be a numeric TypedArray with ${length} values`);
  assertFinite(value, name);
  return value;
}
function safeCoordinates(data, d, name) {
  const bound = Math.sqrt(Number.MAX_VALUE / d) / 4;
  for (const value of data) if (Math.abs(value) > bound) throw new RangeError(`${name} exceeds safe float64 distance range; rescale input and delta`);
}
function validateModelOptions(options) {
  for (const key of ['m', 'alpha', 'beta', 'p', 'excludeSelf', 'includeSelf', 'initMembership']) {
    if (Object.hasOwn(options, key)) throw new RangeError(`${key} is not an RMCM parameter; RMCM uses R = P H without a fuzzifier and includes self`);
  }
  if (options.tolerance != null && options.tolerance !== 0) throw new RangeError('RMCM uses exact full-state convergence; nonzero tolerance is unsupported');
}
function preparationParameters(input, options) {
  validateModelOptions(options);
  if (!input || typeof input !== 'object') throw new TypeError('input must be { data, nSamples, nFeatures }');
  const n = positiveInteger(input.nSamples, 'nSamples'), d = positiveInteger(input.nFeatures, 'nFeatures');
  if (n > MAX_INDEX || !Number.isSafeInteger(n * d)) throw new RangeError('input dimensions exceed CSR index range or safe integer range');
  numericArray(input.data, n * d, 'data');
  safeCoordinates(input.data, d, 'data');
  const delta = finiteNumber(options.delta, 'delta', 0);
  const graphBackend = options.graphBackend ?? 'scalar';
  if (!['scalar', 'wasm-simd', 'grid'].includes(graphBackend)) throw new RangeError("graphBackend must be 'scalar', 'wasm-simd', or 'grid'");
  const backend = options.backend ?? 'adjoint';
  if (!['adjoint', 'reference'].includes(backend)) throw new RangeError("RMCM backend must be 'adjoint' or 'reference'");
  const maxEdges = positiveInteger(options.maxEdges ?? 10_000_000, 'maxEdges');
  if (maxEdges > MAX_INDEX) throw new RangeError('maxEdges exceeds Uint32 CSR index range');
  if (n > maxEdges) throw new RangeError('maxEdges is smaller than the mandatory self-edge count');
  const maxMemoryBytes = finiteNumber(options.maxMemoryBytes ?? DEFAULT_MEMORY, 'maxMemoryBytes', 1);
  const graphBatchPairs = positiveInteger(options.graphBatchPairs ?? 4096, 'graphBatchPairs');
  const blockRows = positiveInteger(options.blockRows ?? 128, 'blockRows');
  // Snapshot X, degree, indptr, fill cursors, adjoint Y/s and optional common mean.
  const fixedBytes = 8 * n * d + 12 * n + 4 + (backend === 'adjoint' ? 8 * n * d + 8 * n : 0) + 8 * d;
  memoryGuard(fixedBytes + 4 * n, maxMemoryBytes);
  checkCancelled(options);
  return { n, d, delta, backend, graphBackend, maxEdges, maxMemoryBytes, graphBatchPairs, blockRows, fixedBytes };
}
function samePoint(data, io, jo, d) {
  for (let f = 0; f < d; ++f) if (data[io + f] !== data[jo + f]) return false;
  return true;
}
function adjacent(data, i, j, d, delta, cutoff) {
  const io = i * d, jo = j * d;
  if (delta === 0) return samePoint(data, io, jo, d);
  let sum = 0;
  for (let f = 0; f < d; ++f) {
    const diff = data[io + f] - data[jo + f]; sum += diff * diff;
    if (sum > cutoff) return false;
  }
  // All data were range-checked, so overflow is impossible. A rejected pair
  // already has a normal partial sum; it cannot conceal underflow either.
  if (sum > 0 && sum < 2.2250738585072014e-308) throw new RangeError('squared distance is subnormal; rescale input');
  if (sum === 0 && !samePoint(data, io, jo, d)) throw new RangeError('squared distance underflow; rescale input');
  return Math.sqrt(sum) <= delta;
}
function completeGraphCertificate(data, n, d, delta) {
  let squared = 0, allIdentical = true, safeSpacing = true;
  const threshold = 2 ** -458;
  for (let f = 0; f < d; ++f) {
    let min = data[f], max = min;
    for (let i = 0; i < n; ++i) {
      const x = data[i * d + f]; min = Math.min(min, x); max = Math.max(max, x);
      if (x !== 0 && Math.abs(x) < threshold) safeSpacing = false;
    }
    allIdentical &&= min === max;
    const span = max - min; squared += span * span;
    if (Math.sqrt(squared) > delta) return false;
  }
  return allIdentical || (safeSpacing && Math.sqrt(squared) * (1 + 16 * Number.EPSILON * d) <= delta);
}

function event(options, phase, completed, total) {
  const e = { algorithm: 'rmcm', phase, completed, total };
  progress(options, e); return e;
}

/** Build fixed symmetric self-including CSR and P^T X / P^T 1 once.
 * Direct pair enumeration is O(N^2 D). A two-pass edge count prevents dense
 * adjacency allocation and enforces maxEdges/maxMemoryBytes before CSR storage.
 */
export function* prepareRMCMSteps(input, options = {}) {
  const cfg = preparationParameters(input, options), { n, d, delta, backend, maxEdges, maxMemoryBytes, graphBatchPairs, blockRows, fixedBytes } = cfg;
  const data = options[OWNED_DATA] ?? Float64Array.from(input.data), degrees = new Uint32Array(n).fill(1);
  if (completeGraphCertificate(data, n, d, delta)) {
    const nEdges = n * n, pairs = n * (n - 1) / 2;
    yield event(options, 'graph-count', 0, pairs);
    // Match whichever per-edge limit the original scan encounters first.
    const edgeFailure = n + 2 * (Math.floor((maxEdges - n) / 2) + 1);
    const memoryLimit = Math.floor((maxMemoryBytes - fixedBytes) / 4);
    const memoryFailure = n + 2 * (Math.floor((memoryLimit - n) / 2) + 1);
    if (Math.min(edgeFailure, memoryFailure) <= nEdges) {
      if (edgeFailure <= memoryFailure) throw new RangeError(`delta graph exceeds maxEdges=${maxEdges} directed edges, including self; reduce delta or raise the limit`);
      memoryGuard(fixedBytes + 4 * memoryFailure, maxMemoryBytes);
    }
    yield event(options, 'graph-count', pairs, pairs);
    degrees.fill(n); cfg.graphBackendUsed = 'scalar';
    const indptr = new Uint32Array(n + 1), indices = new Uint32Array(nEdges), row = Uint32Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < n; ++i) {
      indptr[i] = i * n; indices.set(row, i * n);
      if ((i + 1) % blockRows === 0) { checkCancelled(options); yield { algorithm: 'rmcm', phase: 'graph-fill', completedRows: i + 1, totalRows: n }; }
    }
    indptr[n] = nEdges;
    return new PreparedRMCM(yield* finalizeGraph(cfg, data, degrees, indptr, indices, nEdges, options), PREPARED_TOKEN);
  }
  const cutoff = Math.max(2.2250738585072014e-308, (delta * delta) * (1 + 4 * Number.EPSILON));
  let scanner = cfg.graphBackend === 'wasm-simd' ? createRadiusScanner(data, n, d, delta, graphBatchPairs, maxMemoryBytes - fixedBytes - 4 * n) : null;
  let grid = cfg.graphBackend === 'grid' ? createRadiusGrid(data, n, d, delta, maxMemoryBytes - fixedBytes - 4 * n) : null;
  const preparationBytes = fixedBytes + (scanner?.bytes ?? grid?.bytes ?? 0);
  cfg.graphBackendUsed = scanner ? 'wasm-simd' : grid ? 'grid' : 'scalar';
  let nEdges = n, done = 0;
  const totalPairs = n * (n - 1) / 2;
  // Cache accepted upper-triangle edges only, within the explicit byte budget.
  // If staging would exceed it, release staging and retain the two-pass path.
  const chunkSize = 16384, chunks = [];
  let caching = preparationBytes + 8 * n + 4 <= maxMemoryBytes;
  let upperRows = caching ? new Uint32Array(n + 1) : null;
  let saved = 0, chunk = null, cacheBytes = 0;
  let cachedGraphCapacity = Math.floor((maxMemoryBytes - preparationBytes - (upperRows?.byteLength ?? 0)) / 4);
  const record = (i, j) => {
    nEdges += 2;
    if (nEdges > maxEdges) throw new RangeError(`delta graph exceeds maxEdges=${maxEdges} directed edges, including self; reduce delta or raise the limit`);
    memoryGuard(fixedBytes + 4 * nEdges, maxMemoryBytes);
    // The optional scanner may use space needed by a growing CSR. Retain its
    // last output until the caller finishes consuming this bounded batch.
    ++degrees[i]; ++degrees[j];
    if (caching && nEdges > cachedGraphCapacity) {
      caching = false; chunks.length = 0; chunk = null; upperRows = null;
    }
    if (caching) {
      if (saved % chunkSize === 0) {
        const slots = Math.min(chunkSize, Math.floor((maxEdges - n) / 2) - saved);
        if (preparationBytes + 4 * (n + 1) + 4 * nEdges + 4 * (saved + slots) > maxMemoryBytes) {
          caching = false; chunks.length = 0; chunk = null; upperRows = null;
        } else {
          chunk = new Uint32Array(slots); chunks.push(chunk); cacheBytes += chunk.byteLength;
          cachedGraphCapacity = Math.floor((maxMemoryBytes - preparationBytes - upperRows.byteLength - cacheBytes) / 4);
        }
      }
      if (caching) chunk[saved++ % chunkSize] = j;
    }
  };
  yield event(options, 'graph-count', 0, totalPairs);
  for (let i = 0; i < n; ++i) {
    if (grid) {
      const row = grid.row(i);
      for (let q = 0; q < row.length; ++q) {
        const j = row[q];
        if (adjacent(data, i, j, d, delta, cutoff)) record(i, j);
        if ((q + 1) % graphBatchPairs === 0) yield event(options, 'graph-count', done + j - i, totalPairs);
      }
      done += n - i - 1;
      if ((i + 1) % blockRows === 0) yield event(options, 'graph-count', done, totalPairs);
    } else if (scanner) {
      for (let start = i + 1; start < n; start += scanner.capacity) {
        const end = Math.min(n, start + scanner.capacity), count = scanner.scan(i, start, end);
        for (let q = 0; q < count; ++q) {
          const j = scanner.ids[q];
          if (scanner.exact || adjacent(data, i, j, d, delta, cutoff)) record(i, j);
        }
        if (scanner.errorCode === -1) throw new RangeError('squared distance underflow; rescale input');
        if (scanner.errorCode === -2) throw new RangeError('squared distance is subnormal; rescale input');
        done += end - start;
        yield event(options, 'graph-count', done, totalPairs);
      }
    } else {
      for (let j = i + 1; j < n; ++j) {
        if (adjacent(data, i, j, d, delta, cutoff)) record(i, j);
        if (++done % graphBatchPairs === 0) yield event(options, 'graph-count', done, totalPairs);
      }
    }
    if (caching) upperRows[i + 1] = saved;
  }
  scanner = null; grid = null;
  yield event(options, 'graph-count', totalPairs, totalPairs);
  const indptr = new Uint32Array(n + 1), indices = new Uint32Array(nEdges);
  for (let i = 0; i < n; ++i) indptr[i + 1] = indptr[i] + degrees[i];
  const cursor = indptr.slice(0, n);
  done = 0;
  for (let i = 0; i < n; ++i) {
    // Earlier symmetric insertions contain exactly the neighbors below i.
    // Placing self now and then ascending j produces sorted CSR without sort.
    indices[cursor[i]++] = i;
    if (caching) {
      for (let q = upperRows[i]; q < upperRows[i + 1]; ++q) {
        const j = chunks[Math.floor(q / chunkSize)][q % chunkSize];
        indices[cursor[i]++] = j; indices[cursor[j]++] = i;
        if (++done % graphBatchPairs === 0) yield event(options, 'graph-fill', done, (nEdges - n) / 2);
      }
    } else {
      for (let j = i + 1; j < n; ++j) {
        if (adjacent(data, i, j, d, delta, cutoff)) { indices[cursor[i]++] = j; indices[cursor[j]++] = i; }
        if (++done % graphBatchPairs === 0) yield event(options, 'graph-fill', done, totalPairs);
      }
    }
    if ((i + 1) % blockRows === 0) checkCancelled(options);
  }
  chunks.length = 0; chunk = null;
  yield event(options, 'graph-fill', caching ? (nEdges - n) / 2 : totalPairs, caching ? (nEdges - n) / 2 : totalPairs);
  return new PreparedRMCM(yield* finalizeGraph(cfg, data, degrees, indptr, indices, nEdges, options), PREPARED_TOKEN);
}
function* finalizeGraph(cfg, data, degrees, indptr, indices, nEdges, options) {
  const { n, d, backend, graphBatchPairs } = cfg;
  const full = nEdges === n * n;
  let mean = null;
  if (full) {
    mean = new Float64Array(d);
    for (let i = 0; i < n; ++i) for (let f = 0; f < d; ++f) mean[f] += data[i * d + f];
    for (let f = 0; f < d; ++f) mean[f] /= n;
    assertFinite(mean, 'global mean');
  }
  let Y = null, s = null;
  if (backend === 'adjoint' && !full) {
    Y = new Float64Array(n * d); s = new Float64Array(n);
    let edgesDone = 0;
    const weighted = new Float64Array(d);
    for (let i = 0; i < n; ++i) {
      const inv = 1 / degrees[i], io = i * d;
      for (let f = 0; f < d; ++f) weighted[f] = inv * data[io + f];
      for (let q = indptr[i]; q < indptr[i + 1]; ++q) {
        const j = indices[q], jo = j * d;
        s[j] += inv;
        for (let f = 0; f < d; ++f) Y[jo + f] += weighted[f];
        if (++edgesDone % graphBatchPairs === 0) yield event(options, 'adjoint-precompute', edgesDone, nEdges);
      }
    }
    assertFinite(Y, 'adjoint P^T X'); assertFinite(s, 'adjoint P^T 1');
    yield event(options, 'adjoint-precompute', nEdges, nEdges);
  }
  return { ...cfg, data, degrees, indptr, indices, nEdges, full, mean, Y, s };
}
export function prepareRMCM(input, options = {}) { return consume(prepareRMCMSteps(input, options)); }

function equalArray(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; ++i) if (a[i] !== b[i]) return false;
  return true;
}
function equalState(a, labels, centers) { return equalArray(a?.labels, labels) && equalArray(a?.centers, centers); }
function stateHash(labels, centers) {
  // Hash filters candidates only; full labels AND centers are compared on a hit.
  let hash = 2166136261;
  for (let i = 0; i < labels.length; ++i) hash = Math.imul(hash ^ labels[i], 16777619);
  const bytes = new Uint8Array(centers.buffer, centers.byteOffset, centers.byteLength);
  for (let i = 0; i < bytes.length; ++i) hash = Math.imul(hash ^ bytes[i], 16777619);
  return hash >>> 0;
}
function fitParameters(g, options) {
  validateModelOptions(options);
  if (options.backend != null && options.backend !== g.backend) throw new RangeError('backend is fixed when preparing RMCM');
  if (options.delta != null && options.delta !== g.delta) throw new RangeError('delta is fixed when preparing RMCM');
  const k = positiveInteger(options.nClusters ?? options.initCenters?.length / g.d, 'nClusters');
  if (k > g.n || k > 0x7fffffff) throw new RangeError('nClusters cannot exceed nSamples or the Int32 label range');
  const maxIterations = positiveInteger(options.maxIterations ?? 100, 'maxIterations');
  const cycleWindow = nonnegativeInteger(options.cycleWindow ?? 32, 'cycleWindow');
  const blockRows = positiveInteger(options.blockRows ?? g.blockRows, 'blockRows');
  const returnMembership = options.returnMembership ?? true;
  if (typeof returnMembership !== 'boolean') throw new TypeError('returnMembership must be boolean');
  const maxMemoryBytes = finiteNumber(options.maxMemoryBytes ?? g.maxMemoryBytes, 'maxMemoryBytes', 1);
  const historyStates = Math.min(cycleWindow, maxIterations) + 1;
  const estimatedBytes = g.fixedBytes + 4 * g.nEdges + 8 * g.n + 24 * k * g.d + 8 * k +
    historyStates * (4 * g.n + 8 * k * g.d) + ((g.backend === 'reference' || returnMembership) ? 8 * g.n * k : 0);
  memoryGuard(estimatedBytes, maxMemoryBytes);
  checkCancelled(options);
  return { k, maxIterations, cycleWindow, blockRows, returnMembership, estimatedBytes };
}
function initialization(g, k, options) {
  const centers = new Float64Array(k * g.d);
  let initIndices = null;
  if (options.initCenters != null) {
    numericArray(options.initCenters, centers.length, 'initCenters'); safeCoordinates(options.initCenters, g.d, 'initCenters'); centers.set(options.initCenters);
  } else {
    const rng = seededRandom(options.seed ?? 0), pool = Uint32Array.from({ length: g.n }, (_, i) => i);
    initIndices = new Uint32Array(k);
    for (let c = 0; c < k; ++c) {
      const q = c + Math.floor(rng() * (g.n - c));
      [pool[c], pool[q]] = [pool[q], pool[c]]; initIndices[c] = pool[c];
      centers.set(g.data.subarray(pool[c] * g.d, (pool[c] + 1) * g.d), c * g.d);
    }
  }
  return { centers, initIndices };
}
function* roughMembership(g, labels, k, options, iteration, blockRows, output = null) {
  const R = output ?? new Float64Array(g.n * k);
  R.fill(0);
  for (let i = 0; i < g.n; ++i) {
    const off = i * k;
    for (let q = g.indptr[i]; q < g.indptr[i + 1]; ++q) ++R[off + labels[g.indices[q]]];
    for (let c = 0; c < k; ++c) R[off + c] /= g.degrees[i];
    if ((i + 1) % blockRows === 0) { checkCancelled(options); yield { algorithm: 'rmcm', phase: 'rough-membership', iteration, completedRows: i + 1, totalRows: g.n }; }
  }
  return R;
}

/** Immutable prepared input and graph. Public arrays are independent copies. */
export class PreparedRMCM {
  #g;
  constructor(graph, token) {
    if (token !== PREPARED_TOKEN) throw new TypeError('Use prepareRMCM or prepareRMCMSteps to construct PreparedRMCM');
    this.#g = graph;
    PREPARED_GRAPHS.set(this, graph);
  }
  get nSamples() { return this.#g.n; }
  get nFeatures() { return this.#g.d; }
  get nEdges() { return this.#g.nEdges; }
  get graphBackendUsed() { return this.#g.graphBackendUsed; }
  get delta() { return this.#g.delta; }
  get backend() { return this.#g.backend; }
  get degrees() { return this.#g.degrees.slice(); }
  neighborhoodMatrix() {
    const g = this.#g;
    memoryGuard(g.fixedBytes + 4 * g.nEdges + 4 * (g.n + 1) + 12 * g.nEdges, g.maxMemoryBytes);
    const values = new Float64Array(g.nEdges);
    for (let i = 0; i < g.n; ++i) values.fill(1 / g.degrees[i], g.indptr[i], g.indptr[i + 1]);
    return { indptr: g.indptr.slice(), indices: g.indices.slice(), data: values, shape: [g.n, g.n] };
  }
  fit(options = {}) { return consume(this.fitSteps(options)); }
  *fitSteps(options = {}) {
    const g = this.#g, { n, d, data } = g;
    const { k, maxIterations, cycleWindow, blockRows, returnMembership, estimatedBytes } = fitParameters(g, options);
    let { centers, initIndices } = initialization(g, k, options);
    let labels = new Int32Array(n), membership = null, previous = null;
    const history = [], mass = new Float64Array(k), numerator = new Float64Array(k * d);
    let stopReason = 'max_iter', cycleLength = null, iterations = 0, emptyClusterUpdates = 0;
    for (iterations = 1; iterations <= maxIterations; ++iterations) {
      for (let i = 0; i < n; ++i) {
        let best = 0, bestDistance = Infinity;
        for (let c = 0; c < k; ++c) {
          const value = squaredDistance(data, i * d, centers, c * d, d);
          if (value < bestDistance) { best = c; bestDistance = value; }
        }
        labels[i] = best;
        if ((i + 1) % blockRows === 0) { checkCancelled(options); yield { algorithm: 'rmcm', phase: 'assignment', iteration: iterations, completedRows: i + 1, totalRows: n }; }
      }
      mass.fill(0); numerator.fill(0);
      if (g.full) {
        for (let i = 0; i < n; ++i) ++mass[labels[i]];
        // All occupied centers share exactly the same common reduction.
        for (let c = 0; c < k; ++c) if (mass[c] > 0) centers.set(g.mean, c * d);
      } else if (g.backend === 'adjoint') {
        for (let i = 0; i < n; ++i) {
          const c = labels[i], co = c * d;
          mass[c] += g.s[i];
          for (let f = 0; f < d; ++f) numerator[co + f] += g.Y[i * d + f];
          if ((i + 1) % blockRows === 0) { checkCancelled(options); yield { algorithm: 'rmcm', phase: 'adjoint-centers', iteration: iterations, completedRows: i + 1, totalRows: n }; }
        }
      } else {
        // Deliberately explicit reference: R = P H, then R^T X and R^T 1.
        membership = yield* roughMembership(g, labels, k, options, iterations, blockRows, membership);
        for (let i = 0; i < n; ++i) {
          for (let c = 0; c < k; ++c) {
            const weight = membership[i * k + c]; mass[c] += weight;
            for (let f = 0; f < d; ++f) numerator[c * d + f] += weight * data[i * d + f];
          }
          if ((i + 1) % blockRows === 0) { checkCancelled(options); yield { algorithm: 'rmcm', phase: 'reference-centers', iteration: iterations, completedRows: i + 1, totalRows: n }; }
        }
      }
      for (let c = 0; c < k; ++c) {
        if (mass[c] === 0) ++emptyClusterUpdates;
        else if (!g.full) for (let f = 0; f < d; ++f) centers[c * d + f] = numerator[c * d + f] / mass[c];
      }
      assertFinite(centers, 'centers');
      if (equalState(previous, labels, centers)) stopReason = 'fixed_point';
      const hash = cycleWindow ? stateHash(labels, centers) : 0;
      if (stopReason !== 'fixed_point' && cycleWindow) {
        const repeated = history.find(state => state.hash === hash && equalState(state, labels, centers));
        if (repeated) { stopReason = 'cycle'; cycleLength = iterations - repeated.iteration; }
      }
      checkpoint(options, { algorithm: 'rmcm', centers, labels, membership: null, membershipLayout: 'samples-clusters', iterations, converged: stopReason === 'fixed_point', stopReason: stopReason === 'max_iter' && iterations < maxIterations ? null : stopReason, cycleLength, delta: g.delta, nEdges: g.nEdges, emptyClusterUpdates, nSamples: n, nFeatures: d, nClusters: k, backend: `javascript-float64-${g.backend}`, estimatedPrimaryBytes: estimatedBytes, labelContract: 'hard assignment that produced returned centers; no final reassignment', membershipContract: 'available after finalization; R = P H from returned labels' });
      const e = { algorithm: 'rmcm', iteration: iterations, maxIterations, converged: stopReason === 'fixed_point', stopReason: stopReason === 'max_iter' && iterations < maxIterations ? null : stopReason, cycleLength };
      progress(options, e); yield e;
      if (stopReason !== 'max_iter') break;
      previous = { labels: labels.slice(), centers: centers.slice(), hash, iteration: iterations };
      if (cycleWindow) { history.push(previous); if (history.length > cycleWindow) history.shift(); }
    }
    iterations = Math.min(iterations, maxIterations);
    if (returnMembership) {
      if (g.backend === 'adjoint' || g.full) membership = yield* roughMembership(g, labels, k, options, iterations, blockRows, membership);
    } else membership = null;
    checkCancelled(options);
    return { algorithm: 'rmcm', centers, labels, membership, membershipLayout: 'samples-clusters', iterations,
      converged: stopReason === 'fixed_point', stopReason, cycleLength, initIndices, delta: g.delta, nEdges: g.nEdges,
      emptyClusterUpdates, nSamples: n, nFeatures: d, nClusters: k, backend: `javascript-float64-${g.backend}`,
      estimatedPrimaryBytes: estimatedBytes, labelContract: 'hard assignment that produced returned centers; no final reassignment',
      membershipContract: 'R = P H from returned labels; weighted center update uses R without an exponent' };
  }
}
export function* rmcmSteps(input, options = {}) {
  const prepared = yield* prepareRMCMSteps(input, options);
  return yield* prepared.fitSteps(options);
}
export function rmcm(input, options = {}) { return consume(rmcmSteps(input, options)); }
export function rmcmReference(input, options = {}) {
  if (options.backend != null && options.backend !== 'reference') throw new RangeError('rmcmReference selects reference backend');
  return rmcm(input, { ...options, backend: 'reference' });
}

/** Optional exact candidate cache for slowly moving coordinates.
 * Every frame rebuilds active CSR/weights and adjoint precomputation. The skin
 * only avoids a full pair search while an outward displacement bound proves
 * that all current delta neighbors remain in the cached candidate superset.
 */
function pointIdSnapshot(value, n) {
  if (value == null) return null;
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value instanceof DataView || value.length !== n) throw new TypeError('pointIds must contain one distinct safe integer per row');
  const ids = new Float64Array(n);
  for (let i = 0; i < n; ++i) {
    if (!Number.isSafeInteger(value[i])) throw new RangeError('pointIds must be safe integers');
    ids[i] = value[i];
  }
  const sorted = ids.slice(); sorted.sort();
  for (let i = 1; i < n; ++i) if (sorted[i] === sorted[i - 1]) throw new RangeError('pointIds must be distinct');
  return ids;
}

export class RMCMGraphCache {
  #skin; #maxCacheEdges; #reference = null; #cover = 0;
  #rebuilds = 0; #reuses = 0; #fallbacks = 0;
  #incrementalUpdates = 0; #insertedPoints = 0; #removedPoints = 0;
  constructor({ skin, maxCacheEdges = 10_000_000 } = {}) {
    this.#skin = finiteNumber(skin, 'skin', 0);
    this.#maxCacheEdges = positiveInteger(maxCacheEdges, 'maxCacheEdges');
    if (this.#maxCacheEdges > MAX_INDEX) throw new RangeError('maxCacheEdges exceeds Uint32 CSR index range');
  }
  get stats() {
    const g = this.#reference;
    return { rebuilds: this.#rebuilds, reuses: this.#reuses, fallbacks: this.#fallbacks,
      candidateEdges: g?.nEdges ?? 0, cacheBytes: g ? g.data.byteLength + g.indptr.byteLength + g.indices.byteLength + (g.pointIds?.byteLength ?? 0) : 0,
      incrementalUpdates: this.#incrementalUpdates, insertedPoints: this.#insertedPoints, removedPoints: this.#removedPoints,
      coverRadius: g ? this.#cover : null };
  }
  clear() { this.#reference = null; this.#cover = 0; }
  #canReuse(data, n, d, target, g = this.#reference, cover = this.#cover) {
    if (!g || g.n !== n || g.d !== d) return false;
    let maxSquared = 0, identical = true;
    for (let i = 0; i < n; ++i) {
      let sq = 0;
      for (let f = 0; f < d; ++f) {
        const diff = data[i * d + f] - g.data[i * d + f];
        identical &&= diff === 0; sq += diff * diff;
      }
      maxSquared = Math.max(maxSquared, sq);
    }
    if (identical) return target <= cover;
    if (!Number.isFinite(target) || !Number.isFinite(cover)) return false;
    const upper = maxSquared * (1 + 16 * Number.EPSILON * d) + 2.2250738585072014e-308 * d;
    const displacement = Math.sqrt(upper) * (1 + 4 * Number.EPSILON);
    return (target + 2 * displacement) * (1 + 4 * Number.EPSILON) <= cover;
  }
  *#updatePointIds(data, n, d, target, pointIds, cfg, options) {
    const g = this.#reference, cover = this.#cover;
    if (!g?.pointIds || g.d !== d || n > this.#maxCacheEdges || n > 0x7fffffff || g.n > 0x7fffffff) return false;
    if (equalArray(pointIds, g.pointIds)) {
      if (!this.#canReuse(data, n, d, target, g, cover)) return false;
      g.pointIds = pointIds;
      return true;
    }
    // Old cache, new mixed reference, ID maps, upper-CSR buffers and active
    // graph workspace coexist during staging. Pair chunks are charged below.
    let fixed = cfg.fixedBytes + this.stats.cacheBytes + 8 * n * d + 32 * n + 8 * g.n + 4;
    if (fixed + 21 * g.indices.length > cfg.maxMemoryBytes) return false;
    const sorted = Uint32Array.from({ length: g.n }, (_, i) => i);
    sorted.sort((a, b) => g.pointIds[a] - g.pointIds[b]);
    const oldToNew = new Int32Array(g.n).fill(-1), newToOld = new Int32Array(n).fill(-1);
    let born = 0, surviving = 0;
    for (let i = 0; i < n; ++i) {
      let lo = 0, hi = g.n;
      while (lo < hi) { const mid = lo + Math.floor((hi - lo) / 2); if (g.pointIds[sorted[mid]] < pointIds[i]) lo = mid + 1; else hi = mid; }
      if (lo < g.n && g.pointIds[sorted[lo]] === pointIds[i]) { newToOld[i] = sorted[lo]; oldToNew[sorted[lo]] = i; ++surviving; }
      else ++born;
    }
    if (!surviving || born > Math.max(64, Math.floor(n / 50))) return false;
    const referenceData = data.slice();
    for (let i = 0; i < n; ++i) if (newToOld[i] >= 0) referenceData.set(g.data.subarray(newToOld[i] * d, (newToOld[i] + 1) * d), i * d);
    if (!this.#canReuse(data, n, d, target, { n, d, data: referenceData }, cover)) return false;
    const radius = cover * (1 + 64 * Number.EPSILON * d);
    if (!Number.isFinite(radius)) return false;
    const cutoff = Math.max(2.2250738585072014e-308, radius * radius * (1 + 4 * Number.EPSILON));
    let scanner = cfg.graphBackend === 'wasm-simd' ? createRadiusScanner(referenceData, n, d, radius, cfg.graphBatchPairs, cfg.maxMemoryBytes - fixed - 21 * g.indices.length) : null;
    fixed += scanner?.bytes ?? 0;
    const maxPairs = Math.floor((this.#maxCacheEdges - n) / 2), degree = new Uint32Array(n), chunks = [];
    let count = 0, chunk = null, used = 0;
    const record = (a, b) => {
      if (count >= maxPairs) return false;
      if (!chunk || used === chunk.length) {
        const capacity = Math.min(4096, maxPairs - count, Math.max(1, g.indices.length + born * n - count));
        if (fixed + 21 * (count + capacity) > cfg.maxMemoryBytes) return false;
        chunk = new Uint32Array(2 * capacity); chunks.push(chunk); used = 0;
      }
      const i = Math.min(a, b), j = Math.max(a, b);
      chunk[used++] = i; chunk[used++] = j; ++degree[i]; ++count;
      return true;
    };
    let visited = 0;
    for (let i = 0; i < g.n; ++i) {
      if (oldToNew[i] >= 0) for (let q = g.indptr[i]; q < g.indptr[i + 1]; ++q) {
        const j = g.indices[q];
        if (oldToNew[j] >= 0 && !record(oldToNew[i], oldToNew[j])) return false;
      }
      if ((i + 1) % cfg.blockRows === 0) { checkCancelled(options); yield { algorithm: 'rmcm', phase: 'graph-cache-remap', completedRows: i + 1, totalRows: g.n }; }
    }
    const candidatePair = (i, j, exact = false) => {
      if (i === j || (newToOld[j] < 0 && j < i)) return true;
      if (!exact) {
        let sum = 0;
        for (let f = 0; f < d; ++f) { const diff = referenceData[i * d + f] - referenceData[j * d + f]; sum += diff * diff; if (sum > cutoff) return true; }
        // Defer numeric validation to the active graph's original guard order.
        if (Math.sqrt(sum) > radius) return true;
      }
      return record(i, j);
    };
    for (let i = 0; i < n; ++i) if (newToOld[i] < 0) {
      if (scanner) {
        for (let lo = 0; lo < n; lo += scanner.capacity) {
          const hi = Math.min(n, lo + scanner.capacity), size = scanner.scan(i, lo, hi);
          for (let q = 0; q < size; ++q) if (!candidatePair(i, scanner.ids[q], scanner.exact)) return false;
          if (scanner.errorCode) return false; // ordinary preparation restores exact exception precedence
          visited += hi - lo; checkCancelled(options); yield { algorithm: 'rmcm', phase: 'graph-cache-births', completedPairs: visited, totalPairs: born * n };
        }
      } else {
        for (let j = 0; j < n; ++j) {
          if (!candidatePair(i, j)) return false;
          if (++visited % cfg.graphBatchPairs === 0) { checkCancelled(options); yield { algorithm: 'rmcm', phase: 'graph-cache-births', completedPairs: visited, totalPairs: born * n }; }
        }
      }
    }
    scanner = null;
    const indptr = new Uint32Array(n + 1), indices = new Uint32Array(count);
    for (let i = 0; i < n; ++i) indptr[i + 1] = indptr[i] + degree[i];
    const cursor = indptr.slice(0, n);
    for (let k = 0; k < chunks.length; ++k) {
      const length = k === chunks.length - 1 ? used : chunks[k].length;
      for (let q = 0; q < length; q += 2) indices[cursor[chunks[k][q]]++] = chunks[k][q + 1];
    }
    for (let i = 0; i < n; ++i) indices.subarray(indptr[i], indptr[i + 1]).sort();
    this.#reference = { n, d, data: referenceData, indptr, indices, nEdges: n + 2 * count, pointIds };
    this.#cover = cover; // Commit the proof radius with its matching reference after any yields.
    ++this.#incrementalUpdates; this.#insertedPoints += born; this.#removedPoints += g.n - surviving;
    return true;
  }
  prepare(input, options = {}) { return consume(this.prepareSteps(input, options)); }
  *prepareSteps(input, options = {}) {
    const cfg = preparationParameters(input, options), { n, d, delta, maxEdges, maxMemoryBytes, graphBatchPairs, blockRows } = cfg;
    // Detach an unaffordable previous cache before allocating a new snapshot
    // or ID-validation buffers, including when the caller lowers the budget.
    if (this.#reference && this.stats.cacheBytes + cfg.fixedBytes + 4 * n > maxMemoryBytes) this.clear();
    const data = Float64Array.from(input.data);
    let pointIds = pointIdSnapshot(options.pointIds, n);
    if (!pointIds && this.#reference?.pointIds) this.clear();
    const snapshotInput = { data, nSamples: n, nFeatures: d };
    const ownedOptions = { ...options, [OWNED_DATA]: data };
    const target = delta === 0 ? 0 : delta * (1 + 64 * Number.EPSILON * d);
    const eligible = delta >= 16 * Math.sqrt(2.2250738585072014e-308) && 64 * Number.EPSILON * d < 1e-4;
    if (!eligible || !Number.isFinite(target)) {
      this.clear(); pointIds = null; ++this.#fallbacks;
      return yield* prepareRMCMSteps(snapshotInput, ownedOptions);
    }
    let reused = pointIds ? yield* this.#updatePointIds(data, n, d, target, pointIds, cfg, options) : this.#canReuse(data, n, d, target);
    if (!reused) {
      const cover = (target + this.#skin) * (1 + 4 * Number.EPSILON);
      const radius = cover * (1 + 64 * Number.EPSILON * d);
      if (!Number.isFinite(radius)) {
        this.clear(); pointIds = null; ++this.#fallbacks;
        return yield* prepareRMCMSteps(snapshotInput, ownedOptions);
      }
      // Release an obsolete cache before constructing the next one. Its own
      // byte and edge limits are explicit; an oversized skin falls back.
      this.clear();
      try {
        let candidate = yield* prepareRMCMSteps(snapshotInput, { ...ownedOptions, delta: radius, backend: 'reference', maxEdges: this.#maxCacheEdges });
        let raw = PREPARED_GRAPHS.get(candidate);
        // Persist each undirected candidate once. The active graph is still
        // reconstructed symmetrically in the original canonical row order.
        const pairCount = (raw.nEdges - n) / 2;
        memoryGuard(raw.fixedBytes + 4 * raw.nEdges + 4 * (n + 1) + 4 * pairCount + (pointIds?.byteLength ?? 0), maxMemoryBytes);
        const upperPtr = new Uint32Array(n + 1), upperIds = new Uint32Array(pairCount);
        let cursor = 0;
        for (let i = 0; i < n; ++i) {
          upperPtr[i] = cursor;
          for (let q = raw.indptr[i]; q < raw.indptr[i + 1]; ++q) if (raw.indices[q] > i) upperIds[cursor++] = raw.indices[q];
        }
        upperPtr[n] = cursor;
        this.#reference = { n, d, data: raw.data, indptr: upperPtr, indices: upperIds, nEdges: raw.nEdges, pointIds };
        candidate = null; raw = null;
        this.#cover = cover; ++this.#rebuilds;
      } catch (error) {
        if (!(error instanceof RangeError) || !/maxEdges|estimated primary arrays|underflow|subnormal/.test(error.message)) throw error;
        this.clear(); pointIds = null; ++this.#fallbacks;
        return yield* prepareRMCMSteps(snapshotInput, ownedOptions);
      }
    }
    let reference = this.#reference;
    const cacheBytes = this.stats.cacheBytes;
    // The cache persists beside the prepared graph. Budget for both, including
    // the worst-case accepted edge index array and temporary keep flags.
    const pairs = (reference.nEdges - n) / 2;
    if (cfg.fixedBytes + cacheBytes + 4 * reference.nEdges + reference.indices.length > maxMemoryBytes) {
      this.clear(); reference = null; pointIds = null; ++this.#fallbacks;
      return yield* prepareRMCMSteps(snapshotInput, ownedOptions);
    }
    if (reused) ++this.#reuses;
    cfg.fixedBytes += cacheBytes;
    cfg.graphBackendUsed = 'cache';
    let filter = cfg.graphBackend === 'wasm-simd' ? createCsrFilter(data, reference, delta, maxMemoryBytes - cfg.fixedBytes - 4 * reference.nEdges, maxEdges) : null;
    let degrees, keep, nEdges = n, visited = 0;
    const cutoff = Math.max(2.2250738585072014e-308, delta * delta * (1 + 4 * Number.EPSILON));
    if (filter) {
      keep = filter.keep;
      for (let lo = 0; lo < n; lo += blockRows) {
        const hi = Math.min(n, lo + blockRows), error = filter.filter(lo, hi);
        if (error === -3) throw new RangeError(`delta graph exceeds maxEdges=${maxEdges} directed edges, including self; reduce delta or raise the limit`);
        if (error === -1) throw new RangeError('squared distance underflow; rescale input');
        if (error === -2) throw new RangeError('squared distance is subnormal; rescale input');
        yield event(options, 'graph-cache-filter', Math.floor(pairs * hi / n), pairs);
      }
      degrees = filter.degrees.slice();
      nEdges = degrees.reduce((sum, degree) => sum + degree, 0);
      if (nEdges > maxEdges) throw new RangeError(`delta graph exceeds maxEdges=${maxEdges} directed edges, including self; reduce delta or raise the limit`);
    } else {
      degrees = new Uint32Array(n).fill(1); keep = new Uint8Array(reference.indices.length);
    yield event(options, 'graph-cache-filter', 0, pairs);
    for (let i = 0; i < n; ++i) {
      for (let q = reference.indptr[i]; q < reference.indptr[i + 1]; ++q) {
        const j = reference.indices[q]; if (j <= i) continue;
        if (adjacent(data, i, j, d, delta, cutoff)) {
          nEdges += 2;
          if (nEdges > maxEdges) throw new RangeError(`delta graph exceeds maxEdges=${maxEdges} directed edges, including self; reduce delta or raise the limit`);
          keep[q] = 1; ++degrees[i]; ++degrees[j];
        }
        if (++visited % graphBatchPairs === 0) yield event(options, 'graph-cache-filter', visited, pairs);
      }
      if ((i + 1) % blockRows === 0) checkCancelled(options);
    }
    }
    yield event(options, 'graph-cache-filter', pairs, pairs);
    const indptr = new Uint32Array(n + 1), indices = new Uint32Array(nEdges);
    for (let i = 0; i < n; ++i) indptr[i + 1] = indptr[i] + degrees[i];
    const cursor = indptr.slice(0, n); visited = 0;
    for (let i = 0; i < n; ++i) {
      indices[cursor[i]++] = i;
      for (let q = reference.indptr[i]; q < reference.indptr[i + 1]; ++q) {
        const j = reference.indices[q]; if (j <= i) continue;
        if (keep[q]) { indices[cursor[i]++] = j; indices[cursor[j]++] = i; }
      }
      if ((i + 1) % blockRows === 0) {
        checkCancelled(options);
        yield { algorithm: 'rmcm', phase: 'graph-cache-fill', completedRows: i + 1, totalRows: n };
      }
    }
    const result = new PreparedRMCM(yield* finalizeGraph(cfg, data, degrees, indptr, indices, nEdges, options), PREPARED_TOKEN);
    return result;
  }
}
