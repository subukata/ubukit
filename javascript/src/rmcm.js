import { positiveInteger, finiteNumber, assertFinite, seededRandom, checkCancelled, progress, squaredDistance } from './core.js';

const DEFAULT_MEMORY = 512 * 1024 ** 2;
const MAX_INDEX = 0xffffffff;
const PREPARED_TOKEN = Symbol();
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
  return { n, d, delta, backend, maxEdges, maxMemoryBytes, graphBatchPairs, blockRows, fixedBytes };
}
function samePoint(data, io, jo, d) {
  for (let f = 0; f < d; ++f) if (data[io + f] !== data[jo + f]) return false;
  return true;
}
function adjacent(data, i, j, d, delta) {
  return delta === 0 ? samePoint(data, i * d, j * d, d) : Math.sqrt(squaredDistance(data, i * d, data, j * d, d)) <= delta;
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
  const data = Float64Array.from(input.data), degrees = new Uint32Array(n).fill(1);
  let nEdges = n, done = 0;
  const totalPairs = n * (n - 1) / 2;
  yield event(options, 'graph-count', 0, totalPairs);
  for (let i = 0; i < n; ++i) for (let j = i + 1; j < n; ++j) {
    if (adjacent(data, i, j, d, delta)) {
      nEdges += 2;
      if (nEdges > maxEdges) throw new RangeError(`delta graph exceeds maxEdges=${maxEdges} directed edges, including self; reduce delta or raise the limit`);
      memoryGuard(fixedBytes + 4 * nEdges, maxMemoryBytes);
      ++degrees[i]; ++degrees[j];
    }
    if (++done % graphBatchPairs === 0) yield event(options, 'graph-count', done, totalPairs);
  }
  yield event(options, 'graph-count', totalPairs, totalPairs);
  const indptr = new Uint32Array(n + 1), indices = new Uint32Array(nEdges);
  for (let i = 0; i < n; ++i) indptr[i + 1] = indptr[i] + degrees[i];
  const cursor = indptr.slice(0, n);
  for (let i = 0; i < n; ++i) indices[cursor[i]++] = i;
  done = 0;
  for (let i = 0; i < n; ++i) for (let j = i + 1; j < n; ++j) {
    if (adjacent(data, i, j, d, delta)) { indices[cursor[i]++] = j; indices[cursor[j]++] = i; }
    if (++done % graphBatchPairs === 0) yield event(options, 'graph-fill', done, totalPairs);
  }
  yield event(options, 'graph-fill', totalPairs, totalPairs);
  // Symmetric insertion is sorted except for the self entry; sorting is per row.
  for (let i = 0; i < n; ++i) {
    indices.subarray(indptr[i], indptr[i + 1]).sort();
    if ((i + 1) % blockRows === 0) { checkCancelled(options); yield { algorithm: 'rmcm', phase: 'graph-sort', completedRows: i + 1, totalRows: n }; }
  }
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
    for (let i = 0; i < n; ++i) {
      const inv = 1 / degrees[i], io = i * d;
      for (let q = indptr[i]; q < indptr[i + 1]; ++q) {
        const j = indices[q], jo = j * d;
        s[j] += inv;
        for (let f = 0; f < d; ++f) Y[jo + f] += inv * data[io + f];
        if (++edgesDone % graphBatchPairs === 0) yield event(options, 'adjoint-precompute', edgesDone, nEdges);
      }
    }
    assertFinite(Y, 'adjoint P^T X'); assertFinite(s, 'adjoint P^T 1');
    yield event(options, 'adjoint-precompute', nEdges, nEdges);
  }
  return new PreparedRMCM({ ...cfg, data, degrees, indptr, indices, nEdges, full, mean, Y, s }, PREPARED_TOKEN);
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
  }
  get nSamples() { return this.#g.n; }
  get nFeatures() { return this.#g.d; }
  get nEdges() { return this.#g.nEdges; }
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
