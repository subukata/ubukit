import { somSteps, somBatchSteps, somProject } from './som.js';
import { kmeansSteps, fcmSteps, rcmSteps, exrcmSteps } from './clustering.js';
import { somOlpSteps } from './som-olp.js';
import { prepareRMCMSteps } from './rmcm.js';
import { finiteNumber, labelsFromMembership, normalizeInput, restoreCenters, checkCancelled } from './core.js';
import { SESSION_CHECKPOINT, SESSION_WARM_CENTERS, SESSION_FCM_STATE } from './session-hooks.js';
import { cloneOwned, sameArray, validateSession, sessionAlgorithms } from './session-validation.js';
export { sessionAlgorithms } from './session-validation.js';

const kernels = { kmeans: kmeansSteps, fcm: fcmSteps, rcm: rcmSteps, exrcm: exrcmSteps, 'som-olp': somOlpSteps, som: somSteps, som_batch: somBatchSteps };
const initialKeys = ['initCenters', 'initMembership', 'initialPrototypes', 'initialMemberships'];
const classicSOM = algorithm => algorithm === 'som' || algorithm === 'som_batch';
const graphKeys = ['delta', 'backend', 'maxEdges', 'maxMemoryBytes'];
function integer(value, name, min = 0) {
  if (!Number.isSafeInteger(value) || value < min) throw new RangeError(`${name} must be an integer >= ${min}`);
  return value;
}
function ownedInput(input) {
  const x = normalizeInput(input);
  return { ...x, data: Float64Array.from(x.data) };
}
function copyState(state) { const copy=cloneOwned(state); if(state?.[SESSION_FCM_STATE]) Object.defineProperty(copy, SESSION_FCM_STATE, {value:cloneOwned(state[SESSION_FCM_STATE])}); return copy; }
function clearInitial(options) { for (const key of initialKeys) delete options[key]; }

/** An owned-input, immutable-revision fitting session. See docs/REALTIME.md. */
export class ClusteringSession {
  #algorithm; #input; #options; #cfg; #iterator = null; #prepared = null;
  #committed = null; #revision = 0; #iteration = 0; #totalIterations = 0;
  #done = false; #disposed = false; #busy = false; #phase = 'ready'; #error = null;
  #warm = null; #anchor = null; #invalidation = { kind: 'cold', reason: 'initialization', preparedGraph: 'absent' };
  constructor(algorithm, input, options = {}) {
    if (options == null || typeof options !== 'object') throw new TypeError('options must be an object');
    this.#cfg = validateSession(algorithm, input, options);
    this.#algorithm = algorithm; this.#input = ownedInput(input); this.#options = cloneOwned(options);
  }
  #assertAvailable() {
    if (this.#disposed) throw new Error('Session is disposed');
    if (this.#busy) throw new Error('Session cannot be mutated or stepped during a step callback');
  }
  get status() {
    return { algorithm: this.#algorithm, revision: this.#revision, iteration: this.#iteration,
      totalIterations: this.#totalIterations, done: this.#done, converged: this.#committed?.converged ?? false,
      disposed: this.#disposed, phase: this.#phase, nSamples: this.#input?.nSamples ?? null,
      nFeatures: this.#input?.nFeatures ?? null, ...(classicSOM(this.#algorithm) ? { unit: this.#algorithm === 'som' ? 'sample' : 'epoch', iterationUnit: this.#algorithm === 'som' ? 'sample' : 'epoch' } : {}), nClusters: this.#cfg?.k ?? null,
      invalidation: { ...this.#invalidation }, preparedGraph: this.#prepared != null,
      error: this.#error ? { name: this.#error.name, message: this.#error.message } : null };
  }
  #commit(state, origin) {
    const copy = copyState(state);
    if (origin) restoreCenters(copy.centers, origin);
    if (copy.membership && !copy.labels) copy.labels = labelsFromMembership(copy.membership, copy.nSamples, copy.nClusters);
    copy.backend ??= 'javascript-float64';
    this.#committed = copy; this.#anchor = null; this.#iteration = state.iterations; ++this.#totalIterations;
  }
  *#rmcmIterator(options) {
    if (!this.#prepared) this.#prepared = yield* prepareRMCMSteps(this.#input, options);
    return yield* this.#prepared.fitSteps(options);
  }
  #start() {
    const options = { ...this.#options, maxMemoryBytes: this.#cfg.kernelMemoryBytes,
      [SESSION_CHECKPOINT]: (state, origin) => this.#commit(state, origin) };
    if (this.#warm) {
      clearInitial(options);
      const w = this.#warm;
      if (this.#algorithm === 'fcm' && w.membership) {
        options.initMembership = w.membership; options[SESSION_WARM_CENTERS] = w.centers; options[SESSION_FCM_STATE] = w[SESSION_FCM_STATE];
      } else if (this.#algorithm === 'som-olp' || classicSOM(this.#algorithm)) {
        options.initialPrototypes = w.centers;
        if (w.membership) options.initialMemberships = w.membership;
      } else options.initCenters = w.centers;
    }
    // nClusters inferred from original initialization must survive warm starts.
    if (this.#algorithm !== 'som-olp') options.nClusters = this.#cfg.k;
    this.#iterator = this.#algorithm === 'rmcm' ? this.#rmcmIterator(options) : kernels[this.#algorithm](this.#input, options);
    this.#warm = null;
  }
  /** Advance at most count complete iterations, yielding at existing work chunks.
   * timeBudgetMs is a soft deadline, checked after each chunk; it is not hard RT.
   * Returns metadata only. Explicit snapshot() pays for copied numerical arrays.
   */
  step(count = 1, { timeBudgetMs = 8, maxChunks = 1024 } = {}) {
    this.#assertAvailable(); integer(count, 'count');
    finiteNumber(timeBudgetMs, 'timeBudgetMs', 0); integer(maxChunks, 'maxChunks', 1);
    const start = performance.now(), initialIteration = this.#iteration;
    if (this.#done || count === 0) return { ...this.status, chunks: 0, completedIterations: 0, elapsedMs: 0, softBudget: true };
    let chunks = 0;
    this.#busy = true;
    try {
      checkCancelled(this.#options);
      if (!this.#iterator) this.#start();
      do {
        checkCancelled(this.#options);
        const next = this.#iterator.next(); ++chunks;
        if (next.done) {
          this.#committed = copyState(next.value); this.#iteration = next.value.iterations;
          this.#iterator = null; this.#done = true; this.#phase = 'done'; break;
        }
        this.#phase = next.value?.phase ?? 'iteration';
      } while (this.#iteration - initialIteration < count && chunks < maxChunks && performance.now() - start < timeBudgetMs);
    } catch (error) {
      this.#error = error; this.#phase = error.name === 'AbortError' ? 'cancelled' : 'error';
      this.#iterator?.return?.(); this.#iterator = null; this.#done = true; throw error;
    } finally { this.#busy = false; }
    return { ...this.status, chunks, completedIterations: this.#iteration - initialIteration,
      elapsedMs: performance.now() - start, softBudget: true };
  }
  /** All numerical arrays are copies. During a partial iteration, the result is
   * the previous complete iteration; final-only fields may be null before done.
   */
  snapshot({ project = false } = {}) {
    if (this.#disposed) throw new Error('Session is disposed');
    if (typeof project !== 'boolean') throw new TypeError('project must be boolean');
    const result = cloneOwned(this.#committed);
    if (project && result && classicSOM(this.#algorithm) && result.projectionStatus !== 'current-prototypes') {
      Object.assign(result, somProject(this.#input, result, { maxMemoryBytes: this.#options.maxMemoryBytes, bmuBackend: this.#options.bmuBackend }));
      result.V = result.embedding;
    }
    return { ...this.status, result };
  }
  #replace(input, options, cfg, { warm, graph, kind, reason }) {
    this.#iterator?.return?.(); this.#iterator = null;
    if (!graph) this.#prepared = null;
    this.#input = input; this.#options = options; this.#cfg = cfg; this.#warm = warm; this.#anchor = warm;
    this.#committed = null; this.#iteration = 0; this.#done = false; this.#phase = 'ready'; this.#error = null;
    ++this.#revision;
    this.#invalidation = { kind, reason, preparedGraph: graph ? 'retained' : 'invalidated' };
    return this.status;
  }
  updateData(input, { warmStart = true } = {}) {
    this.#assertAvailable();
    if (typeof warmStart !== 'boolean') throw new TypeError('warmStart must be boolean');
    const nextInput = ownedInput(input), options = cloneOwned(this.#options);
    const same = nextInput.nSamples === this.#input.nSamples && nextInput.nFeatures === this.#input.nFeatures && sameArray(nextInput.data, this.#input.data);
    if (same && warmStart) return this.status;
    const resized = nextInput.nSamples !== this.#input.nSamples || nextInput.nFeatures !== this.#input.nFeatures;
    if (!same) {
      delete options.initMembership; delete options.initialMemberships;
      if (this.#algorithm !== 'som-olp') options.nClusters = this.#cfg.k;
    }
    // Row identity is intentionally not inferred. Movement/reordering/birth/death
    // initializes memberships from retained centers, never from a stale row U.
    if (resized) {
      clearInitial(options);
      if (this.#algorithm !== 'som-olp') options.nClusters = this.#cfg.k;
    }
    const compatible = nextInput.nFeatures === this.#cfg.d;
    const source = this.#committed ?? this.#anchor;
    const warm = warmStart && compatible && source?.centers ? { centers: source.centers.slice() } : null;
    if (warm) { clearInitial(options); if (this.#algorithm === 'som-olp' || classicSOM(this.#algorithm)) options.initialPrototypes = warm.centers; else options.initCenters = warm.centers; }
    const cfg = validateSession(this.#algorithm, nextInput, options);
    return this.#replace(nextInput, options, cfg, { warm, graph: same && this.#prepared != null, kind: warm ? 'centers' : 'cold', reason: resized ? 'data-shape' : 'data-values' });
  }
  updateParameters(patch, { warmStart = true } = {}) {
    this.#assertAvailable();
    if (!patch || typeof patch !== 'object' || Array.isArray(patch)) throw new TypeError('patch must be an object');
    if (typeof warmStart !== 'boolean') throw new TypeError('warmStart must be boolean');
    const options = { ...cloneOwned(this.#options), ...cloneOwned(patch) };
    const explicitInitial = [...initialKeys, 'seed', 'initializer', 'pcaScale'].some(key => Object.hasOwn(patch, key));
    const shapeChange = Object.hasOwn(patch, 'nClusters') && patch.nClusters !== this.#cfg.k || Object.hasOwn(patch, 'grid') || Object.hasOwn(patch, 'gridShape');
    if (shapeChange && !initialKeys.some(key => Object.hasOwn(patch, key))) clearInitial(options);
    if (Object.hasOwn(patch, 'lam') && !Object.hasOwn(patch, 'lambda')) delete options.lambda;
    const cfg = validateSession(this.#algorithm, this.#input, options);
    const compatible = !shapeChange && cfg.k === this.#cfg.k && cfg.d === this.#cfg.d && cfg.q === this.#cfg.q;
    const source = this.#committed ?? this.#anchor;
    const warm = warmStart && compatible && !explicitInitial && source?.centers ? { centers: source.centers.slice(),
      ...(['fcm', 'som-olp'].includes(this.#algorithm) && source.membership ? { membership: source.membership.slice() } : {}) } : null;
    if (warm?.membership && source?.[SESSION_FCM_STATE]) warm[SESSION_FCM_STATE] = source[SESSION_FCM_STATE];
    const graph = this.#prepared != null && graphKeys.every(key => options[key] === this.#options[key]);
    return this.#replace(this.#input, options, cfg, { warm, graph, kind: warm ? (warm.membership ? 'membership-and-centers' : 'centers') : 'cold', reason: shapeChange ? 'model-shape' : 'parameters' });
  }
  /** Atomically replace data and the full option set. Used by bounded latest-
   * request workers; unlike updateParameters, parameters is not a patch. */
  configure(input, parameters, { warmStart = true } = {}) {
    this.#assertAvailable();
    if (!parameters || typeof parameters !== 'object' || Array.isArray(parameters)) throw new TypeError('parameters must be an object');
    if (typeof warmStart !== 'boolean') throw new TypeError('warmStart must be boolean');
    const next = ownedInput(input), options = cloneOwned(parameters);
    const cfg = validateSession(this.#algorithm, next, options);
    const sameData = next.nSamples === this.#input.nSamples && next.nFeatures === this.#input.nFeatures && sameArray(next.data, this.#input.data);
    const sameValue = (a, b) => a === b || ArrayBuffer.isView(a) && ArrayBuffer.isView(b) && sameArray(a, b);
    const initialChanged = [...initialKeys, 'seed', 'initializer', 'pcaScale'].some(key => !sameValue(options[key], this.#options[key]));
    const sameGrid = classicSOM(this.#algorithm) ? cfg.width === this.#cfg.width && cfg.height === this.#cfg.height : this.#algorithm !== 'som-olp' || cfg.k === this.#cfg.k && cfg.q === this.#cfg.q && sameArray(options.grid.data, this.#options.grid.data);
    const compatible = cfg.k === this.#cfg.k && cfg.d === this.#cfg.d && sameGrid;
    const source = this.#committed ?? this.#anchor;
    const warm = warmStart && compatible && !initialChanged && source?.centers ? { centers: source.centers.slice(),
      ...(sameData && ['fcm', 'som-olp'].includes(this.#algorithm) && source.membership ? { membership: source.membership.slice() } : {}) } : null;
    if (warm?.membership && source?.[SESSION_FCM_STATE]) warm[SESSION_FCM_STATE] = source[SESSION_FCM_STATE];
    const graph = this.#prepared != null && sameData && graphKeys.every(key => options[key] === this.#options[key]);
    return this.#replace(next, options, cfg, { warm, graph, kind: warm ? (warm.membership ? 'membership-and-centers' : 'centers') : 'cold', reason: 'configure' });
  }
  /** Restart the current data/configuration from its configured initialization. */
  reset() {
    this.#assertAvailable();
    return this.#replace(this.#input, this.#options, this.#cfg, { warm: null, graph: this.#prepared != null, kind: 'cold', reason: 'reset' });
  }
  dispose() {
    if (this.#disposed) return;
    this.#assertAvailable(); this.#iterator?.return?.(); this.#iterator = null; this.#prepared = null;
    this.#input = null; this.#options = null; this.#cfg = null; this.#warm = null; this.#anchor = null; this.#committed = null;
    this.#disposed = true; this.#done = true; this.#phase = 'disposed';
  }
}
export function createSession(algorithm, input, options = {}) { return new ClusteringSession(algorithm, input, options); }
