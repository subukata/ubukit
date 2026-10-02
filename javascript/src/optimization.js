/** Dependency-free seeded TPE for flat, mixed hyperparameter search spaces. */
const UINT32_SCALE = 4294967296;
const hasOwn = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
const copyParams = params => Object.fromEntries(Object.entries(params));
const copyTrial = trial => ({...trial, params: copyParams(trial.params)});
const fail = message => { throw new RangeError(message); };
function finite(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value)) fail(`${name} must be a finite number`);
  return value;
}
function integer(value, name, minimum = 0) {
  if (!Number.isSafeInteger(value) || value < minimum) fail(`${name} must be a safe integer >= ${minimum}`);
  return value;
}
function scalarKey(value) {
  if (value === null) return 'null';
  if (typeof value === 'string') return `s:${value}`;
  if (typeof value === 'boolean') return value ? 'b:1' : 'b:0';
  if (typeof value === 'number' && Number.isFinite(value)) return `n:${Object.is(value, -0) ? 0 : value}`;
  fail('categorical choices must be null, booleans, strings, or finite numbers');
}
function categoryKey(value) {
  if (typeof value === 'number' && Number.isInteger(value) && !Number.isSafeInteger(value)) fail('integer-valued categorical numbers must be safe integers');
  return scalarKey(value);
}
function makeRandom(seed) {
  integer(seed, 'seed'); if (seed > 0xFFFFFFFF) fail('seed must be in [0, 4294967295]');
  let state = seed >>> 0;
  const random = () => {
    state = (state + 0x6D2B79F5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / UINT32_SCALE;
  };
  random.open = () => random() + .5 / UINT32_SCALE;
  random.getState = () => state >>> 0;
  random.setState = value => { state = value >>> 0; };
  return random;
}

/** Inclusive finite float bounds; log samples uniformly in log space. */
export function floatRange(low, high, {log = false} = {}) {
  const descriptor = {type: 'float', low, high, log};
  validateDomain(descriptor, 'range');
  return descriptor;
}
/** Inclusive safe-integer bounds. */
export function intRange(low, high, {log = false} = {}) {
  const descriptor = {type: 'int', low, high, log};
  validateDomain(descriptor, 'range');
  return descriptor;
}
export function categorical(choices) {
  const domain = validateDomain({type: 'categorical', choices}, 'categorical');
  return {type: 'categorical', choices: domain.choices};
}
function validateDomain(spec, name) {
  if (spec === null || typeof spec !== 'object' || Array.isArray(spec)) fail(`${name} must be a distribution object`);
  if (spec.type === 'categorical') {
    if (Object.keys(spec).some(k => !['type','choices'].includes(k))) fail(`${name} has unsupported categorical fields`);
    if (!Array.isArray(spec.choices) || !spec.choices.length) fail(`${name}.choices must be a nonempty array`);
    const count = spec.choices.length, choices = new Array(count), keys = new Array(count);
    for (let i = 0; i < count; i++) {
      if (!hasOwn(spec.choices, i)) fail(`${name}.choices must be a dense array with own indexed values`);
      const value = spec.choices[i];
      keys[i] = categoryKey(value);
      choices[i] = value;
    }
    if (new Set(keys).size !== keys.length) fail(`${name}.choices must be unique`);
    return {name, type: spec.type, choices, keys};
  }
  if (spec.type !== 'float' && spec.type !== 'int') fail(`${name}.type must be float, int, or categorical`);
  if (Object.keys(spec).some(k => !['type','low','high','log'].includes(k))) fail(`${name} has unsupported range fields`);
  const low = finite(spec.low, `${name}.low`), high = finite(spec.high, `${name}.high`);
  if (low > high) fail(`${name}.low must be <= high`);
  if (spec.log !== undefined && typeof spec.log !== 'boolean') fail(`${name}.log must be a boolean`);
  if (spec.type === 'int' && (!Number.isSafeInteger(low) || !Number.isSafeInteger(high))) fail(`${name} integer bounds must be safe integers`);
  const log = spec.log ?? false;
  if (log && low <= 0) fail(`${name} logarithmic bounds must be positive`);
  const count = spec.type === 'int' ? high - low + 1 : null;
  if (spec.type === 'int' && !Number.isSafeInteger(count)) fail(`${name} integer range size must be a safe integer`);
  const width = high - low;
  const relative = width / low;
  const closeLog = Number.isFinite(relative) && relative <= 1;
  const span = log ? (spec.type === 'int' ? Math.log1p(count / (low - .5)) : low===high ? 1 : (closeLog ? Math.log1p(relative) : Math.log(high) - Math.log(low))) : 1;
  return {name, type: spec.type, low, high, log, count, width, span, closeLog};
}
function validateSpace(space) {
  if (space === null || typeof space !== 'object' || Array.isArray(space)) fail('space must be an object of named distributions');
  const names = Object.keys(space).sort();
  if (!names.length || names.some(name => !name.length)) fail('space must contain nonempty parameter names');
  return names.map(name => validateDomain(space[name], name));
}

// Numerical mixture implementation is shared with the Python reference.

function encode(domain, value) {
  if (domain.type === 'categorical') {
    const index = domain.keys.indexOf(categoryKey(value));
    if (index < 0) fail(`${domain.name} is not an allowed category`);
    return index;
  }
  finite(value, domain.name);
  if (value < domain.low || value > domain.high) fail(`${domain.name} is outside its bounds`);
  if (domain.type === 'int' && !Number.isSafeInteger(value)) fail(`${domain.name} must be a safe integer`);
  if (domain.low === domain.high) return .5;
  if (domain.type === 'int') {
    const offset = value - domain.low + .5;
    return domain.log ? Math.log1p(offset / (domain.low - .5)) / domain.span : offset / domain.count;
  }
  if (domain.log) {
    return (domain.closeLog ? Math.log1p((value-domain.low)/domain.low) : Math.log(value)-Math.log(domain.low)) / domain.span;
  }
  return Math.max(0, Math.min(1, Number.isFinite(domain.width) ? (value - domain.low) / domain.width : (value / 2 - domain.low / 2) / (domain.high / 2 - domain.low / 2)));
}
function decode(domain, u) {
  if (domain.type === 'categorical') return domain.choices[u];
  if (domain.low === domain.high) return domain.low;
  u = Math.min(1,Math.max(0,u));
  if (u===0) return domain.low;
  if (u===1) return domain.high;
  if (domain.type === 'int') {
    const offset = domain.log ? (domain.low - .5) * Math.expm1(u * domain.span) : u * domain.count;
    return domain.low + Math.min(domain.count - 1, Math.max(0, Math.floor(offset)));
  }
  let value;
  if (domain.log) value = domain.closeLog ? domain.low + domain.low * Math.expm1(u * domain.span) : Math.exp(Math.log(domain.low) + u * domain.span);
  else value = (1 - u) * domain.low + u * domain.high;
  return Math.max(domain.low, Math.min(domain.high, value));
}
function binEdges(domain, value) {
  if (domain.low === domain.high) return [0, 1];
  const offset = value - domain.low;
  return domain.log ? [Math.log1p(offset / (domain.low - .5)) / domain.span, Math.log1p((offset + 1) / (domain.low - .5)) / domain.span] : [offset / domain.count, (offset + 1) / domain.count];
}
function binWidth(domain,value) { return domain.log ? Math.log1p(1/(domain.low-.5+value-domain.low))/domain.span : 1/domain.count; }
function validateParams(domains, params) {
  if (params === null || typeof params !== 'object' || Array.isArray(params) || Object.keys(params).length !== domains.length) fail('params must match the search space exactly');
  const entries = [];
  for (const domain of domains) {
    if (!hasOwn(params, domain.name)) fail(`params missing ${domain.name}`);
    const value = params[domain.name];
    encode(domain, value);
    entries.push([domain.name, value]);
  }
  return Object.fromEntries(entries);
}
function paramsKey(domains, params) {
  return JSON.stringify(domains.map(domain => scalarKey(params[domain.name])));
}

/** Raised only when every member of a finite search space has been reserved. */
export class SearchSpaceExhaustedError extends Error {
  constructor(message = 'The finite search space is exhausted') { super(message); this.name = 'SearchSpaceExhaustedError'; }
}

export class ProposalError extends Error {
  constructor(message) { super(message); this.name='ProposalError'; }
}

function normalizeOptions(options) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) fail('options must be an object');
  const o = {
    seed: options.seed ?? 0,
    direction: options.direction ?? 'minimize',
    sampler: options.sampler ?? 'tpe',
    nStartupTrials: options.nStartupTrials ?? 12,
    nCandidates: options.nCandidates ?? 24,
    gamma: options.gamma ?? .15,
    minBandwidth: options.minBandwidth ?? .03,
    multivariate: options.multivariate ?? true,
    weights: options.weights ?? 'ei',
    avoidDuplicates: options.avoidDuplicates ?? !(options.allowDuplicates ?? false),
    maxDuplicateAttempts: options.maxDuplicateAttempts ?? 64
  };
  integer(o.seed, 'seed'); if (o.seed > 0xFFFFFFFF) fail('seed must be in [0, 4294967295]');
  if (options.allowDuplicates !== undefined && typeof options.allowDuplicates !== 'boolean') fail('allowDuplicates must be a boolean');
  if (options.allowDuplicates !== undefined && options.avoidDuplicates !== undefined) fail('use avoidDuplicates or allowDuplicates, not both');
  integer(o.nStartupTrials, 'nStartupTrials', 2);
  integer(o.nCandidates, 'nCandidates', 1);
  integer(o.maxDuplicateAttempts, 'maxDuplicateAttempts', 1);
  if (!['minimize', 'maximize'].includes(o.direction)) fail('direction must be minimize or maximize');
  if (!['tpe', 'random'].includes(o.sampler)) fail('sampler must be tpe or random');
  if (!['ei', 'uniform'].includes(o.weights)) fail('weights must be ei or uniform');
  finite(o.gamma, 'gamma'); if (!(o.gamma > 0 && o.gamma < 1)) fail('gamma must be between 0 and 1');
  finite(o.minBandwidth, 'minBandwidth'); if (!(o.minBandwidth > 0 && o.minBandwidth <= 1)) fail('minBandwidth must be in (0, 1]');
  if (typeof o.multivariate !== 'boolean' || typeof o.avoidDuplicates !== 'boolean') fail('multivariate and avoidDuplicates must be booleans');
  return Object.freeze(o);
}

/** Stateful sequential ask/tell optimizer. Objective code always runs in the caller. */
export class TPEOptimizer {
  constructor(space, options = {}) {
    this._domains = validateSpace(space);
    this.options = normalizeOptions(options);
    this._random = makeRandom(this.options.seed);
    this._history = [];
    this._models = null;
    this.stopReason = 'not_started';
    this._pending = new Map();
    this._seen = new Set();
    this._finiteSize = 1;
    for (const d of this._domains) {
      const count = d.type === 'categorical' ? d.choices.length : d.low === d.high ? 1 : d.type === 'int' ? d.count : Infinity;
      this._finiteSize *= count;
      if (!Number.isSafeInteger(this._finiteSize)) this._finiteSize = Infinity;
    }
  }
  _randomParams() { return Object.fromEntries(this._domains.map(d => [d.name, randomDomain(d,this._random)])); }
  _available(params) { return !this.options.avoidDuplicates || !this._seen.has(paramsKey(this._domains,params)); }
  ask() {
    if (this.options.avoidDuplicates && this._seen.size >= this._finiteSize) throw new SearchSpaceExhaustedError();
    const complete = this._history.filter(t => t.state === 'complete');
    if (this.options.sampler === 'tpe' && complete.length >= this.options.nStartupTrials && complete.some(t=>t.value!==complete[0].value)) {
      if (this._models===null) this._models=buildModel(this._domains,complete,this.options);
      const [good,bad]=this._models;
      let best=null,bestScore=-Infinity;
      for(let i=0;i<this.options.nCandidates;i++) {
        const params=good.sample(this._random);
        if(!this._available(params))continue;
        const score=good.logpdf(params)-bad.logpdf(params);
        if(best===null||score>bestScore){best=params;bestScore=score;}
      }
      if(best!==null)return this._reserve(best);
    }
    for(let attempt=0;attempt<this.options.maxDuplicateAttempts;attempt++) {
      const params=this._randomParams();
      if(this._available(params))return this._reserve(params);
    }
    if(this._finiteSize<=100000) {
      const start=Math.min(this._finiteSize-1,Math.floor(this._random()*this._finiteSize));
      for(let shift=0;shift<this._finiteSize;shift++) {
        let code=(start+shift)%this._finiteSize;
        const entries=[];
        for(let j=this._domains.length-1;j>=0;j--) {
          const d=this._domains[j],n=d.type==='categorical'?d.choices.length:d.low===d.high?1:d.count;
          const offset=code%n;code=Math.floor(code/n);
          entries.push([d.name,d.type==='categorical'?d.choices[offset]:d.type==='int'?d.low+offset:d.low]);
        }
        const params=Object.fromEntries(entries.reverse());
        if(this._available(params))return this._reserve(params);
      }
    }
    throw new ProposalError('Could not find an unseen candidate after bounded retries; allow duplicates or change the space');
  }
  _reserve(params) {
    const key = paramsKey(this._domains, params);
    const trial = {id: this._history.length, params: copyParams(params), state: 'running', value: null, error: null};
    this._history.push(trial);
    this.stopReason = 'running';
    this._pending.set(trial.id, trial);
    this._seen.add(key);
    return copyTrial(trial);
  }
  tell(id, value = null, {state = 'complete', error = null} = {}) {
    integer(id, 'trial id');
    const trial = this._pending.get(id);
    if (!trial) fail('tell requires an outstanding trial id issued by ask');
    if (!['complete', 'fail', 'cancelled'].includes(state)) fail('state must be complete, fail, or cancelled');
    if (state === 'complete') { finite(value, 'objective value'); if (error !== null) fail('a complete trial cannot contain an error'); }
    else if (value !== null) fail('failed/cancelled trials cannot contain a value');
    if (error !== null && typeof error !== 'string') fail('error must be a string or null');
    trial.state = state;
    trial.value = state === 'complete' ? value : null;
    if (error !== null) trial.error = error;
    this._pending.delete(id);
    if (state==='complete') this._models=null;
    return copyTrial(trial);
  }
  /** Import a completed external observation without consuming seeded randomness. */
  addTrial(params, value = null, {state = 'complete', error = null} = {}) {
    const safeParams = validateParams(this._domains, params);
    if (!['complete', 'fail', 'cancelled'].includes(state)) fail('state must be complete, fail, or cancelled');
    if (state === 'complete') { finite(value, 'objective value'); if (error !== null) fail('a complete trial cannot contain an error'); }
    else if (value !== null) fail('failed/cancelled trials cannot contain a value');
    if (error !== null && typeof error !== 'string') fail('error must be a string or null');
    const previousStopReason = this.stopReason;
    const issued = this._reserve(safeParams);
    const completed = this.tell(issued.id, value, {state, error});
    this.stopReason = previousStopReason;
    return completed;
  }
  get history() { return this._history.map(copyTrial); }
  result() {
    let best = null;
    for (const t of this._history) if (t.state === 'complete' && (best === null || (this.options.direction === 'minimize' ? t.value < best.value : t.value > best.value))) best = t;
    return {bestParams: best === null ? null : copyParams(best.params), bestValue: best === null ? null : best.value, bestTrial: best === null ? null : copyTrial(best), bestTrialId: best === null ? null : best.id, history: this.history, nAttempted: this._history.length, nCompleted: this._history.filter(t=>t.state==='complete').length, stopReason: this.stopReason};
  }
}

function checkAbort(signal) {
  if (signal?.aborted) { const error = new Error('Optimization cancelled'); error.name = 'AbortError'; throw error; }
}
function runOptions(objective, options) {
  if (typeof objective !== 'function') throw new TypeError('objective must be a function');
  if (options.nTrials !== undefined && options.budget !== undefined) fail('use nTrials or budget, not both');
  const nTrials = integer(options.nTrials ?? options.budget ?? 100, 'nTrials');
  if (options.continueOnError !== undefined && typeof options.continueOnError !== 'boolean') fail('continueOnError must be a boolean');
  if (options.onTrial !== undefined && typeof options.onTrial !== 'function') fail('onTrial must be a function');
  return nTrials;
}
function beginTrial(optimizer, signal) {
  checkAbort(signal);
  try { return optimizer.ask(); } catch (error) { if (error instanceof SearchSpaceExhaustedError) { optimizer.stopReason='space_exhausted'; return null; } if (error instanceof ProposalError) { optimizer.stopReason='proposal_failed';return null; } throw error; }
}
function recordFailure(optimizer, trial, error, options) {
  const cancelled = error?.name === 'AbortError' || options.signal?.aborted;
  const result = optimizer.tell(trial.id, null, {state: cancelled ? 'cancelled' : 'fail', error: String(error?.message ?? error)});
  if (cancelled || !options.continueOnError) throw error;
  return result;
}
/** Synchronous objective wrapper. Use optimizeAsync for promise-returning functions. */
export function optimize(objective, space, options = {}) {
  const nTrials = runOptions(objective, options), optimizer = new TPEOptimizer(space, options);
  optimizer.stopReason='budget_exhausted';
  for (let i = 0; i < nTrials; i++) {
    const trial = beginTrial(optimizer, options.signal); if (trial === null) break;
    let completed;
    try {
      const value = objective(copyParams(trial.params), {...trial, params: copyParams(trial.params)});
      if (value !== null && typeof value === 'object' && typeof value.then === 'function') {
        // Avoid an unhandled rejection while explaining the synchronous contract.
        Promise.resolve(value).catch(() => {});
        throw new TypeError('A promise objective requires optimizeAsync');
      }
      checkAbort(options.signal);
      completed = optimizer.tell(trial.id, value);
    } catch (error) { completed = recordFailure(optimizer, trial, error, options); }
    if(options.onTrial?.(completed)===false){optimizer.stopReason='callback_stopped';break;}
    optimizer.stopReason='budget_exhausted';
  }
  return optimizer.result();
}
/** Sequential async-objective wrapper; does not create threads or workers. */
export async function optimizeAsync(objective, space, options = {}) {
  const nTrials = runOptions(objective, options), optimizer = new TPEOptimizer(space, options);
  optimizer.stopReason='budget_exhausted';
  for (let i = 0; i < nTrials; i++) {
    const trial = beginTrial(optimizer, options.signal); if (trial === null) break;
    let completed;
    try {
      const value = await objective(copyParams(trial.params), {...trial, params: copyParams(trial.params)});
      checkAbort(options.signal);
      completed = optimizer.tell(trial.id, value);
    } catch (error) { completed = recordFailure(optimizer, trial, error, options); }
    if(await options.onTrial?.(completed)===false){optimizer.stopReason='callback_stopped';break;}
    optimizer.stopReason='budget_exhausted';
  }
  return optimizer.result();
}

const SQRT2PI = 2.5066282746310002;
function normalSfPositive(z) {
  if (z === 0) return .5;
  const t = 1 / (1 + .2316419 * z);
  const p = t * (.319381530 + t * (-.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return Math.exp(-.5 * z * z) * p / SQRT2PI;
}
function normalCdf(x) { return x < 0 ? normalSfPositive(-x) : 1 - normalSfPositive(x); }
// Centered fourth-order normal integral. Under this branch condition, the
// sixth-order relative truncation remainder is below 4e-12 before roundoff.
function normalLogNarrow(mid, width) {
  if (!(width > 0 && width <= .01 && Math.abs(mid) * width <= .1)) return null;
  const mw = mid * width, mw2 = mw * mw, w2 = width * width;
  const correction = (mw2 - w2) / 24 + (mw2 * mw2 - 6 * mw2 * w2 + 3 * w2 * w2) / 1920;
  return -.5 * mid * mid - Math.log(SQRT2PI) + Math.log(width) + Math.log1p(correction);
}
function normalInterval(a, b) {
  if (b <= a) return 0;
  const narrow = normalLogNarrow((a + b) * .5, b - a);
  if (narrow !== null) return Math.exp(narrow);
  const value = a >= 0 ? normalSfPositive(a) - normalSfPositive(b) : b <= 0 ? normalSfPositive(-b) - normalSfPositive(-a) : 1 - normalSfPositive(-a) - normalSfPositive(b);
  return Math.max(0, value);
}
function normalPpf(probability) {
  const p = Math.min(1 - 1e-15, Math.max(1e-15, probability));
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-.007784894002430293, -.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [.007784695709041462, .3224671290700398, 2.445134137142996, 3.754408661907416];
  if (p < .02425 || p > .97575) {
    const q = Math.sqrt(-2 * Math.log(p < .02425 ? p : 1 - p));
    const z = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    return p < .02425 ? z : -z;
  }
  const q = p - .5, r = q*q;
  return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / ((((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r)+1);
}
function logsumexp(values) {
  let maximum = -Infinity;
  for (const value of values) if (value > maximum) maximum = value;
  if (maximum === -Infinity) return maximum;
  let sum = 0;
  for (const value of values) sum += Math.exp(value - maximum);
  return maximum + Math.log(sum);
}
function domainFixed(d) { return d.type === 'categorical' ? d.choices.length === 1 : d.low === d.high; }
function randomDomain(d, random) { return d.type === 'categorical' ? d.choices[Math.min(d.choices.length - 1, Math.floor(random() * d.choices.length))] : decode(d, random.open()); }
function priorLog(d, value) {
  if (domainFixed(d)) return 0;
  if (d.type === 'categorical') return -Math.log(d.choices.length);
  if (d.type === 'int') return Math.log(binWidth(d,value));
  return 0;
}
class KDE {
  constructor(domains, rows, weights, options) {
    this.domains = domains; this.n = rows.length; this.multivariate = options.multivariate;
    const total = weights.reduce((s,w) => s+w,0) + 1;
    this.weights = weights.map(w => w/total); this.weights.push(1/total);
    this.logWeights = this.weights.map(w => w>0 ? Math.log(w) : -Infinity);
    this.centers = domains.map(d => rows.map(row => encode(d,row[d.name])));
    this.sigmas = []; this.norms = []; this.cdfLows = [];
    this.catAlpha = 1/(this.n+1);
    const floor = Math.max(options.minBandwidth,1/(this.n+1)**2);
    for (let j=0;j<domains.length;j++) {
      const d = domains[j], centers = this.centers[j];
      if (d.type === 'categorical' || domainFixed(d)) { this.sigmas.push(null); this.norms.push(null); this.cdfLows.push(null); continue; }
      const ordered = centers.map((x,i) => [x,i]); ordered.push([.5,this.n]); ordered.sort((a,b) => a[0]-b[0] || a[1]-b[1]);
      const sigmas = new Float64Array(this.n);
      for (let pos=0;pos<ordered.length;pos++) {
        const [x,i] = ordered[pos]; if (i===this.n) continue;
        const left = pos ? ordered[pos-1][0] : 0, right = pos+1<ordered.length ? ordered[pos+1][0] : 1;
        sigmas[i] = Math.min(1,Math.max(floor,x-left,right-x));
      }
      this.sigmas.push(sigmas);
      this.cdfLows.push(centers.map((mu,i) => normalCdf(-mu/sigmas[i])));
      this.norms.push(centers.map((mu,i) => normalInterval(-mu/sigmas[i],(1-mu)/sigmas[i])));
    }
  }
  component(random) {
    const u=random(); let total=0;
    for(let i=0;i<this.weights.length;i++){total+=this.weights[i];if(u<total)return i;}
    return this.n;
  }
  sample(random) {
    const component = this.multivariate ? this.component(random) : null, entries = [];
    for(let j=0;j<this.domains.length;j++){
      const d=this.domains[j],i=this.multivariate?component:this.component(random);
      let value;
      if(i===this.n)value=randomDomain(d,random);
      else if(d.type==='categorical'){
        const u=random(),alpha=this.catAlpha;
        const k=u<1-alpha?this.centers[j][i]:Math.min(d.choices.length-1,Math.floor((u-(1-alpha))/alpha*d.choices.length));
        value=decode(d,k);
      }else if(domainFixed(d)){random.open();value=d.low;}
      else{
        const p=this.cdfLows[j][i]+random.open()*this.norms[j][i];
        value=decode(d,this.centers[j][i]+this.sigmas[j][i]*normalPpf(p));
      }
      entries.push([d.name,value]);
    }
    return Object.fromEntries(entries);
  }
  axisLogs(j,value) {
    const d=this.domains[j];
    if(domainFixed(d))return new Float64Array(this.n+1);
    const logs = new Float64Array(this.n+1);
    if(d.type==='categorical'){
      const encoded=encode(d,value),off=this.catAlpha/d.choices.length,on=1-this.catAlpha+off;
      for(let i=0;i<this.n;i++)logs[i]=Math.log(this.centers[j][i]===encoded?on:off);
      logs[this.n]=-Math.log(d.choices.length);return logs;
    }
    if(d.type==='int'){
      const [low,high]=binEdges(d,value),width=binWidth(d,value);
      for(let i=0;i<this.n;i++){
        const mu=this.centers[j][i],sigma=this.sigmas[j][i];
        const narrow=normalLogNarrow(((low+high)*.5-mu)/sigma,width/sigma);
        if(narrow!==null){logs[i]=narrow-Math.log(this.norms[j][i]);continue;}
        const mass=normalInterval((low-mu)/sigma,(high-mu)/sigma);
        logs[i]=mass>0?Math.log(mass/this.norms[j][i]):-Infinity;
      }
      logs[this.n]=priorLog(d,value);return logs;
    }
    const x=encode(d,value);
    for(let i=0;i<this.n;i++){
      const mu=this.centers[j][i],sigma=this.sigmas[j][i],z=(x-mu)/sigma;
      logs[i]=-.5*z**2-Math.log(sigma*SQRT2PI*this.norms[j][i]);
    }
    return logs;
  }
  logpdf(params) {
    if(this.multivariate){
      const components=this.logWeights.slice();
      for(let j=0;j<this.domains.length;j++){
        const logs=this.axisLogs(j,params[this.domains[j].name]);
        for(let i=0;i<=this.n;i++)components[i]+=logs[i];
      }
      return logsumexp(components);
    }
    let value=0;
    for(let j=0;j<this.domains.length;j++){
      const logs=this.axisLogs(j,params[this.domains[j].name]);
      for(let i=0;i<=this.n;i++)logs[i]+=this.logWeights[i];
      value+=logsumexp(logs);
    }
    return value;
  }
}
function buildModel(domains, complete, options) {
  const sign=options.direction==='minimize'?1:-1;
  const ordered=complete.slice().sort((a,b)=>sign*a.value<sign*b.value?-1:sign*a.value>sign*b.value?1:a.id-b.id);
  const split=Math.min(ordered.length-1,Math.max(1,Math.ceil(options.gamma*ordered.length)));
  const good=ordered.slice(0,split),bad=ordered.slice(split);
  let weights=Array(split).fill(1);
  if(options.weights==='ei'){
    let scale=0;for(const t of ordered)scale=Math.max(scale,Math.abs(t.value));scale ||= 1;
    const threshold=sign*bad[0].value/scale;
    const diffs=good.map(t=>Math.max(0,threshold-sign*t.value/scale)),total=diffs.reduce((s,v)=>s+v,0);
    if(total>0){weights=diffs.map(v=>Math.max(1e-12,v/total*split));const mean=weights.reduce((s,w)=>s+w,0)/split;weights=weights.map(w=>w/mean);}
  }
  return [new KDE(domains,good.map(t=>t.params),weights,options),new KDE(domains,bad.map(t=>t.params),Array(bad.length).fill(1),options)];
}
