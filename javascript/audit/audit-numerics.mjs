/** Independent read-only audit of the JS implementation against stored oracles. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
const runtimeURL = new URL(process.env.UBUKIT_AUDIT_SOURCE || '../src/external-metrics.js', import.meta.url);
const {
  adjustedRandScore, adjustedMutualInfoScore, adjustedScores,
  adjusted_rand_score, adjusted_mutual_info_score, adjusted_scores,
  ExternalMetricDomainError,
} = await import(runtimeURL.href);

const reference = JSON.parse(fs.readFileSync(new URL('./numeric-reference.json', import.meta.url), 'utf8'));
const started = performance.now();
const counts = { ari: 0, ami: 0, singular: 0, shared: 0, invariance: 0, invalid: 0, exactHighK: 0, syntheticARI: 0, syntheticConditional: 0, syntheticAMI: 0, earlySampleLimit: 0, symmetricWorkLimit: 0 };
let maxARI = 0, maxAMI = 0, worstAMI = null, maxInvariant = 0;
const methods = ['arithmetic', 'geometric', 'min', 'max'];
const equal = (actual, expected, tolerance, detail) => {
  assert.ok(Number.isFinite(actual), `Nonfinite result: ${detail}`);
  assert.ok(Math.abs(actual - expected) <= tolerance, `${detail}: ${actual} versus ${expected}, error ${Math.abs(actual - expected)}`);
};
for (const item of reference.fixtures) {
  const { name, x, y, ari, ami } = item;
  const actualARI = adjustedRandScore(x, y);
  equal(actualARI, ari, 2e-16, `${name} ARI`);
  maxARI = Math.max(maxARI, Math.abs(actualARI - ari));
  ++counts.ari;
  for (const method of methods) {
    const options = { averageMethod: method };
    if (ami[method] === null) {
      assert.throws(() => adjustedMutualInfoScore(x, y, options),
        error => error instanceof ExternalMetricDomainError && error.code === 'AMI_SINGULAR_NORMALIZATION');
      assert.throws(() => adjustedScores(x, y, options),
        error => error instanceof ExternalMetricDomainError && error.code === 'AMI_SINGULAR_NORMALIZATION');
      ++counts.singular;
      continue;
    }
    const value = adjustedMutualInfoScore(x, y, options);
    const err = Math.abs(value - ami[method]);
    equal(value, ami[method], 2e-13, `${name} AMI ${method}`);
    assert.ok(value <= 1, `${name} AMI ${method} exceeded 1`);
    if (err > maxAMI) { maxAMI = err; worstAMI = { name, method, value, reference: ami[method], error: err }; }
    ++counts.ami;
    const both = adjustedScores(x, y, options);
    assert.equal(both.ari, actualARI);
    assert.equal(both.ami, value);
    ++counts.shared;
    const swapped = adjustedMutualInfoScore(y, x, options);
    const reversed = adjustedMutualInfoScore([...x].reverse(), [...y].reverse(), options);
    const named = adjustedMutualInfoScore(x.map(v => `left:${v}`), y.map(v => `right:${v}`), options);
    for (const [label, changed] of [['swap', swapped], ['order', reversed], ['labels', named]]) {
      equal(changed, value, 2e-13, `${name} ${method} ${label} invariance`);
      maxInvariant = Math.max(maxInvariant, Math.abs(changed - value));
      ++counts.invariance;
    }
  }
}

assert.equal(adjusted_rand_score, adjustedRandScore);
assert.equal(adjusted_mutual_info_score, adjustedMutualInfoScore);
assert.equal(adjusted_scores, adjustedScores);

// Exact high-K answers do not use scikit-learn as a numerical oracle.
const highK = [];
for (const n of [1000, 3000, 100_000]) {
  const x = Uint32Array.from({ length: n }, (_, i) => i);
  const y = x.slice();
  x[1] = x[0]; y[3] = y[2];
  const exact = -2 / (n * (n - 1) - 2);
  equal(adjustedRandScore(x, y), exact, 2e-25, `high-K ${n} ARI`);
  const row = { n, exact, methods: {} };
  for (const averageMethod of methods) {
    const start = performance.now();
    const actual = adjustedMutualInfoScore(x, y, { averageMethod });
    equal(actual, exact, 1e-14, `high-K ${n} ${averageMethod}`);
    row.methods[averageMethod] = { actual, absoluteError: Math.abs(actual - exact), milliseconds: performance.now() - start };
    ++counts.exactHighK;
  }
  highK.push(row);
}

// Read the unchanged source into an isolated data module exposing private
// arithmetic helpers solely for huge synthetic tables; runtime file is unedited.
const source = fs.readFileSync(runtimeURL, 'utf8');
const internals = await import(`data:text/javascript,${encodeURIComponent(source + '\nexport { ariFromTable, pairExpectation, amiFromTable };')}`);
const synthetic = JSON.parse(fs.readFileSync(new URL('./synthetic-reference.json', import.meta.url), 'utf8'));
for (const item of synthetic.ari) {
  equal(internals.ariFromTable(item), item.ari, 6e-17, `synthetic ARI N=${item.n}`);
  ++counts.syntheticARI;
}
for (const { n, first, other, expected } of synthetic.conditional) {
  equal(internals.pairExpectation(n, first, other), expected, Math.max(1e-300, 2e-15 * Math.abs(expected)),
    `synthetic conditional N=${n}, first=${first}, other=${other}`);
  ++counts.syntheticConditional;
}
for (const item of synthetic.ami) {
  for (const [averageMethod, expected] of Object.entries(item.ami)) {
    const actual = internals.amiFromTable(item, { averageMethod, maxExpectedTerms: 1000 });
    equal(actual, expected, 2e-13, `synthetic AMI ${item.n} ${averageMethod}`);
    const err = Math.abs(actual - expected);
    if (err > maxAMI) { maxAMI = err; worstAMI = { name: `synthetic-${item.n}`, method: averageMethod, value: actual, reference: expected, error: err }; }
    ++counts.syntheticAMI;
  }
}

const badLabels = [null, {}, 'abcd', new DataView(new ArrayBuffer(4)), [[1]], [undefined],
  [null], [NaN], [Infinity], [-Infinity], [0.5], [Number.MAX_SAFE_INTEGER + 1], [1n], [true],
  new BigInt64Array([0n]), ['x', 1], new Array(2)];
for (const labels of badLabels) {
  for (const fn of [adjustedRandScore, adjustedMutualInfoScore, adjustedScores]) {
    assert.throws(() => fn(labels, Array(labels?.length ?? 0).fill(0)), { name: /TypeError|RangeError/ });
    ++counts.invalid;
  }
}
for (const fn of [adjustedRandScore, adjustedMutualInfoScore, adjustedScores]) {
  assert.throws(() => fn([0], []), RangeError); ++counts.invalid;
}
for (const fn of [adjustedMutualInfoScore, adjustedScores]) {
  for (const option of [null, 'min', [], { averageMethod: 'median' },
    { maxExpectedTerms: 0 }, { maxExpectedTerms: -1 }, { maxExpectedTerms: 0.5 },
    { maxExpectedTerms: Infinity }, { maxExpectedTerms: Number.MAX_SAFE_INTEGER + 1 }]) {
    assert.throws(() => fn([0, 1], [0, 1], option), { name: /TypeError|RangeError/ });
    ++counts.invalid;
  }
  assert.throws(() => fn([0, 0, 1, 1], [0, 1, 0, 1], { maxExpectedTerms: 1 }),
    error => error instanceof ExternalMetricDomainError && error.code === 'AMI_WORK_LIMIT');
  ++counts.invalid;
  const tooLong = new Proxy([], {
    get(target, key, receiver) {
      if (key === 'length') return 2 ** 26 + 1;
      if (typeof key === 'string' && /^\d+$/.test(key)) throw new Error('Read a label before enforcing sample limit');
      return Reflect.get(target, key, receiver);
    },
  });
  assert.throws(() => fn(tooLong, tooLong), error => error instanceof ExternalMetricDomainError && error.code === 'AMI_SAMPLE_LIMIT');
  ++counts.earlySampleLimit;
}
const entropyTieA = [0, 0, 0, 0, 1, 2, 3, 4], entropyTieB = [0, 0, 1, 1, 2, 2, 3, 3];
const budgetOutcome = (x, y, averageMethod, maxExpectedTerms) => {
  try { return adjustedMutualInfoScore(x, y, { averageMethod, maxExpectedTerms }); }
  catch (error) { return error.code; }
};
for (const method of methods) for (const budget of [3, 4, 5, 100]) {
  assert.equal(budgetOutcome(entropyTieA, entropyTieB, method, budget), budgetOutcome(entropyTieB, entropyTieA, method, budget));
  ++counts.symmetricWorkLimit;
}
for (const Type of [Array, Uint8Array, Int16Array, Uint32Array, Float32Array, Float64Array]) {
  const x = Type.from([0, 0, 1, 1]), y = Type.from([0, 1, 0, 1]);
  const before = [Array.from(x), Array.from(y)];
  const value = adjustedScores(x, y);
  equal(value.ari, -0.5, 0, 'typed ARI');
  equal(value.ami, -0.5, 2e-15, 'typed AMI');
  assert.deepEqual([Array.from(x), Array.from(y)], before);
}
const report = {
  status: 'passed', node: process.version,
  runtimePath: process.env.UBUKIT_AUDIT_SOURCE || '../src/external-metrics.js',
  runtimeSHA256: createHash('sha256').update(source).digest('hex'),
  reference: reference.reference,
  counts, maximumAbsoluteErrors: { ari: maxARI, ami: maxAMI, invariance: maxInvariant },
  worstAMI, highK,
  elapsedSeconds: (performance.now() - started) / 1000,
};
fs.writeFileSync(new URL('./audit-results.json', import.meta.url), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report, null, 2));
