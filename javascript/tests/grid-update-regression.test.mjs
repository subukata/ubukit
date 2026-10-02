import test from 'node:test';
import assert from 'node:assert/strict';
import { createSession } from '../consumer/node_modules/ubukit-js/src/index.js';
const input = values => ({ data: Float64Array.from(values), nSamples: values.length, nFeatures: 1 });
const X = input([0, 1, 4, 5]);
const options = { gridShape: [2, 2], maxIterations: 1, sigma: 0 };
function finish(session) {
  for (let i = 0; !session.status.done; ++i) {
    assert.ok(i < 100, 'session must finish within bounded chunks');
    session.step(10, { timeBudgetMs: 1000, maxChunks: 100 });
  }
  return session.snapshot().result;
}

for (const algorithm of ['som', 'som_batch']) {
  for (const mode of ['moved', 'resized']) {
    for (const trained of [false, true]) {
      test(`${algorithm}: ${mode} data ${trained ? 'after a commit' : 'before stepping'} allows implicit grid-count update`, () => {
        const session = createSession(algorithm, X, options);
        try {
          if (trained) finish(session);
          session.updateData(input(mode === 'moved' ? [.25, 1.25, 4.25, 5.25] : [.25, 1.25, 4.25]));
          const revision = session.status.revision;
          session.updateParameters({ gridShape: [3, 2] });
          assert.equal(session.status.revision, revision + 1);
          assert.equal(session.status.invalidation.kind, 'cold');
          assert.equal(session.status.invalidation.reason, 'model-shape');
          const result = finish(session);
          assert.deepEqual(result.gridShape, [3, 2]);
          assert.equal(result.nClusters, 6);
          assert.equal(result.centers.length, 6);
          assert.equal(result.nSamples, mode === 'moved' ? 4 : 3);
        } finally { session.dispose(); }
      });
    }
  }
  test(`${algorithm}: same-product grid-shape change still cold-restarts`, () => {
    const session = createSession(algorithm, X, options);
    try {
      finish(session);
      session.updateData(input([.25, 1.25, 4.25, 5.25]));
      session.updateParameters({ gridShape: [1, 4] });
      assert.equal(session.status.invalidation.kind, 'cold');
      assert.deepEqual(finish(session).gridShape, [1, 4]);
    } finally { session.dispose(); }
  });
  test(`${algorithm}: caller-explicit existing nClusters remains a constraint`, () => {
    const session = createSession(algorithm, X, { ...options, nClusters: 4 });
    try {
      finish(session);
      session.updateData(input([.25, 1.25, 4.25, 5.25]));
      finish(session);
      const before = session.snapshot();
      assert.throws(() => session.updateParameters({ gridShape: [3, 2] }), /nClusters must equal/);
      assert.deepEqual(session.snapshot(), before);
    } finally { session.dispose(); }
  });
  test(`${algorithm}: explicitly mismatched replacement nClusters is transactional`, () => {
    const session = createSession(algorithm, X, options);
    try {
      finish(session);
      session.updateData(input([.25, 1.25, 4.25, 5.25]));
      finish(session);
      const before = session.snapshot();
      assert.throws(() => session.updateParameters({ gridShape: [3, 2], nClusters: 5 }), /nClusters must equal/);
      assert.deepEqual(session.snapshot(), before);
    } finally { session.dispose(); }
  });
  test(`${algorithm}: explicitly consistent replacement nClusters succeeds`, () => {
    const session = createSession(algorithm, X, { ...options, nClusters: 4 });
    try {
      finish(session);
      session.updateData(input([.25, 1.25, 4.25, 5.25]));
      session.updateParameters({ gridShape: [3, 2], nClusters: 6 });
      assert.equal(session.status.invalidation.kind, 'cold');
      assert.equal(finish(session).nClusters, 6);
    } finally { session.dispose(); }
  });
}
