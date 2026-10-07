// How typed code calls UbuKit: checked by tsc in strict mode against the
// published declarations (types/, from "npm run build:types"), never run;
// "npm run check" runs it. Each @ts-expect-error marks a call that must be
// rejected; tsc reports a directive that is not needed, so a part of the API
// that loses its types (a broken JSDoc becomes `any`) fails the check, and so
// does an option missing from a function's declared options.
import * as ub from '../types/index.js';

const X = [[0, 0], [1, 1], [5, 5]];

/** @returns {number} */
export function fit() {
  return ub.fcm(X, 2, { m: 2, seed: 0 }).nIter;
}

/** @returns {ub.Result} */
export function followMovingData() {
  const run = ub.steps.somOlp(X, [2, 2], { lam: 0.5, gamma: 1, maxIter: Infinity });
  run.next();
  const { value, done } = run.next(X.map(([a, b]) => [a + 1, b]));
  return done ? value : value.result();
}

/** The classes Python exports are types of the package entry. @param {ub.Progress} p @returns {ub.Result} */
export function resultOf(p) {
  return p.result();
}

/** @returns {Promise<ub.TPEResult>} */
export function search() {
  return ub.minimize(p => Number(p.x), { x: ub.uniform(0, 1) });
}

export async function everyOption() {
  const init = [[0, 0], [5, 5]];
  ub.kmeans(X, 2, { init, maxIter: 10, seed: 0 });
  ub.fcm(X, 2, { m: 1.5, init, maxIter: 10, tol: 1e-6, seed: 0 });
  ub.efcm(X, 2, { tau: 1, init, maxIter: 10, tol: 1e-6, seed: 0 });
  ub.rcm(X, 2, { alpha: 1.2, beta: 0.1, p: 2, init, maxIter: 10, seed: 0 });
  ub.rmcm(X, 2, 1, { init, maxIter: 10, maxEdges: 100, seed: 0 });
  ub.som(X, [2, 2], { epochs: 2, sigma: 1, sigmaEnd: 0.5, lr: 0.5, lrEnd: 0.01, init: 'pca', shuffle: false, seed: 0 });
  ub.batchSom(X, [2, 2], { epochs: 2, sigma: 1, sigmaEnd: 0.5, init: 'pca' });
  ub.somOlp(X, [2, 2], { lam: 0.5, gamma: 1, init: 'pca', pcaScale: 2, maxIter: 10, tol: 1e-6 });
  ub.ami([0, 1], [1, 0], { average: 'max' });
  const space = { m: ub.uniform(1.1, 3), k: ub.integer(2, 8, { log: true }) };
  new ub.TPE(space, { seed: 0, nStartup: 5, nCandidates: 8, gamma: 0.2 });
  await ub.minimize(p => Number(p.m), space, { nTrials: 5, seed: 0, nStartup: 2, nCandidates: 8, gamma: 0.2 });
  await ub.runAsync(ub.steps.fcm(X, 2), { signal: AbortSignal.timeout(100), onProgress: p => p.result(), budgetMs: 0 });
}

export function wrongArguments() {
  // @ts-expect-error k is a number
  ub.fcm(X, '2');
  // @ts-expect-error unknown option
  ub.kmeans(X, 2, { maxIterations: 3 });
  // @ts-expect-error lam is required
  ub.steps.somOlp(X, [2, 2], { gamma: 1 });
  // @ts-expect-error the generator takes rows or matrices
  ub.steps.fcm(X, 2).next('rows');
  // @ts-expect-error unknown average
  ub.ami([0, 1], [1, 0], { average: 'median' });
  // @ts-expect-error trustworthiness takes a number of neighbors
  ub.trustworthiness(X, X, '1');
}
