// How typed code calls UbuKit: checked by tsc in strict mode against the
// published declarations (types/, from "npm run build:types"), never run;
// "npm run check" runs it. Each @ts-expect-error marks a call that must be
// rejected; tsc reports a directive that is not needed, so a part of the API
// that loses its types (a broken JSDoc becomes `any`) fails the check.
import * as ub from '../types/index.js';

const X = [[0, 0], [1, 1], [5, 5]];

/** @returns {number} */
export function fit() {
  return ub.fcm(X, 2, { m: 2, seed: 0 }).nIter;
}

/** @returns {import('../types/core.js').Result} */
export function followMovingData() {
  const run = ub.steps.somOlp(X, [2, 2], { lam: 0.5, gamma: 1, maxIter: Infinity });
  run.next();
  const { value, done } = run.next(X.map(([a, b]) => [a + 1, b]));
  return done ? value : value.result();
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
