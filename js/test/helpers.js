// Shared test data: three Gaussian blobs from a fixed linear congruential generator.
// Each test file runs in its own process, so every file sees the same stream.
import * as ub from '../src/index.js';

export const rand = (() => { let s = 7; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32); })();
export const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
export const centers = [[0, 0], [5, 0], [0, 5]];
export const X = centers.flatMap(c => Array.from({ length: 80 }, () => [c[0] + 0.5 * gauss(), c[1] + 0.5 * gauss()]));

/** Largest distance from a true center to the nearest found one. */
export function nearestError(found) {
  const rows = ub.toRows(found);
  return Math.max(...centers.map(c => Math.min(...rows.map(r => Math.hypot(r[0] - c[0], r[1] - c[1])))));
}
