/**
 * UbuKit: fuzzy, rough and classical clustering, self-organizing maps,
 * evaluation metrics and TPE search. Zero dependencies.
 *
 * Fitting functions run synchronously. The generators in `steps` yield
 * { iteration, result } after every iteration, where result() is the Result
 * as if the run had stopped there; drive them by hand or with `runAsync`:
 *   await runAsync(steps.fcm(X, 3), { signal, onProgress: p => draw(p.result()) })
 */
import * as cluster from './cluster.js';
import { run } from './core.js';
import * as maps from './som.js';

/** @typedef {import('./core.js').Result} Result */
/** @typedef {import('./core.js').Progress} Progress */
/** @typedef {import('./tpe.js').TPEResult} TPEResult */

/** Step generators of every fitting function (yield after each iteration or epoch). */
export const steps = Object.freeze({
  kmeans: cluster.kmeans, fcm: cluster.fcm, efcm: cluster.efcm, rcm: cluster.rcm, rmcm: cluster.rmcm,
  som: maps.som, batchSom: maps.batchSom, somOlp: maps.somOlp,
});

/** Lloyd's k-means. @type {(...args: Parameters<typeof cluster.kmeans>) => Result} */
export const kmeans = (...args) => run(cluster.kmeans(...args));

/** Fuzzy c-means. @type {(...args: Parameters<typeof cluster.fcm>) => Result} */
export const fcm = (...args) => run(cluster.fcm(...args));

/** Entropy-regularized fuzzy c-means. @type {(...args: Parameters<typeof cluster.efcm>) => Result} */
export const efcm = (...args) => run(cluster.efcm(...args));

/** Rough c-means (p = 1) and ExRCM. @type {(...args: Parameters<typeof cluster.rcm>) => Result} */
export const rcm = (...args) => run(cluster.rcm(...args));

/** Rough membership c-means. @type {(...args: Parameters<typeof cluster.rmcm>) => Result} */
export const rmcm = (...args) => run(cluster.rmcm(...args));

/** Online SOM. @type {(...args: Parameters<typeof maps.som>) => Result} */
export const som = (...args) => run(maps.som(...args));

/** Batch SOM. @type {(...args: Parameters<typeof maps.batchSom>) => Result} */
export const batchSom = (...args) => run(maps.batchSom(...args));

/** SOM with optimized latent positions. @type {(...args: Parameters<typeof maps.somOlp>) => Result} */
export const somOlp = (...args) => run(maps.somOlp(...args));

export { matrix, run, runAsync, toRows } from './core.js';
export { ami, ari, continuity, trustworthiness } from './metrics.js';
export { choice, integer, loguniform, minimize, TPE, uniform } from './tpe.js';
