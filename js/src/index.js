/**
 * UbuKit: fuzzy, rough and classical clustering, self-organizing maps,
 * evaluation metrics and TPE search. Zero dependencies.
 *
 * Fitting functions run synchronously. The generators in `steps` yield
 * { iteration, result } after every iteration, where result() is the Result
 * as if the run had stopped there; drive them by hand or with `runAsync`:
 *   await runAsync(steps.fcm(X, 3), { signal, onProgress: p => draw(p.result()) })
 * `apply` holds the functions that apply trained prototypes to new data, the
 * prototypes fixed, and `steps.apply` the same as generators.
 */
import * as cluster from './cluster.js';
import { run } from './core.js';
import * as maps from './som.js';

/** @typedef {import('./core.js').Result} Result */
/** @typedef {import('./core.js').Progress} Progress */
/** @typedef {import('./tpe.js').TPEResult} TPEResult */

/**
 * The function that runs a generator to the end and returns its Result.
 * @template {(...args: any[]) => Generator<Progress, Result, any>} G
 * @param {G} generator
 * @returns {(...args: Parameters<G>) => Result}
 */
const runs = generator => (...args) => run(generator(...args));

/** The generators of apply: trained prototypes, fixed, applied to new data. */
const applying = Object.freeze({
  kmeans: cluster.applyKmeans, fcm: cluster.applyFcm, efcm: cluster.applyEfcm, rcm: cluster.applyRcm,
  rmcm: cluster.applyRmcm, som: maps.applySom, batchSom: maps.applyBatchSom, somOlp: maps.applySomOlp,
});

/** Step generators of every fitting function (yield after each iteration or epoch), and of apply. */
export const steps = Object.freeze({
  kmeans: cluster.kmeans, fcm: cluster.fcm, efcm: cluster.efcm, rcm: cluster.rcm, rmcm: cluster.rmcm,
  som: maps.som, batchSom: maps.batchSom, somOlp: maps.somOlp, apply: applying,
});

/**
 * Trained prototypes applied to new data, the prototypes fixed: each function
 * takes the data and the prototypes (centers of a Result), with the parameters
 * they were trained with, and returns the Result of the data under them.
 */
export const apply = Object.freeze({
  kmeans: runs(applying.kmeans), fcm: runs(applying.fcm), efcm: runs(applying.efcm), rcm: runs(applying.rcm),
  rmcm: runs(applying.rmcm), som: runs(applying.som), batchSom: runs(applying.batchSom), somOlp: runs(applying.somOlp),
});

/** Lloyd's k-means. */
export const kmeans = runs(cluster.kmeans);

/** Fuzzy c-means. */
export const fcm = runs(cluster.fcm);

/** Entropy-regularized fuzzy c-means. */
export const efcm = runs(cluster.efcm);

/** Rough c-means (p = 1) and ExRCM. */
export const rcm = runs(cluster.rcm);

/** Rough membership c-means. */
export const rmcm = runs(cluster.rmcm);

/** Online SOM. */
export const som = runs(maps.som);

/** Batch SOM. */
export const batchSom = runs(maps.batchSom);

/** SOM with optimized latent positions. */
export const somOlp = runs(maps.somOlp);

export { matrix, run, runAsync, toRows } from './core.js';
export { ami, ari, continuity, Neighborhood, trustworthiness } from './metrics.js';
export { choice, integer, loguniform, minimize, TPE, uniform } from './tpe.js';
