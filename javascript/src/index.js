import { kmeansSteps, fcmSteps, rcmSteps, exrcmSteps, consumeSteps } from './clustering.js';
import { rmcmSteps } from './rmcm.js';
import { somSteps, somBatchSteps } from './som.js';
export { som, somBatch, som_batch, somSteps, somBatchSteps, somProject } from './som.js';
import { somOlpSteps } from './som-olp.js';
import { entropyFcmSteps } from './entropy-fcm.js';
import { neighborhoodSteps } from './neighborhood.js';
import { checkCancelled, finiteNumber } from './core.js';
export { kmeans, fcm, rcm, exrcm, membershipsFromSquaredDistances, roughAdmissible } from './clustering.js';
export { rmcm, rmcmReference, rmcmSteps, prepareRMCM, prepareRMCMSteps, PreparedRMCM, RMCMGraphCache } from './rmcm.js';
export { somOlp, somOlpSteps } from './som-olp.js';
export { entropyFcm, entropyFcmSteps } from './entropy-fcm.js';
export { neighborhood, neighborhoodSteps } from './neighborhood.js';
export { normalizeInput, seededRandom } from './core.js';
export { createWorkerClient } from './worker-client.js';
export const algorithms = Object.freeze(['kmeans', 'fcm', 'entropy-fcm', 'rcm', 'exrcm', 'rmcm', 'som-olp', 'som', 'som_batch', 'neighborhood']);
const registry = { kmeans: kmeansSteps, fcm: fcmSteps, 'entropy-fcm': entropyFcmSteps, rcm: rcmSteps, exrcm: exrcmSteps, rmcm: rmcmSteps, 'som-olp': somOlpSteps, som: somSteps, som_batch: somBatchSteps, neighborhood: neighborhoodSteps };
export function steps(algorithm, input, options = {}) {
  if (!Object.hasOwn(registry, algorithm)) throw new RangeError(`Unknown algorithm ${algorithm}; choose ${algorithms.join(', ')}`);
  return registry[algorithm](input, options);
}
/** Synchronous CPU API; prefer a Web Worker on a demo's main UI thread. */
export function run(algorithm, input, options = {}) { return consumeSteps(steps(algorithm, input, options)); }
/** Cooperative fallback. Yields between row blocks, not within an individual row. */
export async function runAsync(algorithm, input, options = {}) {
  const timeBudgetMs = finiteNumber(options.timeBudgetMs ?? 8, 'timeBudgetMs', 0);
  const iterator = steps(algorithm, input, options);
  let deadline = performance.now() + timeBudgetMs;
  try {
    for (;;) {
      checkCancelled(options);
      const step = iterator.next(); if (step.done) return step.value;
      if (performance.now() >= deadline) { await new Promise(resolve => setTimeout(resolve, 0)); deadline = performance.now() + timeBudgetMs; }
    }
  } finally { iterator.return?.(); }
}


export { createSession, ClusteringSession, sessionAlgorithms } from './session.js';
export { createRealtimeWorkerClient } from './realtime-worker-client.js';
export { createMetricScheduler } from './metric-scheduler.js';

export { TPEOptimizer, SearchSpaceExhaustedError, ProposalError, floatRange, intRange, categorical, optimize, optimizeAsync } from './optimization.js';

export { adjustedRandScore, adjustedMutualInfoScore, adjustedScores, adjusted_rand_score, adjusted_mutual_info_score, adjusted_scores, ExternalMetricDomainError } from './external-metrics.js';
