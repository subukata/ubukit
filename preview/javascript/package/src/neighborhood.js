import {wasmRows} from './metric-wasm.js';
/** Exact joint trustworthiness / continuity with an explicit portable tie contract.
 * Euclidean Float64 distances, self excluded, equal rounded distances ordered by
 * original sample index. This is NOT NumPy/sklearn's version-dependent tie order.
 * See docs/SOM_AND_NEIGHBORHOOD.md for provenance, complexity and parity limits.
 */
import { normalizeInput, checkCancelled, progress } from './core.js';

const precedes = (d, a, b) => d[a] < d[b] || (d[a] === d[b] && a < b);
function integer(x, name, min = 1) {
  if (!Number.isSafeInteger(x) || x < min) throw new RangeError(`${name} must be an integer >= ${min}`);
  return x;
}

// Max heap, then in-place heap sort. Only K indices, no N-by-K allocation.
function siftDown(heap, size, root, distances) {
  for (;;) {
    const left = 2 * root + 1;
    if (left >= size) return;
    const right = left + 1;
    let worst = left;
    if (right < size && precedes(distances, heap[left], heap[right])) worst = right;
    if (!precedes(distances, heap[root], heap[worst])) return;
    const tmp = heap[root]; heap[root] = heap[worst]; heap[worst] = tmp;
    root = worst;
  }
}
function nearest(distances, self, heap) {
  const k = heap.length;
  let size = 0;
  for (let j = 0; j < distances.length; j++) {
    if (j === self) continue;
    if (size < k) {
      let c = size++;
      heap[c] = j;
      while (c > 0) {
        const p = (c - 1) >> 1;
        if (!precedes(distances, heap[p], heap[c])) break;
        const tmp = heap[p]; heap[p] = heap[c]; heap[c] = tmp; c = p;
      }
    } else if (precedes(distances, j, heap[0])) {
      heap[0] = j; siftDown(heap, k, 0, distances);
    }
  }
  for (let end = k - 1; end > 0; end--) {
    const tmp = heap[end]; heap[end] = heap[0]; heap[0] = tmp;
    siftDown(heap, end, 0, distances);
  }
}
function distanceRow(data, n, d, i, result) {
  const offset = i * d;
  if (d === 2) {
    const x0 = data[offset], x1 = data[offset + 1];
    for (let j = 0; j < n; j++) {
      if (j === i) { result[j] = Infinity; continue; }
      const a = x0 - data[j * 2], b = x1 - data[j * 2 + 1];
      const value = a * a + b * b;
      if (!Number.isFinite(value)) throw new RangeError('Euclidean squared distance overflowed; rescale the input');
      result[j] = Math.sqrt(value);
    }
    return;
  }
  for (let j = 0; j < n; j++) {
    if (j === i) { result[j] = Infinity; continue; }
    let value = 0;
    const other = j * d;
    let f = 0;
    for (; f + 3 < d; f += 4) {
      const a = data[offset+f] - data[other+f]; value += a*a;
      const b = data[offset+f+1] - data[other+f+1]; value += b*b;
      const c = data[offset+f+2] - data[other+f+2]; value += c*c;
      const e = data[offset+f+3] - data[other+f+3]; value += e*e;
    }
    for (; f < d; f++) {
      const delta = data[offset+f] - data[other+f]; value += delta*delta;
    }
    if (!Number.isFinite(value)) throw new RangeError('Euclidean squared distance overflowed; rescale the input');
    result[j] = Math.sqrt(value);
  }
}

// Query-rank histogram: O(N log K), O(K) scratch. Keys (distance,index) are
// unique, so cumulative count <= each query is its exact one-based rank.
function queriedRanks(distances, queries, known, self, order, histogram, ranks, fastHistogram) {
  const countQueries = queries.length;
  let k = 0;
  ranks.fill(0);
  for (let q = 0; q < countQueries; q++) {
    const index = queries[q];
    for (let a = 0; countQueries <= 32 && a < known.length; a++) {
      if (known[a] === index) { ranks[q] = a + 1; break; }
    }
    if (ranks[q] === 0) order[k++] = index;
  }
  if (k === 0) return;
  for (let root = (k >> 1) - 1; root >= 0; root--) siftDown(order, k, root, distances);
  for (let end = k - 1; end > 0; end--) {
    const tmp = order[end]; order[end] = order[0]; order[0] = tmp;
    siftDown(order, end, 0, distances);
  }
  histogram.fill(0);
  const largest = order[k - 1];
  if(fastHistogram) fastHistogram(order,k,histogram,self); else {
  for (let j = 0; j < distances.length; j++) {
    if (j === self || precedes(distances, largest, j)) continue;
    let lo = 0, hi = k;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (precedes(distances, order[mid], j)) lo = mid + 1;
      else hi = mid;
    }
    histogram[lo]++;
  }
  }
  let count = 0;
  for (let q = 0; q < k; q++) { count += histogram[q]; histogram[q] = count; }
  for (let q = 0; q < countQueries; q++) {
    if (ranks[q] !== 0) continue;
    const index = queries[q];
    let lo = 0, hi = k;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (precedes(distances, order[mid], index)) lo = mid + 1;
      else hi = mid;
    }
    ranks[q] = histogram[lo];
  }
}

const littleEndian = new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] === 4;
function radixOrder(distances, indices, temp, buckets) {
  const bits = new Uint32Array(distances.buffer, distances.byteOffset, distances.length * 2);
  for (let i = 0; i < indices.length; i++) indices[i] = i;
  let source = indices, destination = temp;
  for (let pass = 0; pass < 8; pass++) {
    const word = (pass >>> 2) ^ (littleEndian ? 0 : 1), shift = (pass & 3) << 3;
    buckets.fill(0);
    for (let i = 0; i < source.length; i++) buckets[(bits[2*source[i]+word] >>> shift) & 255]++;
    let total = 0;
    for (let b = 0; b < 256; b++) { const n = buckets[b]; buckets[b] = total; total += n; }
    for (let i = 0; i < source.length; i++) { const index = source[i], b = (bits[2*index+word] >>> shift) & 255; destination[buckets[b]++] = index; }
    const swap = source; source = destination; destination = swap;
  }
}

export function* neighborhoodSteps(input, options = {}) {
  const x = normalizeInput(input);
  const y = normalizeInput(options.embedding ?? options.coordinates);
  const n = x.nSamples;
  if (y.nSamples !== n) throw new RangeError('Input and embedding must have the same number of samples');
  if (n < 3) throw new RangeError('Neighborhood metrics need at least 3 samples');
  if (n > 0xffffffff) throw new RangeError('Sample indices must fit Uint32');
  const requested = options.ks ?? options.k ?? 5;
  const ks = [...new Set(typeof requested === 'number' ? [requested] : requested)];
  if (!ks.length || ks.some(k => !Number.isSafeInteger(k) || k < 1 || k >= n / 2)) {
    throw new RangeError('Each k must satisfy 1 <= k < n / 2');
  }
  const maxK = ks.reduce((largest, k) => Math.max(largest, k), 0);
  const blockRows = integer(options.blockRows ?? 16, 'blockRows');
  const maxScratchBytes = integer(options.maxScratchBytes ?? 32 * 1024 * 1024, 'maxScratchBytes');
  // Primary typed buffers: two distance rows, three K-index vectors, two rank
  // vectors, one histogram. Input conversion and returned records are separate.
  const baseScratchBytes = 16 * n + 36 * maxK + 8 + 16 * ks.length;
  const requestedRankMethod = options.rankMethod ?? 'auto';
  if (!['auto','histogram','radix'].includes(requestedRankMethod)) throw new RangeError('rankMethod must be auto, histogram, or radix');
  const radixBytes = 16*n+1024;
  const useRadix = requestedRankMethod !== 'histogram' && (requestedRankMethod === 'radix' || (maxK>=64 && n>=128)) && baseScratchBytes+radixBytes<=maxScratchBytes;
  if(requestedRankMethod==='radix'&&!useRadix) throw new RangeError('Radix metric scratch exceeds maxScratchBytes');
  const scratchBytes = baseScratchBytes + (useRadix ? radixBytes : 0);
  if (scratchBytes > maxScratchBytes) throw new RangeError(`Exact metric row scratch needs ${scratchBytes} bytes`);
  const distanceBackend = options.distanceBackend ?? 'javascript';
  if(!['javascript','wasm'].includes(distanceBackend))throw new RangeError('distanceBackend must be javascript or wasm');
  const wasm = distanceBackend==='wasm' ? wasmRows(x,y,maxScratchBytes-scratchBytes,maxK) : null;
  const ix=useRadix?new Uint32Array(n):null, iy=useRadix?new Uint32Array(n):null, tmp=useRadix?new Uint32Array(n):null, inv=useRadix?new Uint32Array(n):null, buckets=useRadix?new Uint32Array(256):null;
  const dx = wasm?.dx ?? new Float64Array(n), dy = wasm?.dy ?? new Float64Array(n);
  const nx = new Uint32Array(maxK), ny = new Uint32Array(maxK), order = new Uint32Array(maxK);
  const rt = new Float64Array(maxK), rc = new Float64Array(maxK), hist = new Float64Array(maxK + 1);
  const trustPenalties = new Float64Array(ks.length), continuityPenalties = new Float64Array(ks.length);
  for (let start = 0; start < n; start += blockRows) {
    checkCancelled(options);
    const end = Math.min(n, start + blockRows);
    for (let i = start; i < end; i++) {
      checkCancelled(options);
      if(wasm) wasm.row(i); else {
        distanceRow(x.data, n, x.nFeatures, i, dx);
        distanceRow(y.data, n, y.nFeatures, i, dy);
      }
      if(useRadix){
        radixOrder(dx,ix,tmp,buckets);radixOrder(dy,iy,tmp,buckets);
        nx.set(ix.subarray(0,maxK));ny.set(iy.subarray(0,maxK));
        for(let rank=0;rank<n;rank++)inv[ix[rank]]=rank+1;
        for(let q=0;q<maxK;q++)rt[q]=inv[ny[q]];
        for(let rank=0;rank<n;rank++)inv[iy[rank]]=rank+1;
        for(let q=0;q<maxK;q++)rc[q]=inv[nx[q]];
      }else {
      nearest(dx, i, nx); nearest(dy, i, ny);
      if(wasm && maxK<=32 && requestedRankMethod==='auto'){
        for(let q=0;q<maxK;q++){
          let tr=0,cr=0;
          for(let a=0;a<maxK;a++){if(nx[a]===ny[q])tr=a+1;if(ny[a]===nx[q])cr=a+1;}
          rt[q]=tr||wasm.rankX(ny[q]);rc[q]=cr||wasm.rankY(nx[q]);
        }
      }else{
        queriedRanks(dx, ny, nx, i, order, hist, rt, wasm?.histX);
        queriedRanks(dy, nx, ny, i, order, hist, rc, wasm?.histY);
      }
      }
      for (let a = 0; a < ks.length; a++) {
        const k = ks[a];
        let pt = 0, pc = 0;
        for (let q = 0; q < k; q++) {
          pt += Math.max(0, rt[q] - k); pc += Math.max(0, rc[q] - k);
        }
        trustPenalties[a] += pt; continuityPenalties[a] += pc;
        if (!Number.isSafeInteger(trustPenalties[a]) || !Number.isSafeInteger(continuityPenalties[a])) {
          throw new RangeError('Exact integer metric penalties exceed Number.MAX_SAFE_INTEGER');
        }
      }
    }
    const event = { algorithm: 'neighborhood', phase: 'rank', completed: end, total: n, fraction: end / n };
    progress(options, event); yield event;
  }
  const qualities = ks.map((k, a) => {
    const factor = 2 / (n * k * (2 * n - 3 * k - 1));
    return { k, trustworthiness: 1 - trustPenalties[a] * factor,
      continuity: 1 - continuityPenalties[a] * factor,
      trustworthinessPenalty: trustPenalties[a], continuityPenalty: continuityPenalties[a] };
  });
  return { algorithm: 'neighborhood', qualities, nSamples: n,
    stats: { exact: true, distance: 'rooted-float64-direct-differences', tiePolicy: 'distance-then-sample-index',
      sklearnTieParity: false, primaryScratchBytes: scratchBytes+(wasm?.extraBytes??0), wasmDistances:!!wasm, rankMethod:useRadix?'radix':wasm&&maxK<=32&&requestedRankMethod==='auto'?'wasm-scan':'histogram', blockRows,
      distanceMatrixAllocated: false, rankMatrixAllocated: false,
      complexity: 'O(N^2 * (DX + DY + log(Kmax)) + N * sum(ks)) time; O(N + Kmax) scratch' } };
}

export function neighborhood(input, options = {}) {
  const steps = neighborhoodSteps(input, options);
  for (;;) { const step = steps.next(); if (step.done) return step.value; }
}
