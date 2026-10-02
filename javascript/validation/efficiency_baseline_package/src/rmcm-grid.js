/** Bounded-memory low-dimensional candidate grid, never a graph predicate.
 * Float64 coordinates remain authoritative. A 1% enlarged cell and a strict
 * coordinate-range cap make cell-rounding error much smaller than the margin.
 * Only one-cell offsets are needed; every candidate is exactly rechecked.
 */
function createRadiusGridImpl(data, n, d, delta, availableBytes) {
  if (d > 3 || n > 0x3fffffff || delta === 0) return null;
  const width = Math.max(delta * 1.01, 16 * Math.sqrt(2.2250738585072014e-308));
  if (!Number.isFinite(width) || width <= 0) return null;
  const origin = new Float64Array(d).fill(Infinity), high = new Float64Array(d).fill(-Infinity);
  for (let i = 0; i < n; ++i) for (let f = 0; f < d; ++f) {
    const value = data[i * d + f];
    origin[f] = Math.min(origin[f], value); high[f] = Math.max(high[f], value);
  }
  // Subtraction and division error at this cap is below 1e-6 cell, versus
  // the 0.01-cell geometric safety margin. Larger ranges use the scalar path.
  for (let f = 0; f < d; ++f) if ((high[f] - origin[f]) / width > 2 ** 29) return null;
  let size = 1; while (size < 2 * n) size *= 2;
  if (size > 2 ** 30) return null;
  const bytes = 4 * (d + 1) * size + 4 * (d + 2) * n + 16 * d;
  if (!Number.isSafeInteger(bytes) || bytes > availableBytes) return null;
  const keys = Array.from({ length: d }, () => new Int32Array(size));
  const heads = new Int32Array(size).fill(-1), next = new Int32Array(n);
  const cells = Array.from({ length: d }, () => new Int32Array(n));
  const scratch = new Uint32Array(n), mask = size - 1;
  const slot = (x, y, z, create) => {
    let h = (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) & mask;
    for (;;) {
      if (heads[h] === -1) {
        if (!create) return -1;
        keys[0][h] = x; if (d > 1) keys[1][h] = y; if (d > 2) keys[2][h] = z;
        return h;
      }
      if (keys[0][h] === x && (d < 2 || keys[1][h] === y) && (d < 3 || keys[2][h] === z)) return h;
      h = (h + 1) & mask;
    }
  };
  for (let i = 0; i < n; ++i) {
    for (let f = 0; f < d; ++f) cells[f][i] = Math.floor((data[i * d + f] - origin[f]) / width);
    const h = slot(cells[0][i], d > 1 ? cells[1][i] : 0, d > 2 ? cells[2][i] : 0, true);
    next[i] = heads[h]; heads[h] = i;
  }
  return { bytes,
    row(i) {
      const x = cells[0][i], y = d > 1 ? cells[1][i] : 0, z = d > 2 ? cells[2][i] : 0;
      let count = 0;
      for (let dx = -1; dx <= 1; ++dx) {
        if (x + dx < 0) continue;
        for (let dy = d > 1 ? -1 : 0; dy <= (d > 1 ? 1 : 0); ++dy) {
          if (y + dy < 0) continue;
          for (let dz = d > 2 ? -1 : 0; dz <= (d > 2 ? 1 : 0); ++dz) {
            if (z + dz < 0) continue;
            const h = slot(x + dx, y + dy, z + dz, false);
            if (h < 0) continue;
            for (let j = heads[h]; j !== -1; j = next[j]) if (j > i) scratch[count++] = j;
          }
        }
      }
      const result = scratch.subarray(0, count); result.sort(); return result;
    } };
}

export function createRadiusGrid(data, n, d, delta, availableBytes) {
  try { return createRadiusGridImpl(data, n, d, delta, availableBytes); }
  catch (error) { if (error instanceof RangeError) return null; throw error; }
}
