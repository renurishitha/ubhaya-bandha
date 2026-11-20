import type { PreprocessedDataset } from '../types.ts';

/**
 * Computes d = mean(expr_tumor) - mean(expr_normal) for every gene.
 */
export function computeGeneTumorDeltas(ds: PreprocessedDataset): Float64Array {
  const gCount = ds.genes.length;
  const nCells = ds.cells.length;
  const tumorSum = new Float64Array(gCount);
  const normalSum = new Float64Array(gCount);
  let tumorCount = 0;
  let normalCount = 0;

  for (let c = 0; c < nCells; c++) {
    const off = c * gCount;
    if (ds.cells[c].isTumor) {
      tumorCount++;
      for (let g = 0; g < gCount; g++) tumorSum[g] += ds.normExpr[off + g];
    } else {
      normalCount++;
      for (let g = 0; g < gCount; g++) normalSum[g] += ds.normExpr[off + g];
    }
  }

  const deltas = new Float64Array(gCount);
  const tDen = Math.max(1, tumorCount);
  const nDen = Math.max(1, normalCount);
  for (let g = 0; g < gCount; g++) {
    deltas[g] = tumorSum[g] / tDen - normalSum[g] / nDen;
  }
  return deltas;
}

/**
 * Paper formula: score_safe = (dx * dy) / (dx + dy)
 * Guarded for non-positive deltas so high normal-tissue expression cannot
 * yield a spuriously positive safety score via double negation.
 */
export function computeSafetyScore(dx: number, dy: number): number {
  if (dx > 0 && dy > 0) {
    return (dx * dy) / (dx + dy);
  }
  if (dx <= 0 && dy <= 0) {
    const denom = Math.abs(dx) + Math.abs(dy);
    if (denom < 1e-12) return 0;
    return -Math.abs(dx * dy) / denom;
  }
  return Math.min(dx, dy);
}
