import type { CellType, PreprocessedDataset } from '../types.ts';

export const CORR_GROUPS = [
  'overall',
  'exhausted_cd8_t',
  'treg',
  'macrophage_myeloid',
  'epithelial',
] as const;

export interface CorrPrecomputed {
  groupSizes: number[];
  /** Per group -> G x N_group unit-centered vectors for O(N) dot-product Pearson r */
  unitVectors: Float64Array[];
}

export function precomputeCorrStats(ds: PreprocessedDataset): CorrPrecomputed {
  const gCount = ds.genes.length;
  const nCells = ds.cells.length;
  const groupSizes: number[] = [];
  const unitVectors: Float64Array[] = [];

  for (const grp of CORR_GROUPS) {
    const indices: number[] = [];
    for (let c = 0; c < nCells; c++) {
      if (grp === 'overall' || ds.cells[c].cellType === (grp as CellType)) {
        indices.push(c);
      }
    }
    const nGrp = indices.length;
    groupSizes.push(nGrp);

    const vecs = new Float64Array(gCount * nGrp);
    if (nGrp > 1) {
      for (let g = 0; g < gCount; g++) {
        let sum = 0;
        for (let k = 0; k < nGrp; k++) {
          sum += ds.normExpr[indices[k] * gCount + g];
        }
        const mean = sum / nGrp;

        let sq = 0;
        const gOff = g * nGrp;
        for (let k = 0; k < nGrp; k++) {
          const d = ds.normExpr[indices[k] * gCount + g] - mean;
          vecs[gOff + k] = d;
          sq += d * d;
        }
        const norm = Math.sqrt(sq);
        if (norm > 1e-12) {
          for (let k = 0; k < nGrp; k++) vecs[gOff + k] /= norm;
        } else {
          for (let k = 0; k < nGrp; k++) vecs[gOff + k] = 0;
        }
      }
    }
    unitVectors.push(vecs);
  }

  return { groupSizes, unitVectors };
}

export function computePearsonCorrelation(
  x: ArrayLike<number>,
  y: ArrayLike<number>
): number {
  const n = Math.min(x.length, y.length);
  if (n < 2) return 0;
  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < n; i++) {
    sumX += x[i];
    sumY += y[i];
  }
  const mx = sumX / n;
  const my = sumY / n;

  let num = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    num += dx * dy;
    sx += dx * dx;
    sy += dy * dy;
  }
  const den = Math.sqrt(sx * sy);
  if (den < 1e-12) return 0;
  return Math.max(-1, Math.min(1, num / den));
}

export function computePairCorrelationsInto(
  stats: CorrPrecomputed,
  idx1: number,
  idx2: number,
  out: Float64Array,
  outOffset: number
): void {
  let cursor = outOffset;
  for (let grpIdx = 0; grpIdx < CORR_GROUPS.length; grpIdx++) {
    const nGrp = stats.groupSizes[grpIdx];
    if (nGrp < 2) {
      out[cursor++] = 0;
      continue;
    }
    const vecs = stats.unitVectors[grpIdx];
    const off1 = idx1 * nGrp;
    const off2 = idx2 * nGrp;
    let dot = 0;
    for (let k = 0; k < nGrp; k++) {
      dot += vecs[off1 + k] * vecs[off2 + k];
    }
    out[cursor++] = Math.max(-1, Math.min(1, dot));
  }
}
