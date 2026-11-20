import type { CellType, PreprocessedDataset, TissueType } from '../types.ts';

export const ACTIVITY_GROUPS = [
  'overall',
  'tumor_lung',
  'normal_lung',
  'lymph_node',
  'brain_mets',
  'exhausted_cd8_t',
  'treg',
  'macrophage_myeloid',
  'epithelial',
] as const;

export type ActivityGroup = (typeof ACTIVITY_GROUPS)[number];

export interface ActivityPrecomputed {
  threshold: number;
  groupCellIndices: number[][];
  /** Per group -> per gene single positive ratio P_gene */
  singleRatio: Float64Array[];
  /** Per group -> per gene mean expression */
  meanExp: Float64Array[];
  /** Per group -> G x N_group binary indicator (1 if expr > T) */
  binaryFlags: Uint8Array[];
}

export function precomputeActivityStats(
  ds: PreprocessedDataset,
  threshold = 0.5
): ActivityPrecomputed {
  const gCount = ds.genes.length;
  const nCells = ds.cells.length;

  const groupCellIndices: number[][] = ACTIVITY_GROUPS.map((grp) => {
    const indices: number[] = [];
    for (let c = 0; c < nCells; c++) {
      if (grp === 'overall') {
        indices.push(c);
      } else if (
        grp === 'tumor_lung' ||
        grp === 'normal_lung' ||
        grp === 'lymph_node' ||
        grp === 'brain_mets'
      ) {
        if (ds.cells[c].tissue === (grp as TissueType)) indices.push(c);
      } else {
        if (ds.cells[c].cellType === (grp as CellType)) indices.push(c);
      }
    }
    return indices;
  });

  const singleRatio: Float64Array[] = [];
  const meanExp: Float64Array[] = [];
  const binaryFlags: Uint8Array[] = [];

  for (let grpIdx = 0; grpIdx < ACTIVITY_GROUPS.length; grpIdx++) {
    const cellsInGrp = groupCellIndices[grpIdx];
    const nGrp = cellsInGrp.length;
    const sr = new Float64Array(gCount);
    const me = new Float64Array(gCount);
    const bf = new Uint8Array(gCount * nGrp);

    if (nGrp > 0) {
      for (let g = 0; g < gCount; g++) {
        let pos = 0;
        let sum = 0;
        const gOff = g * nGrp;
        for (let k = 0; k < nGrp; k++) {
          const val = ds.normExpr[cellsInGrp[k] * gCount + g];
          sum += val;
          if (val > threshold) {
            pos++;
            bf[gOff + k] = 1;
          }
        }
        sr[g] = pos / nGrp;
        me[g] = sum / nGrp;
      }
    }

    singleRatio.push(sr);
    meanExp.push(me);
    binaryFlags.push(bf);
  }

  return {
    threshold,
    groupCellIndices,
    singleRatio,
    meanExp,
    binaryFlags,
  };
}

/**
 * Hand-testable formula helper: P_gene = sum(I(expr > T)) / N
 */
export function computeSinglePositiveRatio(
  expr: ArrayLike<number>,
  threshold: number
): number {
  if (expr.length === 0) return 0;
  let count = 0;
  for (let i = 0; i < expr.length; i++) {
    if (expr[i] > threshold) count++;
  }
  return count / expr.length;
}

/**
 * Hand-testable formula helper: P_{g1,g2} = sum(I(e1 > T) * I(e2 > T)) / N
 */
export function computeDoublePositiveRatio(
  expr1: ArrayLike<number>,
  expr2: ArrayLike<number>,
  threshold: number
): number {
  const n = Math.min(expr1.length, expr2.length);
  if (n === 0) return 0;
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (expr1[i] > threshold && expr2[i] > threshold) count++;
  }
  return count / n;
}

export function computePairActivityInto(
  stats: ActivityPrecomputed,
  idx1: number,
  idx2: number,
  out: Float64Array,
  outOffset: number
): void {
  let singleRatioMax = 0;
  let doubleRatioMax = 0;
  let cursor = outOffset;

  for (let grpIdx = 0; grpIdx < ACTIVITY_GROUPS.length; grpIdx++) {
    const nGrp = stats.groupCellIndices[grpIdx].length;
    const p1 = stats.singleRatio[grpIdx][idx1];
    const p2 = stats.singleRatio[grpIdx][idx2];
    const e1 = stats.meanExp[grpIdx][idx1];
    const e2 = stats.meanExp[grpIdx][idx2];

    let dbl = 0;
    if (nGrp > 0) {
      const bf = stats.binaryFlags[grpIdx];
      const off1 = idx1 * nGrp;
      const off2 = idx2 * nGrp;
      let both = 0;
      for (let k = 0; k < nGrp; k++) {
        both += bf[off1 + k] & bf[off2 + k];
      }
      dbl = both / nGrp;
    }

    out[cursor++] = dbl; // double_ratio
    out[cursor++] = p1 + p2; // sum_single_ratio
    out[cursor++] = Math.min(p1, p2); // min_single_ratio
    out[cursor++] = e1 + e2; // sum_single_exp
    out[cursor++] = Math.min(e1, e2); // min_single_exp

    if (grpIdx > 0) {
      // Sub-population maxima (tissues + cell types)
      singleRatioMax = Math.max(singleRatioMax, p1, p2);
      doubleRatioMax = Math.max(doubleRatioMax, dbl);
    }
  }

  out[cursor++] = singleRatioMax;
  out[cursor++] = doubleRatioMax;
}
