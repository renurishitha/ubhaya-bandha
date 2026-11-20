import { computeCandidatePairCount } from './data.ts';
import type {
  CellMetadata,
  CellType,
  PreprocessedDataset,
  RawSyntheticDataset,
  TissueType,
} from './types.ts';

export interface PreprocessConfig {
  minGenes: number;
  maxMitoPct: number;
  targetSum: number;
  nTopHvg: number;
  nPcaComps: number;
}

export const DEFAULT_PREPROCESS_CONFIG: PreprocessConfig = {
  minGenes: 40,
  maxMitoPct: 20.0,
  targetSum: 10000,
  nTopHvg: 150,
  nPcaComps: 8,
};

/**
 * Scanpy-equivalent single-cell preprocessing in TypeScript:
 * 1. QC filtering on detected genes and mitochondrial percentage
 * 2. Library-size normalization (targetSum=10,000) + log1p
 * 3. Highly variable gene (HVG) dispersion selection
 * 4. Power-iteration PCA variance explained summary
 */
export function preprocessScrnaDataset(
  raw: RawSyntheticDataset,
  config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG
): PreprocessedDataset {
  const gCount = raw.genes.length;
  const keptIndices: number[] = [];
  const keptCells: CellMetadata[] = [];

  for (let c = 0; c < raw.cells.length; c++) {
    const cell = raw.cells[c];
    if (cell.nGenes >= config.minGenes && cell.mitoPct <= config.maxMitoPct && cell.totalCounts > 0) {
      keptIndices.push(c);
      keptCells.push(cell);
    }
  }

  const nKept = keptCells.length;
  const normExpr = new Float64Array(nKept * gCount);

  for (let r = 0; r < nKept; r++) {
    const origCell = keptIndices[r];
    const total = raw.cells[origCell].totalCounts || 1;
    const scale = config.targetSum / total;
    const srcOffset = origCell * gCount;
    const dstOffset = r * gCount;
    for (let g = 0; g < gCount; g++) {
      const scaled = raw.counts[srcOffset + g] * scale;
      normExpr[dstOffset + g] = Math.log1p(scaled);
    }
  }

  // Compute mean and variance per gene for HVG selection
  const geneMeans = new Float64Array(gCount);
  const geneDispersions: Array<{ gene: string; disp: number }> = [];

  for (let g = 0; g < gCount; g++) {
    let sum = 0;
    for (let r = 0; r < nKept; r++) {
      sum += normExpr[r * gCount + g];
    }
    const mean = sum / Math.max(1, nKept);
    geneMeans[g] = mean;

    let sqDiff = 0;
    for (let r = 0; r < nKept; r++) {
      const d = normExpr[r * gCount + g] - mean;
      sqDiff += d * d;
    }
    const variance = sqDiff / Math.max(1, nKept - 1);
    const disp = mean > 1e-8 ? variance / mean : 0;
    geneDispersions.push({ gene: raw.genes[g].gene, disp });
  }

  geneDispersions.sort((a, b) => b.disp - a.disp);
  const hvgGenes = geneDispersions.slice(0, Math.min(config.nTopHvg, gCount)).map((d) => d.gene);

  // Estimate PCA variance ratio on top HVGs via deflated power iteration
  const pcaVarianceExplained = computeTopPcaVarianceRatios(
    normExpr,
    nKept,
    gCount,
    geneMeans,
    Math.min(config.nPcaComps, 8)
  );

  const tissueCounts: Record<TissueType, number> = {
    tumor_lung: 0,
    normal_lung: 0,
    lymph_node: 0,
    brain_mets: 0,
  };
  const cellTypeCounts: Record<CellType, number> = {
    exhausted_cd8_t: 0,
    treg: 0,
    macrophage_myeloid: 0,
    epithelial: 0,
  };
  for (const cell of keptCells) {
    tissueCounts[cell.tissue]++;
    cellTypeCounts[cell.cellType]++;
  }

  return {
    genes: raw.genes,
    geneIndex: raw.geneIndex,
    cells: keptCells,
    normExpr,
    hvgGenes,
    pcaVarianceExplained,
    qcStats: {
      rawCells: raw.cells.length,
      filteredCells: raw.cells.length - nKept,
      retainedCells: nKept,
      geneCount: gCount,
      totalCandidatePairs: computeCandidatePairCount(gCount),
      tissueCounts,
      cellTypeCounts,
    },
    seed: raw.seed,
    datasetId: raw.datasetId,
  };
}

function computeTopPcaVarianceRatios(
  normExpr: Float64Array,
  nCells: number,
  nGenes: number,
  geneMeans: Float64Array,
  nComps: number
): number[] {
  if (nCells === 0 || nGenes === 0) return [];
  // Center copy
  const X = new Float64Array(nCells * nGenes);
  let totalVar = 0;
  for (let r = 0; r < nCells; r++) {
    for (let g = 0; g < nGenes; g++) {
      const v = normExpr[r * nGenes + g] - geneMeans[g];
      X[r * nGenes + g] = v;
      totalVar += v * v;
    }
  }
  if (totalVar <= 1e-12) return Array(nComps).fill(0);

  const ratios: number[] = [];
  for (let k = 0; k < nComps; k++) {
    let v = new Float64Array(nGenes);
    for (let g = 0; g < nGenes; g++) v[g] = ((g + k + 1) % 7) - 3;
    normalizeVec(v);

    const u = new Float64Array(nCells);
    for (let iter = 0; iter < 12; iter++) {
      // u = X * v
      for (let r = 0; r < nCells; r++) {
        let s = 0;
        const off = r * nGenes;
        for (let g = 0; g < nGenes; g++) s += X[off + g] * v[g];
        u[r] = s;
      }
      // v_new = X^T * u
      const vNew = new Float64Array(nGenes);
      for (let r = 0; r < nCells; r++) {
        const ur = u[r];
        const off = r * nGenes;
        for (let g = 0; g < nGenes; g++) vNew[g] += X[off + g] * ur;
      }
      normalizeVec(vNew);
      v = vNew;
    }

    let sigmaSq = 0;
    for (let r = 0; r < nCells; r++) {
      let s = 0;
      const off = r * nGenes;
      for (let g = 0; g < nGenes; g++) s += X[off + g] * v[g];
      u[r] = s;
      sigmaSq += s * s;
    }
    ratios.push(Math.round((sigmaSq / totalVar) * 10000) / 10000);

    // Deflate X = X - u * v^T
    for (let r = 0; r < nCells; r++) {
      const ur = u[r];
      const off = r * nGenes;
      for (let g = 0; g < nGenes; g++) X[off + g] -= ur * v[g];
    }
  }
  return ratios;
}

function normalizeVec(v: Float64Array): void {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  if (norm > 1e-12) {
    for (let i = 0; i < v.length; i++) v[i] /= norm;
  }
}
