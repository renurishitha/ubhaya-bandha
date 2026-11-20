import { Mulberry32 } from '../prng.ts';
import type { PreprocessedDataset } from '../types.ts';

/**
 * Learns deterministic Gene2Vec-style co-expression embeddings (dim=16)
 * from the positive gene-gene co-expression correlation matrix across single cells
 * using deflated power iteration (deterministic spectral factorization).
 */
export function buildGeneEmbeddings(
  ds: PreprocessedDataset,
  dim = 16
): Float64Array {
  const nCells = ds.cells.length;
  const gCount = ds.genes.length;
  const rng = new Mulberry32(ds.seed + 101);

  // Build standardized (zero-mean, unit-norm across cells) gene matrix G x C
  const Z = new Float64Array(gCount * nCells);
  for (let g = 0; g < gCount; g++) {
    let mean = 0;
    for (let c = 0; c < nCells; c++) mean += ds.normExpr[c * gCount + g];
    mean /= Math.max(1, nCells);

    let normSq = 0;
    const gOff = g * nCells;
    for (let c = 0; c < nCells; c++) {
      const v = ds.normExpr[c * gCount + g] - mean;
      Z[gOff + c] = v;
      normSq += v * v;
    }
    const invNorm = normSq > 1e-12 ? 1 / Math.sqrt(normSq) : 0;
    for (let c = 0; c < nCells; c++) Z[gOff + c] *= invNorm;
  }

  // Gene-gene cosine co-expression matrix (G x G), thresholded at >= 0 (PPMI-like)
  const C = new Float64Array(gCount * gCount);
  for (let i = 0; i < gCount; i++) {
    const iOff = i * nCells;
    C[i * gCount + i] = 1.0;
    for (let j = i + 1; j < gCount; j++) {
      const jOff = j * nCells;
      let dot = 0;
      for (let c = 0; c < nCells; c++) dot += Z[iOff + c] * Z[jOff + c];
      const val = Math.max(0, dot);
      C[i * gCount + j] = val;
      C[j * gCount + i] = val;
    }
  }

  const emb = new Float64Array(gCount * dim);
  const Cwork = new Float64Array(C);

  for (let d = 0; d < dim; d++) {
    const v = new Float64Array(gCount);
    for (let g = 0; g < gCount; g++) v[g] = rng.normal(0, 1);
    normalize(v);

    const w = new Float64Array(gCount);
    for (let iter = 0; iter < 15; iter++) {
      for (let i = 0; i < gCount; i++) {
        let s = 0;
        const off = i * gCount;
        for (let j = 0; j < gCount; j++) s += Cwork[off + j] * v[j];
        w[i] = s;
      }
      normalize(w);
      v.set(w);
    }

    // Eigenvalue lambda = v^T Cwork v
    let lambda = 0;
    for (let i = 0; i < gCount; i++) {
      let s = 0;
      const off = i * gCount;
      for (let j = 0; j < gCount; j++) s += Cwork[off + j] * v[j];
      lambda += v[i] * s;
    }
    const scale = Math.sqrt(Math.max(0, lambda));
    for (let g = 0; g < gCount; g++) {
      emb[g * dim + d] = v[g] * scale;
    }

    // Deflate Cwork -= lambda * v * v^T
    for (let i = 0; i < gCount; i++) {
      const off = i * gCount;
      const lvi = lambda * v[i];
      for (let j = 0; j < gCount; j++) {
        Cwork[off + j] -= lvi * v[j];
      }
    }
  }

  return emb;
}

/**
 * Euclidean distance d(x, y) = sqrt(sum_i (xi - yi)^2) ("vector_score").
 */
export function computeEuclideanDistance(
  emb: Float64Array,
  idx1: number,
  idx2: number,
  dim = 16
): number {
  const off1 = idx1 * dim;
  const off2 = idx2 * dim;
  let sq = 0;
  for (let d = 0; d < dim; d++) {
    const diff = emb[off1 + d] - emb[off2 + d];
    sq += diff * diff;
  }
  return Math.sqrt(sq);
}

export function computeEuclideanArrays(a: number[], b: number[]): number {
  let sq = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sq += d * d;
  }
  return Math.sqrt(sq);
}

function normalize(v: Float64Array): void {
  let sq = 0;
  for (let i = 0; i < v.length; i++) sq += v[i] * v[i];
  const n = Math.sqrt(sq);
  if (n > 1e-12) {
    for (let i = 0; i < v.length; i++) v[i] /= n;
  }
}
