import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import test from 'node:test';
import {
  generateSyntheticScrnaDataset,
  loadCandidateGenes,
  loadPathwayDatabase,
} from '../../src/engine/data.ts';
import {
  computeSinglePairFeatures,
  createFeatureContext,
} from '../../src/engine/features/index.ts';
import { preprocessScrnaDataset } from '../../src/engine/preprocess.ts';

test('TypeScript and Python pipelines produce identical 60-dim features for shared seed=42', () => {
  const seed = 42;
  const pairs: Array<[string, string]> = [
    ['CD274', 'CTLA4'],
    ['PDCD1', 'CTLA4'],
    ['EGFR', 'MET'],
  ];

  // 1. Run TypeScript pipeline
  const genes = loadCandidateGenes();
  const pathways = loadPathwayDatabase();
  const raw = generateSyntheticScrnaDataset(genes, seed, 720);
  const prep = preprocessScrnaDataset(raw);
  const ctx = createFeatureContext(prep, pathways, 0.5, 16);

  const tsFeatures: Record<string, number[]> = {};
  for (const [g1, g2] of pairs) {
    const idx1 = prep.geneIndex.get(g1)!;
    const idx2 = prep.geneIndex.get(g2)!;
    tsFeatures[`${g1}__${g2}`] = Array.from(computeSinglePairFeatures(ctx, idx1, idx2));
  }

  // 2. Run Python pipeline via subprocess with identical seed=42
  const pyScript = `
import json
from bspai.data import generate_synthetic_scrna_dataset, load_candidate_genes, load_pathway_database
from bspai.preprocess import preprocess_scrna_dataset
from bspai.features import FeatureContext

genes = load_candidate_genes()
pathways = load_pathway_database()
raw = generate_synthetic_scrna_dataset(genes, seed=${seed}, n_cells=720)
prep = preprocess_scrna_dataset(raw, use_scanpy=False)
ctx = FeatureContext(prep, pathways, activity_threshold=0.5, embedding_dim=16)
out = {
    "retained_cells": prep.qc_stats["retained_cells"],
    "total_pairs": prep.qc_stats["total_candidate_pairs"],
    "pairs": {
        "CD274__CTLA4": ctx.compute_pair_features("CD274", "CTLA4"),
        "PDCD1__CTLA4": ctx.compute_pair_features("PDCD1", "CTLA4"),
        "EGFR__MET": ctx.compute_pair_features("EGFR", "MET"),
    },
}
print(json.dumps(out))
`;

  const stdout = execFileSync('python3', ['-c', pyScript], {
    cwd: process.cwd(),
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  const pyOut = JSON.parse(stdout) as {
    retained_cells: number;
    total_pairs: number;
    pairs: Record<string, number[]>;
  };

  assert.equal(pyOut.retained_cells, prep.qcStats.retainedCells);
  assert.equal(pyOut.total_pairs, prep.qcStats.totalCandidatePairs);

  for (const [g1, g2] of pairs) {
    const key = `${g1}__${g2}`;
    const tsVec = tsFeatures[key];
    const pyVec = pyOut.pairs[key];
    assert.equal(tsVec.length, pyVec.length);
    for (let i = 0; i < tsVec.length; i++) {
      const diff = Math.abs(tsVec[i] - pyVec[i]);
      assert.ok(
        diff < 1e-5,
        `Parity mismatch for ${key} feature[${i}] (${ctx.featureNames[i]}): TS=${tsVec[i]} vs PY=${pyVec[i]} (diff=${diff})`
      );
    }
  }
});
