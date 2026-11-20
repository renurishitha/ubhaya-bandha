import assert from 'node:assert/strict';
import test from 'node:test';
import {
  computeCandidatePairCount,
  loadCandidateGenes,
} from '../../src/engine/data.ts';
import {
  applyDiscretizer,
  fitFeatureDiscretizer,
} from '../../src/engine/discretize.ts';
import {
  computeDoublePositiveRatio,
  computeSinglePositiveRatio,
} from '../../src/engine/features/activity.ts';
import { computePearsonCorrelation } from '../../src/engine/features/corr.ts';
import { computeEuclideanArrays } from '../../src/engine/features/embedding.ts';
import { computeSafetyScore } from '../../src/engine/features/safety.ts';
import {
  predictXgbScores,
  trainPairwiseXGBoost,
} from '../../src/engine/models.ts';
import {
  generateGroundedRagReport,
  resolveAndValidateGeminiModel,
} from '../../src/engine/rag.ts';

test('safety score matches paper formula (dx*dy)/(dx+dy) and guards edge cases', () => {
  assert.ok(Math.abs(computeSafetyScore(2.0, 6.0) - 1.5) < 1e-9);
  assert.ok(Math.abs(computeSafetyScore(3.0, 3.0) - 1.5) < 1e-9);
  assert.equal(computeSafetyScore(0, 0), 0);
  assert.ok(computeSafetyScore(-1.2, 2.5) < 0);
  assert.ok(computeSafetyScore(-2.0, -3.0) < 0);
});

test('single positive and double positive activity ratios match hand-computed values', () => {
  const e1 = [0.2, 0.8, 1.5, 0.5, 1.1];
  const e2 = [0.9, 0.7, 0.1, 1.2, 0.6];
  // Strictly > 0.5: e1 has 0.8, 1.5, 1.1 -> 3/5 = 0.6
  assert.ok(Math.abs(computeSinglePositiveRatio(e1, 0.5) - 0.6) < 1e-9);
  // Both > 0.5: index 1 (0.8, 0.7) and index 4 (1.1, 0.6) -> 2/5 = 0.4
  assert.ok(Math.abs(computeDoublePositiveRatio(e1, e2, 0.5) - 0.4) < 1e-9);
});

test('Euclidean distance and Pearson correlation match hand-computed values', () => {
  assert.ok(Math.abs(computeEuclideanArrays([1, 2, 3], [4, 6, 3]) - 5.0) < 1e-9);
  assert.ok(Math.abs(computePearsonCorrelation([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-9);
  assert.ok(Math.abs(computePearsonCorrelation([1, 2, 3, 4], [8, 6, 4, 2]) + 1.0) < 1e-9);
});

test('dynamic candidate pair count equals G*(G-1)/2 from loaded CSV', () => {
  const genes = loadCandidateGenes();
  assert.equal(computeCandidatePairCount(genes.length), (genes.length * (genes.length - 1)) / 2);
  assert.equal(computeCandidatePairCount(223), 24753);
});

test('pairwise XGBoost respects clinical stage ordering', () => {
  const X = [
    [2.5, 0.9, 0.8, 0.4], // Stage 1 (rel=5)
    [2.3, 0.8, 0.7, 0.35],
    [1.6, 0.6, 0.5, 0.25], // Stage 2 (rel=4)
    [1.4, 0.55, 0.45, 0.2],
    [0.8, 0.3, 0.3, 0.1], // Stage 4 (rel=2)
    [0.6, 0.25, 0.2, 0.08],
    [-0.5, 0.05, 0.05, 0.01], // Unlabeled (rel=0)
    [-0.8, 0.02, 0.01, 0.0],
  ];
  const rel = [5, 5, 4, 4, 2, 2, 0, 0];
  const model = trainPairwiseXGBoost(X, rel, 42, 40, 0.15, 4);
  const scores = predictXgbScores(model, X);

  assert.ok(scores[0] > scores[2], `Expected Stage 1 (${scores[0]}) > Stage 2 (${scores[2]})`);
  assert.ok(scores[2] > scores[4], `Expected Stage 2 (${scores[2]}) > Stage 4 (${scores[4]})`);
  assert.ok(scores[4] > scores[6], `Expected Stage 4 (${scores[4]}) > Unlabeled (${scores[6]})`);
});

test('GEMINI_MODEL validator accepts current models and rejects deprecated aliases', () => {
  assert.equal(resolveAndValidateGeminiModel('gemini-4-hydrogen'), 'gemini-4-hydrogen');
  assert.equal(resolveAndValidateGeminiModel('gemini-4-argon'), 'gemini-4-argon');
  assert.throws(() => resolveAndValidateGeminiModel('gemini-1.5-flash'), /deprecated/);
  assert.throws(() => resolveAndValidateGeminiModel('gemini-2.0-flash'), /deprecated/);
  assert.throws(() => resolveAndValidateGeminiModel('unknown-model-x'), /invalid/);
});

test('discretizer and RAG template fallback include provenance stamp', async () => {
  const names = ['vector_score', 'score_safe', 'pearson_overall', 'single_ratio_max', 'double_ratio_max', 'pathway_cooccur'];
  const matrix = new Float64Array([
    1.0, 0.1, -0.2, 0.2, 0.05, 0,
    1.5, 0.3, 0.0, 0.4, 0.1, 1,
    2.0, 0.5, 0.2, 0.6, 0.2, 2,
    2.5, 0.8, 0.5, 0.8, 0.4, 3,
    3.0, 1.2, 0.8, 0.95, 0.6, 4,
  ]);
  const imps = names.map((feature, i) => ({
    feature,
    weight: 20 - i,
    group: 'safety' as const,
  }));
  const specs = fitFeatureDiscretizer(names, matrix, [0, 1, 2, 3, 4], imps, 6, 5);
  assert.equal(specs.length, 6);
  const applied = applyDiscretizer(specs, {
    vector_score: 1.05,
    score_safe: 1.15,
    pearson_overall: 0.75,
    single_ratio_max: 0.9,
    double_ratio_max: 0.55,
    pathway_cooccur: 4,
  });
  assert.equal(applied[0].binLabel, 'very similar');
  assert.equal(applied[1].binLabel, 'high safety');

  const prevKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = '';
  try {
    const report = await generateGroundedRagReport({
      prediction: {
        pairKey: 'CD274__CTLA4',
        gene1: 'CD274',
        gene2: 'CTLA4',
        score: 5.25,
        ndcgScore: 0.785,
        rank: 28,
        totalPairs: 53301,
        mechanismType: 'tumor-immunotherapy',
        inClinicalTrial: true,
        clinicalStage: 2,
        clinicalDrugs: ['KN046'],
        needsVerification: true,
        features: {
          vector_score: 1.05,
          score_safe: 1.15,
          pearson_overall: 0.32,
          single_ratio_max: 0.85,
          double_ratio_max: 0.41,
          pathway_cooccur: 3,
          kegg_cooccur: 1,
          go_cooccur: 2,
          same_pathway: 1,
        },
      },
      gene1Info: { gene: 'CD274', category: 'immune_checkpoint', cell_bias: 'epithelial', description: 'PD-L1' },
      gene2Info: { gene: 'CTLA4', category: 'immune_checkpoint', cell_bias: 'treg', description: 'CTLA-4' },
      binSpecs: specs,
      datasetId: 'GSE131907 (NSCLC scRNA-seq)',
      seed: 42,
      modelVersion: 'BSPAI-PairwiseXGB-v1.0.0',
      dataMode: 'DEMO_SYNTHETIC',
    });

    assert.equal(report.provenance.generationMode, 'deterministic_template');
    assert.equal(report.provenance.seed, 42);
    assert.ok(report.reportMarkdown.includes('GSE131907 (NSCLC scRNA-seq)'));
    assert.ok(report.reportMarkdown.includes('BSPAI-PairwiseXGB-v1.0.0'));
    assert.ok(report.reportMarkdown.includes('#28 / 53,301') || report.reportMarkdown.includes('#28 / 53301'));
    assert.ok(report.plainEnglishReport.includes('Pairwise Druggability Score of 5.2500'));
  } finally {
    process.env.GEMINI_API_KEY = prevKey;
  }
});
