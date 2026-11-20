import type {
  PathwayDatabase,
  PreprocessedDataset,
} from '../types.ts';
import {
  ACTIVITY_GROUPS,
  computePairActivityInto,
  precomputeActivityStats,
  type ActivityPrecomputed,
} from './activity.ts';
import {
  CORR_GROUPS,
  computePairCorrelationsInto,
  precomputeCorrStats,
  type CorrPrecomputed,
} from './corr.ts';
import {
  buildGeneEmbeddings,
  computeEuclideanDistance,
} from './embedding.ts';
import {
  buildPathwayIndex,
  computeMechanismFeatures,
  type PathwayIndex,
} from './pathway.ts';
import { computeGeneTumorDeltas, computeSafetyScore } from './safety.ts';

export type FeatureGroupName =
  | 'safety'
  | 'mechanism'
  | 'embedding'
  | 'activity'
  | 'correlation';

export interface FeatureEngineContext {
  ds: PreprocessedDataset;
  deltas: Float64Array;
  pathwayIndex: PathwayIndex;
  embeddings: Float64Array;
  embeddingDim: number;
  activityStats: ActivityPrecomputed;
  corrStats: CorrPrecomputed;
  featureNames: string[];
  featureGroups: FeatureGroupName[];
}

export function getFeatureSchema(): {
  names: string[];
  groups: FeatureGroupName[];
} {
  const names: string[] = [];
  const groups: FeatureGroupName[] = [];

  const push = (name: string, grp: FeatureGroupName) => {
    names.push(name);
    groups.push(grp);
  };

  // 1. Safety (3)
  push('dx_tumor_minus_normal', 'safety');
  push('dy_tumor_minus_normal', 'safety');
  push('score_safe', 'safety');

  // 2. Mechanism (4)
  push('kegg_cooccur', 'mechanism');
  push('go_cooccur', 'mechanism');
  push('pathway_cooccur', 'mechanism');
  push('same_pathway', 'mechanism');

  // 3. Gene embedding (1)
  push('vector_score', 'embedding');

  // 4. Target activity (9 groups * 5 + 2 summary = 47)
  for (const grp of ACTIVITY_GROUPS) {
    push(`double_ratio_${grp}`, 'activity');
    push(`sum_single_ratio_${grp}`, 'activity');
    push(`min_single_ratio_${grp}`, 'activity');
    push(`sum_single_exp_${grp}`, 'activity');
    push(`min_single_exp_${grp}`, 'activity');
  }
  push('single_ratio_max', 'activity');
  push('double_ratio_max', 'activity');

  // 5. Expression correlation (5)
  for (const grp of CORR_GROUPS) {
    push(`pearson_${grp}`, 'correlation');
  }

  return { names, groups };
}

export function createFeatureContext(
  ds: PreprocessedDataset,
  pathways: PathwayDatabase,
  activityThreshold = 0.5,
  embeddingDim = 16
): FeatureEngineContext {
  const deltas = computeGeneTumorDeltas(ds);
  const pathwayIndex = buildPathwayIndex(pathways);
  const embeddings = buildGeneEmbeddings(ds, embeddingDim);
  const activityStats = precomputeActivityStats(ds, activityThreshold);
  const corrStats = precomputeCorrStats(ds);
  const schema = getFeatureSchema();

  return {
    ds,
    deltas,
    pathwayIndex,
    embeddings,
    embeddingDim,
    activityStats,
    corrStats,
    featureNames: schema.names,
    featureGroups: schema.groups,
  };
}

export function computeSinglePairFeatures(
  ctx: FeatureEngineContext,
  idx1: number,
  idx2: number
): Float64Array {
  const nFeat = ctx.featureNames.length;
  const out = new Float64Array(nFeat);
  computePairFeaturesInto(ctx, idx1, idx2, out, 0);
  return out;
}

export function computePairFeaturesInto(
  ctx: FeatureEngineContext,
  idx1: number,
  idx2: number,
  out: Float64Array,
  offset: number
): void {
  const g1 = ctx.ds.genes[idx1].gene;
  const g2 = ctx.ds.genes[idx2].gene;

  // 1. Safety
  const dx = ctx.deltas[idx1];
  const dy = ctx.deltas[idx2];
  out[offset + 0] = dx;
  out[offset + 1] = dy;
  out[offset + 2] = computeSafetyScore(dx, dy);

  // 2. Mechanism
  const mech = computeMechanismFeatures(g1, g2, ctx.pathwayIndex);
  out[offset + 3] = mech.kegg_cooccur;
  out[offset + 4] = mech.go_cooccur;
  out[offset + 5] = mech.pathway_cooccur;
  out[offset + 6] = mech.same_pathway;

  // 3. Embedding
  out[offset + 7] = computeEuclideanDistance(
    ctx.embeddings,
    idx1,
    idx2,
    ctx.embeddingDim
  );

  // 4. Activity (47)
  computePairActivityInto(ctx.activityStats, idx1, idx2, out, offset + 8);

  // 5. Correlation (5)
  computePairCorrelationsInto(ctx.corrStats, idx1, idx2, out, offset + 8 + 47);
}

/**
 * Computes the feature matrix for all unique candidate pairs G*(G-1)/2.
 */
export function buildAllCandidateFeatureMatrix(ctx: FeatureEngineContext): {
  pairIndicesA: Uint16Array;
  pairIndicesB: Uint16Array;
  featureMatrix: Float64Array;
  numPairs: number;
  numFeatures: number;
} {
  const gCount = ctx.ds.genes.length;
  const numPairs = (gCount * (gCount - 1)) / 2;
  const numFeatures = ctx.featureNames.length;
  const pairIndicesA = new Uint16Array(numPairs);
  const pairIndicesB = new Uint16Array(numPairs);
  const featureMatrix = new Float64Array(numPairs * numFeatures);

  let p = 0;
  for (let i = 0; i < gCount; i++) {
    for (let j = i + 1; j < gCount; j++) {
      pairIndicesA[p] = i;
      pairIndicesB[p] = j;
      computePairFeaturesInto(ctx, i, j, featureMatrix, p * numFeatures);
      p++;
    }
  }

  return {
    pairIndicesA,
    pairIndicesB,
    featureMatrix,
    numPairs,
    numFeatures,
  };
}
