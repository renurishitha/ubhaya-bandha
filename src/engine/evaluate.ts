import type { FeatureEngineContext, FeatureGroupName } from './features/index.ts';
import type { TrainingDatasetIndices } from './labels.ts';
import {
  predictDecisionTree,
  predictDnnMlp,
  predictLogisticRegression,
  predictRandomForest,
  predictXgbScores,
  trainDecisionTree,
  trainDnnMlp,
  trainLogisticRegression,
  trainPairwiseXGBoost,
  trainPointwiseXGBoost,
  trainRandomForest,
  type TrainedXgbModel,
} from './models.ts';
import { Mulberry32, seededShuffle } from './prng.ts';
import type {
  AblationStep,
  FeatureImportanceItem,
  ModelBenchmarkRow,
  RocCurveSeries,
} from './types.ts';

export function computeRocAndAuc(
  yTrue: Array<0 | 1>,
  yScore: number[]
): { auc: number; fpr: number[]; tpr: number[] } {
  const nPos = yTrue.filter((y) => y === 1).length;
  const nNeg = yTrue.length - nPos;
  if (nPos === 0 || nNeg === 0) {
    return { auc: 0.5, fpr: [0, 1], tpr: [0, 1] };
  }

  const order = yScore
    .map((s, idx) => ({ s, y: yTrue[idx] }))
    .sort((a, b) => b.s - a.s);

  const fpr: number[] = [0];
  const tpr: number[] = [0];
  let tp = 0;
  let fp = 0;

  let i = 0;
  while (i < order.length) {
    const score = order[i].s;
    while (i < order.length && order[i].s === score) {
      if (order[i].y === 1) tp++;
      else fp++;
      i++;
    }
    fpr.push(fp / nNeg);
    tpr.push(tp / nPos);
  }

  let auc = 0;
  for (let k = 1; k < fpr.length; k++) {
    auc += 0.5 * (fpr[k] - fpr[k - 1]) * (tpr[k] + tpr[k - 1]);
  }
  return { auc, fpr, tpr };
}

export function computeNdcgAtK(
  relevances: number[],
  scores: number[],
  k: number
): number {
  const n = Math.min(k, relevances.length, scores.length);
  if (n <= 0) return 0;

  const predOrder = scores
    .map((s, idx) => ({ s, rel: relevances[idx] }))
    .sort((a, b) => b.s - a.s);

  const idealOrder = [...relevances].sort((a, b) => b - a);

  let dcg = 0;
  let idcg = 0;
  for (let i = 0; i < n; i++) {
    const denom = Math.log2(i + 2);
    dcg += (Math.pow(2, predOrder[i].rel) - 1) / denom;
    idcg += (Math.pow(2, idealOrder[i]) - 1) / denom;
  }
  return idcg > 0 ? dcg / idcg : 0;
}

export function computePairNdcgScore(
  rank: number,
  rawScore: number
): number {
  const normMargin = 1 / (1 + Math.exp(-rawScore / 3.5));
  const rankDiscount = 1 / Math.log2(Math.max(1, rank) + 1);
  const val = 0.72 * normMargin + 0.28 * rankDiscount;
  return Math.round(Math.min(0.9999, Math.max(0.0001, val)) * 10000) / 10000;
}

function interpolateRoc(
  fpr: number[],
  tpr: number[],
  grid: number[]
): number[] {
  return grid.map((x) => {
    for (let k = 1; k < fpr.length; k++) {
      if (fpr[k] >= x) {
        const span = fpr[k] - fpr[k - 1];
        if (span < 1e-12) return tpr[k];
        const t = (x - fpr[k - 1]) / span;
        return tpr[k - 1] + t * (tpr[k] - tpr[k - 1]);
      }
    }
    return 1;
  });
}

export function createStratifiedKFoldIndices(
  binaryLabels: Array<0 | 1>,
  k = 5,
  seed = 42
): number[][] {
  const rng = new Mulberry32(seed + 909);
  const posIdx: number[] = [];
  const negIdx: number[] = [];
  binaryLabels.forEach((y, i) => (y === 1 ? posIdx.push(i) : negIdx.push(i)));

  const posShuf = seededShuffle(posIdx, rng);
  const negShuf = seededShuffle(negIdx, rng);
  const folds: number[][] = Array.from({ length: k }, () => []);

  posShuf.forEach((idx, i) => folds[i % k].push(idx));
  negShuf.forEach((idx, i) => folds[i % k].push(idx));
  return folds;
}

function extractSubsetRows(
  featureMatrix: Float64Array,
  numFeatures: number,
  rowIndices: number[],
  colIndices: number[]
): number[][] {
  return rowIndices.map((r) => {
    const off = r * numFeatures;
    return colIndices.map((c) => featureMatrix[off + c]);
  });
}

const PAPER_REF_AUC: Record<string, number> = {
  'Pairwise XGBoost (BSPAI)': 0.8929,
  'Pointwise XGBoost': 0.841,
  'GBDT': 0.828,
  'Random Forest': 0.819,
  'DNN (MLP 256-128-64-32-1)': 0.794,
  'Logistic Regression': 0.742,
  'Decision Tree': 0.688,
  'Single Feature (double_ratio)': 0.5476,
};

export interface EvaluationSummary {
  benchmarks: ModelBenchmarkRow[];
  rocCurves: RocCurveSeries[];
  ablation: AblationStep[];
  topFeatureImportances: FeatureImportanceItem[];
  fullPairwiseModel: TrainedXgbModel;
}

export function runFullEvaluation(
  ctx: FeatureEngineContext,
  featureMatrix: Float64Array,
  training: TrainingDatasetIndices,
  seed = 42
): EvaluationSummary {
  const numFeatures = ctx.featureNames.length;
  const allCols = Array.from({ length: numFeatures }, (_, i) => i);
  const doubleRatioOverallCol = ctx.featureNames.indexOf('double_ratio_overall');
  const Xall = extractSubsetRows(
    featureMatrix,
    numFeatures,
    training.pairRowIndices,
    allCols
  );
  const yAll = training.binaryLabels;
  const relAll = training.relevances;
  const folds = createStratifiedKFoldIndices(yAll, 5, seed);

  const modelNames = [
    'Pairwise XGBoost (BSPAI)',
    'Pointwise XGBoost',
    'GBDT',
    'Random Forest',
    'DNN (MLP 256-128-64-32-1)',
    'Logistic Regression',
    'Decision Tree',
    'Single Feature (double_ratio)',
  ];

  const foldScoresByModel = new Map<string, number[][]>();
  const foldAucsByModel = new Map<string, number[]>();
  const foldNdcg10ByModel = new Map<string, number[]>();
  const foldNdcg20ByModel = new Map<string, number[]>();
  const foldNdcg100ByModel = new Map<string, number[]>();
  for (const name of modelNames) {
    foldScoresByModel.set(name, []);
    foldAucsByModel.set(name, []);
    foldNdcg10ByModel.set(name, []);
    foldNdcg20ByModel.set(name, []);
    foldNdcg100ByModel.set(name, []);
  }

  const rocGrid = Array.from({ length: 41 }, (_, i) => i / 40);

  for (let f = 0; f < folds.length; f++) {
    const valSet = new Set(folds[f]);
    const trainIdx: number[] = [];
    const valIdx: number[] = [];
    for (let i = 0; i < Xall.length; i++) {
      if (valSet.has(i)) valIdx.push(i);
      else trainIdx.push(i);
    }

    const Xtr = trainIdx.map((i) => Xall[i]);
    const ytr = trainIdx.map((i) => yAll[i]);
    const rtr = trainIdx.map((i) => relAll[i]);
    const Xva = valIdx.map((i) => Xall[i]);
    const yva = valIdx.map((i) => yAll[i]);
    const rva = valIdx.map((i) => relAll[i]);

    // 1. Pairwise XGBoost
    const pwXgb = trainPairwiseXGBoost(Xtr, rtr, seed + f, 100, 0.1, 10);
    const sPwXgb = predictXgbScores(pwXgb, Xva);

    // 2. Pointwise XGBoost
    const ptXgb = trainPointwiseXGBoost(Xtr, ytr, seed + f, 100, 0.1, 10, 'pointwise_xgb');
    const sPtXgb = predictXgbScores(ptXgb, Xva);

    // 3. GBDT
    const gbdt = trainPointwiseXGBoost(Xtr, ytr, seed + f, 100, 0.1, 4, 'gbdt');
    const sGbdt = predictXgbScores(gbdt, Xva);

    // 4. Random Forest
    const rf = trainRandomForest(Xtr, ytr, seed + f, 100, 10);
    const sRf = predictRandomForest(rf, Xva);

    // 5. DNN
    const dnn = trainDnnMlp(Xtr, ytr, seed + f, 28);
    const sDnn = predictDnnMlp(dnn, Xva);

    // 6. Logistic Regression
    const lr = trainLogisticRegression(Xtr, ytr, 0.1, 1000);
    const sLr = predictLogisticRegression(lr, Xva);

    // 7. Decision Tree
    const dt = trainDecisionTree(Xtr, ytr, 5, 10);
    const sDt = predictDecisionTree(dt, Xva);

    // 8. Single Feature
    const sSingle = Xva.map((row) => row[doubleRatioOverallCol]);

    const preds: Record<string, number[]> = {
      'Pairwise XGBoost (BSPAI)': sPwXgb,
      'Pointwise XGBoost': sPtXgb,
      'GBDT': sGbdt,
      'Random Forest': sRf,
      'DNN (MLP 256-128-64-32-1)': sDnn,
      'Logistic Regression': sLr,
      'Decision Tree': sDt,
      'Single Feature (double_ratio)': sSingle,
    };

    for (const name of modelNames) {
      const roc = computeRocAndAuc(yva, preds[name]);
      foldAucsByModel.get(name)!.push(roc.auc);
      foldScoresByModel.get(name)!.push(interpolateRoc(roc.fpr, roc.tpr, rocGrid));
      foldNdcg10ByModel.get(name)!.push(computeNdcgAtK(rva, preds[name], 10));
      foldNdcg20ByModel.get(name)!.push(computeNdcgAtK(rva, preds[name], 20));
      foldNdcg100ByModel.get(name)!.push(computeNdcgAtK(rva, preds[name], 100));
    }
  }

  const benchmarks: ModelBenchmarkRow[] = [];
  const rocCurves: RocCurveSeries[] = [];

  for (const name of modelNames) {
    const aucs = foldAucsByModel.get(name)!;
    const meanAuc = aucs.reduce((a, b) => a + b, 0) / aucs.length;
    const variance =
      aucs.reduce((a, b) => a + (b - meanAuc) * (b - meanAuc), 0) / aucs.length;
    const stdAuc = Math.sqrt(variance);

    const ndcg10s = foldNdcg10ByModel.get(name)!;
    const ndcg20s = foldNdcg20ByModel.get(name)!;
    const ndcg100s = foldNdcg100ByModel.get(name)!;
    const meanNdcg10 = ndcg10s.reduce((a, b) => a + b, 0) / ndcg10s.length;
    const meanNdcg20 = ndcg20s.reduce((a, b) => a + b, 0) / ndcg20s.length;
    const meanNdcg100 = ndcg100s.reduce((a, b) => a + b, 0) / ndcg100s.length;

    const tprMatrix = foldScoresByModel.get(name)!;
    const meanTpr = rocGrid.map((_, gi) => {
      let s = 0;
      for (const row of tprMatrix) s += row[gi];
      return Math.round((s / tprMatrix.length) * 10000) / 10000;
    });
    meanTpr[0] = 0;
    meanTpr[meanTpr.length - 1] = 1;

    benchmarks.push({
      model: name,
      meanAuc: Math.round(meanAuc * 10000) / 10000,
      stdAuc: Math.round(stdAuc * 10000) / 10000,
      foldAucs: aucs.map((v) => Math.round(v * 10000) / 10000),
      paperRefAuc: PAPER_REF_AUC[name],
      meanNdcg10: Math.round(meanNdcg10 * 10000) / 10000,
      meanNdcg20: Math.round(meanNdcg20 * 10000) / 10000,
      meanNdcg100: Math.round(meanNdcg100 * 10000) / 10000,
    });

    rocCurves.push({
      model: name,
      auc: Math.round(meanAuc * 10000) / 10000,
      fpr: rocGrid,
      tpr: meanTpr,
    });
  }

  // Incremental feature ablation (6 steps matching paper)
  const safetyCols = allCols.filter((c) => ctx.featureGroups[c] === 'safety');
  const mechCols = allCols.filter((c) => ctx.featureGroups[c] === 'mechanism');
  const embCols = allCols.filter((c) => ctx.featureGroups[c] === 'embedding');
  const overallActCols = allCols.filter(
    (c) =>
      ctx.featureGroups[c] === 'activity' &&
      (ctx.featureNames[c].endsWith('_overall') ||
        ctx.featureNames[c].endsWith('_max'))
  );
  const overallCorrCol = ctx.featureNames.indexOf('pearson_overall');

  const ablationDefs: Array<{
    step: string;
    cols: number[];
    paperRefAuc: number;
    singleFeatureOnly?: boolean;
  }> = [
    {
      step: '1. Double-positive only',
      cols: [doubleRatioOverallCol],
      paperRefAuc: 0.5476,
      singleFeatureOnly: true,
    },
    {
      step: '2. + Safety (score_safe)',
      cols: [doubleRatioOverallCol, ...safetyCols],
      paperRefAuc: 0.6706,
    },
    {
      step: '3. + Mechanism (KEGG/GO)',
      cols: [doubleRatioOverallCol, ...safetyCols, ...mechCols],
      paperRefAuc: 0.7174,
    },
    {
      step: '4. + Gene embedding (vector_score)',
      cols: [doubleRatioOverallCol, ...safetyCols, ...mechCols, ...embCols],
      paperRefAuc: 0.7372,
    },
    {
      step: '5. + Single-cell overall expression',
      cols: Array.from(
        new Set([
          doubleRatioOverallCol,
          ...safetyCols,
          ...mechCols,
          ...embCols,
          ...overallActCols,
          overallCorrCol,
        ])
      ),
      paperRefAuc: 0.8619,
    },
    {
      step: '6. + Grouped tissue & cell-type features',
      cols: allCols,
      paperRefAuc: 0.8929,
    },
  ];

  const ablation: AblationStep[] = ablationDefs.map((def) => {
    const foldAucs: number[] = [];
    for (let f = 0; f < folds.length; f++) {
      const valSet = new Set(folds[f]);
      const trainIdx: number[] = [];
      const valIdx: number[] = [];
      for (let i = 0; i < Xall.length; i++) {
        if (valSet.has(i)) valIdx.push(i);
        else trainIdx.push(i);
      }

      if (def.singleFeatureOnly) {
        const scores = valIdx.map((i) => Xall[i][def.cols[0]]);
        const yva = valIdx.map((i) => yAll[i]);
        foldAucs.push(computeRocAndAuc(yva, scores).auc);
      } else {
        const Xtr = trainIdx.map((i) => def.cols.map((c) => Xall[i][c]));
        const rtr = trainIdx.map((i) => relAll[i]);
        const Xva = valIdx.map((i) => def.cols.map((c) => Xall[i][c]));
        const yva = valIdx.map((i) => yAll[i]);
        const mdl = trainPairwiseXGBoost(Xtr, rtr, seed + 17 + f, 100, 0.1, 10);
        const scores = predictXgbScores(mdl, Xva);
        foldAucs.push(computeRocAndAuc(yva, scores).auc);
      }
    }
    const mean = foldAucs.reduce((a, b) => a + b, 0) / foldAucs.length;
    const varr =
      foldAucs.reduce((a, b) => a + (b - mean) * (b - mean), 0) / foldAucs.length;
    return {
      step: def.step,
      featureCount: def.cols.length,
      meanAuc: Math.round(mean * 10000) / 10000,
      stdAuc: Math.round(Math.sqrt(varr) * 10000) / 10000,
      paperRefAuc: def.paperRefAuc,
    };
  });

  // Train final full Pairwise XGBoost model on all training pairs
  const fullPairwiseModel = trainPairwiseXGBoost(Xall, relAll, seed, 100, 0.1, 10);

  // Extract Top-40 feature frequency ("weight") importances
  const topFeatureImportances: FeatureImportanceItem[] = ctx.featureNames
    .map((feat, idx) => ({
      feature: feat,
      weight: fullPairwiseModel.splitCounts[idx],
      group: ctx.featureGroups[idx] as FeatureGroupName,
    }))
    .sort((a, b) => b.weight - a.weight || a.feature.localeCompare(b.feature))
    .slice(0, 40);

  return {
    benchmarks,
    rocCurves,
    ablation,
    topFeatureImportances,
    fullPairwiseModel,
  };
}
