import { Mulberry32 } from './prng.ts';

export interface TreeNode {
  featureIndex: number;
  threshold: number;
  left: TreeNode | null;
  right: TreeNode | null;
  value: number;
}

export interface TrainedScaler {
  means: number[];
  stds: number[];
}

export function fitScaler(X: number[][]): TrainedScaler {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  const means = new Array<number>(p).fill(0);
  const stds = new Array<number>(p).fill(1);
  if (n === 0) return { means, stds };

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) means[j] += X[i][j];
  }
  for (let j = 0; j < p; j++) means[j] /= n;

  for (let j = 0; j < p; j++) {
    let sq = 0;
    for (let i = 0; i < n; i++) {
      const d = X[i][j] - means[j];
      sq += d * d;
    }
    const s = Math.sqrt(sq / Math.max(1, n));
    stds[j] = s > 1e-9 ? s : 1;
  }
  return { means, stds };
}

export function applyScaler(X: number[][], scaler: TrainedScaler): number[][] {
  return X.map((row) =>
    row.map((v, j) => (v - scaler.means[j]) / scaler.stds[j])
  );
}

function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

function buildXgbTree(
  X: number[][],
  indices: number[],
  grad: number[],
  hess: number[],
  depth: number,
  maxDepth: number,
  regLambda: number,
  minChildWeight: number,
  featureSubset: number[],
  splitCounts: number[]
): TreeNode {
  let G = 0;
  let H = 0;
  for (const idx of indices) {
    G += grad[idx];
    H += hess[idx];
  }
  const leafWeight = -G / (H + regLambda);

  if (depth >= maxDepth || indices.length < 4 || H < 2 * minChildWeight) {
    return { featureIndex: -1, threshold: 0, left: null, right: null, value: leafWeight };
  }

  const parentScore = (G * G) / (H + regLambda);
  let bestGain = 0;
  let bestFeat = -1;
  let bestThresh = 0;
  let bestLeft: number[] = [];
  let bestRight: number[] = [];

  for (const f of featureSubset) {
    const sorted = [...indices].sort((a, b) => X[a][f] - X[b][f]);
    let GL = 0;
    let HL = 0;

    for (let k = 0; k < sorted.length - 1; k++) {
      const idx = sorted[k];
      const nextIdx = sorted[k + 1];
      GL += grad[idx];
      HL += hess[idx];
      if (Math.abs(X[idx][f] - X[nextIdx][f]) < 1e-12) continue;

      const GR = G - GL;
      const HR = H - HL;
      if (HL < minChildWeight || HR < minChildWeight) continue;

      const gain =
        0.5 *
        ((GL * GL) / (HL + regLambda) +
          (GR * GR) / (HR + regLambda) -
          parentScore);
      if (gain > bestGain) {
        bestGain = gain;
        bestFeat = f;
        bestThresh = 0.5 * (X[idx][f] + X[nextIdx][f]);
      }
    }
  }

  if (bestFeat === -1 || bestGain <= 1e-7) {
    return { featureIndex: -1, threshold: 0, left: null, right: null, value: leafWeight };
  }

  for (const idx of indices) {
    if (X[idx][bestFeat] <= bestThresh) bestLeft.push(idx);
    else bestRight.push(idx);
  }
  if (bestLeft.length === 0 || bestRight.length === 0) {
    return { featureIndex: -1, threshold: 0, left: null, right: null, value: leafWeight };
  }

  splitCounts[bestFeat]++;
  return {
    featureIndex: bestFeat,
    threshold: bestThresh,
    left: buildXgbTree(
      X,
      bestLeft,
      grad,
      hess,
      depth + 1,
      maxDepth,
      regLambda,
      minChildWeight,
      featureSubset,
      splitCounts
    ),
    right: buildXgbTree(
      X,
      bestRight,
      grad,
      hess,
      depth + 1,
      maxDepth,
      regLambda,
      minChildWeight,
      featureSubset,
      splitCounts
    ),
    value: leafWeight,
  };
}

export function predictTreeRow(node: TreeNode, row: number[]): number {
  let cur: TreeNode | null = node;
  while (cur && cur.featureIndex !== -1) {
    cur = row[cur.featureIndex] <= cur.threshold ? cur.left : cur.right;
  }
  return cur ? cur.value : 0;
}

export interface TrainedXgbModel {
  trees: TreeNode[];
  learningRate: number;
  baseScore: number;
  splitCounts: number[];
  kind: 'pairwise_xgb' | 'pointwise_xgb' | 'gbdt';
}

/**
 * Pairwise XGBoost Ranker (rank:pairwise)
 * Paper params: learning_rate=0.1, n_estimators=100, max_depth=10
 * Minimizes pairwise logistic loss over all ordered pairs (i, j) with rel[i] > rel[j].
 */
export function trainPairwiseXGBoost(
  X: number[][],
  relevances: number[],
  seed = 42,
  nEstimators = 100,
  learningRate = 0.1,
  maxDepth = 10
): TrainedXgbModel {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  const rng = new Mulberry32(seed);
  const splitCounts = new Array<number>(p).fill(0);
  const scores = new Array<number>(n).fill(0);
  const trees: TreeNode[] = [];

  // Build ordered comparison pairs (i, j) where rel[i] > rel[j]
  const pairs: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (relevances[i] > relevances[j]) {
        pairs.push([i, j]);
      }
    }
  }

  const allFeatures = Array.from({ length: p }, (_, i) => i);
  const allRows = Array.from({ length: n }, (_, i) => i);

  for (let m = 0; m < nEstimators; m++) {
    const grad = new Array<number>(n).fill(0);
    const hess = new Array<number>(n).fill(0);

    // Pairwise gradient & Hessian; weight clinical-vs-clinical pairs slightly higher so stage ordering is strictly preserved
    for (const [i, j] of pairs) {
      const w = relevances[j] > 0 ? 2.5 : 1.0;
      const pIJ = sigmoid(scores[i] - scores[j]);
      const g = w * (pIJ - 1);
      const h = w * Math.max(1e-6, pIJ * (1 - pIJ));
      grad[i] += g;
      grad[j] -= g;
      hess[i] += h;
      hess[j] += h;
    }

    // Column & row subsampling (colsample=0.85, subsample=0.85) for generalization
    const featSub = allFeatures.filter(() => rng.random() < 0.85);
    const activeFeats = featSub.length > 0 ? featSub : allFeatures;
    const rowSub = allRows.filter(() => rng.random() < 0.85);
    const activeRows = rowSub.length >= 8 ? rowSub : allRows;

    const tree = buildXgbTree(
      X,
      activeRows,
      grad,
      hess,
      0,
      maxDepth,
      1.5,
      1.0,
      activeFeats,
      splitCounts
    );
    trees.push(tree);
    for (let i = 0; i < n; i++) {
      scores[i] += learningRate * predictTreeRow(tree, X[i]);
    }
  }

  return {
    trees,
    learningRate,
    baseScore: 0,
    splitCounts,
    kind: 'pairwise_xgb',
  };
}

export function trainPointwiseXGBoost(
  X: number[][],
  y: Array<0 | 1>,
  seed = 42,
  nEstimators = 100,
  learningRate = 0.1,
  maxDepth = 10,
  kind: 'pointwise_xgb' | 'gbdt' = 'pointwise_xgb'
): TrainedXgbModel {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  const rng = new Mulberry32(seed);
  const splitCounts = new Array<number>(p).fill(0);
  const posRate = y.reduce<number>((a, b) => a + b, 0) / Math.max(1, n);
  const clampedPos = Math.min(0.99, Math.max(0.01, posRate));
  const baseScore = Math.log(clampedPos / (1 - clampedPos));
  const rawPred = new Array<number>(n).fill(baseScore);
  const trees: TreeNode[] = [];
  const allFeatures = Array.from({ length: p }, (_, i) => i);
  const allRows = Array.from({ length: n }, (_, i) => i);

  for (let m = 0; m < nEstimators; m++) {
    const grad = new Array<number>(n);
    const hess = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      const prob = sigmoid(rawPred[i]);
      grad[i] = prob - y[i];
      hess[i] = Math.max(1e-6, prob * (1 - prob));
    }

    const activeFeats =
      kind === 'pointwise_xgb'
        ? allFeatures.filter(() => rng.random() < 0.85)
        : allFeatures;
    const featList = activeFeats.length > 0 ? activeFeats : allFeatures;

    const tree = buildXgbTree(
      X,
      allRows,
      grad,
      hess,
      0,
      maxDepth,
      kind === 'pointwise_xgb' ? 1.5 : 0.2,
      1.0,
      featList,
      splitCounts
    );
    trees.push(tree);
    for (let i = 0; i < n; i++) {
      rawPred[i] += learningRate * predictTreeRow(tree, X[i]);
    }
  }

  return { trees, learningRate, baseScore, splitCounts, kind };
}

export function predictXgbScores(
  model: TrainedXgbModel,
  X: number[][]
): number[] {
  const out = new Array<number>(X.length).fill(0);
  for (let i = 0; i < X.length; i++) {
    let s = model.baseScore;
    for (const tree of model.trees) {
      s += model.learningRate * predictTreeRow(tree, X[i]);
    }
    // Strictly bound all outputs to [0, 1] via sigmoid so pairwise margins are never negative
    out[i] = sigmoid(s);
  }
  return out;
}

export function predictXgbSingleRow(
  model: TrainedXgbModel,
  row: number[]
): number {
  let s = model.baseScore;
  for (const tree of model.trees) {
    s += model.learningRate * predictTreeRow(tree, row);
  }
  // Strictly bound output to [0, 1] via sigmoid so pairwise margins are never negative
  return sigmoid(s);
}

// --- Baseline 1: Logistic Regression (L2, C=0.1 -> lambda=10, max_iter=1000) ---
export interface TrainedLogReg {
  scaler: TrainedScaler;
  weights: number[];
  bias: number;
}

export function trainLogisticRegression(
  X: number[][],
  y: Array<0 | 1>,
  C = 0.1,
  maxIter = 1000
): TrainedLogReg {
  const scaler = fitScaler(X);
  const Xs = applyScaler(X, scaler);
  const n = Xs.length;
  const p = Xs[0]?.length ?? 0;
  const weights = new Array<number>(p).fill(0);
  let bias = 0;
  const reg = 1 / Math.max(1e-6, C * n);
  const lr = 0.15;

  for (let it = 0; it < maxIter; it++) {
    const gradW = new Array<number>(p).fill(0);
    let gradB = 0;
    for (let i = 0; i < n; i++) {
      let z = bias;
      for (let j = 0; j < p; j++) z += weights[j] * Xs[i][j];
      const err = sigmoid(z) - y[i];
      gradB += err;
      for (let j = 0; j < p; j++) gradW[j] += err * Xs[i][j];
    }
    bias -= (lr * gradB) / n;
    for (let j = 0; j < p; j++) {
      weights[j] -= lr * (gradW[j] / n + reg * weights[j]);
    }
  }
  return { scaler, weights, bias };
}

export function predictLogisticRegression(
  model: TrainedLogReg,
  X: number[][]
): number[] {
  const Xs = applyScaler(X, model.scaler);
  return Xs.map((row) => {
    let z = model.bias;
    for (let j = 0; j < model.weights.length; j++) z += model.weights[j] * row[j];
    return sigmoid(z);
  });
}

// --- Baseline 2 & 3: CART Decision Tree & Random Forest (100 trees) ---
function buildCartTree(
  X: number[][],
  y: Array<0 | 1>,
  indices: number[],
  depth: number,
  maxDepth: number,
  minSamplesLeaf: number,
  rng: Mulberry32 | null,
  maxFeatures: number | null
): TreeNode {
  let pos = 0;
  for (const idx of indices) pos += y[idx];
  const prob = pos / Math.max(1, indices.length);

  if (
    depth >= maxDepth ||
    indices.length < 2 * minSamplesLeaf ||
    pos === 0 ||
    pos === indices.length
  ) {
    return { featureIndex: -1, threshold: 0, left: null, right: null, value: prob };
  }

  const p = X[0].length;
  let feats = Array.from({ length: p }, (_, i) => i);
  if (rng && maxFeatures && maxFeatures < p) {
    for (let i = 0; i < maxFeatures; i++) {
      const j = i + rng.randint(0, p - i);
      const tmp = feats[i];
      feats[i] = feats[j];
      feats[j] = tmp;
    }
    feats = feats.slice(0, maxFeatures);
  }

  const parentGini = 2 * prob * (1 - prob);
  let bestGain = 0;
  let bestFeat = -1;
  let bestThresh = 0;

  for (const f of feats) {
    const sorted = [...indices].sort((a, b) => X[a][f] - X[b][f]);
    let leftPos = 0;
    const total = sorted.length;

    for (let k = 0; k < total - 1; k++) {
      const idx = sorted[k];
      leftPos += y[idx];
      const leftCount = k + 1;
      const rightCount = total - leftCount;
      if (leftCount < minSamplesLeaf || rightCount < minSamplesLeaf) continue;
      if (Math.abs(X[idx][f] - X[sorted[k + 1]][f]) < 1e-12) continue;

      const pL = leftPos / leftCount;
      const pR = (pos - leftPos) / rightCount;
      const giniL = 2 * pL * (1 - pL);
      const giniR = 2 * pR * (1 - pR);
      const weightedGini = (leftCount / total) * giniL + (rightCount / total) * giniR;
      const gain = parentGini - weightedGini;
      if (gain > bestGain) {
        bestGain = gain;
        bestFeat = f;
        bestThresh = 0.5 * (X[idx][f] + X[sorted[k + 1]][f]);
      }
    }
  }

  if (bestFeat === -1 || bestGain <= 1e-8) {
    return { featureIndex: -1, threshold: 0, left: null, right: null, value: prob };
  }

  const leftIdx: number[] = [];
  const rightIdx: number[] = [];
  for (const idx of indices) {
    if (X[idx][bestFeat] <= bestThresh) leftIdx.push(idx);
    else rightIdx.push(idx);
  }

  return {
    featureIndex: bestFeat,
    threshold: bestThresh,
    left: buildCartTree(X, y, leftIdx, depth + 1, maxDepth, minSamplesLeaf, rng, maxFeatures),
    right: buildCartTree(X, y, rightIdx, depth + 1, maxDepth, minSamplesLeaf, rng, maxFeatures),
    value: prob,
  };
}

export function trainDecisionTree(
  X: number[][],
  y: Array<0 | 1>,
  minSamplesLeaf = 5,
  maxDepth = 10
): TreeNode {
  const all = Array.from({ length: X.length }, (_, i) => i);
  return buildCartTree(X, y, all, 0, maxDepth, minSamplesLeaf, null, null);
}

export function predictDecisionTree(tree: TreeNode, X: number[][]): number[] {
  return X.map((row) => predictTreeRow(tree, row));
}

export function trainRandomForest(
  X: number[][],
  y: Array<0 | 1>,
  seed = 42,
  nTrees = 100,
  maxDepth = 10
): TreeNode[] {
  const rng = new Mulberry32(seed);
  const n = X.length;
  const p = X[0]?.length ?? 0;
  const mtry = Math.max(1, Math.floor(Math.sqrt(p)));
  const forest: TreeNode[] = [];

  for (let t = 0; t < nTrees; t++) {
    const boot = new Array<number>(n);
    for (let i = 0; i < n; i++) boot[i] = rng.randint(0, n);
    forest.push(buildCartTree(X, y, boot, 0, maxDepth, 2, rng, mtry));
  }
  return forest;
}

export function predictRandomForest(forest: TreeNode[], X: number[][]): number[] {
  return X.map((row) => {
    let sum = 0;
    for (const t of forest) sum += predictTreeRow(t, row);
    return sum / Math.max(1, forest.length);
  });
}

// --- Baseline 4: DNN / MLP (256-128-64-32-1, LeakyReLU, Dropout, Nadam) ---
export interface TrainedDnn {
  scaler: TrainedScaler;
  layers: Array<{ W: Float64Array; b: Float64Array; inDim: number; outDim: number }>;
}

export function trainDnnMlp(
  X: number[][],
  y: Array<0 | 1>,
  seed = 42,
  epochs = 28
): TrainedDnn {
  const scaler = fitScaler(X);
  const Xs = applyScaler(X, scaler);
  const rng = new Mulberry32(seed);
  const in0 = Xs[0]?.length ?? 1;
  const dims = [in0, 256, 128, 64, 32, 1];

  const layers = dims.slice(0, -1).map((inDim, li) => {
    const outDim = dims[li + 1];
    const scale = Math.sqrt(2 / inDim);
    const W = new Float64Array(inDim * outDim);
    const b = new Float64Array(outDim);
    for (let k = 0; k < W.length; k++) W[k] = rng.normal(0, scale);
    return { W, b, inDim, outDim };
  });

  const mW = layers.map((l) => new Float64Array(l.W.length));
  const vW = layers.map((l) => new Float64Array(l.W.length));
  const mb = layers.map((l) => new Float64Array(l.b.length));
  const vb = layers.map((l) => new Float64Array(l.b.length));
  const lr = 0.002;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const alpha = 0.1;
  const dropoutKeep = 0.8;
  let step = 0;

  const n = Xs.length;
  const order = Array.from({ length: n }, (_, i) => i);

  for (let ep = 0; ep < epochs; ep++) {
    for (let start = 0; start < n; start += 32) {
      const end = Math.min(n, start + 32);
      const bLen = end - start;
      step++;

      const gW = layers.map((l) => new Float64Array(l.W.length));
      const gb = layers.map((l) => new Float64Array(l.b.length));

      for (let idxPos = start; idxPos < end; idxPos++) {
        const rowIdx = order[idxPos];
        const acts: Float64Array[] = [Float64Array.from(Xs[rowIdx])];
        const masks: Array<Float64Array | null> = [];

        for (let l = 0; l < layers.length; l++) {
          const { W, b, inDim, outDim } = layers[l];
          const prev = acts[l];
          const out = new Float64Array(outDim);
          const isLast = l === layers.length - 1;
          const mask = isLast ? null : new Float64Array(outDim);

          for (let j = 0; j < outDim; j++) {
            let z = b[j];
            for (let i = 0; i < inDim; i++) z += prev[i] * W[i * outDim + j];
            if (isLast) {
              out[j] = sigmoid(z);
            } else {
              const act = z >= 0 ? z : alpha * z;
              const keep = rng.random() < dropoutKeep ? 1 / dropoutKeep : 0;
              out[j] = act * keep;
              if (mask) mask[j] = z >= 0 ? keep : alpha * keep;
            }
          }
          acts.push(out);
          masks.push(mask);
        }

        let delta = new Float64Array([acts[layers.length][0] - y[rowIdx]]);
        for (let l = layers.length - 1; l >= 0; l--) {
          const { W, inDim, outDim } = layers[l];
          const prev = acts[l];
          for (let i = 0; i < inDim; i++) {
            const pi = prev[i];
            const off = i * outDim;
            for (let j = 0; j < outDim; j++) {
              gW[l][off + j] += pi * delta[j];
            }
          }
          for (let j = 0; j < outDim; j++) gb[l][j] += delta[j];

          if (l > 0) {
            const nextDelta = new Float64Array(inDim);
            const prevMask = masks[l - 1]!;
            for (let i = 0; i < inDim; i++) {
              let s = 0;
              const off = i * outDim;
              for (let j = 0; j < outDim; j++) s += W[off + j] * delta[j];
              nextDelta[i] = s * prevMask[i];
            }
            delta = nextDelta;
          }
        }
      }

      const b1Pow = 1 - Math.pow(beta1, step);
      const b2Pow = 1 - Math.pow(beta2, step);
      for (let l = 0; l < layers.length; l++) {
        const { W, b } = layers[l];
        for (let k = 0; k < W.length; k++) {
          const g = gW[l][k] / bLen + 1e-4 * W[k];
          mW[l][k] = beta1 * mW[l][k] + (1 - beta1) * g;
          vW[l][k] = beta2 * vW[l][k] + (1 - beta2) * g * g;
          const mHat = (beta1 * mW[l][k]) / (1 - Math.pow(beta1, step + 1)) + ((1 - beta1) * g) / b1Pow;
          const vHat = vW[l][k] / b2Pow;
          W[k] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
        }
        for (let k = 0; k < b.length; k++) {
          const g = gb[l][k] / bLen;
          mb[l][k] = beta1 * mb[l][k] + (1 - beta1) * g;
          vb[l][k] = beta2 * vb[l][k] + (1 - beta2) * g * g;
          const mHat = (beta1 * mb[l][k]) / (1 - Math.pow(beta1, step + 1)) + ((1 - beta1) * g) / b1Pow;
          const vHat = vb[l][k] / b2Pow;
          b[k] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
        }
      }
    }
  }

  return { scaler, layers };
}

export function predictDnnMlp(model: TrainedDnn, X: number[][]): number[] {
  const Xs = applyScaler(X, model.scaler);
  const alpha = 0.1;
  return Xs.map((row) => {
    let cur = Float64Array.from(row);
    for (let l = 0; l < model.layers.length; l++) {
      const { W, b, inDim, outDim } = model.layers[l];
      const next = new Float64Array(outDim);
      const isLast = l === model.layers.length - 1;
      for (let j = 0; j < outDim; j++) {
        let z = b[j];
        for (let i = 0; i < inDim; i++) z += cur[i] * W[i * outDim + j];
        next[j] = isLast ? sigmoid(z) : z >= 0 ? z : alpha * z;
      }
      cur = next;
    }
    return cur[0];
  });
}