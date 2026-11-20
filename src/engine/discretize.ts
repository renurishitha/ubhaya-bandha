import type {
  BinEdgeSpec,
  FeatureImportanceItem,
} from './types.ts';

const GLOSSARY_MAP: Array<{
  match: (f: string) => boolean;
  labels: string[];
  glossaryTerm: string;
  plainDescription: string;
}> = [
  {
    match: (f) => f === 'vector_score',
    labels: ['very similar', 'similar', 'moderate similarity', 'dissimilar', 'very dissimilar'],
    glossaryTerm: 'Gene2Vec Euclidean Distance',
    plainDescription:
      'Euclidean distance between co-expression embedding vectors; smaller distance indicates closer functional co-expression.',
  },
  {
    match: (f) => f === 'score_safe',
    labels: ['unsafe', 'low safety', 'moderate safety', 'good safety', 'high safety'],
    glossaryTerm: 'Tumor-Normal Safety Index',
    plainDescription:
      'Harmonic tumor-minus-normal expression enrichment score; higher values reflect stronger tumor selectivity over normal tissue.',
  },
  {
    match: (f) => f.startsWith('pearson_'),
    labels: [
      'strongly negative',
      'moderately negative',
      'near zero',
      'moderately positive',
      'strongly positive',
    ],
    glossaryTerm: 'Single-Cell Pearson Correlation',
    plainDescription:
      'Linear co-expression correlation across single cells; positive values indicate co-expression within the same cells.',
  },
  {
    match: (f) => f.includes('double_ratio'),
    labels: ['very low', 'low', 'moderate', 'high', 'very high'],
    glossaryTerm: 'Double-Positive Cell Ratio',
    plainDescription:
      'Fraction of cells co-expressing both targets above threshold T; higher values support cis-co-engagement on single cells.',
  },
  {
    match: (f) => f.includes('single_ratio'),
    labels: ['very low', 'low', 'moderate', 'high', 'very high'],
    glossaryTerm: 'Single-Target Positive Ratio',
    plainDescription:
      'Proportion of cells in a population expressing at least one target above threshold T.',
  },
  {
    match: (f) => f.includes('pathway') || f.includes('kegg') || f.includes('go_'),
    labels: ['no overlap', 'minimal overlap', 'moderate overlap', 'high overlap', 'very high overlap'],
    glossaryTerm: 'KEGG / GO Pathway Co-occurrence',
    plainDescription:
      'Count of shared KEGG signaling pathways and GO biological process annotations between the two targets.',
  },
  {
    match: () => true,
    labels: ['very low', 'low', 'moderate', 'high', 'very high'],
    glossaryTerm: 'Single-Cell Activity Metric',
    plainDescription:
      'Quantitative single-cell expression or target-activity summary across tumor and immune compartments.',
  },
];

function getFeatureBinMeta(feature: string) {
  return GLOSSARY_MAP.find((m) => m.match(feature)) ?? GLOSSARY_MAP[GLOSSARY_MAP.length - 1];
}

/**
 * Selects top 6 features from Pairwise XGBoost split weights (guaranteed distinct)
 * and learns up to 5 equal-frequency (quantile) bin edges strictly on training data.
 */
export function fitFeatureDiscretizer(
  featureNames: string[],
  featureMatrix: Float64Array,
  trainingRowIndices: number[],
  importances: FeatureImportanceItem[],
  topK = 6,
  nBins = 5
): BinEdgeSpec[] {
  const numFeatures = featureNames.length;
  const chosen: string[] = [];

  for (const item of importances) {
    if (item.weight > 0 && !chosen.includes(item.feature)) {
      chosen.push(item.feature);
      if (chosen.length >= topK) break;
    }
  }

  // Ensure canonical interpretable features fill any remaining slots if splits were concentrated
  const canonicalFallback = [
    'vector_score',
    'score_safe',
    'pearson_overall',
    'single_ratio_max',
    'double_ratio_max',
    'pathway_cooccur',
  ];
  for (const cand of canonicalFallback) {
    if (chosen.length >= topK) break;
    if (featureNames.includes(cand) && !chosen.includes(cand)) {
      chosen.push(cand);
    }
  }

  return chosen.map((featName) => {
    const colIdx = featureNames.indexOf(featName);
    const vals = trainingRowIndices
      .map((r) => featureMatrix[r * numFeatures + colIdx])
      .sort((a, b) => a - b);

    const rawEdges: number[] = [vals[0]];
    for (let b = 1; b < nBins; b++) {
      const qPos = (b / nBins) * (vals.length - 1);
      const lo = Math.floor(qPos);
      const hi = Math.ceil(qPos);
      const frac = qPos - lo;
      const edge = vals[lo] + frac * (vals[hi] - vals[lo]);
      rawEdges.push(edge);
    }
    rawEdges.push(vals[vals.length - 1]);

    // Deduplicate collapsed quantiles while preserving at least 2 distinct bin edges
    const edges: number[] = [rawEdges[0]];
    for (let i = 1; i < rawEdges.length; i++) {
      if (rawEdges[i] > edges[edges.length - 1] + 1e-9) {
        edges.push(rawEdges[i]);
      }
    }
    if (edges.length < 2) {
      edges.push(edges[0] + 1e-6);
    }

    const binCount = Math.max(1, edges.length - 1);
    const meta = getFeatureBinMeta(featName);
    const labels =
      binCount === 5
        ? meta.labels
        : Array.from({ length: binCount }, (_, bi) => {
            const mappedIdx = Math.round((bi / Math.max(1, binCount - 1)) * (meta.labels.length - 1));
            return meta.labels[mappedIdx];
          });

    return {
      feature: featName,
      edges: edges.map((v) => Math.round(v * 1e6) / 1e6),
      labels,
      glossaryTerm: meta.glossaryTerm,
      plainDescription: meta.plainDescription,
    };
  });
}

export function applyDiscretizer(
  specs: BinEdgeSpec[],
  featureValues: Record<string, number>
): Array<{
  feature: string;
  value: number;
  binIndex: number;
  binLabel: string;
  description: string;
}> {
  return specs.map((spec) => {
    const val = featureValues[spec.feature] ?? 0;
    const nBins = spec.labels.length;
    let binIdx = nBins - 1;
    for (let b = 0; b < nBins; b++) {
      const upper = spec.edges[b + 1];
      if (val <= upper || b === nBins - 1) {
        binIdx = b;
        break;
      }
    }
    return {
      feature: spec.feature,
      value: val,
      binIndex: binIdx,
      binLabel: spec.labels[binIdx],
      description: spec.plainDescription,
    };
  });
}
