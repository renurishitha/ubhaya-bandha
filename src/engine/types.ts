export type TissueType = 'tumor_lung' | 'normal_lung' | 'lymph_node' | 'brain_mets';

export type CellType =
  | 'exhausted_cd8_t'
  | 'treg'
  | 'macrophage_myeloid'
  | 'epithelial';

export type MechanismType = 'bridge-two-cells' | 'TAA+TAA' | 'tumor-immunotherapy';

export interface GeneInfo {
  gene: string;
  category: string;
  cell_bias: CellType;
  description: string;
}

export interface ClinicalPair {
  drug: string;
  gene1: string;
  gene2: string;
  clinical_stage: 1 | 2 | 3 | 4 | 5;
  mechanism_type: MechanismType;
  indication_notes: string;
  needs_verification: boolean;
}

export interface CellMetadata {
  cellId: string;
  tissue: TissueType;
  cellType: CellType;
  isTumor: boolean;
  mitoPct: number;
  totalCounts: number;
  nGenes: number;
}

export interface RawSyntheticDataset {
  genes: GeneInfo[];
  geneIndex: Map<string, number>;
  cells: CellMetadata[];
  /** Row-major expression matrix: cells.length * genes.length */
  counts: Float64Array;
  seed: number;
  datasetId: string;
}

export interface PreprocessedDataset {
  genes: GeneInfo[];
  geneIndex: Map<string, number>;
  cells: CellMetadata[];
  /** Log1p normalized expression matrix: cells.length * genes.length */
  normExpr: Float64Array;
  hvgGenes: string[];
  pcaVarianceExplained: number[];
  qcStats: {
    rawCells: number;
    filteredCells: number;
    retainedCells: number;
    geneCount: number;
    totalCandidatePairs: number;
    tissueCounts: Record<TissueType, number>;
    cellTypeCounts: Record<CellType, number>;
  };
  seed: number;
  datasetId: string;
}

export interface PathwayDatabase {
  kegg: Record<string, string[]>;
  go: Record<string, string[]>;
}

export interface FeatureVectorResult {
  names: string[];
  /** Pair count * feature count row-major matrix */
  matrix: Float64Array;
}

export interface FeatureImportanceItem {
  feature: string;
  weight: number;
  group: 'safety' | 'mechanism' | 'embedding' | 'activity' | 'correlation';
}

export interface ModelBenchmarkRow {
  model: string;
  meanAuc: number;
  stdAuc: number;
  foldAucs: number[];
  paperRefAuc?: number;
  meanNdcg10?: number;
  meanNdcg20?: number;
  meanNdcg100?: number;
}

export interface RocCurveSeries {
  model: string;
  auc: number;
  fpr: number[];
  tpr: number[];
}

export interface AblationStep {
  step: string;
  featureCount: number;
  meanAuc: number;
  stdAuc: number;
  paperRefAuc: number;
}

export interface BinEdgeSpec {
  feature: string;
  edges: number[];
  labels: string[];
  glossaryTerm: string;
  plainDescription: string;
}

export interface CandidatePrediction {
  pairKey: string;
  gene1: string;
  gene2: string;
  score: number;
  ndcgScore: number;
  rank: number;
  totalPairs: number;
  mechanismType: MechanismType;
  inClinicalTrial: boolean;
  clinicalStage: number | null;
  clinicalDrugs: string[];
  needsVerification: boolean;
  features: Record<string, number>;
}

export interface ProvenanceStamp {
  dataset: string;
  seed: number;
  modelVersion: string;
  dataMode: 'DEMO_SYNTHETIC' | 'REAL_GEO';
  generationMode: 'gemini' | 'deterministic_template';
  geminiModel?: string;
  webSearchEnabled: boolean;
  generatedAt: string;
}

export interface RagReportResponse {
  pairKey: string;
  gene1: string;
  gene2: string;
  score: number;
  ndcgScore: number;
  rank: number;
  totalPairs: number;
  mechanismType: MechanismType;
  discretizedFeatures: Array<{
    feature: string;
    value: number;
    binIndex: number;
    binLabel: string;
    description: string;
  }>;
  plainEnglishReport: string;
  reportMarkdown: string;
  provenance: ProvenanceStamp;
}
