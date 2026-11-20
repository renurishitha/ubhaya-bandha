export type ActiveTab = 'home' | 'predict' | 'leaderboard' | 'model-card' | 'glossary';

export type MechanismType = 'bridge-two-cells' | 'TAA+TAA' | 'tumor-immunotherapy';

export interface GeneInfo {
  gene: string;
  category: string;
  cell_bias: string;
  description: string;
}

export type SecondaryStructure = 'H' | 'E' | 'C';

export interface ProteinResidue {
  resSeq: number;
  resName: string;
  chain: string;
  x: number;
  y: number;
  z: number;
  plddt: number;
  ss: SecondaryStructure;
  domain: string;
}

export interface ProteinDomainSpan {
  name: string;
  start: number;
  end: number;
  kind: 'signal' | 'ectodomain' | 'transmembrane' | 'cytoplasmic' | 'secreted';
  color: string;
}

export interface ResidueContactEdge {
  source: number;
  target: number;
  distance: number;
  seqDist: number;
  isSpatialContact: boolean;
}

export interface GnnGraphStats {
  nodeCount: number;
  edgeCount: number;
  spatialContactEdges: number;
  avgDegree: number;
  contactCutoffAngstrom: number;
  ectodomainAccessibleRatio: number;
}

export interface AlphaFoldStructurePayload {
  gene: string;
  uniprotId: string;
  entryId: string;
  organism: string;
  sequenceLength: number;
  meanPlddt: number;
  confidentFraction: number;
  veryHighFraction: number;
  sourceMode: 'ALPHAFOLD_DB_LIVE' | 'ALPHAFOLD_DB_PRIOR';
  pdbUrl: string;
  alphafoldUrl: string;
  foldClass: string;
  domains: ProteinDomainSpan[];
  residues: ProteinResidue[];
  gnn: {
    embeddingDim: number;
    embedding: number[];
    residueSalience: number[];
    graphStats: GnnGraphStats;
    contactEdges: ResidueContactEdge[];
  };
}

export interface PairGnnTargetSummary {
  gene: string;
  uniprotId: string;
  entryId: string;
  sourceMode: 'ALPHAFOLD_DB_LIVE' | 'ALPHAFOLD_DB_PRIOR';
  meanPlddt: number;
  confidentFraction: number;
  sequenceLength: number;
  foldClass: string;
  alphafoldUrl: string;
  graphStats: GnnGraphStats;
}

export type BindingVerdict = 'strong-bind' | 'conditional-bind' | 'blocked';

export interface BindingFactorCheck {
  id: 'pocket-exposure' | 'pocket-confidence' | 'dual-arm-fit' | 'steric-clearance';
  label: string;
  status: 'pass' | 'caution' | 'block';
  detail: string;
}

export interface TargetBindingSummary {
  gene: string;
  canBind: boolean;
  pocketDomain: string;
  pocketResidueRange: string;
  pocketMeanPlddt: number;
  pocketExposure: number;
  pocketEctoFraction: number;
  isMultiPass: boolean;
  barrierReason?: string;
}

export interface GnnBindingAssessment {
  verdict: BindingVerdict;
  canBind: boolean;
  fitPercent: number;
  headline: string;
  summary: string;
  whyNote: string;
  target1: TargetBindingSummary;
  target2: TargetBindingSummary;
  checks: BindingFactorCheck[];
}

export interface PairGnnMetrics {
  gnnCosineSim: number;
  gnnEuclideanDist: number;
  structuralComplementarity: number;
  ectodomainAccessibility: number;
  predictedInterfaceCompatibility: number;
  fusedEmbeddingDist: number;
  embeddingDim: number;
  embedding1: number[];
  embedding2: number[];
  target1: PairGnnTargetSummary;
  target2: PairGnnTargetSummary;
  bindingAssessment: GnnBindingAssessment;
}

export interface PredictionResult {
  pairKey: string;
  gene1: string;
  gene2: string;
  score: number;
  ndcgScore?: number;
  rank: number;
  totalPairs: number;
  mechanismType: MechanismType;
  inClinicalTrial: boolean;
  clinicalStage: number | null;
  clinicalDrugs: string[];
  needsVerification: boolean;
  features: Record<string, number>;
  gene1Info?: GeneInfo;
  gene2Info?: GeneInfo;
  structuralGnn?: PairGnnMetrics;
  dataMode?: 'DEMO_SYNTHETIC' | 'REAL_GEO';
  datasetId?: string;
  seed?: number;
  modelVersion?: string;
}

export interface DiscretizedFeatureItem {
  feature: string;
  value: number;
  binIndex: number;
  binLabel: string;
  description: string;
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

export interface RagReportResult {
  pairKey: string;
  gene1: string;
  gene2: string;
  score: number;
  ndcgScore?: number;
  rank: number;
  totalPairs: number;
  mechanismType: MechanismType;
  discretizedFeatures: DiscretizedFeatureItem[];
  plainEnglishReport?: string;
  reportMarkdown: string;
  provenance: ProvenanceStamp;
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

export interface FeatureImportanceItem {
  feature: string;
  weight: number;
  group: 'safety' | 'mechanism' | 'embedding' | 'activity' | 'correlation';
}

export interface BinEdgeSpec {
  feature: string;
  edges: number[];
  labels: string[];
  glossaryTerm: string;
  plainDescription: string;
}

export interface MetricsResponse {
  datasetId: string;
  dataMode: 'DEMO_SYNTHETIC' | 'REAL_GEO';
  seed: number;
  modelVersion: string;
  geminiModel: string;
  geneCount: number;
  totalCandidatePairs: number;
  qcStats: {
    rawCells: number;
    filteredCells: number;
    retainedCells: number;
    geneCount: number;
    totalCandidatePairs: number;
    tissueCounts: Record<string, number>;
    cellTypeCounts: Record<string, number>;
  };
  pcaVarianceExplained: number[];
  hvgCount: number;
  clinicalPairCount: number;
  benchmarks: ModelBenchmarkRow[];
  rocCurves: RocCurveSeries[];
  ablation: AblationStep[];
  topFeatureImportances: FeatureImportanceItem[];
  binSpecs: BinEdgeSpec[];
}
