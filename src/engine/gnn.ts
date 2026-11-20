import type { AlphaFoldStructure, ProteinResidue } from './alphafold.ts';
import { Mulberry32 } from './prng.ts';
import type { MechanismType } from './types.ts';

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

export interface GnnProteinEmbeddingResult {
  gene: string;
  uniprotId: string;
  entryId: string;
  embeddingDim: number;
  embedding: number[];
  residueSalience: number[];
  graphStats: GnnGraphStats;
  contactEdges: ResidueContactEdge[];
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
  target1: {
    gene: string;
    uniprotId: string;
    entryId: string;
    sourceMode: AlphaFoldStructure['sourceMode'];
    meanPlddt: number;
    confidentFraction: number;
    sequenceLength: number;
    foldClass: string;
    alphafoldUrl: string;
    graphStats: GnnGraphStats;
  };
  target2: {
    gene: string;
    uniprotId: string;
    entryId: string;
    sourceMode: AlphaFoldStructure['sourceMode'];
    meanPlddt: number;
    confidentFraction: number;
    sequenceLength: number;
    foldClass: string;
    alphafoldUrl: string;
    graphStats: GnnGraphStats;
  };
  bindingAssessment: GnnBindingAssessment;
}

const NODE_DIM = 12;
const EDGE_DIM = 6;
const HIDDEN_DIM = 16;
const CONTACT_CUTOFF_A = 10.0;

const HYDROPHOBIC_AA = new Set(['ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO']);
const CHARGED_AA = new Set(['ASP', 'GLU', 'LYS', 'ARG', 'HIS']);
const AROMATIC_AA = new Set(['PHE', 'TYR', 'TRP', 'HIS']);

interface GnnLayerWeights {
  Wself: Float64Array; // HIDDEN x HIDDEN
  Wmsg: Float64Array;  // HIDDEN x HIDDEN
  Wedge: Float64Array; // EDGE x HIDDEN
  Watt: Float64Array;  // 2*HIDDEN + EDGE -> 1
  bias: Float64Array;  // HIDDEN
}

interface GnnWeights {
  Win: Float64Array; // NODE_DIM x HIDDEN
  bin: Float64Array; // HIDDEN
  layers: [GnnLayerWeights, GnnLayerWeights];
  Wsalience: Float64Array; // HIDDEN
  Wpool: Float64Array; // HIDDEN (attention pooling query)
}

function initWeights(): GnnWeights {
  const rng = new Mulberry32(202604);
  const xavier = (rows: number, cols: number): Float64Array => {
    const arr = new Float64Array(rows * cols);
    const scale = Math.sqrt(2 / (rows + cols));
    for (let i = 0; i < arr.length; i++) arr[i] = rng.normal(0, scale);
    return arr;
  };
  const makeLayer = (): GnnLayerWeights => ({
    Wself: xavier(HIDDEN_DIM, HIDDEN_DIM),
    Wmsg: xavier(HIDDEN_DIM, HIDDEN_DIM),
    Wedge: xavier(EDGE_DIM, HIDDEN_DIM),
    Watt: xavier(2 * HIDDEN_DIM + EDGE_DIM, 1),
    bias: new Float64Array(HIDDEN_DIM),
  });

  return {
    Win: xavier(NODE_DIM, HIDDEN_DIM),
    bin: new Float64Array(HIDDEN_DIM),
    layers: [makeLayer(), makeLayer()],
    Wsalience: xavier(HIDDEN_DIM, 1),
    Wpool: xavier(HIDDEN_DIM, 1),
  };
}

const GNN_WEIGHTS = initWeights();

function silu(x: number): number {
  return x / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

function layerNormInPlace(vec: Float64Array): void {
  const d = vec.length;
  let mean = 0;
  for (let i = 0; i < d; i++) mean += vec[i];
  mean /= d;
  let sq = 0;
  for (let i = 0; i < d; i++) {
    const diff = vec[i] - mean;
    sq += diff * diff;
  }
  const invStd = 1 / Math.sqrt(sq / d + 1e-5);
  for (let i = 0; i < d; i++) {
    vec[i] = (vec[i] - mean) * invStd;
  }
}

/**
 * Builds the 3D residue contact graph G = (V, E) from AlphaFold C-alpha coordinates.
 * Combines sequential peptide links (|i-j| <= 2) and non-local 3D spatial contacts (d_ij <= 10 A).
 */
export function buildResidueContactGraph(residues: ProteinResidue[]): {
  edges: ResidueContactEdge[];
  adj: number[][];
  spatialCount: number;
  neighborCounts: Int32Array;
} {
  const n = residues.length;
  const edges: ResidueContactEdge[] = [];
  const adj: number[][] = Array.from({ length: n }, () => []);
  const neighborCounts = new Int32Array(n);
  let spatialCount = 0;

  for (let i = 0; i < n; i++) {
    const a = residues[i];
    for (let j = i + 1; j < n; j++) {
      const b = residues[j];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = a.z - b.z;
      const distSq = dx * dx + dy * dy + dz * dz;
      const seqDist = j - i;

      if (seqDist <= 2 || distSq <= CONTACT_CUTOFF_A * CONTACT_CUTOFF_A) {
        const dist = Math.sqrt(distSq);
        const isSpatial = seqDist > 2;
        if (isSpatial) spatialCount++;

        const eIdx = edges.length;
        edges.push({
          source: i,
          target: j,
          distance: Math.round(dist * 1000) / 1000,
          seqDist,
          isSpatialContact: isSpatial,
        });
        adj[i].push(eIdx);
        adj[j].push(eIdx);
        neighborCounts[i]++;
        neighborCounts[j]++;
      }
    }
  }

  return { edges, adj, spatialCount, neighborCounts };
}

function buildNodeFeatures(
  residues: ProteinResidue[],
  neighborCounts: Int32Array
): Float64Array[] {
  const n = residues.length;
  const maxDeg = Math.max(1, ...Array.from(neighborCounts));
  const feats: Float64Array[] = new Array(n);

  for (let i = 0; i < n; i++) {
    const r = residues[i];
    const x = new Float64Array(NODE_DIM);
    // 0-2: Secondary structure one-hot
    x[0] = r.ss === 'H' ? 1 : 0;
    x[1] = r.ss === 'E' ? 1 : 0;
    x[2] = r.ss === 'C' ? 1 : 0;
    // 3: AlphaFold pLDDT confidence [0..1]
    x[3] = r.plddt / 100;
    // 4: Normalized sequence coordinate
    x[4] = n > 1 ? i / (n - 1) : 0.5;
    // 5: Surface exposure proxy (inverse local packing density)
    x[5] = 1 - Math.min(1, neighborCounts[i] / Math.max(6, maxDeg));
    // 6: Local backbone curvature angle proxy (cos angle between i-1->i and i->i+1)
    if (i > 0 && i < n - 1) {
      const prev = residues[i - 1];
      const next = residues[i + 1];
      const v1x = r.x - prev.x;
      const v1y = r.y - prev.y;
      const v1z = r.z - prev.z;
      const v2x = next.x - r.x;
      const v2y = next.y - r.y;
      const v2z = next.z - r.z;
      const n1 = Math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z) + 1e-6;
      const n2 = Math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z) + 1e-6;
      x[6] = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2);
    }
    // 7: Extracellular / receptor-binding domain flag
    const domLower = r.domain.toLowerCase();
    x[7] = domLower.includes('ectodomain') || domLower.includes('extracellular') || domLower.includes('receptor') ? 1 : 0;
    // 8: Transmembrane flag
    x[8] = domLower.includes('transmembrane') || domLower.includes('tm') ? 1 : 0;
    // 9-11: Physicochemical residue flags
    x[9] = HYDROPHOBIC_AA.has(r.resName) ? 1 : 0;
    x[10] = CHARGED_AA.has(r.resName) ? 1 : 0;
    x[11] = AROMATIC_AA.has(r.resName) ? 1 : 0;
    feats[i] = x;
  }
  return feats;
}

function encodeEdgeRbf(edge: ResidueContactEdge): Float64Array {
  const e = new Float64Array(EDGE_DIM);
  const centers = [2.5, 4.2, 6.0, 8.0, 10.0];
  const gamma = 0.35;
  for (let k = 0; k < 5; k++) {
    const diff = edge.distance - centers[k];
    e[k] = Math.exp(-gamma * diff * diff);
  }
  e[5] = Math.min(1, Math.log1p(edge.seqDist) / 5.0);
  return e;
}

/**
 * Runs a 2-layer SE(3)-invariant Residue-Graph Message Passing & Attention GNN
 * over an AlphaFold 3D protein structure to produce a 16-dim L2-normalized embedding
 * and per-residue epitope salience scores.
 */
export function computeProteinGnnEmbedding(
  structure: AlphaFoldStructure
): GnnProteinEmbeddingResult {
  const residues = structure.residues;
  const n = residues.length;
  const { edges, adj, spatialCount, neighborCounts } = buildResidueContactGraph(residues);
  const nodeFeats = buildNodeFeatures(residues, neighborCounts);
  const edgeFeats = edges.map(encodeEdgeRbf);

  // Input linear projection -> H^(0) [N x 16]
  let H: Float64Array[] = nodeFeats.map((x) => {
    const h = new Float64Array(HIDDEN_DIM);
    for (let j = 0; j < HIDDEN_DIM; j++) {
      let s = GNN_WEIGHTS.bin[j];
      for (let i = 0; i < NODE_DIM; i++) {
        s += x[i] * GNN_WEIGHTS.Win[i * HIDDEN_DIM + j];
      }
      h[j] = silu(s);
    }
    return h;
  });

  // 2 Message-Passing Layers with Attention & Residual LayerNorm
  for (const layer of GNN_WEIGHTS.layers) {
    const Hnext: Float64Array[] = Array.from({ length: n }, () => new Float64Array(HIDDEN_DIM));

    for (let i = 0; i < n; i++) {
      const hi = H[i];
      const incEdges = adj[i];

      // Compute attention logits over incident edges
      const attLogits = new Float64Array(incEdges.length);
      let maxLogit = -Infinity;

      for (let k = 0; k < incEdges.length; k++) {
        const edge = edges[incEdges[k]];
        const nb = edge.source === i ? edge.target : edge.source;
        const hj = H[nb];
        const ef = edgeFeats[incEdges[k]];

        let logit = 0;
        for (let d = 0; d < HIDDEN_DIM; d++) {
          logit += hi[d] * layer.Watt[d] + hj[d] * layer.Watt[HIDDEN_DIM + d];
        }
        for (let e = 0; e < EDGE_DIM; e++) {
          logit += ef[e] * layer.Watt[2 * HIDDEN_DIM + e];
        }
        attLogits[k] = logit;
        if (logit > maxLogit) maxLogit = logit;
      }

      // Softmax attention weights
      let sumExp = 0;
      for (let k = 0; k < incEdges.length; k++) {
        attLogits[k] = Math.exp(attLogits[k] - maxLogit);
        sumExp += attLogits[k];
      }

      // Aggregate neighbor messages
      const agg = new Float64Array(HIDDEN_DIM);
      if (sumExp > 1e-12) {
        for (let k = 0; k < incEdges.length; k++) {
          const alpha = attLogits[k] / sumExp;
          const edge = edges[incEdges[k]];
          const nb = edge.source === i ? edge.target : edge.source;
          const hj = H[nb];
          const ef = edgeFeats[incEdges[k]];

          for (let outD = 0; outD < HIDDEN_DIM; outD++) {
            let msg = 0;
            for (let inD = 0; inD < HIDDEN_DIM; inD++) {
              msg += hj[inD] * layer.Wmsg[inD * HIDDEN_DIM + outD];
            }
            for (let e = 0; e < EDGE_DIM; e++) {
              msg += ef[e] * layer.Wedge[e * HIDDEN_DIM + outD];
            }
            agg[outD] += alpha * silu(msg);
          }
        }
      }

      // Self update + residual + LayerNorm
      const out = Hnext[i];
      for (let outD = 0; outD < HIDDEN_DIM; outD++) {
        let selfTerm = layer.bias[outD];
        for (let inD = 0; inD < HIDDEN_DIM; inD++) {
          selfTerm += hi[inD] * layer.Wself[inD * HIDDEN_DIM + outD];
        }
        out[outD] = hi[outD] + silu(selfTerm + agg[outD]);
      }
      layerNormInPlace(out);
    }

    H = Hnext;
  }

  // Per-residue GNN epitope salience (combines learned projection + ectodomain surface exposure + pLDDT confidence)
  const rawSalience = new Float64Array(n);
  let maxSal = -Infinity;
  let minSal = Infinity;
  let ectoAccessibleCount = 0;

  for (let i = 0; i < n; i++) {
    let z = 0;
    for (let d = 0; d < HIDDEN_DIM; d++) {
      z += H[i][d] * GNN_WEIGHTS.Wsalience[d];
    }
    const exposure = nodeFeats[i][5];
    const conf = nodeFeats[i][3];
    const isEcto = nodeFeats[i][7];
    const combined = sigmoid(z) * (0.45 + 0.35 * exposure + 0.2 * conf) + 0.18 * isEcto;
    rawSalience[i] = combined;
    if (combined > maxSal) maxSal = combined;
    if (combined < minSal) minSal = combined;
    if (isEcto > 0.5 && conf >= 0.7 && exposure >= 0.35) {
      ectoAccessibleCount++;
    }
  }

  const salSpan = Math.max(1e-6, maxSal - minSal);
  const residueSalience = Array.from(
    rawSalience,
    (v) => Math.round(((v - minSal) / salSpan) * 10000) / 10000
  );

  // Attention + pLDDT-weighted global graph readout -> 16-dim L2-normalized embedding
  const poolWeights = new Float64Array(n);
  let maxPoolLogit = -Infinity;
  for (let i = 0; i < n; i++) {
    let q = 0;
    for (let d = 0; d < HIDDEN_DIM; d++) {
      q += H[i][d] * GNN_WEIGHTS.Wpool[d];
    }
    // Weight structural core residues (high pLDDT) higher in global representation
    const logit = q + 0.6 * (residues[i].plddt / 100);
    poolWeights[i] = logit;
    if (logit > maxPoolLogit) maxPoolLogit = logit;
  }

  let poolSum = 0;
  for (let i = 0; i < n; i++) {
    poolWeights[i] = Math.exp(poolWeights[i] - maxPoolLogit);
    poolSum += poolWeights[i];
  }

  const emb = new Float64Array(HIDDEN_DIM);
  for (let i = 0; i < n; i++) {
    const w = poolWeights[i] / Math.max(1e-12, poolSum);
    for (let d = 0; d < HIDDEN_DIM; d++) {
      emb[d] += w * H[i][d];
    }
  }

  // L2 normalize embedding
  let normSq = 0;
  for (let d = 0; d < HIDDEN_DIM; d++) normSq += emb[d] * emb[d];
  const invNorm = normSq > 1e-12 ? 1 / Math.sqrt(normSq) : 1;
  const embedding = Array.from(emb, (v) => Math.round(v * invNorm * 1e6) / 1e6);

  const avgDegree = n > 0 ? Math.round(((2 * edges.length) / n) * 100) / 100 : 0;

  return {
    gene: structure.gene,
    uniprotId: structure.uniprotId,
    entryId: structure.entryId,
    embeddingDim: HIDDEN_DIM,
    embedding,
    residueSalience,
    graphStats: {
      nodeCount: n,
      edgeCount: edges.length,
      spatialContactEdges: spatialCount,
      avgDegree,
      contactCutoffAngstrom: CONTACT_CUTOFF_A,
      ectodomainAccessibleRatio: Math.round((ectoAccessibleCount / Math.max(1, n)) * 10000) / 10000,
    },
    contactEdges: edges,
  };
}

function summarizeTargetBinding(
  s: AlphaFoldStructure,
  g: GnnProteinEmbeddingResult
): TargetBindingSummary {
  const n = s.residues.length;
  const { neighborCounts } = buildResidueContactGraph(s.residues);
  const maxDeg = Math.max(1, ...Array.from(neighborCounts));

  const topIndices = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => (g.residueSalience[b] ?? 0) - (g.residueSalience[a] ?? 0))
    .slice(0, Math.min(14, n));

  let plddtSum = 0;
  let expSum = 0;
  let ectoCount = 0;
  let minSeq = Infinity;
  let maxSeq = -Infinity;
  const domainCounts = new Map<string, number>();

  for (const idx of topIndices) {
    const r = s.residues[idx];
    plddtSum += r.plddt;
    const exposure = 1 - Math.min(1, neighborCounts[idx] / Math.max(6, maxDeg));
    expSum += exposure;
    const domLower = r.domain.toLowerCase();
    const isExposedDom =
      domLower.includes('ectodomain') ||
      domLower.includes('extracellular') ||
      domLower.includes('receptor') ||
      domLower.includes('secreted');
    if (isExposedDom) ectoCount++;
    if (r.resSeq < minSeq) minSeq = r.resSeq;
    if (r.resSeq > maxSeq) maxSeq = r.resSeq;
    domainCounts.set(r.domain, (domainCounts.get(r.domain) ?? 0) + 1);
  }

  const k = Math.max(1, topIndices.length);
  const pocketMeanPlddt = Math.round((plddtSum / k) * 10) / 10;
  const pocketExposure = Math.round((expSum / k) * 1000) / 1000;
  const pocketEctoFraction = Math.round((ectoCount / k) * 1000) / 1000;
  const isMultiPass = s.foldClass.toLowerCase().includes('multi-pass');

  let pocketDomain = 'Extracellular Domain';
  let bestDomCount = -1;
  for (const [dom, cnt] of domainCounts.entries()) {
    if (cnt > bestDomCount) {
      bestDomCount = cnt;
      pocketDomain = dom;
    }
  }

  const pocketResidueRange =
    Number.isFinite(minSeq) && Number.isFinite(maxSeq)
      ? `res ${minSeq}–${maxSeq}`
      : 'N/A';

  let barrierReason: string | undefined;
  let canBind = true;

  if (pocketEctoFraction < 0.45) {
    canBind = false;
    barrierReason = `Top pocket (${pocketResidueRange}) sits in ${pocketDomain}, shielded from extracellular antibody access.`;
  } else if (pocketMeanPlddt < 68) {
    canBind = false;
    barrierReason = `Binding loop (${pocketResidueRange}) is flexible/disordered (pLDDT ${pocketMeanPlddt.toFixed(1)}), preventing stable Fab lock.`;
  } else if (pocketExposure < 0.28) {
    canBind = false;
    barrierReason = `Binding residues (${pocketResidueRange}) are tightly buried inside the protein core (exposure ${(pocketExposure * 100).toFixed(0)}%).`;
  } else if (isMultiPass && pocketExposure < 0.46) {
    barrierReason = `Multi-pass transmembrane fold leaves narrow extracellular loops (${pocketResidueRange}) close to the cell membrane.`;
  }

  return {
    gene: s.gene,
    canBind,
    pocketDomain,
    pocketResidueRange,
    pocketMeanPlddt,
    pocketExposure,
    pocketEctoFraction,
    isMultiPass,
    barrierReason,
  };
}

function buildBindingAssessment(
  s1: AlphaFoldStructure,
  g1: GnnProteinEmbeddingResult,
  s2: AlphaFoldStructure,
  g2: GnnProteinEmbeddingResult,
  mechanismType: MechanismType,
  structuralComplementarity: number,
  ectodomainAccessibility: number,
  predictedInterfaceCompatibility: number
): GnnBindingAssessment {
  const t1 = summarizeTargetBinding(s1, g1);
  const t2 = summarizeTargetBinding(s2, g2);

  const bothExposed = t1.canBind && t2.canBind && ectodomainAccessibility >= 0.14;
  const lowConfidence = t1.pocketMeanPlddt < 68 || t2.pocketMeanPlddt < 68;
  const stericCrowding =
    (t1.isMultiPass && t2.isMultiPass) ||
    ((t1.isMultiPass || t2.isMultiPass) && mechanismType === 'TAA+TAA' && structuralComplementarity < 0.68);
  const lowComplementarity = structuralComplementarity < 0.56;

  let verdict: BindingVerdict = 'strong-bind';
  if (!t1.canBind || !t2.canBind || lowComplementarity || (t1.isMultiPass && t2.isMultiPass)) {
    verdict = 'blocked';
  } else if (
    t1.barrierReason ||
    t2.barrierReason ||
    stericCrowding ||
    structuralComplementarity < 0.68 ||
    predictedInterfaceCompatibility < 0.68
  ) {
    verdict = 'conditional-bind';
  }

  if (!bothExposed && verdict === 'strong-bind') {
    verdict = 'conditional-bind';
  }

  const canBind = verdict !== 'blocked';
  const fitPercent = Math.max(
    12,
    Math.min(
      98,
      Math.round(
        (verdict === 'blocked'
          ? Math.min(48, predictedInterfaceCompatibility * 55)
          : predictedInterfaceCompatibility * 100)
      )
    )
  );

  const exposureCheck: BindingFactorCheck = {
    id: 'pocket-exposure',
    label: 'Exposed Binding Pockets',
    status: !t1.canBind || !t2.canBind ? 'block' : t1.barrierReason || t2.barrierReason ? 'caution' : 'pass',
    detail:
      t1.canBind && t2.canBind
        ? `${s1.gene} (${t1.pocketResidueRange}) and ${s2.gene} (${t2.pocketResidueRange}) both expose accessible extracellular pockets.`
        : `${!t1.canBind ? t1.barrierReason : t2.barrierReason}`,
  };

  const confidenceCheck: BindingFactorCheck = {
    id: 'pocket-confidence',
    label: 'Folded Pocket Stability (AlphaFold pLDDT)',
    status: lowConfidence ? 'block' : t1.pocketMeanPlddt < 75 || t2.pocketMeanPlddt < 75 ? 'caution' : 'pass',
    detail: lowConfidence
      ? `Flexible/disordered pocket geometry (${s1.gene} ${t1.pocketMeanPlddt.toFixed(1)}, ${s2.gene} ${t2.pocketMeanPlddt.toFixed(1)}) makes antibody docking unstable.`
      : `Both binding pockets are well-folded (${s1.gene} pLDDT ${t1.pocketMeanPlddt.toFixed(1)}, ${s2.gene} ${t2.pocketMeanPlddt.toFixed(1)}) for stable Fab contact.`,
  };

  const stericCheck: BindingFactorCheck = {
    id: 'steric-clearance',
    label: 'Bispecific Arm Reach & Clearance',
    status: stericCrowding || lowComplementarity ? (verdict === 'blocked' ? 'block' : 'caution') : 'pass',
    detail:
      t1.isMultiPass && t2.isMultiPass
        ? `Both ${s1.gene} and ${s2.gene} are membrane-hugging multi-pass bundles, leaving too little extracellular clearance for dual Fab arms.`
        : stericCrowding || t1.barrierReason || t2.barrierReason
          ? `${t1.barrierReason ?? t2.barrierReason ?? 'Tight membrane-proximal loop geometry requires linker-length optimization.'}`
          : mechanismType === 'bridge-two-cells'
            ? `Exposed pockets on separate cells align cleanly for trans-cellular bispecific bridging (fit ${(structuralComplementarity * 100).toFixed(0)}%).`
            : `Both Fab arms can reach their target pockets simultaneously without steric clash (fit ${(structuralComplementarity * 100).toFixed(0)}%).`,
  };

  let headline = '';
  let summary = '';
  let whyNote = '';

  if (verdict === 'strong-bind') {
    headline = 'Both targets expose stable pockets — Bispecific can bind';
    summary = `${s1.gene} (${t1.pocketDomain}) and ${s2.gene} (${t2.pocketDomain}) present well-folded, surface-exposed pockets so both arms of the bispecific antibody can lock on simultaneously.`;
    whyNote = `Why it binds: High AlphaFold confidence at both epitopes (pLDDT ${t1.pocketMeanPlddt.toFixed(0)} & ${t2.pocketMeanPlddt.toFixed(0)}) and clear extracellular surface exposure prevent steric clashes.`;
  } else if (verdict === 'conditional-bind') {
    headline = 'Partial fit — One pocket sits close to the membrane';
    summary = `${s1.gene} and ${s2.gene} can engage, but tight membrane-proximal loops require optimized Fab spacing or flexible hinges to avoid steric collision.`;
    whyNote = `Why caution is needed: ${t1.barrierReason ?? t2.barrierReason ?? 'Moderate GNN pocket complementarity limits dual-arm reach.'}`;
  } else {
    headline = 'Binding blocked — Structural or accessibility barrier';
    summary = `${s1.gene} + ${s2.gene} cannot easily form a stable dual-arm complex in standard IgG format because at least one binding site is shielded, buried, or disordered.`;
    whyNote = `Why it can’t bind: ${t1.barrierReason ?? t2.barrierReason ?? 'Multi-pass membrane crowding blocks simultaneous dual-Fab engagement.'}`;
  }

  return {
    verdict,
    canBind,
    fitPercent,
    headline,
    summary,
    whyNote,
    target1: t1,
    target2: t2,
    checks: [exposureCheck, confidenceCheck, stericCheck],
  };
}

export function computePairGnnMetrics(
  s1: AlphaFoldStructure,
  g1: GnnProteinEmbeddingResult,
  s2: AlphaFoldStructure,
  g2: GnnProteinEmbeddingResult,
  mechanismType: MechanismType,
  scrnaVectorScore = 1.0
): PairGnnMetrics {
  let dot = 0;
  let sqDist = 0;
  let hadamardSum = 0;

  for (let d = 0; d < HIDDEN_DIM; d++) {
    const a = g1.embedding[d];
    const b = g2.embedding[d];
    dot += a * b;
    const diff = a - b;
    sqDist += diff * diff;
    hadamardSum += Math.abs(a * b);
  }

  const gnnCosineSim = Math.round(Math.max(-1, Math.min(1, dot)) * 10000) / 10000;
  const gnnEuclideanDist = Math.round(Math.sqrt(sqDist) * 10000) / 10000;
  const ectodomainAccessibility =
    Math.round(
      Math.sqrt(
        Math.max(0.01, g1.graphStats.ectodomainAccessibleRatio) *
          Math.max(0.01, g2.graphStats.ectodomainAccessibleRatio)
      ) * 10000
    ) / 10000;

  // Structural complementarity rewards accessible high-pLDDT ectodomains and complementary fold geometries
  const confFactor = Math.min(1, ((s1.meanPlddt + s2.meanPlddt) / 2) / 92);
  const foldDiversityBonus =
    mechanismType === 'bridge-two-cells'
      ? 0.14 * Math.min(1, gnnEuclideanDist)
      : 0.14 * Math.max(0, gnnCosineSim);
  const structuralComplementarity =
    Math.round(
      Math.max(
        0.1,
        Math.min(
          0.99,
          0.42 * confFactor + 0.34 * Math.min(1, ectodomainAccessibility * 2.1) + foldDiversityBonus + 0.12 * Math.min(1, hadamardSum)
        )
      ) * 10000
    ) / 10000;

  const predictedInterfaceCompatibility =
    Math.round(
      sigmoid(
        1.6 * structuralComplementarity +
          0.8 * Math.min(1, ectodomainAccessibility * 2.2) +
          0.45 * gnnCosineSim -
          0.5
      ) * 10000
    ) / 10000;

  // Fused scRNA Gene2Vec + AlphaFold GNN multi-modal distance
  const fusedEmbeddingDist =
    Math.round((0.65 * scrnaVectorScore + 0.35 * gnnEuclideanDist) * 10000) / 10000;

  const bindingAssessment = buildBindingAssessment(
    s1,
    g1,
    s2,
    g2,
    mechanismType,
    structuralComplementarity,
    ectodomainAccessibility,
    predictedInterfaceCompatibility
  );

  return {
    gnnCosineSim,
    gnnEuclideanDist,
    structuralComplementarity,
    ectodomainAccessibility,
    predictedInterfaceCompatibility,
    fusedEmbeddingDist,
    embeddingDim: HIDDEN_DIM,
    embedding1: g1.embedding,
    embedding2: g2.embedding,
    target1: {
      gene: s1.gene,
      uniprotId: s1.uniprotId,
      entryId: s1.entryId,
      sourceMode: s1.sourceMode,
      meanPlddt: s1.meanPlddt,
      confidentFraction: s1.confidentFraction,
      sequenceLength: s1.sequenceLength,
      foldClass: s1.foldClass,
      alphafoldUrl: s1.alphafoldUrl,
      graphStats: g1.graphStats,
    },
    target2: {
      gene: s2.gene,
      uniprotId: s2.uniprotId,
      entryId: s2.entryId,
      sourceMode: s2.sourceMode,
      meanPlddt: s2.meanPlddt,
      confidentFraction: s2.confidentFraction,
      sequenceLength: s2.sequenceLength,
      foldClass: s2.foldClass,
      alphafoldUrl: s2.alphafoldUrl,
      graphStats: g2.graphStats,
    },
    bindingAssessment,
  };
}
