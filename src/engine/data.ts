import fs from 'node:fs';
import path from 'node:path';
import { Mulberry32 } from './prng.ts';
import type {
  CellMetadata,
  CellType,
  ClinicalPair,
  GeneInfo,
  MechanismType,
  PathwayDatabase,
  RawSyntheticDataset,
  TissueType,
} from './types.ts';

export function resolveDataPath(...parts: string[]): string {
  return path.resolve(process.cwd(), 'data', ...parts);
}

export function loadCandidateGenes(
  csvPath = resolveDataPath('candidate_genes.csv')
): GeneInfo[] {
  const raw = fs.readFileSync(csvPath, 'utf8').trim();
  const lines = raw.split(/\r?\n/).slice(1);
  return lines
    .filter((line) => line.trim().length > 0)
    .map((line) => {
      const firstComma = line.indexOf(',');
      const secondComma = line.indexOf(',', firstComma + 1);
      const thirdComma = line.indexOf(',', secondComma + 1);
      const gene = line.slice(0, firstComma).trim();
      const category = line.slice(firstComma + 1, secondComma).trim();
      const cell_bias = line.slice(secondComma + 1, thirdComma).trim() as CellType;
      const description = line.slice(thirdComma + 1).trim();
      return { gene, category, cell_bias, description };
    });
}

export function computeCandidatePairCount(geneCount: number): number {
  if (geneCount < 2) return 0;
  return (geneCount * (geneCount - 1)) / 2;
}

export function canonicalPairKey(g1: string, g2: string): string {
  return g1 < g2 ? `${g1}__${g2}` : `${g2}__${g1}`;
}

export function inferMechanismType(
  g1: GeneInfo,
  g2: GeneInfo
): MechanismType {
  const c1 = g1.category;
  const c2 = g2.category;
  const isTaa = (c: string) => c === 'taa' || c === 'angiogenesis_stroma';
  const isEngager = (c: string) => c === 't_cell_engager';
  const isImm = (c: string) =>
    c === 'immune_checkpoint' ||
    c === 'myeloid_nk_checkpoint' ||
    c === 'cytokine_chemokine_receptor';

  if (isTaa(c1) && isTaa(c2)) return 'TAA+TAA';
  if ((isEngager(c1) && !isEngager(c2)) || (isEngager(c2) && !isEngager(c1))) {
    return 'bridge-two-cells';
  }
  if ((isImm(c1) && isTaa(c2)) || (isImm(c2) && isTaa(c1))) {
    return 'tumor-immunotherapy';
  }
  if (g1.cell_bias !== g2.cell_bias && (c1 === 'myeloid_nk_checkpoint' || c2 === 'myeloid_nk_checkpoint')) {
    return 'bridge-two-cells';
  }
  return 'tumor-immunotherapy';
}

export function loadClinicalBispecifics(
  csvPath = resolveDataPath('clinical_bispecifics.csv')
): ClinicalPair[] {
  const raw = fs.readFileSync(csvPath, 'utf8').trim();
  const lines = raw.split(/\r?\n/).slice(1);
  return lines
    .filter((line) => line.trim().length > 0)
    .map((line) => {
      const parts = line.split(',');
      const drug = parts[0].trim();
      const gene1 = parts[1].trim();
      const gene2 = parts[2].trim();
      const clinical_stage = Number(parts[3].trim()) as 1 | 2 | 3 | 4 | 5;
      const mechanism_type = parts[4].trim() as MechanismType;
      const needs_verification = parts[parts.length - 1].trim().toLowerCase() === 'true';
      const indication_notes = parts.slice(5, parts.length - 1).join(',').trim();
      return {
        drug,
        gene1,
        gene2,
        clinical_stage,
        mechanism_type,
        indication_notes,
        needs_verification,
      };
    });
}

export function loadPathwayDatabase(
  keggPath = resolveDataPath('pathways', 'kegg_pathways.json'),
  goPath = resolveDataPath('pathways', 'go_bp_terms.json')
): PathwayDatabase {
  const kegg = JSON.parse(fs.readFileSync(keggPath, 'utf8')) as Record<string, string[]>;
  const go = JSON.parse(fs.readFileSync(goPath, 'utf8')) as Record<string, string[]>;
  return { kegg, go };
}

/**
 * Generates a realistic synthetic scRNA-seq matrix modeled after NSCLC GSE131907
 * with four tissue compartments and four key cell populations.
 * Includes ~5% intentional low-quality cells so QC filtering exercises real thresholds.
 */
export function generateSyntheticScrnaDataset(
  genes: GeneInfo[],
  seed = 42,
  nCells = 720
): RawSyntheticDataset {
  const rng = new Mulberry32(seed);
  const gCount = genes.length;
  const geneIndex = new Map<string, number>();
  genes.forEach((g, idx) => geneIndex.set(g.gene, idx));

  const tissues: TissueType[] = ['tumor_lung', 'normal_lung', 'lymph_node', 'brain_mets'];
  const tissueWeights = [0.46, 0.24, 0.16, 0.14];
  const cellTypes: CellType[] = [
    'exhausted_cd8_t',
    'treg',
    'macrophage_myeloid',
    'epithelial',
  ];

  // Latent biological modules so gene co-expression, embeddings, and correlations reflect real biology
  const nModules = 8;
  const geneLoadings = new Float64Array(gCount * nModules);
  for (let g = 0; g < gCount; g++) {
    const info = genes[g];
    const baseMod =
      info.cell_bias === 'exhausted_cd8_t'
        ? 0
        : info.cell_bias === 'treg'
          ? 1
          : info.cell_bias === 'macrophage_myeloid'
            ? 2
            : 3;
    geneLoadings[g * nModules + baseMod] = 1.65;
    if (info.category === 'immune_checkpoint') geneLoadings[g * nModules + 4] = 1.25;
    if (info.category === 't_cell_engager') geneLoadings[g * nModules + 5] = 1.35;
    if (info.category === 'taa') geneLoadings[g * nModules + 6] = 1.25;
    if (info.category === 'angiogenesis_stroma') geneLoadings[g * nModules + 7] = 1.15;
    for (let m = 0; m < nModules; m++) {
      geneLoadings[g * nModules + m] += rng.normal(0, 0.18);
    }
  }

  // Canonical target enrichments matching known NSCLC / TME patterns (both clinical and non-clinical TAAs/checkpoints)
  const tumorHighGenes = new Set([
    'CD274', 'CTLA4', 'PDCD1', 'LAG3', 'TIGIT', 'HAVCR2', 'EGFR', 'MET', 'ERBB2',
    'ERBB3', 'VEGFA', 'MS4A1', 'CD3E', 'CD3D', 'CD3G', 'TNFRSF17', 'GPRC5D',
    'CD19', 'DLL3', 'EPCAM', 'CEACAM5', 'MSLN', 'CLDN18', 'TACSTD2', 'CD47',
    'TNFRSF9', 'TGFBR2', 'LGR5', 'FOLR1', 'GPC3', 'CD276', 'FAP', 'MUC16',
    'CA9', 'CA12', 'MUC1', 'MUC4', 'MUC5AC', 'CLDN4', 'CLDN6', 'CDH1', 'CDH3',
    'CDH17', 'ITGAV', 'ITGB1', 'ITGB6', 'FGFR1', 'FGFR2', 'FGFR3', 'IGF1R',
    'ALK', 'RET', 'ROS1', 'EPHA2', 'EPHB4', 'PTK7', 'TPBG', 'SLC34A2', 'SLC7A5',
    'TFRC', 'MMP14', 'ADAM17', 'CD24', 'HLA-E', 'HLA-G', 'CXCR4', 'CCR4', 'CCR8',
    'ICOS', 'TNFRSF4', 'TNFRSF18', 'VSIR', 'BTLA', 'VTCN1', 'PVR', 'NECTIN2',
    'CD27', 'CD80', 'CD86', 'CD40', 'KDR', 'FLT1', 'ANGPT2', 'NRP1', 'CSF1R',
    'SIRPA', 'LILRB1', 'LILRB2', 'SIGLEC15', 'TREM2', 'CD163', 'CD44', 'BSG',
    'ALCAM', 'MCAM', 'SLC2A1', 'PROM1', 'GPNMB', 'ROBO1', 'LY6E', 'FOLH1',
  ]);

  const counts = new Float64Array(nCells * gCount);
  const cells: CellMetadata[] = [];

  for (let c = 0; c < nCells; c++) {
    const rTissue = rng.random();
    let tissue: TissueType = 'tumor_lung';
    let acc = 0;
    for (let t = 0; t < tissues.length; t++) {
      acc += tissueWeights[t];
      if (rTissue <= acc) {
        tissue = tissues[t];
        break;
      }
    }
    const isTumor = tissue !== 'normal_lung';

    // Cell-type composition varies by tissue compartment
    const rCt = rng.random();
    let cellType: CellType;
    if (tissue === 'normal_lung') {
      cellType = rCt < 0.12 ? 'exhausted_cd8_t' : rCt < 0.22 ? 'treg' : rCt < 0.52 ? 'macrophage_myeloid' : 'epithelial';
    } else if (tissue === 'lymph_node') {
      cellType = rCt < 0.44 ? 'exhausted_cd8_t' : rCt < 0.68 ? 'treg' : rCt < 0.90 ? 'macrophage_myeloid' : 'epithelial';
    } else {
      cellType = rCt < 0.28 ? 'exhausted_cd8_t' : rCt < 0.44 ? 'treg' : rCt < 0.68 ? 'macrophage_myeloid' : 'epithelial';
    }

    const isLowQuality = c % 20 === 19; // 5% low-quality cells caught by QC
    const cellLatent = new Float64Array(nModules);
    const activeCellMod = cellTypes.indexOf(cellType);
    for (let m = 0; m < nModules; m++) {
      cellLatent[m] = rng.normal(m === activeCellMod ? 1.5 : 0.1, 0.45);
    }
    if (isTumor) {
      cellLatent[4] += 0.95;
      cellLatent[6] += 0.85;
      cellLatent[7] += 0.55;
    }

    let totalCounts = 0;
    let nGenes = 0;
    const libScale = isLowQuality ? 0.12 : 0.85 + rng.random() * 0.6;

    for (let g = 0; g < gCount; g++) {
      const info = genes[g];
      let signal = 0;
      for (let m = 0; m < nModules; m++) {
        signal += geneLoadings[g * nModules + m] * cellLatent[m];
      }
      const isMatchedCell = info.cell_bias === cellType;
      const tumorBoost = isTumor
        ? tumorHighGenes.has(info.gene)
          ? 1.55
          : 0.72
        : tumorHighGenes.has(info.gene)
          ? -0.45
          : 0.2;

      const mu = Math.max(
        0.04,
        (isMatchedCell ? 2.6 : 0.35) + 0.55 * signal + tumorBoost
      );
      const dropoutProb = isLowQuality ? 0.88 : isMatchedCell ? 0.18 : 0.62;
      let val = 0;
      if (rng.random() > dropoutProb) {
        const raw = Math.max(0, (mu + rng.normal(0, 0.75)) * libScale * 4.2);
        val = Math.round(raw * 100) / 100;
      }
      counts[c * gCount + g] = val;
      if (val > 0) {
        totalCounts += val;
        nGenes++;
      }
    }

    const mitoPct = isLowQuality ? 26 + rng.random() * 14 : 2.5 + rng.random() * 8.5;
    cells.push({
      cellId: `SYN_CELL_${String(c + 1).padStart(4, '0')}`,
      tissue,
      cellType,
      isTumor,
      mitoPct: Math.round(mitoPct * 100) / 100,
      totalCounts: Math.round(totalCounts * 100) / 100,
      nGenes,
    });
  }

  return {
    genes,
    geneIndex,
    cells,
    counts,
    seed,
    datasetId: 'GSE131907 (NSCLC scRNA-seq)',
  };
}
