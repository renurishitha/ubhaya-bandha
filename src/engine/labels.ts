import { canonicalPairKey, inferMechanismType } from './data.ts';
import { Mulberry32, seededShuffle } from './prng.ts';
import type { ClinicalPair, GeneInfo, MechanismType } from './types.ts';

export interface PairLabelRecord {
  pairKey: string;
  bestStage: number; // 1..5 for clinical, 6 for unlabeled
  relevance: number; // 6 - bestStage (5..1 for clinical, 0 for unlabeled)
  binaryPositive: 0 | 1; // 1 if stage <= 3, else 0
  drugs: string[];
  needsVerification: boolean;
  mechanismType: MechanismType;
}

export interface TrainingDatasetIndices {
  pairRowIndices: number[];
  relevances: number[];
  binaryLabels: Array<0 | 1>;
}

/**
 * Aggregates clinical bispecifics by canonical gene pair, taking the most advanced
 * (lowest) clinical stage reached by any drug on that pair.
 */
export function buildClinicalLabelMap(
  clinicalPairs: ClinicalPair[],
  genes: GeneInfo[]
): Map<string, PairLabelRecord> {
  const geneMap = new Map<string, GeneInfo>();
  for (const g of genes) geneMap.set(g.gene, g);

  const map = new Map<string, PairLabelRecord>();
  for (const row of clinicalPairs) {
    const g1 = geneMap.get(row.gene1);
    const g2 = geneMap.get(row.gene2);
    if (!g1 || !g2 || row.gene1 === row.gene2) continue;

    const key = canonicalPairKey(row.gene1, row.gene2);
    const existing = map.get(key);
    if (!existing) {
      map.set(key, {
        pairKey: key,
        bestStage: row.clinical_stage,
        relevance: 6 - row.clinical_stage,
        binaryPositive: row.clinical_stage <= 3 ? 1 : 0,
        drugs: [row.drug],
        needsVerification: row.needs_verification,
        mechanismType: row.mechanism_type || inferMechanismType(g1, g2),
      });
    } else {
      if (row.clinical_stage < existing.bestStage) {
        existing.bestStage = row.clinical_stage;
        existing.relevance = 6 - row.clinical_stage;
        existing.binaryPositive = row.clinical_stage <= 3 ? 1 : 0;
      }
      if (!existing.drugs.includes(row.drug)) {
        existing.drugs.push(row.drug);
      }
      existing.needsVerification = existing.needsVerification || row.needs_verification;
    }
  }
  return map;
}

/**
 * Selects all labeled clinical pairs plus a deterministic stratified sample
 * of unlabeled background pairs (relevance=0, binary=0) for training & 5-fold CV.
 */
export function buildTrainingPairSubset(
  pairIndicesA: Uint16Array,
  pairIndicesB: Uint16Array,
  genes: GeneInfo[],
  labelMap: Map<string, PairLabelRecord>,
  seed = 42,
  unlabeledSampleCount = 160
): TrainingDatasetIndices {
  const rng = new Mulberry32(seed + 777);
  const labeledRows: number[] = [];
  const unlabeledRows: number[] = [];

  for (let r = 0; r < pairIndicesA.length; r++) {
    const g1 = genes[pairIndicesA[r]].gene;
    const g2 = genes[pairIndicesB[r]].gene;
    const key = canonicalPairKey(g1, g2);
    if (labelMap.has(key)) {
      labeledRows.push(r);
    } else {
      unlabeledRows.push(r);
    }
  }

  const sampledUnlabeled = seededShuffle(unlabeledRows, rng).slice(
    0,
    Math.min(unlabeledSampleCount, unlabeledRows.length)
  );

  const pairRowIndices = [...labeledRows, ...sampledUnlabeled];
  const relevances: number[] = [];
  const binaryLabels: Array<0 | 1> = [];

  for (const r of pairRowIndices) {
    const g1 = genes[pairIndicesA[r]].gene;
    const g2 = genes[pairIndicesB[r]].gene;
    const rec = labelMap.get(canonicalPairKey(g1, g2));
    if (rec) {
      relevances.push(rec.relevance);
      binaryLabels.push(rec.binaryPositive);
    } else {
      relevances.push(0);
      binaryLabels.push(0);
    }
  }

  return { pairRowIndices, relevances, binaryLabels };
}
