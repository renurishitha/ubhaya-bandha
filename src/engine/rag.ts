import { GoogleGenAI } from '@google/genai';
import { applyDiscretizer } from './discretize.ts';
import type { PairGnnMetrics } from './gnn.ts';
import type {
  BinEdgeSpec,
  CandidatePrediction,
  GeneInfo,
  ProvenanceStamp,
  RagReportResponse,
} from './types.ts';

export const VALID_GEMINI_MODELS = [
  'gemini-4-hydrogen',
  'gemini-4-argon',
  'gemini-4-carbon',
  'gemini-3.1-flash-lite',
] as const;

export type ValidGeminiModel = (typeof VALID_GEMINI_MODELS)[number];

const DEPRECATED_GEMINI_MODELS = new Set([
  'gemini-1.5-flash',
  'gemini-1.5-pro',
  'gemini-pro',
  'gemini-2.0-flash',
  'gemini-2.0-pro',
  'gemini-2.0-flash-thinking',
]);

const CLINICAL_STAGE_LABELS: Record<number, string> = {
  1: 'Stage 1 · Approved / BLA',
  2: 'Stage 2 · Phase 3 / 2–3',
  3: 'Stage 3 · Phase 1 / 2',
  4: 'Stage 4 · Preclinical / IND',
  5: 'Stage 5 · Discontinued',
};

const FEATURE_DEFINITIONS: Record<string, string> = {
  score_safe:
    'Harmonic mean of tumor-minus-normal expression differentials (dx, dy); flags healthy-tissue off-tumor risk.',
  double_ratio_max:
    'Peak fraction of cells within key tumor microenvironment compartments co-expressing both targets.',
  single_ratio_max:
    'Peak single-target positive cell fraction across evaluated compartments.',
  pearson_overall:
    'Correlation of expression across cells/compartments.',
  vector_score:
    'Structural/embedding distance between target representations.',
  same_pathway:
    'Shared pathway or ontology co-occurrence across KEGG/GO annotations.',
};

const RAG_SYSTEM_INSTRUCTION = [
  'You are a computational biology research assistant evaluating candidate bispecific antibody target pairs for BSPAI / Ubhaya-Bandha.',
  'Rely strictly on the provided ML viability score (bounded in [0, 1]), rank, 5-tier clinical progression stage (Stage 1-5 or Stage 0 Novel Discovery Candidate), discretized feature tags, feature definitions, 3D-GNN metrics, and optional retrieved search context.',
  'Never invent clinical trial numbers, NCT IDs, or unsupported claims.',
  'Write clearly for computational biologists and translational researchers.',
].join(' ');

function formatStageSummary(pred: CandidatePrediction): {
  stageTitle: string;
  stageDetail: string;
} {
  if (pred.inClinicalTrial && pred.clinicalStage) {
    const stageTitle =
      CLINICAL_STAGE_LABELS[pred.clinicalStage] ?? `Stage ${pred.clinicalStage}`;
    const drugs =
      pred.clinicalDrugs?.length > 0
        ? pred.clinicalDrugs.join(', ')
        : 'Clinical bispecific benchmark record';
    return {
      stageTitle,
      stageDetail: `${drugs}${pred.needsVerification ? ' (flagged for verification)' : ''}`,
    };
  }
  return {
    stageTitle: 'Stage 0 · Novel Discovery Candidate',
    stageDetail: 'Unbenchmarked pair scored via 5-stage pairwise hierarchy',
  };
}

export function resolveAndValidateGeminiModel(
  rawModel = process.env.GEMINI_MODEL
): ValidGeminiModel {
  const candidate = (rawModel ?? 'gemini-4-hydrogen').trim() || 'gemini-4-hydrogen';
  if (DEPRECATED_GEMINI_MODELS.has(candidate)) {
    throw new Error(
      `GEMINI_MODEL "${candidate}" is deprecated and unsupported. Valid models: ${VALID_GEMINI_MODELS.join(', ')}.`
    );
  }
  if (!(VALID_GEMINI_MODELS as readonly string[]).includes(candidate)) {
    throw new Error(
      `GEMINI_MODEL "${candidate}" is invalid. Valid models: ${VALID_GEMINI_MODELS.join(', ')}.`
    );
  }
  return candidate as ValidGeminiModel;
}

function hasUsableApiKey(): boolean {
  const key = (process.env.GEMINI_API_KEY ?? '').trim();
  return key.length > 0 && key !== 'MY_GEMINI_API_KEY';
}

const reportCache = new Map<string, RagReportResponse>();
const modelCooldownUntil = new Map<string, number>();

function isQuotaOrRateLimitError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return (
    msg.includes('429') ||
    msg.includes('RESOURCE_EXHAUSTED') ||
    msg.toLowerCase().includes('quota') ||
    msg.toLowerCase().includes('rate limit')
  );
}

function formatProvenanceHeader(p: ProvenanceStamp): string {
  return [
    `> **Research Aid Notice**: Automated BSPAI / Ubhaya-Bandha target-pair synthesis. Not clinical or medical advice.`,
    `> **Provenance**: Dataset \`${p.dataset}\` | Seed \`${p.seed}\` | Model \`${p.modelVersion}\` | Mode \`${p.generationMode}\` | Generated \`${p.generatedAt}\``,
  ].join('\n');
}

export interface RagReportInput {
  prediction: CandidatePrediction;
  gene1Info: GeneInfo;
  gene2Info: GeneInfo;
  binSpecs: BinEdgeSpec[];
  datasetId: string;
  seed: number;
  modelVersion: string;
  dataMode: 'DEMO_SYNTHETIC' | 'REAL_GEO';
  enableWebSearch?: boolean;
  structuralGnn?: PairGnnMetrics;
}

export async function generateGroundedRagReport(
  input: RagReportInput
): Promise<RagReportResponse> {
  const {
    prediction,
    gene1Info,
    gene2Info,
    binSpecs,
    datasetId,
    seed,
    modelVersion,
    dataMode,
    enableWebSearch = process.env.BSPAI_ENABLE_WEB_SEARCH === 'true',
    structuralGnn,
  } = input;

  const discretized = applyDiscretizer(binSpecs, prediction.features);
  const percentile = Math.max(
    0.01,
    (prediction.rank / Math.max(1, prediction.totalPairs)) * 100
  ).toFixed(2);
  const validatedModel = resolveAndValidateGeminiModel();

  const cacheKey = `${prediction.pairKey}__${modelVersion}__${validatedModel}__${enableWebSearch ? 'web' : 'local'}__${hasUsableApiKey() ? 'key' : 'nokey'}`;
  const cached = reportCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  const plainEnglishReport = buildReadableEnglishReport(
    prediction,
    gene1Info,
    gene2Info,
    percentile,
    structuralGnn
  );

  const ragPromptPayload = buildRetrievedPromptContext(
    prediction,
    gene1Info,
    gene2Info,
    discretized,
    percentile,
    datasetId,
    seed,
    modelVersion,
    dataMode,
    structuralGnn
  );

  if (hasUsableApiKey()) {
    const now = Date.now();
    const candidateModels: ValidGeminiModel[] =
      validatedModel === 'gemini-3.1-flash-lite'
        ? ['gemini-3.1-flash-lite']
        : [validatedModel, 'gemini-3.1-flash-lite'];

    const ai = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });

    for (const modelId of candidateModels) {
      const cooldownEnd = modelCooldownUntil.get(modelId) ?? 0;
      if (now < cooldownEnd) {
        continue;
      }

      try {
        const response = await ai.models.generateContent({
          model: modelId,
          contents: ragPromptPayload,
          config: {
            systemInstruction: RAG_SYSTEM_INSTRUCTION,
            ...(enableWebSearch ? { tools: [{ googleSearch: {} }] } : {}),
          },
        });

        const text = (response.text ?? '').trim();
        if (text.length > 80) {
          const provenance: ProvenanceStamp = {
            dataset: datasetId,
            seed,
            modelVersion,
            dataMode,
            generationMode: 'gemini',
            geminiModel: modelId,
            webSearchEnabled: enableWebSearch,
            generatedAt: new Date().toISOString(),
          };

          const result: RagReportResponse = {
            pairKey: prediction.pairKey,
            gene1: prediction.gene1,
            gene2: prediction.gene2,
            score: prediction.score,
            ndcgScore: prediction.ndcgScore,
            rank: prediction.rank,
            totalPairs: prediction.totalPairs,
            mechanismType: prediction.mechanismType,
            discretizedFeatures: discretized,
            plainEnglishReport,
            reportMarkdown: `${formatProvenanceHeader(provenance)}\n\n${text}`,
            provenance,
          };
          reportCache.set(cacheKey, result);
          return result;
        }
      } catch (err) {
        if (isQuotaOrRateLimitError(err)) {
          modelCooldownUntil.set(modelId, Date.now() + 30 * 60 * 1000);
          continue;
        }
        break;
      }
    }
  }

  const provenance: ProvenanceStamp = {
    dataset: datasetId,
    seed,
    modelVersion,
    dataMode,
    generationMode: 'deterministic_template',
    geminiModel: validatedModel,
    webSearchEnabled: false,
    generatedAt: new Date().toISOString(),
  };

  const fallbackResult: RagReportResponse = {
    pairKey: prediction.pairKey,
    gene1: prediction.gene1,
    gene2: prediction.gene2,
    score: prediction.score,
    ndcgScore: prediction.ndcgScore,
    rank: prediction.rank,
    totalPairs: prediction.totalPairs,
    mechanismType: prediction.mechanismType,
    discretizedFeatures: discretized,
    plainEnglishReport,
    reportMarkdown: buildDeterministicTemplateReport(
      prediction,
      gene1Info,
      gene2Info,
      discretized,
      percentile,
      provenance,
      structuralGnn
    ),
    provenance,
  };

  reportCache.set(cacheKey, fallbackResult);
  return fallbackResult;
}

function describeMechanismPlain(mech: string, g1: string, g2: string): string {
  if (mech === 'bridge-two-cells') {
    return `${g1} and ${g2} act as a two-cell bridge, pairing a tumor-facing recognition arm with an immune-cell engagement arm.`;
  }
  if (mech === 'TAA+TAA') {
    return `${g1} and ${g2} co-occur on tumor cells, enabling dual-target blockade or co-targeting across shared tumor pathways.`;
  }
  return `${g1} and ${g2} operate across tumor-immune modulation pathways relevant to dual-target therapeutic design.`;
}

function buildReadableEnglishReport(
  pred: CandidatePrediction,
  g1: GeneInfo,
  g2: GeneInfo,
  percentile: string,
  structuralGnn?: PairGnnMetrics
): string {
  const f = pred.features;
  const safety = f.score_safe ?? 0;
  const doubleMax = f.double_ratio_max ?? 0;
  const pearson = f.pearson_overall ?? 0;
  const stageInfo = formatStageSummary(pred);

  const tier =
    pred.score >= 0.8
      ? 'Top-tier score'
      : pred.score >= 0.55
        ? 'High score'
        : pred.score >= 0.25
          ? 'Moderate score'
          : 'Lower score';

  const safetyNote =
    safety >= 0.35
      ? `Tumor-minus-normal safety index is ${safety.toFixed(2)}, indicating stronger tumor vs. healthy-tissue expression.`
      : safety > 0.05
        ? `Tumor-minus-normal safety index is ${safety.toFixed(2)}, showing modest selectivity that warrants dose and affinity checks.`
        : `Safety index is ${safety.toFixed(2)}, flagging potential healthy-tissue overlap to evaluate carefully.`;

  const coexpNote =
    doubleMax >= 0.25
      ? `Up to ${(doubleMax * 100).toFixed(1)}% of cells in top compartments co-express both targets.`
      : `${(doubleMax * 100).toFixed(1)}% double-positive peak co-expression across compartments; expression correlation r = ${pearson.toFixed(2)}.`;

  const stageNote = pred.inClinicalTrial
    ? `**${stageInfo.stageTitle}** (${stageInfo.stageDetail}).`
    : `**${stageInfo.stageTitle}** (${stageInfo.stageDetail}).`;

  const gnnNote = structuralGnn
    ? `\n- **3D Structural Fit**: ${structuralGnn.bindingAssessment.fitPercent}% (${structuralGnn.bindingAssessment.verdict}) — ${structuralGnn.bindingAssessment.summary}`
    : '';

  return [
    `### Summary: ${pred.gene1} + ${pred.gene2}`,
    `**Score (${tier})**: **${pred.score.toFixed(4)}** (rank **#${pred.rank.toLocaleString()} / ${pred.totalPairs.toLocaleString()}**, top **${percentile}%**).`,
    '',
    `- **Targets & Mechanism**: ${pred.gene1} (${g1.description}) + ${pred.gene2} (${g2.description}). ${describeMechanismPlain(pred.mechanismType, pred.gene1, pred.gene2)}`,
    `- **Clinical Stage**: ${stageNote}`,
    `- **Safety & Co-expression**: ${safetyNote} ${coexpNote}${gnnNote}`,
  ].join('\n');
}

function buildRetrievedPromptContext(
  pred: CandidatePrediction,
  g1: GeneInfo,
  g2: GeneInfo,
  discretized: ReturnType<typeof applyDiscretizer>,
  percentile: string,
  datasetId: string,
  seed: number,
  modelVersion: string,
  dataMode: string,
  structuralGnn?: PairGnnMetrics
): string {
  const f = pred.features;
  const stageInfo = formatStageSummary(pred);

  const discretizedLines = discretized
    .map((d) => {
      const defText = FEATURE_DEFINITIONS[d.feature] ?? d.description;
      return `- ${d.feature} = ${d.value.toFixed(4)} -> [${d.binLabel}] | Meaning: ${defText}`;
    })
    .join('\n');

  const gnnSection = structuralGnn
    ? `
3D Structural / GNN Context:
- Targets: ${structuralGnn.target1.gene} (${structuralGnn.target1.entryId}, pLDDT=${structuralGnn.target1.meanPlddt}) + ${structuralGnn.target2.gene} (${structuralGnn.target2.entryId}, pLDDT=${structuralGnn.target2.meanPlddt})
- Verdict & Fit: ${structuralGnn.bindingAssessment.verdict} (${structuralGnn.bindingAssessment.fitPercent}% fit) — ${structuralGnn.bindingAssessment.summary}
- Note: ${structuralGnn.bindingAssessment.whyNote}
- Cosine Sim: ${structuralGnn.gnnCosineSim.toFixed(4)}, Complementarity: ${structuralGnn.structuralComplementarity.toFixed(4)}, Ectodomain Access: ${structuralGnn.ectodomainAccessibility.toFixed(4)}`
    : '';

  return `{ml_result}
Candidate Pair: [${pred.gene1}] + [${pred.gene2}]
Provenance: Dataset=${datasetId} (${dataMode}), Seed=${seed}, Model=${modelVersion}
Viability Score: ${pred.score.toFixed(4)} (bounded [0, 1])
Global Ranking: #${pred.rank} of ${pred.totalPairs} pairs (top ${percentile}%)
Mechanism Class: ${pred.mechanismType}
Clinical Progression Stage: ${stageInfo.stageTitle} (${stageInfo.stageDetail})

Target Metadata:
- ${pred.gene1}: ${g1.category} | bias=${g1.cell_bias} | ${g1.description}
- ${pred.gene2}: ${g2.category} | bias=${g2.cell_bias} | ${g2.description}

Discretized Biological & Quantitative Drivers:
${discretizedLines}

Raw Metrics:
- score_safe: ${(f.score_safe ?? 0).toFixed(4)} (dx=${(f.dx_tumor_minus_normal ?? 0).toFixed(4)}, dy=${(f.dy_tumor_minus_normal ?? 0).toFixed(4)})
- double_ratio_max: ${(f.double_ratio_max ?? 0).toFixed(4)}, double_ratio_overall: ${(f.double_ratio_overall ?? 0).toFixed(4)}, single_ratio_max: ${(f.single_ratio_max ?? 0).toFixed(4)}
- pearson_overall: ${(f.pearson_overall ?? 0).toFixed(4)}, vector_score: ${(f.vector_score ?? 0).toFixed(4)}
- pathway_cooccur: ${f.pathway_cooccur ?? 0} (KEGG: ${f.kegg_cooccur ?? 0}, GO: ${f.go_cooccur ?? 0}, same_pathway: ${f.same_pathway ?? 0})${gnnSection}

Task:
Generate a concise, grounded markdown report covering:
1. Candidate Pair, Clinical Progression Stage & Mechanism Overview
2. Key Ranking Drivers & Discretized Biological/Structural Profile
3. Safety Risks & Experimental Follow-Up Considerations`;
}

function buildDeterministicTemplateReport(
  pred: CandidatePrediction,
  g1: GeneInfo,
  g2: GeneInfo,
  discretized: ReturnType<typeof applyDiscretizer>,
  percentile: string,
  provenance: ProvenanceStamp,
  structuralGnn?: PairGnnMetrics
): string {
  const f = pred.features;
  const stageInfo = formatStageSummary(pred);

  const tagSummary = discretized
    .map((d) => `- **${d.feature}**: \`${d.value.toFixed(4)}\` **[${d.binLabel}]**`)
    .join('\n');

  const gnnBlock = structuralGnn
    ? `\n- **3D Structural Fit**: **${structuralGnn.bindingAssessment.fitPercent}%** (\`${structuralGnn.bindingAssessment.verdict}\`), cosine similarity \`${structuralGnn.gnnCosineSim.toFixed(4)}\`, ectodomain accessibility \`${structuralGnn.ectodomainAccessibility.toFixed(4)}\`.`
    : '';

  return `${formatProvenanceHeader(provenance)}

### 1. Candidate & Clinical Progression Stage
- **Pair**: **${pred.gene1} + ${pred.gene2}** (\`${pred.mechanismType}\`)
- **Stage**: **${stageInfo.stageTitle}** — ${stageInfo.stageDetail}
- **Targets**: ${pred.gene1} (${g1.description}); ${pred.gene2} (${g2.description})

### 2. Ranking & Discretized Drivers
- **Viability Score**: **${pred.score.toFixed(4)}** (Rank **#${pred.rank.toLocaleString()} / ${pred.totalPairs.toLocaleString()}**, top **${percentile}%**)
${tagSummary}${gnnBlock}

### 3. Safety & Next Steps
- **Safety Differential (\`score_safe\`)**: \`${(f.score_safe ?? 0).toFixed(4)}\` (dx=\`${(f.dx_tumor_minus_normal ?? 0).toFixed(4)}\`, dy=\`${(f.dy_tumor_minus_normal ?? 0).toFixed(4)}\`).
- **Follow-up**: Verify surface expression in tumor vs. normal cells, assess structural/linker constraints, and benchmark against relevant comparator controls.`;
}