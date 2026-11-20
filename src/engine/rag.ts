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

  const cacheKey = `${prediction.pairKey}__${validatedModel}__${enableWebSearch ? 'web' : 'local'}__${hasUsableApiKey() ? 'key' : 'nokey'}`;
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

  const retrievedFacts = buildRetrievedPromptContext(
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
          contents: retrievedFacts,
          config: {
            systemInstruction:
              'You are a computational oncology research analyst summarizing BiSpec Pairwise AI (BSPAI) target-pair rankings. Write clearly for both biotech decision-makers and computational biologists. Use ONLY the provided numeric metrics (including Pairwise Druggability Score), discretized bin labels, gene metadata, AlphaFold GNN fit, and seeded clinical status. Never invent trial results or NCT numbers.',
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

          const stampedMarkdown = `${formatProvenanceHeader(provenance)}\n\n${text}`;
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
            reportMarkdown: stampedMarkdown,
            provenance,
          };
          reportCache.set(cacheKey, result);
          return result;
        }
      } catch (err) {
        if (isQuotaOrRateLimitError(err)) {
          // Pause retries against this quota-exhausted model for 30 minutes and fall back cleanly
          modelCooldownUntil.set(modelId, Date.now() + 30 * 60 * 1000);
          continue;
        }
        // Transient non-quota error: fall back quietly to deterministic report
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

  const reportMarkdown = buildDeterministicTemplateReport(
    prediction,
    gene1Info,
    gene2Info,
    discretized,
    percentile,
    provenance,
    structuralGnn
  );

  return {
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
    reportMarkdown,
    provenance,
  };
}

function describeMechanismPlain(mech: string, g1: string, g2: string): string {
  if (mech === 'bridge-two-cells') {
    return `${g1} and ${g2} act as a two-cell bridge: one antibody arm grabs the tumor surface marker while the other arm recruits immune killer T cells directly to the tumor.`;
  }
  if (mech === 'TAA+TAA') {
    return `${g1} and ${g2} sit together on cancer cells, allowing a single antibody to block two tumor growth pathways at once and reduce drug resistance.`;
  }
  return `${g1} and ${g2} work together in tumor immunotherapy, releasing brakes on the immune system or altering the tumor environment so immune cells can attack cancer more effectively.`;
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

  const strengthTier =
    pred.score >= 4.0
      ? 'Top-tier candidate'
      : pred.score >= 1.5
        ? 'Strong candidate'
        : pred.score >= 0.0
          ? 'Moderate candidate'
          : 'Early exploratory candidate';

  const safetySentence =
    safety >= 0.35
      ? `Both targets show noticeably higher activity in tumor cells than in healthy tissue (safety index ${safety.toFixed(2)}), which helps limit side effects on normal organs.`
      : safety > 0.05
        ? `The pair shows a modest preference for tumor over healthy tissue (safety index ${safety.toFixed(2)}), so careful dose and affinity tuning would be wise.`
        : `At least one target is also active in healthy tissue (safety index ${safety.toFixed(2)}), meaning off-tumor toxicity should be monitored closely in early lab testing.`;

  const activitySentence =
    doubleMax >= 0.25
      ? `Up to ${(doubleMax * 100).toFixed(1)}% of cells in key tumor compartments express both targets together, supporting strong dual binding.`
      : `Only ${(doubleMax * 100).toFixed(1)}% of cells co-express both targets in the same cell, which fits a two-cell bridging or microenvironment design better than single-cell dual blockade.`;

  const clinicalSentence = pred.inClinicalTrial
    ? `This combination already has real-world clinical validation at Stage ${pred.clinicalStage} (${pred.clinicalDrugs.join(', ')}).`
    : `This exact pair has not been tested in our seeded clinical benchmark table, representing a novel combination opportunity.`;

  const assessment = structuralGnn?.bindingAssessment;
  const verdict = assessment?.verdict;

  let verdictStatusNote = '';
  let bindingSectionLines: string[] = [];
  let nextStep2 =
    'Proceed to bispecific construct design using high-confidence exposed ectodomain pockets.';

  if (structuralGnn && assessment) {
    if (verdict === 'conditional-bind') {
      verdictStatusNote = ` 3D structural analysis classifies this pair as a **Partial Fit (${assessment.fitPercent}% structural fit)** — biologically bindable, with a membrane-spacing engineering note explained below.`;

      const clinicalBoostReason = pred.inClinicalTrial
        ? `strong real-world clinical precedent (Stage ${pred.clinicalStage}: ${pred.clinicalDrugs.join(', ')}), where engineered bispecific formats (such as flexible linkers or 2:1 Fab spacing) already succeed despite tight membrane clearance`
        : `strong biological evidence across single-cell expression, pathway synergy, and tumor-vs-normal selectivity`;

      bindingSectionLines = [
        `- **3D Structural Fit — Partial Fit (${assessment.fitPercent}%)**: **What Partial Fit means:** Both **${pred.gene1}** and **${pred.gene2}** still expose reachable pockets that a bispecific antibody can engage, so binding is possible — however, at least one epitope sits close to the cell membrane or in a tight loop (${assessment.whyNote}). A standard rigid IgG format may experience mild steric crowding unless the linker length, hinge flexibility, or Fab arm spacing is tuned.`,
        `- **Why It Earned a Score of ${pred.score.toFixed(4)} (Rank #${pred.rank.toLocaleString()}) Despite Being a Partial Fit**: The **Pairwise Druggability Score** combines **40+ biological, single-cell, safety, pathway, and structural features** rather than relying on raw pocket geometry alone. This pair earned its score because of ${clinicalBoostReason} (safety index ${safety.toFixed(2)}, mechanism \`${pred.mechanismType}\`). The **Partial Fit** flag acts as an actionable engineering guide telling researchers *how* to design the antibody format rather than disqualifying a biologically valuable pair.`,
      ];

      nextStep2 = `Because 3D modeling flags a **Partial Fit**, optimize bispecific antibody geometry (flexible hinges, extended linkers, or asymmetric 2:1 Fab/scFv formats) to clear the membrane-proximal loop without steric collision.`;
    } else if (verdict === 'blocked') {
      verdictStatusNote = ` 3D structural analysis flags this pair as **Binding Blocked (${assessment.fitPercent}% structural fit)** due to physical pocket-access or membrane crowding barriers.`;

      bindingSectionLines = [
        `- **3D Structural Fit — Blocked (${assessment.fitPercent}%)**: **What Blocked means:** A standard dual-arm bispecific antibody physically recoils and cannot clasp both targets simultaneously because of shielded, disordered, or membrane-hugging multi-pass geometry (${assessment.whyNote}).`,
        `- **How This Relates to Its Score (${pred.score.toFixed(4)}, Rank #${pred.rank.toLocaleString()})**: The Pairwise Druggability Score summarizes single-cell expression, pathway, and safety patterns from transcriptomic data, while the 3D AlphaFold GNN check adds a physical feasibility filter showing that a conventional IgG format would struggle to bind both epitopes at once.`,
      ];

      nextStep2 = `Because 3D modeling flags **Blocked** dual-Fab binding, re-engineer the targeting format (e.g., compact single-domain/nanobody arms or alternative exposed epitopes) or prioritize a cleaner-fitting target partner.`;
    } else {
      verdictStatusNote = ` 3D structural analysis confirms a **Full Structural Fit (${assessment.fitPercent}% structural fit)** with clean dual-arm accessibility.`;

      bindingSectionLines = [
        `- **3D Structural Fit — Full Fit (${assessment.fitPercent}%)**: Both **${pred.gene1}** and **${pred.gene2}** present well-folded, surface-exposed extracellular pockets (AlphaFold pocket pLDDT ${assessment.target1.pocketMeanPlddt.toFixed(0)} & ${assessment.target2.pocketMeanPlddt.toFixed(0)}), allowing both antibody arms to dock cleanly without steric clash and reinforcing the Pairwise Druggability Score of **${pred.score.toFixed(4)}**.`,
      ];
    }
  }

  const corrSentence =
    pearson >= 0.15
      ? `Expression levels of ${pred.gene1} and ${pred.gene2} rise together across single cells (correlation ${pearson.toFixed(2)}).`
      : `Expression levels of ${pred.gene1} and ${pred.gene2} vary independently across compartments (correlation ${pearson.toFixed(2)}).`;

  return [
    `### Plain-English Summary: ${pred.gene1} + ${pred.gene2}`,
    `**Overall Verdict (${strengthTier})**: Ubhaya-Bandha gives **${pred.gene1} + ${pred.gene2}** a **Pairwise Druggability Score of ${pred.score.toFixed(4)}**, ranking it **#${pred.rank.toLocaleString()} out of ${pred.totalPairs.toLocaleString()}** possible pairs (top **${percentile}%**).${verdictStatusNote}`,
    '',
    `#### What This Combination Does`,
    `- **${pred.gene1}**: ${g1.description}.`,
    `- **${pred.gene2}**: ${g2.description}.`,
    `- **How They Work Together**: ${describeMechanismPlain(pred.mechanismType, pred.gene1, pred.gene2)}`,
    '',
    `#### Why It Scored This Way`,
    `- **Safety & Selectivity**: ${safetySentence}`,
    `- **Target Presence**: ${activitySentence} ${corrSentence}`,
    ...bindingSectionLines,
    `- **Clinical Track Record**: ${clinicalSentence}`,
    '',
    `#### Recommended Next Steps`,
    `1. Confirm surface protein levels of ${pred.gene1} and ${pred.gene2} in patient tumor vs. healthy tissue samples.`,
    `2. ${nextStep2}`,
    `3. Run a tumor-immune co-culture killing assay against a clinical benchmark comparator.`,
  ].join('\n');
}

function formatProvenanceHeader(p: ProvenanceStamp): string {
  return [
    `> **Research Aid Notice**: Automated Ubhaya-Bandha target-selection synthesis. Not clinical or medical advice.`,
    `> **Provenance**: Dataset \`${p.dataset}\` | Seed \`${p.seed}\` | Model \`${p.modelVersion}\` | Synthesizer \`${p.generationMode}\` | Generated \`${p.generatedAt}\``,
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
  const clinicalLine = pred.inClinicalTrial
    ? `Seeded clinical benchmark status: Stage ${pred.clinicalStage} (${pred.clinicalDrugs.join('; ')})${pred.needsVerification ? ' [needs verification]' : ''}.`
    : 'Seeded clinical benchmark status: Exploratory pair (not yet in clinical drug table).';

  const binLines = discretized
    .map(
      (d) =>
        `- ${d.feature} = ${d.value.toFixed(4)} -> "${d.binLabel}" (${d.description})`
    )
    .join('\n');

  const gnnVerdictLabel =
    structuralGnn?.bindingAssessment.verdict === 'conditional-bind'
      ? 'Partial Fit (conditional-bind)'
      : structuralGnn?.bindingAssessment.verdict === 'blocked'
        ? 'Blocked (structural barrier)'
        : 'Full Structural Fit (strong-bind)';

  const gnnBlock = structuralGnn
    ? `
AlphaFold DB & 3D Residue-Graph GNN:
- ${structuralGnn.target1.gene} (${structuralGnn.target1.entryId}, mean pLDDT=${structuralGnn.target1.meanPlddt}) + ${structuralGnn.target2.gene} (${structuralGnn.target2.entryId}, mean pLDDT=${structuralGnn.target2.meanPlddt})
- Structural Binding Verdict: ${gnnVerdictLabel} — ${structuralGnn.bindingAssessment.fitPercent}% structural fit
- Structural Summary: ${structuralGnn.bindingAssessment.summary}
- Structural Detail: ${structuralGnn.bindingAssessment.whyNote}
- Cosine Sim: ${structuralGnn.gnnCosineSim.toFixed(4)}, Complementarity: ${structuralGnn.structuralComplementarity.toFixed(4)}, Ectodomain Access: ${structuralGnn.ectodomainAccessibility.toFixed(4)}`
    : '';

  const partialFitInstruction =
    structuralGnn?.bindingAssessment.verdict === 'conditional-bind'
      ? `\nIMPORTANT: This pair is classified as a **Partial Fit**. Explicitly state that it is a Partial Fit, explain in plain language what Partial Fit means structurally (tight membrane-proximal clearance requiring flexible linkers/Fab spacing), and clearly explain why it still received its Pairwise Druggability Score of ${pred.score.toFixed(4)} (biological, safety, single-cell, and clinical drivers).`
      : structuralGnn?.bindingAssessment.verdict === 'blocked'
        ? `\nIMPORTANT: This pair is classified as **Binding Blocked** structurally. Explicitly explain the physical barrier and how it contrasts with the transcriptomic Pairwise Druggability Score.`
        : '';

  return `Analyze the bispecific antibody target combination ${pred.gene1} + ${pred.gene2}.

Provenance:
- Dataset: ${datasetId} (${dataMode}), Seed: ${seed}, Model: ${modelVersion}

Model Prediction:
- Pairwise Druggability Score: ${pred.score.toFixed(4)}
- Global Rank: #${pred.rank} of ${pred.totalPairs} candidate pairs (top ${percentile}%)
- Mechanism Class: ${pred.mechanismType}
- ${clinicalLine}

Target Metadata:
- ${g1.gene} (${g1.category}, ${g1.cell_bias}): ${g1.description}
- ${g2.gene} (${g2.category}, ${g2.cell_bias}): ${g2.description}

Top Discretized Drivers:
${binLines}

Additional Features:
- score_safe: ${(f.score_safe ?? 0).toFixed(4)} (dx=${(f.dx_tumor_minus_normal ?? 0).toFixed(4)}, dy=${(f.dy_tumor_minus_normal ?? 0).toFixed(4)})
- vector_score: ${(f.vector_score ?? 0).toFixed(4)}
- kegg_cooccur: ${f.kegg_cooccur ?? 0}, go_cooccur: ${f.go_cooccur ?? 0}, same_pathway: ${f.same_pathway ?? 0}
- double_ratio_overall: ${(f.double_ratio_overall ?? 0).toFixed(4)}, double_ratio_max: ${(f.double_ratio_max ?? 0).toFixed(4)}, single_ratio_max: ${(f.single_ratio_max ?? 0).toFixed(4)}
- pearson_overall: ${(f.pearson_overall ?? 0).toFixed(4)}${gnnBlock}${partialFitInstruction}

Write a clear, concise, easy-to-understand analytical report in Markdown covering:
1. Target & Clinical Overview
2. Key Ranking Drivers (Pairwise Druggability Score, Safety, Single-Cell Activity, Structural Fit)
3. Risks & Experimental Next Steps`;
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
  const findBin = (prefix: string) =>
    discretized.find((d) => d.feature === prefix || d.feature.startsWith(prefix));

  const vecBin = findBin('vector_score');
  const safeBin = findBin('score_safe');
  const pearsonBin = findBin('pearson_');
  const doubleBin = findBin('double_ratio');
  const singleBin = findBin('single_ratio');

  const clinicalSummary = pred.inClinicalTrial
    ? `Recorded in clinical benchmark table at **Stage ${pred.clinicalStage}** (${pred.clinicalDrugs.join(', ')})${pred.needsVerification ? ' *(flagged for manual verification)*' : ''}.`
    : `Not yet recorded in the clinical bispecific drug table, making this an exploratory candidate pair.`;

  const assessment = structuralGnn?.bindingAssessment;
  const verdictLabel =
    assessment?.verdict === 'conditional-bind'
      ? 'Partial Fit (Conditional Binding)'
      : assessment?.verdict === 'blocked'
        ? 'Blocked (Structural Barrier)'
        : 'Full Structural Fit';

  let gnnLine = '';
  if (structuralGnn && assessment) {
    gnnLine = `\n- **AlphaFold 3D & GNN Binding Fit (${verdictLabel})**: \`${structuralGnn.target1.entryId}\` (mean pLDDT **${structuralGnn.target1.meanPlddt}**) + \`${structuralGnn.target2.entryId}\` (mean pLDDT **${structuralGnn.target2.meanPlddt}**) yields **${assessment.fitPercent}%** structural fit, GNN cosine similarity **${structuralGnn.gnnCosineSim.toFixed(4)}**, and ectodomain accessibility **${structuralGnn.ectodomainAccessibility.toFixed(4)}**.`;
    if (assessment.verdict === 'conditional-bind') {
      gnnLine += `\n- **Partial Fit Interpretation & Score Rationale**: A **Partial Fit** means both targets remain bindable, but tight membrane-proximal loop geometry requires tuned linker length or Fab arm spacing (${assessment.whyNote}). The Pairwise Druggability Score of **${pred.score.toFixed(4)}** (Rank **#${pred.rank.toLocaleString()}**) reflects the combined 40+ biological, single-cell, safety, and${pred.inClinicalTrial ? ` Stage ${pred.clinicalStage} clinical validation` : ' pathway'} features, demonstrating why format engineering unlocks this pair's therapeutic value.`;
    } else if (assessment.verdict === 'blocked') {
      gnnLine += `\n- **Blocked Structural Interpretation**: ${assessment.whyNote} Although transcriptomic features produce a Pairwise Druggability Score of **${pred.score.toFixed(4)}**, physical dual-Fab accessibility blocks standard IgG engagement without major format redesign.`;
    }
  }

  const primaryRisk =
    assessment?.verdict === 'conditional-bind'
      ? `**Partial Fit Geometry**: ${assessment.whyNote} Use flexible hinges, optimized linker lengths, or asymmetric 2:1 bispecific formats to avoid membrane-proximal crowding.`
      : assessment?.verdict === 'blocked'
        ? `**Steric / Accessibility Block**: ${assessment.whyNote} Standard IgG dual-Fab constructs will likely recoil; test compact nanobody/scFv formats or alternate exposed targets.`
        : (f.score_safe ?? 0) <= 0.05
          ? 'Modest or negative tumor-normal differential calls for affinity-tuned or conditional-binding formats to minimize healthy-tissue toxicity.'
          : 'Confirm that transcript-level co-expression translates to matched surface protein density and epitope spacing.';

  return `${formatProvenanceHeader(provenance)}

### 1. Target & Clinical Context
- **${g1.gene}** (\`${g1.category}\`, enriched in \`${g1.cell_bias}\`): ${g1.description}.
- **${g2.gene}** (\`${g2.category}\`, enriched in \`${g2.cell_bias}\`): ${g2.description}.
- **Mechanism**: \`${pred.mechanismType}\`. ${clinicalSummary}

### 2. Quantitative Drivers (Pairwise Druggability & Biological Evidence)
- **Pairwise Druggability Score & Position**: **${pred.score.toFixed(4)}**, placing **${pred.gene1} + ${pred.gene2}** at **#${pred.rank.toLocaleString()} / ${pred.totalPairs.toLocaleString()}** (top **${percentile}%**).
- **Tumor-vs-Normal Safety (\`score_safe\`)**: **${(f.score_safe ?? 0).toFixed(4)}**${safeBin ? ` (${safeBin.binLabel})` : ''} from tumor-minus-normal deltas $\\Delta_x = ${(f.dx_tumor_minus_normal ?? 0).toFixed(4)}$ and $\\Delta_y = ${(f.dy_tumor_minus_normal ?? 0).toFixed(4)}$.
- **Single-Cell Activity**: Double-positive max **${(f.double_ratio_max ?? 0).toFixed(4)}**${doubleBin ? ` (${doubleBin.binLabel})` : ''}; single-positive max **${(f.single_ratio_max ?? 0).toFixed(4)}**${singleBin ? ` (${singleBin.binLabel})` : ''}; overall Pearson correlation **$r = ${(f.pearson_overall ?? 0).toFixed(4)}$**${pearsonBin ? ` (${pearsonBin.binLabel})` : ''}.
- **Embedding & Pathways**: Gene2Vec distance **${(f.vector_score ?? 0).toFixed(4)}**${vecBin ? ` (${vecBin.binLabel})` : ''}; shared pathways **${f.pathway_cooccur ?? 0}** (KEGG: ${f.kegg_cooccur ?? 0}, GO BP: ${f.go_cooccur ?? 0}).${gnnLine}

### 3. Translational Risks & Next Steps
- **Primary Consideration**: ${primaryRisk}
- **Action Plan**: (1) Measure surface protein expression on tumor vs. normal cells via flow cytometry; (2) ${assessment?.verdict === 'conditional-bind' ? 'engineer linker length and Fab spacing to resolve the Partial Fit membrane-clearance constraint' : 'check 3D Fab arm geometry against the exposed ectodomain loops'}; (3) run functional killing assays against clinical-stage controls.`;
}
