import 'dotenv/config';
import { buildPredictionObject, runAndPersistPipeline } from '../src/engine/pipeline.ts';
import { generateGroundedRagReport } from '../src/engine/rag.ts';

async function main() {
  const seed = Number(process.env.BSPAI_SEED ?? 42);
  const force = process.argv.includes('--rebuild');

  console.log(`[BSPAI] Running TypeScript pipeline (seed=${seed}, forceRebuild=${force})...`);
  const t0 = Date.now();
  const rt = runAndPersistPipeline(seed, force);
  const elapsedMs = Date.now() - t0;

  console.log(
    `[BSPAI] Dataset=${rt.datasetId} | Genes=${rt.genes.length} | Candidate Pairs=${rt.totalPairs.toLocaleString()} | Retained QC Cells=${rt.preprocessed.qcStats.retainedCells}/${rt.preprocessed.qcStats.rawCells} | Elapsed=${elapsedMs}ms`
  );

  console.log('\n--- 5-Fold CV Model Benchmark ---');
  for (const row of rt.benchmarks) {
    console.log(
      `${row.model.padEnd(32)} AUC=${row.meanAuc.toFixed(4)} ± ${row.stdAuc.toFixed(4)} (Paper Ref: ${row.paperRefAuc ?? 'N/A'})`
    );
  }

  console.log('\n--- 6-Step Incremental Feature Ablation ---');
  for (const step of rt.ablation) {
    console.log(
      `${step.step.padEnd(42)} feats=${String(step.featureCount).padStart(2)} | AUC=${step.meanAuc.toFixed(4)} ± ${step.stdAuc.toFixed(4)} (Paper Ref: ${step.paperRefAuc.toFixed(4)})`
    );
  }

  console.log('\n--- Top-10 Ranked Candidate Pairs ---');
  for (const pair of rt.rankedPairs.slice(0, 10)) {
    const trial = pair.inClinicalTrial ? `Stage ${pair.clinicalStage} (${pair.clinicalDrugs[0]})` : 'Exploratory';
    console.log(
      `#${String(pair.rank).padStart(2)} ${pair.gene1} + ${pair.gene2} | score=${pair.score.toFixed(4)} | ${pair.mechanismType} | ${trial}`
    );
  }

  const cd274Ctla4 = buildPredictionObject(rt, 'CD274', 'CTLA4');
  const report = await generateGroundedRagReport({
    prediction: cd274Ctla4,
    gene1Info: rt.geneMap.get('CD274')!,
    gene2Info: rt.geneMap.get('CTLA4')!,
    binSpecs: rt.binSpecs,
    datasetId: rt.datasetId,
    seed: rt.seed,
    modelVersion: rt.modelVersion,
    dataMode: rt.dataMode,
  });

  console.log('\n--- Sample RAG Report (CD274 + CTLA4) ---');
  console.log(
    `Rank #${report.rank} / ${report.totalPairs} | Score=${report.score.toFixed(4)} | Mode=${report.provenance.generationMode}`
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
