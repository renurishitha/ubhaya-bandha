import assert from 'node:assert/strict';
import { once } from 'node:events';
import type { AddressInfo } from 'node:net';
import test from 'node:test';
import { createBspaiApp } from '../../server.ts';
import { runAndPersistPipeline } from '../../src/engine/pipeline.ts';

test('Express API endpoints smoke test (genes, predict, features, top-pairs, metrics, report, CSV export)', async () => {
  const rt = runAndPersistPipeline(42, false);
  const app = createBspaiApp(rt);
  const server = app.listen(0, '127.0.0.1');
  await once(server, 'listening');

  try {
    const addr = server.address() as AddressInfo;
    const baseUrl = `http://127.0.0.1:${addr.port}`;

    // 1. /api/genes
    const genesRes = await fetch(`${baseUrl}/api/genes?q=CD274`);
    assert.equal(genesRes.status, 200);
    const genesBody = (await genesRes.json()) as {
      genes: Array<{ gene: string }>;
      geneCount: number;
      totalCandidatePairs: number;
    };
    assert.ok(genesBody.genes.some((g) => g.gene === 'CD274'));
    assert.equal(
      genesBody.totalCandidatePairs,
      (genesBody.geneCount * (genesBody.geneCount - 1)) / 2
    );

    // 2. /api/predict (valid + invalid)
    const predRes = await fetch(`${baseUrl}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gene1: 'CD274', gene2: 'CTLA4' }),
    });
    assert.equal(predRes.status, 200);
    const predBody = (await predRes.json()) as {
      gene1: string;
      gene2: string;
      rank: number;
      totalPairs: number;
      score: number;
      ndcgScore: number;
      features: Record<string, number>;
    };
    assert.equal(predBody.gene1, 'CD274');
    assert.equal(predBody.gene2, 'CTLA4');
    assert.ok(predBody.rank >= 1 && predBody.rank <= predBody.totalPairs);
    assert.ok(predBody.ndcgScore > 0 && predBody.ndcgScore <= 1);
    assert.ok('score_safe' in predBody.features);

    const badPred = await fetch(`${baseUrl}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gene1: 'NOT_A_GENE', gene2: 'CTLA4' }),
    });
    assert.equal(badPred.status, 400);

    // 3. /api/features/:pair
    const featRes = await fetch(`${baseUrl}/api/features/CD274__CTLA4`);
    assert.equal(featRes.status, 200);

    // 4. /api/top-pairs
    const topRes = await fetch(`${baseUrl}/api/top-pairs?type=all&n=20`);
    assert.equal(topRes.status, 200);
    const topBody = (await topRes.json()) as { pairs: Array<{ ndcgScore?: number }> };
    assert.equal(topBody.pairs.length, 20);
    assert.ok((topBody.pairs[0].ndcgScore ?? 0) > 0);

    // 5. /api/metrics
    const metRes = await fetch(`${baseUrl}/api/metrics`);
    assert.equal(metRes.status, 200);
    const metBody = (await metRes.json()) as {
      benchmarks: Array<{ meanNdcg10?: number }>;
      ablation: unknown[];
      topFeatureImportances: unknown[];
    };
    assert.equal(metBody.benchmarks.length, 8);
    assert.ok((metBody.benchmarks[0].meanNdcg10 ?? 0) > 0);
    assert.equal(metBody.ablation.length, 6);
    assert.ok(metBody.topFeatureImportances.length >= 20);

    // 6. /api/report
    const repRes = await fetch(`${baseUrl}/api/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gene1: 'CD274', gene2: 'CTLA4' }),
    });
    assert.equal(repRes.status, 200);
    const repBody = (await repRes.json()) as {
      plainEnglishReport?: string;
      reportMarkdown: string;
      provenance: { dataset: string; seed: number; modelVersion: string };
    };
    assert.equal(repBody.provenance.seed, 42);
    assert.ok(repBody.reportMarkdown.includes(repBody.provenance.modelVersion));
    assert.ok(
      repBody.plainEnglishReport &&
        repBody.plainEnglishReport.includes('Pairwise Druggability Score')
    );

    // 7. Excel (.xlsx) & CSV exports
    const xlsxLb = await fetch(`${baseUrl}/api/export/leaderboard.xlsx?n=10`);
    assert.equal(xlsxLb.status, 200);
    assert.ok(
      (xlsxLb.headers.get('content-type') ?? '').includes('spreadsheetml.sheet')
    );
    const xlsxBytes = new Uint8Array(await xlsxLb.arrayBuffer());
    assert.equal(xlsxBytes[0], 0x50); // 'P'
    assert.equal(xlsxBytes[1], 0x4b); // 'K'

    const csvLb = await fetch(`${baseUrl}/api/export/leaderboard.csv?n=10`);
    assert.equal(csvLb.status, 200);
    const csvLbText = await csvLb.text();
    assert.ok(csvLbText.startsWith('rank,total_pairs,gene1,gene2'));

    const csvFeat = await fetch(`${baseUrl}/api/export/features/CD274__CTLA4.csv`);
    assert.equal(csvFeat.status, 200);
    const csvFeatText = await csvFeat.text();
    assert.ok(csvFeatText.includes('score_safe'));
  } finally {
    server.close();
  }
});
