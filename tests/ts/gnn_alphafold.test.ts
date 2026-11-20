import assert from 'node:assert/strict';
import { once } from 'node:events';
import type { AddressInfo } from 'node:net';
import test from 'node:test';
import { createBspaiApp } from '../../server.ts';
import {
  buildAlphaFoldPriorStructure,
  getAlphaFoldStructure,
  parseAlphaFoldPdb,
} from '../../src/engine/alphafold.ts';
import {
  computePairGnnMetrics,
  computeProteinGnnEmbedding,
} from '../../src/engine/gnn.ts';
import { runAndPersistPipeline } from '../../src/engine/pipeline.ts';

test('AlphaFold PDB parser extracts C-alpha coordinates and pLDDT scores', () => {
  const lines: string[] = [];
  for (let i = 1; i <= 24; i++) {
    const x = (i * 3.8).toFixed(3).padStart(8);
    const y = (Math.sin(i * 0.7) * 2.1).toFixed(3).padStart(8);
    const z = (Math.cos(i * 0.7) * 2.1).toFixed(3).padStart(8);
    const b = (88.5 - (i % 5) * 4).toFixed(2).padStart(6);
    const seq = String(i).padStart(4);
    lines.push(
      `ATOM  ${String(i).padStart(5)}  CA  ALA A${seq}    ${x}${y}${z}  1.00${b}           C`
    );
  }
  const parsed = parseAlphaFoldPdb(lines.join('\n'), 'CD274', 'Q9NZQ7');
  assert.ok(parsed !== null);
  assert.equal(parsed.gene, 'CD274');
  assert.equal(parsed.uniprotId, 'Q9NZQ7');
  assert.equal(parsed.sourceMode, 'ALPHAFOLD_DB_LIVE');
  assert.equal(parsed.residues.length, 24);
  assert.ok(parsed.meanPlddt > 70 && parsed.meanPlddt < 95);
});

test('GNN structural embedding is L2-normalized and SE(3) rotation-invariant', async () => {
  const s1 = await getAlphaFoldStructure('CD274', {
    gene: 'CD274',
    category: 'immune_checkpoint',
    cell_bias: 'epithelial',
    description: 'PD-L1 immune checkpoint ligand',
  });
  const s2 = buildAlphaFoldPriorStructure('CTLA4', {
    gene: 'CTLA4',
    category: 'immune_checkpoint',
    cell_bias: 'treg',
    description: 'CTLA-4 inhibitory checkpoint receptor',
  });

  assert.equal(s1.uniprotId, 'Q9NZQ7');
  assert.equal(s2.uniprotId, 'P16410');

  const g1 = computeProteinGnnEmbedding(s1);
  const g2 = computeProteinGnnEmbedding(s2);

  assert.equal(g1.embeddingDim, 16);
  assert.equal(g1.embedding.length, 16);
  assert.ok(g1.graphStats.spatialContactEdges > 10);

  const norm1 = Math.sqrt(g1.embedding.reduce((acc, v) => acc + v * v, 0));
  assert.ok(Math.abs(norm1 - 1.0) < 1e-4, `Expected unit L2 norm, got ${norm1}`);

  for (const sal of g1.residueSalience) {
    assert.ok(sal >= 0 && sal <= 1);
  }

  // Rotate + translate all 3D coordinates of s1 and verify SE(3) invariance
  const theta = 0.73;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const s1Rotated = {
    ...s1,
    residues: s1.residues.map((r) => ({
      ...r,
      x: r.x * c - r.z * s + 14.2,
      y: r.y - 7.5,
      z: r.x * s + r.z * c + 3.1,
    })),
  };
  const g1Rot = computeProteinGnnEmbedding(s1Rotated);
  for (let d = 0; d < 16; d++) {
    assert.ok(
      Math.abs(g1.embedding[d] - g1Rot.embedding[d]) < 1e-4,
      `SE(3) invariance violated at dim ${d}: ${g1.embedding[d]} vs ${g1Rot.embedding[d]}`
    );
  }

  const pairMetrics = computePairGnnMetrics(s1, g1, s2, g2, 'tumor-immunotherapy', 1.1);
  assert.ok(pairMetrics.gnnCosineSim >= -1 && pairMetrics.gnnCosineSim <= 1);
  assert.ok(pairMetrics.structuralComplementarity > 0 && pairMetrics.structuralComplementarity <= 1);
  assert.ok(pairMetrics.bindingAssessment);
  assert.equal(pairMetrics.bindingAssessment.checks.length, 3);
  assert.equal(pairMetrics.bindingAssessment.canBind, true);
  assert.ok(pairMetrics.bindingAssessment.headline.length > 5);

  // Two multi-pass membrane bundles trigger steric crowding verdict
  const m1 = buildAlphaFoldPriorStructure('MS4A1', {
    gene: 'MS4A1',
    category: 'taa',
    cell_bias: 'epithelial',
    description: 'CD20 multi-pass membrane protein',
  });
  const m2 = buildAlphaFoldPriorStructure('CLDN18', {
    gene: 'CLDN18',
    category: 'taa',
    cell_bias: 'epithelial',
    description: 'Claudin-18 tight junction multi-pass protein',
  });
  const blockedMetrics = computePairGnnMetrics(
    m1,
    computeProteinGnnEmbedding(m1),
    m2,
    computeProteinGnnEmbedding(m2),
    'TAA+TAA',
    1.2
  );
  assert.equal(blockedMetrics.bindingAssessment.verdict, 'blocked');
  assert.equal(blockedMetrics.bindingAssessment.canBind, false);
});

test('AlphaFold structure & GNN API routes return valid 3D structures and pair embeddings', async () => {
  const rt = runAndPersistPipeline(42, false);
  const app = createBspaiApp(rt);
  const server = app.listen(0, '127.0.0.1');
  await once(server, 'listening');

  try {
    const addr = server.address() as AddressInfo;
    const baseUrl = `http://127.0.0.1:${addr.port}`;

    // 1. GET /api/structure/:gene
    const structRes = await fetch(`${baseUrl}/api/structure/EGFR`);
    assert.equal(structRes.status, 200);
    const structBody = (await structRes.json()) as {
      gene: string;
      uniprotId: string;
      entryId: string;
      residues: unknown[];
      gnn: { embedding: number[]; residueSalience: number[]; graphStats: { nodeCount: number } };
    };
    assert.equal(structBody.gene, 'EGFR');
    assert.equal(structBody.uniprotId, 'P00533');
    assert.equal(structBody.entryId, 'AF-P00533-F1');
    assert.ok(structBody.residues.length > 50);
    assert.equal(structBody.gnn.embedding.length, 16);
    assert.equal(structBody.gnn.residueSalience.length, structBody.residues.length);

    // 2. POST /api/gnn/pair
    const gnnRes = await fetch(`${baseUrl}/api/gnn/pair`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gene1: 'EGFR', gene2: 'MET' }),
    });
    assert.equal(gnnRes.status, 200);
    const gnnBody = (await gnnRes.json()) as {
      pairKey: string;
      metrics: {
        gnnCosineSim: number;
        structuralComplementarity: number;
        target1: { uniprotId: string };
        bindingAssessment: { verdict: string; canBind: boolean; checks: unknown[] };
      };
    };
    assert.equal(gnnBody.pairKey, 'EGFR__MET');
    assert.equal(gnnBody.metrics.target1.uniprotId, 'P00533');
    assert.equal(gnnBody.metrics.bindingAssessment.canBind, true);
    assert.equal(gnnBody.metrics.bindingAssessment.checks.length, 3);

    // 3. POST /api/predict includes structuralGnn
    const predRes = await fetch(`${baseUrl}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gene1: 'CD274', gene2: 'CTLA4' }),
    });
    assert.equal(predRes.status, 200);
    const predBody = (await predRes.json()) as {
      structuralGnn?: { gnnCosineSim: number; target1: { entryId: string }; bindingAssessment: { canBind: boolean } };
    };
    assert.ok(predBody.structuralGnn);
    assert.equal(predBody.structuralGnn.target1.entryId, 'AF-Q9NZQ7-F1');
    assert.equal(predBody.structuralGnn.bindingAssessment.canBind, true);
  } finally {
    server.close();
  }
});
