import 'dotenv/config';
import ExcelJS from 'exceljs';
import express, { type Request, type Response } from 'express';
import path from 'node:path';
import { createServer as createViteServer } from 'vite';
import { computePairNdcgScore } from './src/engine/evaluate.ts';
import {
  getAlphaFoldStructure,
  type AlphaFoldStructure,
} from './src/engine/alphafold.ts';
import {
  computePairGnnMetrics,
  computeProteinGnnEmbedding,
  type GnnProteinEmbeddingResult,
  type PairGnnMetrics,
} from './src/engine/gnn.ts';
import {
  buildPredictionObject,
  runAndPersistPipeline,
  type PipelineRuntime,
} from './src/engine/pipeline.ts';
import {
  generateGroundedRagReport,
  resolveAndValidateGeminiModel,
} from './src/engine/rag.ts';
import type { GeneInfo } from './src/engine/types.ts';

function csvEscape(val: unknown): string {
  const s = String(val ?? '');
  if (s.includes(',') || s.includes('"') || s.includes('\n')) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

export function createBspaiApp(rt: PipelineRuntime) {
  const app = express();
  app.use(express.json({ limit: '1mb' }));

  // Allow cross-origin local frontend connections (e.g., Vite on another port)
  app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type,Authorization');
    if (req.method === 'OPTIONS') {
      res.status(204).end();
      return;
    }
    next();
  });

  // Optional server-side proxy to an external Python/FastAPI backend
  const externalBackend = (process.env.EXTERNAL_BACKEND_URL ?? '').trim().replace(/\/+$/, '');
  if (externalBackend) {
    console.log(`[BSPAI] External backend proxy enabled -> ${externalBackend}`);
    app.use(['/api/predict', '/api/report'], async (req, res, next) => {
      if (req.method !== 'POST') {
        next();
        return;
      }
      const basePath = req.baseUrl;
      const stripped = basePath.replace(/^\/api/, '');
      const body = req.body ?? {};
      const g1 = String(body.gene1 ?? body.gene_a ?? body.target1 ?? '').trim().toUpperCase();
      const g2 = String(body.gene2 ?? body.gene_b ?? body.target2 ?? '').trim().toUpperCase();
      const payload = JSON.stringify({
        ...body,
        gene1: g1,
        gene2: g2,
        gene_a: g1,
        gene_b: g2,
        target1: g1,
        target2: g2,
      });

      for (const candidate of [`${externalBackend}${basePath}`, `${externalBackend}${stripped}`]) {
        try {
          const upstream = await fetch(candidate, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: payload,
          });
          if (upstream.status === 404 || upstream.status === 405) continue;
          const text = await upstream.text();
          if (upstream.ok && text) {
            try {
              const parsed = JSON.parse(text);
              res.json(parsed);
              return;
            } catch {
              // Fall through if non-JSON
            }
          }
        } catch (err) {
          console.warn(
            `[BSPAI Proxy] Upstream ${candidate} failed, falling back to local handler:`,
            err instanceof Error ? err.message : String(err)
          );
        }
      }
      next();
    });
  }

  const getOrSynthesizeGeneInfo = (geneSymbol: string): GeneInfo => {
    const g = geneSymbol.trim().toUpperCase();
    return (
      rt.geneMap.get(g) ?? {
        gene: g,
        category: 'scRNA-seq cohort target',
        cell_bias: 'epithelial',
        description: `Custom HGNC target (${g}) evaluated from single-cell cohort`,
      }
    );
  };

  const gnnEmbedCache = new Map<
    string,
    { structure: AlphaFoldStructure; gnn: GnnProteinEmbeddingResult }
  >();

  const resolveGeneStructureAndGnn = async (
    geneSymbol: string,
    tryLive = false
  ) => {
    const g = geneSymbol.trim().toUpperCase();
    const info = getOrSynthesizeGeneInfo(g);
    const existing = gnnEmbedCache.get(g);
    if (existing && (!tryLive || existing.structure.sourceMode === 'ALPHAFOLD_DB_LIVE')) {
      return existing;
    }
    const structure = await getAlphaFoldStructure(g, info, { tryLive });
    const gnn = computeProteinGnnEmbedding(structure);
    const entry = { structure, gnn };
    gnnEmbedCache.set(g, entry);
    return entry;
  };

  const resolvePairStructuralGnn = async (
    gene1: string,
    gene2: string,
    mechanismType: 'bridge-two-cells' | 'TAA+TAA' | 'tumor-immunotherapy',
    vectorScore: number,
    tryLive = false
  ): Promise<PairGnnMetrics> => {
    const [r1, r2] = await Promise.all([
      resolveGeneStructureAndGnn(gene1, tryLive),
      resolveGeneStructureAndGnn(gene2, tryLive),
    ]);
    return computePairGnnMetrics(
      r1.structure,
      r1.gnn,
      r2.structure,
      r2.gnn,
      mechanismType,
      vectorScore
    );
  };

  const handleGenes = (req: Request, res: Response) => {
    const rawQ = String(req.query.q ?? '').trim().toUpperCase();
    const filtered = rt.genes.filter(
      (g) =>
        !rawQ ||
        g.gene.includes(rawQ) ||
        g.description.toUpperCase().includes(rawQ)
    );
    res.json({
      genes: rawQ ? filtered.slice(0, 100) : filtered,
      allGeneSymbols: rt.genes.map((g) => g.gene),
      geneCount: rt.genes.length,
      totalCandidatePairs: rt.totalPairs,
    });
  };

  const handlePredict = async (req: Request, res: Response) => {
    try {
      const { gene1, gene2, liveAlphaFold } = req.body ?? {};
      if (!gene1 || !gene2) {
        res.status(400).json({ error: 'Both gene1 and gene2 are required.' });
        return;
      }
      const g1 = String(gene1).trim().toUpperCase();
      const g2 = String(gene2).trim().toUpperCase();

      const pred = buildPredictionObject(rt, g1, g2);
      const structuralGnn = await resolvePairStructuralGnn(
        pred.gene1,
        pred.gene2,
        pred.mechanismType,
        Number(pred.features.vector_score ?? 1.0),
        Boolean(liveAlphaFold)
      );
      const ndcgScore = pred.ndcgScore ?? computePairNdcgScore(pred.rank, pred.score);

      res.json({
        ...pred,
        ndcgScore,
        gene1Info: getOrSynthesizeGeneInfo(pred.gene1),
        gene2Info: getOrSynthesizeGeneInfo(pred.gene2),
        structuralGnn,
        dataMode: rt.dataMode,
        datasetId: rt.datasetId,
        seed: rt.seed,
        modelVersion: rt.modelVersion,
      });
    } catch (err) {
      res.status(400).json({
        error: err instanceof Error ? err.message : 'Prediction failed',
      });
    }
  };

  const handleStructure = async (req: Request, res: Response) => {
    try {
      const gene = String(req.params.gene ?? '');
      const tryLive = req.query.live === '1' || req.query.live === 'true';
      const { structure, gnn } = await resolveGeneStructureAndGnn(gene, tryLive);
      res.json({
        ...structure,
        gnn: {
          embeddingDim: gnn.embeddingDim,
          embedding: gnn.embedding,
          residueSalience: gnn.residueSalience,
          graphStats: gnn.graphStats,
          contactEdges: gnn.contactEdges,
        },
      });
    } catch (err) {
      res.status(400).json({
        error: err instanceof Error ? err.message : 'Structure lookup failed',
      });
    }
  };

  const handleGnnPair = async (req: Request, res: Response) => {
    try {
      const { gene1, gene2, liveAlphaFold, includeResidues } = req.body ?? {};
      if (!gene1 || !gene2) {
        res.status(400).json({ error: 'Both gene1 and gene2 are required.' });
        return;
      }
      const g1 = String(gene1).trim().toUpperCase();
      const g2 = String(gene2).trim().toUpperCase();
      const pred = buildPredictionObject(rt, g1, g2);
      const tryLive = Boolean(liveAlphaFold);
      const [r1, r2] = await Promise.all([
        resolveGeneStructureAndGnn(pred.gene1, tryLive),
        resolveGeneStructureAndGnn(pred.gene2, tryLive),
      ]);
      const metrics = computePairGnnMetrics(
        r1.structure,
        r1.gnn,
        r2.structure,
        r2.gnn,
        pred.mechanismType,
        pred.features.vector_score ?? 1.0
      );
      res.json({
        pairKey: pred.pairKey,
        metrics,
        ...(includeResidues
          ? {
              structure1: {
                ...r1.structure,
                gnn: r1.gnn,
              },
              structure2: {
                ...r2.structure,
                gnn: r2.gnn,
              },
            }
          : {}),
      });
    } catch (err) {
      res.status(400).json({
        error: err instanceof Error ? err.message : 'GNN pair evaluation failed',
      });
    }
  };

  const handleFeatures = (req: Request, res: Response) => {
    try {
      const rawPair = String(req.params.pair ?? '');
      const parts = rawPair.replace(/\+/g, '__').replace(/-/g, '__').split('__');
      if (parts.length !== 2) {
        res.status(400).json({
          error: 'Pair path parameter must be formatted as GENE1__GENE2 or GENE1+GENE2.',
        });
        return;
      }
      const pred = buildPredictionObject(
        rt,
        parts[0].trim().toUpperCase(),
        parts[1].trim().toUpperCase()
      );
      res.json({
        ...pred,
        featureGroups: rt.featureCtx.featureNames.map((name, i) => ({
          name,
          group: rt.featureCtx.featureGroups[i],
          value: pred.features[name],
        })),
        dataMode: rt.dataMode,
      });
    } catch (err) {
      res.status(400).json({
        error: err instanceof Error ? err.message : 'Feature lookup failed',
      });
    }
  };

  const handleTopPairs = (req: Request, res: Response) => {
    const typeFilter = String(req.query.type ?? 'all');
    const n = Math.min(250, Math.max(1, Number(req.query.n ?? 100) || 100));
    const filtered =
      typeFilter === 'all'
        ? rt.rankedPairs
        : rt.rankedPairs.filter((p) => p.mechanismType === typeFilter);
    res.json({
      pairs: filtered.slice(0, n),
      totalCandidatePairs: rt.totalPairs,
      geneCount: rt.genes.length,
      dataMode: rt.dataMode,
      datasetId: rt.datasetId,
      seed: rt.seed,
      modelVersion: rt.modelVersion,
    });
  };

  const handleMetrics = (_req: Request, res: Response) => {
    res.json({
      datasetId: rt.datasetId,
      dataMode: rt.dataMode,
      seed: rt.seed,
      modelVersion: rt.modelVersion,
      geminiModel: resolveAndValidateGeminiModel(),
      geneCount: rt.genes.length,
      totalCandidatePairs: rt.totalPairs,
      qcStats: rt.preprocessed.qcStats,
      pcaVarianceExplained: rt.preprocessed.pcaVarianceExplained,
      hvgCount: rt.preprocessed.hvgGenes.length,
      clinicalPairCount: rt.labelMap.size,
      benchmarks: rt.benchmarks,
      rocCurves: rt.rocCurves,
      ablation: rt.ablation,
      topFeatureImportances: rt.topFeatureImportances,
      binSpecs: rt.binSpecs,
    });
  };

  const handleReport = async (req: Request, res: Response) => {
    try {
      const { gene1, gene2, enableWebSearch } = req.body ?? {};
      if (!gene1 || !gene2) {
        res.status(400).json({ error: 'Both gene1 and gene2 are required.' });
        return;
      }
      const g1 = String(gene1).trim().toUpperCase();
      const g2 = String(gene2).trim().toUpperCase();
      const pred = buildPredictionObject(rt, g1, g2);
      const structuralGnn = await resolvePairStructuralGnn(
        pred.gene1,
        pred.gene2,
        pred.mechanismType,
        pred.features.vector_score ?? 1.0
      );
      const report = await generateGroundedRagReport({
        prediction: pred,
        gene1Info: getOrSynthesizeGeneInfo(pred.gene1),
        gene2Info: getOrSynthesizeGeneInfo(pred.gene2),
        binSpecs: rt.binSpecs,
        datasetId: rt.datasetId,
        seed: rt.seed,
        modelVersion: rt.modelVersion,
        dataMode: rt.dataMode,
        enableWebSearch: Boolean(enableWebSearch),
        structuralGnn,
      });
      res.json(report);
    } catch (err) {
      res.status(400).json({
        error: err instanceof Error ? err.message : 'Report generation failed',
      });
    }
  };

  const handleExportLeaderboardCsv = (req: Request, res: Response) => {
    const typeFilter = String(req.query.type ?? 'all');
    const n = Math.min(250, Math.max(1, Number(req.query.n ?? 100) || 100));
    const rows = (
      typeFilter === 'all'
        ? rt.rankedPairs
        : rt.rankedPairs.filter((p) => p.mechanismType === typeFilter)
    ).slice(0, n);

    const header = [
      'rank',
      'total_pairs',
      'gene1',
      'gene2',
      'ndcg_score',
      'score',
      'mechanism_type',
      'in_clinical_trial',
      'clinical_stage',
      'clinical_drugs',
      'needs_verification',
      'score_safe',
      'vector_score',
      'pathway_cooccur',
      'double_ratio_overall',
      'double_ratio_max',
      'single_ratio_max',
      'pearson_overall',
      'dataset',
      'seed',
      'model_version',
    ];
    const lines = [header.join(',')];
    for (const r of rows) {
      const f = r.features;
      lines.push(
        [
          r.rank,
          r.totalPairs,
          r.gene1,
          r.gene2,
          r.ndcgScore ?? computePairNdcgScore(r.rank, r.score),
          r.score,
          r.mechanismType,
          r.inClinicalTrial,
          r.clinicalStage ?? '',
          r.clinicalDrugs.join('; '),
          r.needsVerification,
          f.score_safe ?? '',
          f.vector_score ?? '',
          f.pathway_cooccur ?? '',
          f.double_ratio_overall ?? '',
          f.double_ratio_max ?? '',
          f.single_ratio_max ?? '',
          f.pearson_overall ?? '',
          rt.datasetId,
          rt.seed,
          rt.modelVersion,
        ]
          .map(csvEscape)
          .join(',')
      );
    }
    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader(
      'Content-Disposition',
      `attachment; filename="bspai_leaderboard_${typeFilter}_top${n}.csv"`
    );
    res.send(lines.join('\n'));
  };

  const buildLeaderboardXlsxBuffer = async (typeFilter: string, n: number) => {
    const rows = (
      typeFilter === 'all'
        ? rt.rankedPairs
        : rt.rankedPairs.filter((p) => p.mechanismType === typeFilter)
    ).slice(0, n);

    const workbook = new ExcelJS.Workbook();
    workbook.creator = 'Ubhaya-Bandha (Dual Binding)';
    workbook.created = new Date();

    const sheet = workbook.addWorksheet(`Top ${n} Target Pairs`, {
      views: [{ state: 'frozen', ySplit: 1 }],
    });

    sheet.columns = [
      { header: 'Rank', key: 'rank', width: 9 },
      { header: 'Target Pair', key: 'pair', width: 20 },
      { header: 'Gene 1', key: 'gene1', width: 12 },
      { header: 'Gene 2', key: 'gene2', width: 12 },
      { header: 'Pairwise Druggability Score', key: 'score', width: 26 },
      { header: 'Mechanism Type', key: 'mechanism', width: 22 },
      { header: 'In Clinical Trial', key: 'inTrial', width: 16 },
      { header: 'Clinical Stage', key: 'stage', width: 15 },
      { header: 'Clinical Drugs', key: 'drugs', width: 30 },
      { header: 'Safety (score_safe)', key: 'scoreSafe', width: 18 },
      { header: 'Embedding (vector_score)', key: 'vectorScore', width: 22 },
      { header: 'Double Positive Max', key: 'doubleMax', width: 19 },
      { header: 'Pearson Overall', key: 'pearson', width: 16 },
    ];

    const headerRow = sheet.getRow(1);
    headerRow.font = { bold: true, color: { argb: 'FFFFFFFF' } };
    headerRow.fill = {
      type: 'pattern',
      pattern: 'solid',
      fgColor: { argb: 'FF2D241E' },
    };
    headerRow.alignment = { vertical: 'middle', horizontal: 'left' };
    headerRow.height = 24;

    for (const r of rows) {
      const f = r.features;
      sheet.addRow({
        rank: r.rank,
        pair: `${r.gene1} + ${r.gene2}`,
        gene1: r.gene1,
        gene2: r.gene2,
        score: Number(r.score.toFixed(4)),
        mechanism: r.mechanismType,
        inTrial: r.inClinicalTrial ? 'Yes' : 'Exploratory',
        stage: r.clinicalStage ?? '',
        drugs: r.clinicalDrugs.join('; '),
        scoreSafe: Number((f.score_safe ?? 0).toFixed(4)),
        vectorScore: Number((f.vector_score ?? 0).toFixed(4)),
        doubleMax: Number((f.double_ratio_max ?? 0).toFixed(4)),
        pearson: Number((f.pearson_overall ?? 0).toFixed(4)),
      });
    }

    const metaSheet = workbook.addWorksheet('Summary & Provenance');
    metaSheet.columns = [
      { header: 'Field', key: 'field', width: 28 },
      { header: 'Value', key: 'value', width: 48 },
    ];
    metaSheet.getRow(1).font = { bold: true };
    metaSheet.addRows([
      { field: 'Application', value: 'Ubhaya-Bandha (Dual Binding)' },
      { field: 'Dataset ID', value: rt.datasetId },
      { field: 'Data Mode', value: rt.dataMode },
      { field: 'Seed', value: rt.seed },
      { field: 'Model Version', value: rt.modelVersion },
      { field: 'Total Candidate Pairs', value: rt.totalPairs },
      { field: 'Mechanism Filter', value: typeFilter },
      { field: 'Rows Exported', value: rows.length },
      { field: 'Generated At (UTC)', value: new Date().toISOString() },
    ]);

    const rawBuffer = await workbook.xlsx.writeBuffer();
    const safeFilter = typeFilter.replace(/[^a-zA-Z0-9_-]+/g, '_');
    const filename = `ubhaya_bandha_leaderboard_${safeFilter}_top${n}.xlsx`;
    return { buffer: Buffer.from(rawBuffer), filename };
  };

  const handleExportLeaderboardXlsx = async (req: Request, res: Response) => {
    try {
      const typeFilter = String(req.query.type ?? 'all');
      const n = Math.min(250, Math.max(1, Number(req.query.n ?? 100) || 100));
      const { buffer, filename } = await buildLeaderboardXlsxBuffer(typeFilter, n);

      res.setHeader(
        'Content-Type',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      );
      res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
      res.setHeader('Cache-Control', 'no-store');
      res.send(buffer);
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Excel export failed',
      });
    }
  };

  const handleExportLeaderboardXlsxJson = async (req: Request, res: Response) => {
    try {
      const typeFilter = String(req.query.type ?? 'all');
      const n = Math.min(250, Math.max(1, Number(req.query.n ?? 100) || 100));
      const { buffer, filename } = await buildLeaderboardXlsxBuffer(typeFilter, n);

      res.setHeader('Cache-Control', 'no-store');
      res.json({
        filename,
        mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        base64: buffer.toString('base64'),
      });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Excel export failed',
      });
    }
  };

  const handleExportFeaturesCsv = (req: Request, res: Response) => {
    try {
      const rawPair = String(req.params.pair ?? '');
      const parts = rawPair.replace(/\+/g, '__').replace(/-/g, '__').split('__');
      if (parts.length !== 2) {
        res.status(400).json({ error: 'Invalid pair format' });
        return;
      }
      const pred = buildPredictionObject(rt, parts[0], parts[1]);
      const lines = ['feature,group,value,pair,score,rank,total_pairs,dataset,seed,model_version'];
      rt.featureCtx.featureNames.forEach((name, idx) => {
        lines.push(
          [
            name,
            rt.featureCtx.featureGroups[idx],
            pred.features[name],
            pred.pairKey,
            pred.score,
            pred.rank,
            pred.totalPairs,
            rt.datasetId,
            rt.seed,
            rt.modelVersion,
          ]
            .map(csvEscape)
            .join(',')
        );
      });
      res.setHeader('Content-Type', 'text/csv; charset=utf-8');
      res.setHeader(
        'Content-Disposition',
        `attachment; filename="bspai_features_${pred.pairKey}.csv"`
      );
      res.send(lines.join('\n'));
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : 'Export failed' });
    }
  };

  app.get('/api/health', (_req, res) =>
    res.json({
      ok: true,
      datasetId: rt.datasetId,
      dataMode: rt.dataMode,
      seed: rt.seed,
      modelVersion: rt.modelVersion,
      geneCount: rt.genes.length,
      totalCandidatePairs: rt.totalPairs,
    })
  );

  for (const prefix of ['/api', '']) {
    app.get(`${prefix}/genes`, handleGenes);
    app.post(`${prefix}/predict`, handlePredict);
    app.get(`${prefix}/structure/:gene`, handleStructure);
    app.post(`${prefix}/gnn/pair`, handleGnnPair);
    app.get(`${prefix}/features/:pair`, handleFeatures);
    app.get(`${prefix}/top-pairs`, handleTopPairs);
    app.get(`${prefix}/metrics`, handleMetrics);
    app.post(`${prefix}/report`, handleReport);
    app.get(`${prefix}/export/leaderboard.csv`, handleExportLeaderboardCsv);
    app.get(`${prefix}/export/leaderboard.xlsx`, handleExportLeaderboardXlsx);
    app.get(`${prefix}/export/leaderboard-xlsx-json`, handleExportLeaderboardXlsxJson);
    app.get(`${prefix}/export/features/:pair.csv`, handleExportFeaturesCsv);
  }

  return app;
}

async function startServer() {
  const validatedModel = resolveAndValidateGeminiModel();
  const seed = Number(process.env.BSPAI_SEED ?? 42);
  console.log(`[BSPAI] Initializing pipeline (seed=${seed}, geminiModel=${validatedModel})...`);
  const runtime = runAndPersistPipeline(seed, false);
  console.log(
    `[BSPAI] Ready: ${runtime.genes.length} genes, ${runtime.totalPairs.toLocaleString()} candidate pairs.`
  );

  const app = createBspaiApp(runtime);
  const port = Number(process.env.PORT ?? 3000);

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(port, '0.0.0.0', () => {
    console.log(`[BSPAI] Server listening on http://0.0.0.0:${port}`);
  });
}

const isEntrypoint =
  process.argv[1] && path.resolve(process.argv[1]).endsWith('server.ts');
if (isEntrypoint) {
  startServer().catch((err) => {
    console.error('[BSPAI] Fatal server error:', err);
    process.exit(1);
  });
}