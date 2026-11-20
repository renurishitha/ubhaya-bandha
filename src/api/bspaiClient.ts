import type {
  AlphaFoldStructurePayload,
  GeneInfo,
  MetricsResponse,
  PairGnnMetrics,
  PredictionResult,
  RagReportResult,
} from '../types.ts';

const API_BASE = (
  (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_API_BASE_URL ?? ''
)
  .trim()
  .replace(/\/+$/, '');

function externalUrl(path: string): string {
  return `${API_BASE}${path}`;
}

async function readJsonSafe(res: Response): Promise<Record<string, unknown> | null> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return null;
  }
}

async function requestLocalJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  const body = await readJsonSafe(res);
  if (!res.ok || !body) {
    const msg =
      typeof body?.error === 'string'
        ? body.error
        : typeof body?.detail === 'string'
          ? body.detail
          : `HTTP ${res.status} (${path})`;
    throw new Error(msg);
  }
  return body as T;
}

/**
 * Queries an external backend (when VITE_API_BASE_URL is set), trying both
 * `/api/...` and prefix-less `/...` routes to support FastAPI/Flask backends.
 */
async function tryExternalJson(
  path: string,
  init?: RequestInit
): Promise<Record<string, unknown> | null> {
  if (!API_BASE) return null;

  const candidates = path.startsWith('/api/')
    ? [externalUrl(path), externalUrl(path.slice(4))]
    : [externalUrl(path)];

  for (const url of candidates) {
    try {
      const res = await fetch(url, init);
      if (res.status === 404 || res.status === 405) {
        continue;
      }
      const body = await readJsonSafe(res);
      if (res.ok && body) {
        return body;
      }
    } catch {
      // Network/CORS error on external URL — try next candidate or local fallback
    }
  }
  return null;
}

function toNumber(val: unknown, fallback = 0): number {
  const n = Number(val);
  return Number.isFinite(n) ? n : fallback;
}

function normalizePrediction(
  raw: Record<string, unknown>,
  gene1: string,
  gene2: string
): PredictionResult {
  const g1 = String(raw.gene1 ?? raw.gene_a ?? raw.target1 ?? raw.target_a ?? gene1).toUpperCase();
  const g2 = String(raw.gene2 ?? raw.gene_b ?? raw.target2 ?? raw.target_b ?? gene2).toUpperCase();
  const score = toNumber(
    raw.score ??
      raw.druggability_score ??
      raw.pairwise_score ??
      raw.predicted_score ??
      raw.prediction ??
      raw.ndcgScore ??
      raw.ndcg_score,
    0
  );
  const rank = Math.max(1, Math.round(toNumber(raw.rank ?? raw.global_rank, 1)));
  const totalPairs = Math.max(rank, Math.round(toNumber(raw.totalPairs ?? raw.total_pairs, 53301)));
  const rawMech = String(raw.mechanismType ?? raw.mechanism_type ?? raw.mechanism ?? 'tumor-immunotherapy');
  const mechanismType =
    rawMech === 'bridge-two-cells' || rawMech === 'TAA+TAA' || rawMech === 'tumor-immunotherapy'
      ? rawMech
      : 'tumor-immunotherapy';

  return {
    pairKey: String(raw.pairKey ?? raw.pair_key ?? `${g1}__${g2}`),
    gene1: g1,
    gene2: g2,
    score,
    ndcgScore: raw.ndcgScore !== undefined ? toNumber(raw.ndcgScore) : undefined,
    rank,
    totalPairs,
    mechanismType,
    inClinicalTrial: Boolean(raw.inClinicalTrial ?? raw.in_clinical_trial ?? false),
    clinicalStage:
      raw.clinicalStage !== undefined && raw.clinicalStage !== null
        ? toNumber(raw.clinicalStage)
        : raw.clinical_stage !== undefined && raw.clinical_stage !== null
          ? toNumber(raw.clinical_stage)
          : null,
    clinicalDrugs: Array.isArray(raw.clinicalDrugs)
      ? (raw.clinicalDrugs as string[])
      : Array.isArray(raw.clinical_drugs)
        ? (raw.clinical_drugs as string[])
        : [],
    needsVerification: Boolean(raw.needsVerification ?? raw.needs_verification ?? false),
    features:
      raw.features && typeof raw.features === 'object'
        ? (raw.features as Record<string, number>)
        : {},
    gene1Info: raw.gene1Info as GeneInfo | undefined,
    gene2Info: raw.gene2Info as GeneInfo | undefined,
    structuralGnn: raw.structuralGnn as PairGnnMetrics | undefined,
  };
}

function normalizeReport(
  raw: Record<string, unknown>,
  gene1: string,
  gene2: string
): RagReportResult {
  const reportMarkdown = String(
    raw.reportMarkdown ??
      raw.report_markdown ??
      raw.report ??
      raw.explanation ??
      raw.analysis ??
      raw.answer ??
      raw.response ??
      ''
  );
  const plainEnglishReport = String(
    raw.plainEnglishReport ??
      raw.plain_english_report ??
      raw.plain_summary ??
      raw.summary ??
      reportMarkdown
  );
  const score = toNumber(
    raw.score ?? raw.druggability_score ?? raw.pairwise_score ?? raw.predicted_score,
    0
  );

  return {
    pairKey: String(raw.pairKey ?? raw.pair_key ?? `${gene1}__${gene2}`),
    gene1: String(raw.gene1 ?? raw.gene_a ?? gene1).toUpperCase(),
    gene2: String(raw.gene2 ?? raw.gene_b ?? gene2).toUpperCase(),
    score,
    rank: Math.max(1, Math.round(toNumber(raw.rank, 1))),
    totalPairs: Math.max(1, Math.round(toNumber(raw.totalPairs ?? raw.total_pairs, 53301))),
    mechanismType: (raw.mechanismType as RagReportResult['mechanismType']) ?? 'tumor-immunotherapy',
    discretizedFeatures: Array.isArray(raw.discretizedFeatures)
      ? (raw.discretizedFeatures as RagReportResult['discretizedFeatures'])
      : [],
    plainEnglishReport,
    reportMarkdown: reportMarkdown || plainEnglishReport,
    provenance: (raw.provenance as RagReportResult['provenance']) ?? {
      dataset: 'EXTERNAL_BACKEND',
      seed: 42,
      modelVersion: 'External-API',
      dataMode: 'EXTERNAL',
      generationMode: 'external',
      geminiModel: '',
      webSearchEnabled: false,
      generatedAt: new Date().toISOString(),
    },
  };
}

export interface GenesApiResponse {
  genes: GeneInfo[];
  allGeneSymbols?: string[];
  geneCount: number;
  totalCandidatePairs: number;
}

export interface TopPairsApiResponse {
  pairs: PredictionResult[];
  totalCandidatePairs: number;
  geneCount?: number;
}

export const bspaiClient = {
  getGenes: async (): Promise<GenesApiResponse> => {
    const ext = await tryExternalJson('/api/genes');
    if (ext && Array.isArray(ext.genes)) {
      return ext as unknown as GenesApiResponse;
    }
    return requestLocalJson<GenesApiResponse>('/api/genes');
  },

  getMetrics: async (): Promise<MetricsResponse> => {
    const ext = await tryExternalJson('/api/metrics');
    if (ext && Array.isArray(ext.benchmarks) && Array.isArray(ext.rocCurves)) {
      return ext as unknown as MetricsResponse;
    }
    return requestLocalJson<MetricsResponse>('/api/metrics');
  },

  predictPair: async (
    gene1: string,
    gene2: string,
    liveAlphaFold = false
  ): Promise<PredictionResult> => {
    let raw: Record<string, unknown> | null = null;
    if (API_BASE) {
      raw = await tryExternalJson('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          gene1,
          gene2,
          gene_a: gene1,
          gene_b: gene2,
          target1: gene1,
          target2: gene2,
          liveAlphaFold,
        }),
      });
    }

    if (!raw) {
      raw = await requestLocalJson<Record<string, unknown>>('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gene1, gene2, liveAlphaFold }),
      });
    }

    const pred = normalizePrediction(raw, gene1, gene2);
    if (!pred.structuralGnn) {
      try {
        pred.structuralGnn = await requestLocalJson<PairGnnMetrics>('/api/gnn/pair', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ gene1: pred.gene1, gene2: pred.gene2 }),
        });
      } catch {
        // Optional 3D GNN hydration — safe to skip if gene isn't in local catalog
      }
    }
    return pred;
  },

  getTopPairs: async (type = 'all', n = 100): Promise<TopPairsApiResponse> => {
    const query = `/api/top-pairs?type=${encodeURIComponent(type)}&n=${n}`;
    const ext = await tryExternalJson(query);
    if (ext && Array.isArray(ext.pairs)) {
      const pairs = (ext.pairs as Record<string, unknown>[]).map((row, idx) => {
        const g1 = String(row.gene1 ?? row.gene_a ?? row.target1 ?? 'GENE1');
        const g2 = String(row.gene2 ?? row.gene_b ?? row.target2 ?? 'GENE2');
        const p = normalizePrediction(row, g1, g2);
        if (!row.rank) p.rank = idx + 1;
        return p;
      });
      return {
        pairs,
        totalCandidatePairs: toNumber(ext.totalCandidatePairs ?? ext.total_pairs, pairs.length),
      };
    }
    return requestLocalJson<TopPairsApiResponse>(query);
  },

  getReport: async (
    gene1: string,
    gene2: string,
    enableWebSearch = false
  ): Promise<RagReportResult> => {
    let raw: Record<string, unknown> | null = null;
    if (API_BASE) {
      raw = await tryExternalJson('/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          gene1,
          gene2,
          gene_a: gene1,
          gene_b: gene2,
          target1: gene1,
          target2: gene2,
          enableWebSearch,
        }),
      });
    }

    if (!raw) {
      raw = await requestLocalJson<Record<string, unknown>>('/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gene1, gene2, enableWebSearch }),
      });
    }

    return normalizeReport(raw, gene1, gene2);
  },

  getAlphaFoldStructure: async (
    gene: string,
    live = false
  ): Promise<AlphaFoldStructurePayload> => {
    const path = `/api/structure/${encodeURIComponent(gene)}${live ? '?live=1' : ''}`;
    const ext = await tryExternalJson(path);
    if (ext && Array.isArray(ext.residues)) {
      return ext as unknown as AlphaFoldStructurePayload;
    }
    return requestLocalJson<AlphaFoldStructurePayload>(path);
  },

  getLeaderboardExcelUrl: (type = 'all', n = 100) =>
    `/api/export/leaderboard.xlsx?type=${encodeURIComponent(type)}&n=${n}`,

  downloadLeaderboardExcel: async (type = 'all', n = 100): Promise<void> => {
    const XLSX_MIME =
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
    const safeFilter = type.replace(/[^a-zA-Z0-9_-]+/g, '_');
    const fallbackFilename = `ubhaya_bandha_leaderboard_${safeFilter}_top${n}.xlsx`;

    const isValidXlsxBytes = (bytes: Uint8Array) =>
      bytes.length > 4 && bytes[0] === 0x50 && bytes[1] === 0x4b;

    const triggerBlobDownload = (bytes: Uint8Array, filename: string) => {
      const arrayBuffer = new ArrayBuffer(bytes.byteLength);
      new Uint8Array(arrayBuffer).set(bytes);
      const blob = new Blob([arrayBuffer], { type: XLSX_MIME });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.setTimeout(() => URL.revokeObjectURL(url), 1500);
    };

    // 1. Primary path: fetch binary XLSX using the authenticated same-origin fetch session
    try {
      const res = await fetch(
        `/api/export/leaderboard.xlsx?type=${encodeURIComponent(type)}&n=${n}`,
        {
          method: 'GET',
          credentials: 'same-origin',
          headers: { Accept: XLSX_MIME },
        }
      );
      const contentType = (res.headers.get('content-type') ?? '').toLowerCase();
      if (res.ok && !contentType.includes('text/html')) {
        const bytes = new Uint8Array(await res.arrayBuffer());
        if (isValidXlsxBytes(bytes)) {
          const disp = res.headers.get('content-disposition') ?? '';
          const match = disp.match(/filename="?([^"]+)"?/i);
          triggerBlobDownload(bytes, match?.[1] ?? fallbackFilename);
          return;
        }
      }
    } catch {
      // Fall through to JSON base64 endpoint
    }

    // 2. Fallback path: JSON base64 endpoint (immune to binary/extension proxy interception)
    const payload = await requestLocalJson<{
      filename?: string;
      base64?: string;
    }>(
      `/api/export/leaderboard-xlsx-json?type=${encodeURIComponent(type)}&n=${n}`,
      { credentials: 'same-origin' }
    );

    if (!payload.base64) {
      throw new Error('Server did not return Excel workbook data.');
    }

    const binary = window.atob(payload.base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    if (!isValidXlsxBytes(bytes)) {
      throw new Error('Downloaded Excel workbook failed integrity verification.');
    }

    triggerBlobDownload(bytes, payload.filename ?? fallbackFilename);
  },
};
