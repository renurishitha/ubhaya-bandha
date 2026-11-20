import { ArrowUpRight, Download, Search } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { bspaiClient } from '../api/bspaiClient.ts';
import type { MechanismType, PredictionResult } from '../types.ts';

interface LeaderboardViewProps {
  onInspectPair: (gene1: string, gene2: string) => void;
}

type FilterOption = 'all' | MechanismType;

const FILTER_OPTIONS: Array<{ id: FilterOption; label: string }> = [
  { id: 'all', label: 'All Mechanisms' },
  { id: 'bridge-two-cells', label: 'Bridge Two Cells' },
  { id: 'TAA+TAA', label: 'TAA + TAA' },
  { id: 'tumor-immunotherapy', label: 'Tumor Immunotherapy' },
];

const MECHANISM_TEXT_COLOR: Record<string, string> = {
  'bridge-two-cells': 'text-[#4F6D4A] font-semibold',
  'TAA+TAA': 'text-[#8B5E3C] font-semibold',
  'tumor-immunotherapy': 'text-[#7A543E] font-semibold',
};

const STAGE_LABELS: Record<number, string> = {
  1: 'Stage 1 · Approved/BLA',
  2: 'Stage 2 · Phase 3 / 2-3',
  3: 'Stage 3 · Phase 1 / 2',
  4: 'Stage 4 · Preclinical/IND',
  5: 'Stage 5 · Discontinued',
};

export function LeaderboardView({ onInspectPair }: LeaderboardViewProps) {
  const [filter, setFilter] = useState<FilterOption>('all');
  const [clinicalOnly, setClinicalOnly] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [pairs, setPairs] = useState<PredictionResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleExportExcel = async () => {
    if (exporting) return;
    setExporting(true);
    setError(null);
    try {
      await bspaiClient.downloadLeaderboardExcel(filter, 100);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to export Excel workbook'
      );
    } finally {
      setExporting(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    bspaiClient
      .getTopPairs(filter, 100)
      .then((data) => {
        if (cancelled) return;
        setPairs(data.pairs ?? []);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Failed to load leaderboard');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [filter]);

  const visiblePairs = useMemo(() => {
    const q = searchQuery.trim().toUpperCase();
    return pairs.filter((p) => {
      if (clinicalOnly && !p.inClinicalTrial) return false;
      if (!q) return true;
      return (
        p.gene1.toUpperCase().includes(q) ||
        p.gene2.toUpperCase().includes(q) ||
        `${p.gene1} + ${p.gene2}`.toUpperCase().includes(q) ||
        p.mechanismType.toUpperCase().includes(q) ||
        p.clinicalDrugs.some((d) => d.toUpperCase().includes(q))
      );
    });
  }, [pairs, clinicalOnly, searchQuery]);

  const scoreBounds = useMemo(() => {
    if (visiblePairs.length === 0) return { min: 0, max: 1 };
    const scores = visiblePairs.map((p) => p.score);
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    return { min, max: max === min ? min + 1 : max };
  }, [visiblePairs]);

  return (
    <section className="space-y-6 pb-8">
      {/* Header */}
      <div className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl p-7 sm:p-8 flex flex-col lg:flex-row lg:items-end lg:justify-between gap-5">
        <div className="space-y-1.5 max-w-xl">
          <div className="text-xs font-medium text-[#7C5333]">
            Ranked candidate combinations
          </div>
          <h1 className="text-2xl sm:text-3xl font-semibold text-[#231D18]">
            Top-100 Target Pair Leaderboard
          </h1>
          <p className="text-sm text-[#574D43] leading-relaxed">
            Candidate combinations ordered by <strong className="font-semibold text-[#231D18]">Pairwise Druggability Score</strong>. Click any row to open its report and 3D binding simulations.
          </p>
        </div>

        <button
          type="button"
          onClick={handleExportExcel}
          disabled={exporting}
          className="inline-flex items-center justify-center gap-2 px-4 py-2.5 text-xs font-medium text-[#FFFEFC] bg-[#2D241E] hover:bg-[#1E1713] disabled:opacity-60 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Download className="w-3.5 h-3.5" />
          {exporting ? 'Preparing Excel...' : 'Export Top-100 Excel (.xlsx)'}
        </button>
      </div>

      {/* Filters + Search Toolbar */}
      <div className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl p-4 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex flex-wrap items-center gap-1 p-1 bg-[#F4EFE7] rounded-lg border border-[#EAE2D6]">
            {FILTER_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => setFilter(opt.id)}
                className={`px-3 py-1.5 text-xs rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  filter === opt.id
                    ? 'bg-[#FFFEFC] text-[#231D18] font-semibold shadow-xs'
                    : 'text-[#665B4F] hover:text-[#231D18] font-medium'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          <label className="flex items-center gap-2 text-xs font-medium text-[#574D43] cursor-pointer select-none px-1">
            <input
              type="checkbox"
              checked={clinicalOnly}
              onChange={(e) => setClinicalOnly(e.target.checked)}
              className="rounded border-[#DFD4C3] accent-[#2D241E]"
            />
            Clinical benchmarks only
          </label>
        </div>

        <div className="relative w-full lg:w-72">
          <Search className="w-3.5 h-3.5 text-[#8B7F72] absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="search"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search gene or drug..."
            aria-label="Search leaderboard"
            className="w-full h-9 pl-9 pr-3 bg-[#FAF6F0] border border-[#DFD4C3] rounded-lg text-xs text-[#231D18] focus:outline-none focus:bg-[#FFFEFC] focus:border-[#7C5333]"
          />
        </div>
      </div>

      {error && (
        <div className="bg-[#FDF4F0] border border-[#E5C1B5] rounded-xl p-4 text-sm text-[#853826]">
          Failed to load leaderboard: {error}
        </div>
      )}

      {/* Ranked Table */}
      <div className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-[#FAF6F0] border-b border-[#E3D9CA] text-xs font-semibold text-[#574D43]">
                <th className="py-3.5 px-4 w-20">Rank</th>
                <th className="py-3.5 px-4">Target Pair</th>
                <th className="py-3.5 px-4 text-right w-52">
                  Pairwise Druggability Score
                </th>
                <th className="py-3.5 px-4">Mechanism</th>
                <th className="py-3.5 px-4">Clinical Benchmark</th>
                <th className="py-3.5 px-4 text-right">Safety</th>
                <th className="py-3.5 px-4 text-right">Double Pos.</th>
                <th className="py-3.5 px-4 text-right">Report</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#EFE4D4] text-sm">
              {loading ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-[#786A5B]">
                    Loading ranked candidate pairs...
                  </td>
                </tr>
              ) : visiblePairs.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-[#786A5B]">
                    No target pairs match your current filter.
                  </td>
                </tr>
              ) : (
                visiblePairs.map((pair) => {
                  const norm =
                    (pair.score - scoreBounds.min) /
                    (scoreBounds.max - scoreBounds.min);
                  const barPct = Math.max(12, Math.min(100, norm * 100));
                  return (
                    <tr
                      key={pair.pairKey}
                      onClick={() => onInspectPair(pair.gene1, pair.gene2)}
                      className="hover:bg-[#FAF4EA] transition-colors cursor-pointer"
                    >
                      <td className="py-3.5 px-4 font-mono text-xs font-bold text-[#8B5E3C]">
                        #{pair.rank}
                      </td>
                      <td className="py-3.5 px-4 font-mono font-semibold text-[#2B231C]">
                        {pair.gene1}{' '}
                        <span className="text-[#B07D52]">+</span>{' '}
                        {pair.gene2}
                      </td>
                      <td className="py-3.5 px-4 text-right font-mono font-semibold text-[#8B5E3C]">
                        <div className="flex flex-col items-end gap-1">
                          <span>{pair.score.toFixed(4)}</span>
                          <div className="w-28 h-1.5 bg-[#EFE4D4] rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full"
                              style={{
                                width: `${barPct}%`,
                                background:
                                  'linear-gradient(90deg, #C49A6C, #8B5E3C)',
                              }}
                            />
                          </div>
                        </div>
                      </td>
                      <td
                        className={`py-3.5 px-4 text-xs ${
                          MECHANISM_TEXT_COLOR[pair.mechanismType] ??
                          'text-[#5E5245]'
                        }`}
                      >
                        {pair.mechanismType}
                      </td>
                      <td className="py-3.5 px-4 text-xs">
                        {pair.inClinicalTrial ? (
                          <div className="space-y-0.5">
                            <div className="font-semibold text-[#4F6D4A]">
                              {STAGE_LABELS[pair.clinicalStage ?? 3] ??
                                `Stage ${pair.clinicalStage}`}
                            </div>
                            <div className="text-[#786A5B] truncate max-w-[240px]">
                              {pair.clinicalDrugs.join(', ')}
                              {pair.needsVerification ? ' · Verify' : ''}
                            </div>
                          </div>
                        ) : (
                          <span className="text-[#8A7B6A]">Exploratory</span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-right font-mono text-xs text-[#4F6D4A] font-medium">
                        {(pair.features.score_safe ?? 0).toFixed(4)}
                      </td>
                      <td className="py-3.5 px-4 text-right font-mono text-xs text-[#5E5245]">
                        {(pair.features.double_ratio_max ?? 0).toFixed(4)}
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <span className="inline-flex items-center gap-1 text-xs font-semibold text-[#8B5E3C]">
                          Open
                          <ArrowUpRight className="w-3.5 h-3.5" />
                        </span>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
