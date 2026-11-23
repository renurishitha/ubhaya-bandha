import { useEffect, useState } from 'react';
import { bspaiClient } from './api/bspaiClient.ts';
import { GlossaryView } from './components/GlossaryView.tsx';
import { HomeView } from './components/HomeView.tsx';
import { LeaderboardView } from './components/LeaderboardView.tsx';
import { ModelCardView } from './components/ModelCardView.tsx';
import { PredictView } from './components/PredictView.tsx';
import { TopBar } from './components/TopBar.tsx';
import {
  UserAuthHistory,
  type SearchHistoryEntry,
} from './components/UserAuthHistory.tsx';
import type { ActiveTab, GeneInfo, MetricsResponse } from './types.ts';

const CURRENT_USER_KEY = 'bspai_current_user_v1';
const HISTORY_STORAGE_KEY = 'bspai_history_v1';

function loadInitialUser(): string | null {
  try {
    return localStorage.getItem(CURRENT_USER_KEY);
  } catch {
    return null;
  }
}

function loadInitialHistory(): Record<string, SearchHistoryEntry[]> {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Record<string, SearchHistoryEntry[]>) : {};
  } catch {
    return {};
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('home');
  const [genes, setGenes] = useState<GeneInfo[]>([]);
  const [totalPairs, setTotalPairs] = useState<number>(0);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [selectedGene1, setSelectedGene1] = useState<string>('CD274');
  const [selectedGene2, setSelectedGene2] = useState<string>('CTLA4');
  const [inspectNonce, setInspectNonce] = useState<number>(0);
  const [bootError, setBootError] = useState<string | null>(null);

  const [currentUserEmail, setCurrentUserEmail] = useState<string | null>(() =>
    loadInitialUser()
  );
  const [historyByUser, setHistoryByUser] = useState<
    Record<string, SearchHistoryEntry[]>
  >(() => loadInitialHistory());

  useEffect(() => {
    try {
      if (currentUserEmail) {
        localStorage.setItem(CURRENT_USER_KEY, currentUserEmail);
      } else {
        localStorage.removeItem(CURRENT_USER_KEY);
      }
    } catch {
      // Ignore storage quota/access issues
    }
  }, [currentUserEmail]);

  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(historyByUser));
    } catch {
      // Ignore storage quota/access issues
    }
  }, [historyByUser]);

  const recordSearchForUser = (g1: string, g2: string) => {
    if (!currentUserEmail) return;
    const clean1 = g1.trim().toUpperCase();
    const clean2 = g2.trim().toUpperCase();
    if (!clean1 || !clean2) return;

    const entry: SearchHistoryEntry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      gene1: clean1,
      gene2: clean2,
      searchedAt: new Date().toISOString(),
    };

    setHistoryByUser((prev) => {
      const existing = prev[currentUserEmail] ?? [];
      const deduped = existing.filter(
        (item) => !(item.gene1 === clean1 && item.gene2 === clean2)
      );
      return {
        ...prev,
        [currentUserEmail]: [entry, ...deduped].slice(0, 30),
      };
    });
  };

  const handleClearHistory = (email: string) => {
    setHistoryByUser((prev) => ({
      ...prev,
      [email]: [],
    }));
  };

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([
      bspaiClient.getGenes(),
      bspaiClient.getMetrics(),
    ]).then(([genesOutcome, metricsOutcome]) => {
      if (cancelled) return;

      if (genesOutcome.status === 'fulfilled') {
        const genesRes = genesOutcome.value;
        const fullSymbols: string[] = genesRes.allGeneSymbols ?? [];
        const loadedGenes: GeneInfo[] = genesRes.genes ?? [];
        const byName = new Map(loadedGenes.map((g) => [g.gene, g]));
        const allGenes: GeneInfo[] =
          fullSymbols.length > loadedGenes.length
            ? fullSymbols.map(
                (sym) =>
                  byName.get(sym) ?? {
                    gene: sym,
                    category: 'candidate',
                    cell_bias: 'epithelial',
                    description: 'Curated membrane target candidate',
                  }
              )
            : loadedGenes;

        setGenes(allGenes);
        setTotalPairs(genesRes.totalCandidatePairs ?? 0);
      }

      if (metricsOutcome.status === 'fulfilled') {
        const metricsRes = metricsOutcome.value;
        setMetrics(metricsRes as MetricsResponse);
        setTotalPairs((prev) => prev || metricsRes.totalCandidatePairs || 0);
      }

      if (genesOutcome.status === 'rejected' && metricsOutcome.status === 'rejected') {
        const err = genesOutcome.reason;
        setBootError(err instanceof Error ? err.message : 'Failed to initialize backend API');
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const navigateTab = (tab: ActiveTab) => {
    setActiveTab(tab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleInspectPair = (g1: string, g2: string) => {
    setSelectedGene1(g1);
    setSelectedGene2(g2);
    setInspectNonce((n) => n + 1);
    recordSearchForUser(g1, g2);
    setActiveTab('predict');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen flex flex-col text-[#231D18]">
      <TopBar activeTab={activeTab} onSelectTab={navigateTab} />

      <main className="flex-1 max-w-[1240px] w-full mx-auto px-6 py-10 space-y-8">
        {bootError && (
          <div className="bg-[#FDF4F0] border border-[#E5C1B5] rounded-xl p-4 text-sm text-[#853826]">
            Initialization error: {bootError}
          </div>
        )}

        {activeTab === 'home' && (
          <HomeView
            metrics={metrics}
            onNavigateTab={navigateTab}
            onInspectPair={handleInspectPair}
          />
        )}

        {activeTab === 'predict' && (
          <PredictView
            genes={genes}
            totalPairs={totalPairs}
            selectedGene1={selectedGene1}
            selectedGene2={selectedGene2}
            inspectNonce={inspectNonce}
            onChangePair={(g1, g2) => {
              setSelectedGene1(g1);
              setSelectedGene2(g2);
              recordSearchForUser(g1, g2);
            }}
          />
        )}

        {activeTab === 'leaderboard' && (
          <LeaderboardView onInspectPair={handleInspectPair} />
        )}

        {activeTab === 'model-card' && <ModelCardView metrics={metrics} />}

        {activeTab === 'glossary' && <GlossaryView />}

        <UserAuthHistory
          currentUserEmail={currentUserEmail}
          onUserChange={setCurrentUserEmail}
          historyByUser={historyByUser}
          onSelectHistoryItem={handleInspectPair}
          onClearHistory={handleClearHistory}
        />
      </main>

      <footer className="border-t border-[#E3D9CA] bg-[#FAF6F0]/80">
        <div className="max-w-[1240px] mx-auto px-6 py-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-[#6E6356]">
          <span>
            <strong className="font-semibold text-[#231D18]">Ubhaya-Bandha (Dual Binding)</strong> · Bispecific antibody target-pair discovery workbench
          </span>
          <div className="flex items-center gap-5">
            <button
              type="button"
              onClick={() => navigateTab('home')}
              className="hover:text-[#231D18] transition-colors cursor-pointer"
            >
              Home
            </button>
            <button
              type="button"
              onClick={() => navigateTab('predict')}
              className="hover:text-[#231D18] transition-colors cursor-pointer"
            >
              Predict pairs
            </button>
            <button
              type="button"
              onClick={() => navigateTab('leaderboard')}
              className="hover:text-[#231D18] transition-colors cursor-pointer"
            >
              Leaderboard
            </button>
            <button
              type="button"
              onClick={() => navigateTab('model-card')}
              className="hover:text-[#231D18] transition-colors cursor-pointer"
            >
              Model Card
            </button>
            <button
              type="button"
              onClick={() => navigateTab('glossary')}
              className="hover:text-[#231D18] transition-colors cursor-pointer"
            >
              Glossary
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}