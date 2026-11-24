import { ArrowRight, Info } from 'lucide-react';
import type { ActiveTab, MetricsResponse } from '../types.ts';

interface HomeViewProps {
  metrics: MetricsResponse | null;
  onNavigateTab: (tab: ActiveTab) => void;
  onInspectPair: (gene1: string, gene2: string) => void;
}

const WORKFLOW_STEPS = [
  {
    number: '01',
    title: 'Biological data',
    description:
      'Single-cell gene expression and known bispecific antibody development records.',
  },
  {
    number: '02',
    title: 'Feature engineering',
    description:
      'Expression, tumor safety, biological pathways, correlation, and gene embeddings.',
  },
  {
    number: '03',
    title: 'Pairwise learning',
    description:
      'XGBoost compares candidate pairs to compute a Pairwise Druggability Score.',
  },
  {
    number: '04',
    title: 'Report & 3D check',
    description:
      'Generates a plain-English report and checks 3D binding geometry below.',
  },
];

const CLINICAL_STAGES = [
  {
    stage: 'Stage 1',
    badge: 'Highest weight',
    title: 'Approved / BLA',
    description:
      'FDA/EMA-approved bispecific antibodies or Biologics License Application filings with established clinical efficacy and safety in patients.',
  },
  {
    stage: 'Stage 2',
    badge: 'Late clinical',
    title: 'Phase 3 / Phase 2–3',
    description:
      'Pivotal late-stage clinical trials confirming therapeutic benefit across large patient cohorts prior to regulatory review.',
  },
  {
    stage: 'Stage 3',
    badge: 'Early–mid clinical',
    title: 'Phase 1 / Phase 2',
    description:
      'Human trials evaluating dosing, pharmacokinetics, safety profile, and early signals of anti-tumor activity.',
  },
  {
    stage: 'Stage 4',
    badge: 'Pre-human',
    title: 'Preclinical / IND',
    description:
      'Lab-validated candidates progressing through animal models and IND-enabling toxicology ahead of first-in-human dosing.',
  },
  {
    stage: 'Stage 5',
    badge: 'Cautionary',
    title: 'Discontinued',
    description:
      'Programs halted during development due to toxicity, limited efficacy, or portfolio changes — useful as risk benchmarks.',
  },
];

const QUICK_STARTS = [
  { label: 'CD274 + CTLA4', note: 'Dual checkpoint', g1: 'CD274', g2: 'CTLA4' },
  { label: 'MS4A1 + CD3E', note: 'Two-cell bridge', g1: 'MS4A1', g2: 'CD3E' },
  { label: 'CLDN18 + MS4A1', note: 'Dual membrane', g1: 'CLDN18', g2: 'MS4A1' },
];

export function HomeView({
  onNavigateTab,
  onInspectPair,
}: HomeViewProps) {
  const scrollToAbout = () => {
    const el = document.getElementById('how-bsab-works');
    el?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const goToTab = (tab: ActiveTab) => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
    onNavigateTab(tab);
  };

  const inspectWithTopScroll = (g1: string, g2: string) => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
    onInspectPair(g1, g2);
  };

  return (
    <section className="space-y-14 pb-8">
      {/* 1. Editorial Research Hero */}
      <div className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl p-8 lg:p-12">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-12 items-center">
          <div className="lg:col-span-6 space-y-6">
            <div className="space-y-3">
              <div className="text-xs font-medium text-[#7C5333] tracking-wide">
                Bispecific antibody target-pair discovery
              </div>

              <div className="space-y-1.5">
                <h1 className="text-4xl sm:text-5xl font-semibold text-[#231D18] leading-[1.08]">
                  Ubhaya-Bandha
                  <span className="block text-xl sm:text-2xl font-normal italic text-[#7C5333] mt-1">
                    Dual Binding
                  </span>
                </h1>
              </div>

              <p className="text-base sm:text-lg font-medium text-[#3B322A] pt-1">
                Two targets. One smarter therapeutic strategy.
              </p>
            </div>

            <p className="text-[15px] text-[#574D43] leading-relaxed max-w-xl">
              Evaluate candidate target combinations for bispecific antibodies using single-cell tumor biology, pairwise machine learning, and 3D AlphaFold structural fit — distilled into clear decision reports for R&amp;D teams.
            </p>

            <div className="flex flex-wrap items-center gap-3 pt-1">
              <button
                type="button"
                onClick={() => goToTab('predict')}
                className="inline-flex items-center gap-2 px-6 py-3 text-sm font-medium text-[#FFFEFC] bg-[#2D241E] hover:bg-[#1E1713] rounded-lg transition-colors cursor-pointer"
              >
                Open Pair Explorer
                <ArrowRight className="w-4 h-4" />
              </button>

              <button
                type="button"
                onClick={scrollToAbout}
                className="inline-flex items-center gap-2 px-5 py-3 text-sm font-medium text-[#231D18] bg-[#F7F2EA] hover:bg-[#EFE7DC] border border-[#DFD4C3] rounded-lg transition-colors cursor-pointer"
              >
                How it works
              </button>
            </div>

            <div className="pt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[#7A6E61]">
              <span>Single-cell biology</span>
              <span aria-hidden="true">·</span>
              <span>Pairwise XGBoost + 3D GNN</span>
              <span aria-hidden="true">·</span>
              <span>Plain-English reports</span>
            </div>
          </div>

          <div className="lg:col-span-6">
            <figure className="rounded-xl bg-[#FAF6F0] border border-[#E6DDD0] p-6 sm:p-8 flex flex-col justify-between min-h-[330px]">
              <div className="flex items-center justify-between text-xs text-[#7A6E61] pb-3 border-b border-[#EAE2D6]">
                <span className="font-mono text-[11px] text-[#574D43]">
                  Fig. 1 — Dual-Specificity Architecture
                </span>
                <span>Two distinct cell targets</span>
              </div>

              <svg
                viewBox="0 0 380 210"
                className="w-full max-w-[360px] mx-auto my-4 h-auto"
                role="img"
                aria-label="Scientific schematic of a Y-shaped bispecific antibody engaging Target A (CD3) and Target B (EGFR)"
              >
                <circle
                  cx="190"
                  cy="112"
                  r="88"
                  fill="none"
                  stroke="#E5DCCE"
                  strokeWidth="1"
                  strokeDasharray="3 5"
                />
                <circle
                  cx="190"
                  cy="112"
                  r="58"
                  fill="none"
                  stroke="#ECE4D8"
                  strokeWidth="1"
                />

                <line
                  x1="126"
                  y1="76"
                  x2="94"
                  y2="54"
                  stroke="#C9B79F"
                  strokeWidth="1.2"
                  strokeDasharray="2 3"
                />
                <line
                  x1="254"
                  y1="76"
                  x2="286"
                  y2="54"
                  stroke="#AFC4B3"
                  strokeWidth="1.2"
                  strokeDasharray="2 3"
                />

                <circle cx="94" cy="54" r="27" fill="#F5EDE1" stroke="#7C5333" strokeWidth="1.75" />
                <text
                  x="94"
                  y="52"
                  textAnchor="middle"
                  className="fill-[#231D18] text-[12px] font-semibold font-mono"
                >
                  CD3
                </text>
                <text
                  x="94"
                  y="66"
                  textAnchor="middle"
                  className="fill-[#7A6E61] text-[9px]"
                >
                  Target A
                </text>

                <circle cx="286" cy="54" r="27" fill="#EDF4EE" stroke="#2F5A3B" strokeWidth="1.75" />
                <text
                  x="286"
                  y="52"
                  textAnchor="middle"
                  className="fill-[#231D18] text-[12px] font-semibold font-mono"
                >
                  EGFR
                </text>
                <text
                  x="286"
                  y="66"
                  textAnchor="middle"
                  className="fill-[#4E6955] text-[9px]"
                >
                  Target B
                </text>

                <path
                  d="M190 124 L126 76"
                  stroke="#7C5333"
                  strokeWidth="10"
                  strokeLinecap="round"
                />
                <path
                  d="M190 124 L254 76"
                  stroke="#2F5A3B"
                  strokeWidth="10"
                  strokeLinecap="round"
                />
                <path
                  d="M190 124 L190 186"
                  stroke="#2D241E"
                  strokeWidth="11"
                  strokeLinecap="round"
                />
                <circle cx="190" cy="124" r="8.5" fill="#FFFEFC" stroke="#2D241E" strokeWidth="3" />

                <text x="148" y="114" textAnchor="end" className="fill-[#7C5333] text-[9px] font-mono">
                  Fab Arm 1
                </text>
                <text x="232" y="114" textAnchor="start" className="fill-[#2F5A3B] text-[9px] font-mono">
                  Fab Arm 2
                </text>
                <text x="206" y="170" textAnchor="start" className="fill-[#6E6356] text-[9px] font-mono">
                  Fc Stem
                </text>
              </svg>

              <figcaption className="pt-3 border-t border-[#EAE2D6] flex flex-wrap items-center justify-between gap-2 text-xs text-[#665B4F]">
                <span>Simultaneous dual-epitope engagement</span>
                <span className="font-mono text-[11px] text-[#2F5A3B]">
                  Pairwise Druggability Scoring
                </span>
              </figcaption>
            </figure>
          </div>
        </div>
      </div>

      {/* 2. Unified 4-Column Benchmark Strip */}
      <div className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl grid grid-cols-2 lg:grid-cols-4 divide-y lg:divide-y-0 lg:divide-x divide-[#EAE2D6]">
        <div className="p-6">
          <div className="text-3xl font-mono font-semibold text-[#231D18] tabular-nums">
            2
          </div>
          <div className="mt-1.5 text-xs text-[#665B4F] leading-snug">
            target proteins bound by one antibody
          </div>
        </div>
        <div className="p-6">
          <div className="text-3xl font-mono font-semibold text-[#231D18] tabular-nums">
            63
          </div>
          <div className="mt-1.5 text-xs text-[#665B4F] leading-snug">
            biological &amp; structural features per pair
          </div>
        </div>
        <div className="p-6 bg-[#F6FAF7]">
          <div className="text-3xl font-mono font-semibold text-[#2F5A3B] tabular-nums">
            0.9410
          </div>
          <div className="mt-1.5 text-xs text-[#435D4A] font-medium leading-snug">
            5-fold NDCG score with GNN embeddings
          </div>
        </div>
        <div className="p-6">
          <div className="text-3xl font-mono font-semibold text-[#7C5333] tabular-nums">
            1
          </div>
          <div className="mt-1.5 text-xs text-[#665B4F] leading-snug">
            goal: smarter early target discovery
          </div>
        </div>
      </div>

      {/* 3. The Science, Simplified */}
      <div id="how-bsab-works" className="space-y-6 scroll-mt-24">
        <div className="space-y-2 max-w-2xl">
          <div className="text-xs font-medium text-[#7C5333]">
            Background &amp; biology
          </div>
          <h2 className="text-2xl sm:text-3xl font-semibold text-[#231D18]">
            What are bispecific antibodies and antigens?
          </h2>
          <p className="text-sm text-[#574D43] leading-relaxed">
            Think of antigens as surface markers on cells, and bispecific antibodies as two-armed proteins engineered to recognize two different markers at the same time.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <article className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl p-6 space-y-3">
            <div className="text-xs font-mono text-[#7C5333]">01. Basics</div>
            <h3 className="text-base font-semibold text-[#231D18]">
              One antibody, two targets
            </h3>
            <p className="text-sm text-[#574D43] leading-relaxed">
              An <strong className="font-semibold text-[#231D18]">antigen</strong> is a protein marker on a cell surface. A standard antibody binds only one antigen, whereas a <strong className="font-semibold text-[#231D18]">bispecific antibody</strong> uses two distinct arms to engage two antigens simultaneously.
            </p>
          </article>

          <article className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl p-6 space-y-3">
            <div className="text-xs font-mono text-[#2F5A3B]">02. Rationale</div>
            <h3 className="text-base font-semibold text-[#231D18]">
              Why pair two targets?
            </h3>
            <p className="text-sm text-[#574D43] leading-relaxed">
              Dual targeting enables one therapeutic to physically bridge an immune killer cell onto a tumor cell, or shut down two cancer growth pathways at once — while weighing healthy-tissue selectivity to manage toxicity.
            </p>
          </article>

          <article className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl p-6 space-y-3">
            <div className="text-xs font-mono text-[#7C5333]">03. Discovery</div>
            <h3 className="text-base font-semibold text-[#231D18]">
              Where AI helps R&amp;D teams
            </h3>
            <p className="text-sm text-[#574D43] leading-relaxed">
              Rather than screening vast combinations blindly at the bench, Ubhaya-Bandha evaluates candidate pairs across biological and structural features to compute a <strong className="font-semibold text-[#231D18]">Pairwise Druggability Score</strong> with clear explanations.
            </p>
          </article>
        </div>

        <div className="flex items-start gap-3 border-t border-b border-[#E3D9CA] py-4 px-2 text-xs sm:text-sm text-[#574D43]">
          <Info className="w-4 h-4 text-[#7C5333] shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <strong className="font-semibold text-[#231D18]">Research disclaimer:</strong> A high computational Pairwise Druggability Score is an early prioritization signal, not proof of clinical safety or efficacy. Experimental wet-lab and clinical validation remain essential.
          </p>
        </div>
      </div>

      {/* 4. Workflow */}
      <div className="space-y-6">
        <div className="space-y-1.5 max-w-2xl">
          <div className="text-xs font-medium text-[#7C5333]">
            Methodology
          </div>
          <h2 className="text-2xl sm:text-3xl font-semibold text-[#231D18]">
            How the evaluation pipeline works
          </h2>
          <p className="text-sm text-[#574D43]">
            Four structured stages from single-cell expression data to a plain-English report and 3D binding simulation.
          </p>
        </div>

        <div className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-[#EAE2D6]">
          {WORKFLOW_STEPS.map((step) => (
            <article key={step.number} className="p-6 space-y-2.5">
              <div className="text-xs font-mono font-medium text-[#7C5333]">
                Step {step.number}
              </div>
              <h3 className="text-base font-semibold text-[#231D18]">
                {step.title}
              </h3>
              <p className="text-xs sm:text-[13px] text-[#574D43] leading-relaxed">
                {step.description}
              </p>
            </article>
          ))}
        </div>
      </div>

      {/* 4b. Clinical Progression Taxonomy */}
      <div className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl p-8 lg:p-10 space-y-6">
        <div className="space-y-2 max-w-3xl">
          <div className="text-xs font-medium text-[#7C5333]">
            Clinical progression taxonomy
          </div>
          <h2 className="text-2xl sm:text-3xl font-semibold text-[#231D18]">
            Understanding the 5 clinical stages
          </h2>
          <p className="text-sm text-[#574D43] leading-relaxed">
            Throughout the Leaderboard and Target Reports, known bispecific antibody combinations from the 791-drug clinical trial dataset are labeled across five clinical development stages — ordered from regulatory-approved therapies down to discontinued programs — which guide how our pairwise model learns therapeutic relevance.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {CLINICAL_STAGES.map((item) => (
            <article
              key={item.stage}
              className="bg-[#FAF6F0] border border-[#E6DDD0] rounded-xl p-5 space-y-2"
            >
              <div className="flex items-center justify-between gap-2 text-xs font-mono">
                <span className="font-semibold text-[#231D18]">{item.stage}</span>
                <span className="text-[11px] text-[#7A6E61]">{item.badge}</span>
              </div>
              <h3 className="text-sm font-semibold text-[#231D18]">
                {item.title}
              </h3>
              <p className="text-xs text-[#574D43] leading-relaxed">
                {item.description}
              </p>
            </article>
          ))}
        </div>

        <div className="pt-4 border-t border-[#EAE2D6] text-xs sm:text-[13px] text-[#574D43] leading-relaxed">
          <strong className="font-semibold text-[#231D18]">
            What about Stage 0 (Novel Discovery Candidates)?
          </strong>{' '}
          Pairs labeled <span className="font-mono text-[#231D18]">Stage 0 · Novel Discovery Candidate</span> do not yet appear in the 791-drug clinical trial dataset — these represent unbenchmarked target combinations scored through the 5-stage pairwise hierarchy to surface promising de novo bispecific candidates.
        </div>
      </div>

      {/* 5. Why the NDCG Score Matters for Researchers */}
      <div className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl p-8 lg:p-10 space-y-8">
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6 border-b border-[#EAE2D6] pb-6">
          <div className="space-y-2 max-w-2xl">
            <div className="text-xs font-medium text-[#2F5A3B]">
              Model validation
            </div>
            <h2 className="text-2xl sm:text-3xl font-semibold text-[#231D18]">
              Why the NDCG score matters for researchers
            </h2>
            <p className="text-sm text-[#574D43] leading-relaxed">
              Wet-lab teams cannot test tens of thousands of gene combinations — they can only advance a small shortlist at the very top.{' '}
              <strong className="font-semibold text-[#231D18]">
                NDCG (Normalized Discounted Cumulative Gain)
              </strong>{' '}
              measures how accurately our ranker places true clinical-grade target pairs at the top of the list on a <span className="font-mono text-[#231D18]">0.0–1.0</span> scale.
            </p>
          </div>

          <div className="bg-[#F3F8F4] border border-[#CDE0D1] rounded-xl px-5 py-4 shrink-0">
            <div className="text-xs font-medium text-[#2F5A3B]">
              5-Fold Cross-Validation
            </div>
            <div className="mt-1 flex items-baseline gap-2">
              <span className="text-3xl font-mono font-semibold text-[#1F4229] tabular-nums">
                0.9410
              </span>
              <span className="text-xs font-mono text-[#2F5A3B]">NDCG</span>
            </div>
            <div className="mt-0.5 text-xs text-[#4C6854]">
              Pairwise XGBoost + 3D GNN Embeddings
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="space-y-2">
            <div className="text-xs font-mono text-[#7C5333]">01. Top-rank focus</div>
            <h3 className="text-base font-semibold text-[#231D18]">
              Early ranks carry the highest weight
            </h3>
            <p className="text-sm text-[#574D43] leading-relaxed">
              Standard accuracy treats rank #1 and rank #500 equally. NDCG applies a logarithmic discount (<span className="font-mono text-xs text-[#231D18]">1 / log₂(rank + 1)</span>), heavily rewarding the model for surfacing the strongest pairs into the top shortlist scientists actually validate.
            </p>
          </div>

          <div className="space-y-2">
            <div className="text-xs font-mono text-[#2F5A3B]">02. Graded relevance</div>
            <h3 className="text-base font-semibold text-[#231D18]">
              Respects clinical maturity stages
            </h3>
            <p className="text-sm text-[#574D43] leading-relaxed">
              Target combinations are not simply binary “good” or “bad.” NDCG accounts for multi-level clinical evidence — rewarding approved and Phase III clinical pairs above early preclinical or novel candidates.
            </p>
          </div>

          <div className="space-y-2">
            <div className="text-xs font-mono text-[#7C5333]">03. Practical meaning</div>
            <h3 className="text-base font-semibold text-[#231D18]">
              What 0.9410 means in R&amp;D
            </h3>
            <p className="text-sm text-[#574D43] leading-relaxed">
              An NDCG of <strong className="font-semibold text-[#231D18]">0.9410</strong> (near the ideal <span className="font-mono text-xs">1.0</span>) means our XGBoost + GNN model closely recovers the optimal ranking across held-out folds — reducing screening cost and improving bench hit rates.
            </p>
          </div>
        </div>

        <div className="pt-6 border-t border-[#EAE2D6] flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
          <div className="flex-1 space-y-3">
            <div className="text-xs font-medium text-[#574D43]">
              Logarithmic position weight across discovery ranks
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
              <div className="space-y-1.5">
                <div className="flex items-baseline justify-between text-xs font-mono">
                  <span className="font-semibold text-[#231D18]">Rank #1</span>
                  <span className="text-[#2F5A3B] font-semibold">100%</span>
                </div>
                <div className="h-1.5 rounded-full bg-[#ECE4D8] overflow-hidden">
                  <div className="h-full w-full bg-[#2F5A3B] rounded-full" />
                </div>
                <div className="text-[11px] text-[#7A6E61]">Highest shortlist impact</div>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-baseline justify-between text-xs font-mono">
                  <span className="font-semibold text-[#231D18]">Rank #5</span>
                  <span className="text-[#7C5333] font-semibold">39%</span>
                </div>
                <div className="h-1.5 rounded-full bg-[#ECE4D8] overflow-hidden">
                  <div className="h-full w-[39%] bg-[#7C5333] rounded-full" />
                </div>
                <div className="text-[11px] text-[#7A6E61]">Early-review priority</div>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-baseline justify-between text-xs font-mono">
                  <span className="font-semibold text-[#231D18]">Rank #50</span>
                  <span className="text-[#8B7F72] font-semibold">18%</span>
                </div>
                <div className="h-1.5 rounded-full bg-[#ECE4D8] overflow-hidden">
                  <div className="h-full w-[18%] bg-[#B5A898] rounded-full" />
                </div>
                <div className="text-[11px] text-[#7A6E61]">Discounted lower zone</div>
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={() => goToTab('model-card')}
            className="inline-flex items-center justify-center gap-2 px-5 py-2.5 text-xs font-medium text-[#231D18] bg-[#F7F2EA] hover:bg-[#EFE7DC] border border-[#DFD4C3] rounded-lg transition-colors cursor-pointer shrink-0 self-start lg:self-center"
          >
            Inspect 5-fold validation
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* 6. Restrained Closing Exploration Strip */}
      <div className="bg-[#EFE7DC] border border-[#DFD4C3] rounded-2xl p-8 sm:p-10 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
        <div className="space-y-2 max-w-xl">
          <h2 className="text-2xl sm:text-3xl font-semibold text-[#231D18]">
            Evaluate a target combination
          </h2>
          <p className="text-sm text-[#574D43] leading-relaxed">
            Select any two membrane or secreted targets to generate a Pairwise Druggability Score, decision report, and 3D structural binding walkthrough.
          </p>

          <div className="pt-2 flex flex-wrap items-center gap-2 text-xs">
            <span className="text-[#6E6356]">Reference pairs:</span>
            {QUICK_STARTS.map((item) => (
              <button
                key={item.label}
                type="button"
                onClick={() => inspectWithTopScroll(item.g1, item.g2)}
                className="px-3 py-1.5 rounded-lg bg-[#FFFEFC] hover:bg-[#FAF6F0] border border-[#DFD4C3] text-[#231D18] font-mono text-xs transition-colors cursor-pointer"
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={() => goToTab('predict')}
            className="inline-flex items-center gap-2 px-5 py-3 text-sm font-medium text-[#FFFEFC] bg-[#2D241E] hover:bg-[#1E1713] rounded-lg transition-colors cursor-pointer"
          >
            Explore target pairs
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            type="button"
            onClick={() => goToTab('leaderboard')}
            className="inline-flex items-center gap-2 px-5 py-3 text-sm font-medium text-[#231D18] bg-[#FFFEFC] hover:bg-[#FAF6F0] border border-[#DFD4C3] rounded-lg transition-colors cursor-pointer"
          >
            View Leaderboard
          </button>
        </div>
      </div>
    </section>
  );
}