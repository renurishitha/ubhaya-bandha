import { ArrowLeftRight, ArrowUp, Check, ChevronDown, Download, FileText, RefreshCw } from 'lucide-react';
import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import { bspaiClient } from '../api/bspaiClient.ts';
import type { GeneInfo, PredictionResult, RagReportResult } from '../types.ts';
import { AlphaFold3DViewer } from './AlphaFold3DViewer.tsx';
import { BispecificBindingAnimation } from './BispecificBindingAnimation.tsx';

interface PredictViewProps {
  genes: GeneInfo[];
  totalPairs: number;
  selectedGene1: string;
  selectedGene2: string;
  inspectNonce: number;
  onChangePair: (g1: string, g2: string) => void;
}

const QUICK_PAIRS: Array<{ label: string; g1: string; g2: string }> = [
  { label: 'CD274 + CTLA4', g1: 'CD274', g2: 'CTLA4' },
  { label: 'EGFR + MET', g1: 'EGFR', g2: 'MET' },
  { label: 'MS4A1 + CD3E', g1: 'MS4A1', g2: 'CD3E' },
  { label: 'CLDN18 + MS4A1', g1: 'CLDN18', g2: 'MS4A1' },
  { label: 'PDCD1 + VEGFA', g1: 'PDCD1', g2: 'VEGFA' },
];

const STAGE_LABELS: Record<number, string> = {
  1: 'Stage 1 · Approved / BLA',
  2: 'Stage 2 · Phase 3 / 2–3',
  3: 'Stage 3 · Phase 1 / 2',
  4: 'Stage 4 · Preclinical / IND',
  5: 'Stage 5 · Discontinued',
};

function formatClinicalStage(prediction: PredictionResult): {
  title: string;
  subtitle: string;
  isClinical: boolean;
} {
  if (prediction.inClinicalTrial && prediction.clinicalStage) {
    const label =
      STAGE_LABELS[prediction.clinicalStage] ?? `Stage ${prediction.clinicalStage}`;
    const drugs =
      prediction.clinicalDrugs?.length > 0
        ? prediction.clinicalDrugs.join(', ')
        : 'Validated clinical bispecific benchmark';
    return {
      title: label,
      subtitle: `${drugs}${prediction.needsVerification ? ' · Verify' : ''}`,
      isClinical: true,
    };
  }
  return {
    title: 'Stage 0 · Novel Candidate',
    subtitle: 'Unbenchmarked pair scored via 5-stage hierarchy',
    isClinical: false,
  };
}

function renderInlineFormatted(text: string) {
  const cleaned = text
    .replace(/\$\\Delta_x\s*=\s*([^$]+)\$/g, 'Δx = $1')
    .replace(/\$\\Delta_y\s*=\s*([^$]+)\$/g, 'Δy = $1')
    .replace(/\$([^$]+)\$/g, '$1');

  const tokens = cleaned.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g);
  return tokens.map((tok, i) => {
    if (tok.startsWith('**') && tok.endsWith('**') && tok.length > 4) {
      return (
        <strong key={i} className="font-semibold text-[#2B231C]">
          {tok.slice(2, -2)}
        </strong>
      );
    }
    if (tok.startsWith('`') && tok.endsWith('`') && tok.length > 2) {
      return (
        <code
          key={i}
          className="px-1.5 py-0.5 bg-[#F3EADF] text-[#4E3D2E] rounded font-mono text-[11px]"
        >
          {tok.slice(1, -1)}
        </code>
      );
    }
    if (tok.startsWith('*') && tok.endsWith('*') && tok.length > 2) {
      return (
        <em key={i} className="italic text-[#6E6052]">
          {tok.slice(1, -1)}
        </em>
      );
    }
    return <Fragment key={i}>{tok}</Fragment>;
  });
}

function FormattedReportContent({ markdown }: { markdown: string }) {
  const lines = markdown.split('\n');
  const elements: Array<{
    type: 'h3' | 'h4' | 'quote' | 'bullet' | 'numbered' | 'p';
    text: string;
    num?: string;
  }> = [];

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith('#### ')) {
      elements.push({ type: 'h4', text: line.slice(5).trim() });
    } else if (line.startsWith('### ')) {
      elements.push({ type: 'h3', text: line.slice(4).trim() });
    } else if (line.startsWith('## ')) {
      elements.push({ type: 'h3', text: line.slice(3).trim() });
    } else if (line.startsWith('# ')) {
      elements.push({ type: 'h3', text: line.slice(2).trim() });
    } else if (line.startsWith('> ')) {
      elements.push({ type: 'quote', text: line.slice(2).trim() });
    } else if (line.startsWith('- ') || line.startsWith('* ')) {
      elements.push({ type: 'bullet', text: line.slice(2).trim() });
    } else {
      const numMatch = line.match(/^(\d+)\.\s+(.*)$/);
      if (numMatch) {
        elements.push({ type: 'numbered', num: numMatch[1], text: numMatch[2].trim() });
      } else {
        elements.push({ type: 'p', text: line });
      }
    }
  }

  return (
    <div className="space-y-3 text-sm text-[#4C4136] leading-relaxed">
      {elements.map((el, idx) => {
        if (el.type === 'h3') {
          return (
            <h3
              key={idx}
              className="pt-2 first:pt-0 text-base font-bold text-[#2B231C] border-b border-[#EDE2D3] pb-1.5"
            >
              {renderInlineFormatted(el.text)}
            </h3>
          );
        }
        if (el.type === 'h4') {
          return (
            <h4 key={idx} className="pt-1.5 text-xs font-bold uppercase tracking-wider text-[#8B5E3C]">
              {renderInlineFormatted(el.text)}
            </h4>
          );
        }
        if (el.type === 'quote') {
          return (
            <div
              key={idx}
              className="px-4 py-2.5 bg-[#F6EFE6] border border-[#E5D8C5] rounded-xl text-xs text-[#6E6052]"
            >
              {renderInlineFormatted(el.text)}
            </div>
          );
        }
        if (el.type === 'bullet') {
          return (
            <div key={idx} className="flex items-start gap-2.5 pl-1">
              <span className="mt-2 w-1.5 h-1.5 rounded-full bg-[#8B5E3C] shrink-0" />
              <div>{renderInlineFormatted(el.text)}</div>
            </div>
          );
        }
        if (el.type === 'numbered') {
          return (
            <div key={idx} className="flex items-start gap-2.5 pl-1">
              <span className="mt-0.5 w-5 h-5 rounded-md bg-[#F3EADF] border border-[#DFCEB8] text-[#8B5E3C] font-mono text-[11px] font-semibold flex items-center justify-center shrink-0">
                {el.num}
              </span>
              <div>{renderInlineFormatted(el.text)}</div>
            </div>
          );
        }
        return <p key={idx}>{renderInlineFormatted(el.text)}</p>;
      })}
    </div>
  );
}

function GeneAutocomplete({
  id,
  label,
  value,
  genes,
  onSelect,
  onSubmitCustom,
}: {
  id: string;
  label: string;
  value: string;
  genes: GeneInfo[];
  onSelect: (gene: string) => void;
  onSubmitCustom?: (gene: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [query, setQuery] = useState(value);
  const [isFiltering, setIsFiltering] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    setQuery(value);
    setIsFiltering(false);
  }, [value]);

  useEffect(() => {
    const handleOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
        setIsFiltering(false);
        const trimmed = query.trim().toUpperCase();
        if (trimmed) {
          onSelect(trimmed);
        } else {
          setQuery(value);
        }
      }
    };
    document.addEventListener('mousedown', handleOutside);
    return () => document.removeEventListener('mousedown', handleOutside);
  }, [query, value, onSelect]);

  const sortedGenes = useMemo(
    () => [...genes].sort((a, b) => a.gene.localeCompare(b.gene)),
    [genes]
  );

  const matches = useMemo(() => {
    const q = query.trim().toUpperCase();
    if (!isFiltering || !q) return sortedGenes;
    return sortedGenes.filter(
      (g) =>
        g.gene.toUpperCase().includes(q) ||
        g.description.toUpperCase().includes(q) ||
        g.category.toUpperCase().includes(q)
    );
  }, [query, isFiltering, sortedGenes]);

  const exactMatch = useMemo(() => {
    const q = query.trim().toUpperCase();
    return sortedGenes.some((g) => g.gene.toUpperCase() === q);
  }, [query, sortedGenes]);

  const commit = (sym: string) => {
    const clean = sym.trim().toUpperCase();
    if (!clean) return;
    setQuery(clean);
    setIsFiltering(false);
    setOpen(false);
    onSelect(clean);
  };

  const toggleDropdown = () => {
    if (open) {
      setOpen(false);
      setIsFiltering(false);
    } else {
      setIsFiltering(false);
      setOpen(true);
      inputRef.current?.focus();
    }
  };

  const cleanQuery = query.trim().toUpperCase();

  return (
    <div ref={containerRef} className="relative w-full">
      <label
        htmlFor={id}
        className="block text-xs font-medium text-[#574D43] mb-1.5"
      >
        {label}
      </label>
      <div className="relative flex items-center">
        <input
          ref={inputRef}
          id={id}
          type="text"
          role="combobox"
          aria-expanded={open}
          aria-controls={`${id}-listbox`}
          aria-autocomplete="list"
          value={query}
          autoComplete="off"
          onFocus={() => {
            setIsFiltering(false);
            setOpen(true);
          }}
          onChange={(e) => {
            const next = e.target.value.toUpperCase();
            setQuery(next);
            setIsFiltering(true);
            setOpen(true);
            onSelect(next);
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              if (cleanQuery) {
                commit(cleanQuery);
                onSubmitCustom?.(cleanQuery);
              }
            } else if (e.key === 'Escape') {
              setOpen(false);
              setIsFiltering(false);
            }
          }}
          placeholder="Select or type any gene symbol (e.g. CD274, TP53)..."
          className="w-full h-11 pl-3.5 pr-10 bg-[#FFFEFC] border border-[#DFD4C3] rounded-lg font-mono text-sm text-[#231D18] focus:outline-none focus:border-[#7C5333] focus:ring-2 focus:ring-[#7C5333]/15"
        />
        <button
          type="button"
          aria-label={`Toggle ${label} dropdown`}
          onClick={toggleDropdown}
          className="absolute right-1.5 p-2 rounded-md text-[#6E6356] hover:text-[#231D18] hover:bg-[#F4EFE7] transition-colors cursor-pointer"
        >
          <ChevronDown
            className={`w-4 h-4 transition-transform duration-150 ${
              open ? 'rotate-180' : ''
            }`}
          />
        </button>
      </div>

      {open && (
        <div
          id={`${id}-listbox`}
          role="listbox"
          className="absolute z-30 mt-1.5 w-full bg-[#FFFEFC] border border-[#DFD4C3] rounded-xl shadow-lg overflow-hidden"
        >
          <div className="px-3.5 py-2 bg-[#FAF6F0] border-b border-[#EAE2D6] flex items-center justify-between text-[11px] text-[#6E6356]">
            <span>
              {isFiltering && cleanQuery
                ? `${matches.length} curated match${matches.length === 1 ? '' : 'es'}`
                : `Curated targets (${matches.length}) · or type any scRNA-seq gene`}
            </span>
            <span className="font-mono text-[10px]">Free text supported</span>
          </div>

          {cleanQuery && !exactMatch && (
            <button
              type="button"
              onMouseDown={(e) => {
                e.preventDefault();
                commit(cleanQuery);
              }}
              className="w-full text-left px-3.5 py-2.5 bg-[#F6FAF7] hover:bg-[#EDF5EF] border-b border-[#DCE8DF] flex items-center justify-between gap-3 transition-colors cursor-pointer"
            >
              <span className="font-mono text-xs font-semibold text-[#2F5A3B]">
                Use “{cleanQuery}” (scRNA-seq cohort lookup)
              </span>
              <span className="text-[11px] text-[#4C6854]">
                Custom symbol
              </span>
            </button>
          )}

          {matches.length === 0 ? (
            <div className="px-3.5 py-3 text-xs text-[#6E6356]">
              Not in the 327 curated surface targets — click above or press{' '}
              <strong className="font-semibold text-[#231D18]">Evaluate Pair</strong>{' '}
              to query <span className="font-mono text-[#231D18]">{cleanQuery}</span> directly from the single-cell matrix.
            </div>
          ) : (
            <ul className="max-h-64 overflow-y-auto divide-y divide-[#F0E9DD]">
              {matches.map((item) => {
                const isSelected = item.gene === value;
                return (
                  <li key={item.gene} role="option" aria-selected={isSelected}>
                    <button
                      type="button"
                      onMouseDown={(e) => {
                        e.preventDefault();
                        commit(item.gene);
                      }}
                      className={`w-full text-left px-3.5 py-2.5 flex items-center justify-between gap-3 transition-colors cursor-pointer ${
                        isSelected
                          ? 'bg-[#F4EFE7] text-[#231D18]'
                          : 'hover:bg-[#FAF6F0] text-[#231D18]'
                      }`}
                    >
                      <span className="flex items-center gap-2 shrink-0">
                        <span className="font-mono text-xs font-semibold">
                          {item.gene}
                        </span>
                        {isSelected && (
                          <Check className="w-3.5 h-3.5 text-[#2F5A3B]" />
                        )}
                      </span>
                      <span className="text-xs text-[#6E6356] truncate">
                        {item.category} · {item.description}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

export function PredictView({
  genes,
  totalPairs,
  selectedGene1,
  selectedGene2,
  inspectNonce,
  onChangePair,
}: PredictViewProps) {
  const selectorRef = useRef<HTMLDivElement | null>(null);
  const reportRef = useRef<HTMLDivElement | null>(null);

  const [draftG1, setDraftG1] = useState(selectedGene1);
  const [draftG2, setDraftG2] = useState(selectedGene2);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predLoading, setPredLoading] = useState(true);
  const [predError, setPredError] = useState<string | null>(null);
  const [evalTrigger, setEvalTrigger] = useState(0);

  const [report, setReport] = useState<RagReportResult | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);
  const [reportMode, setReportMode] = useState<'plain' | 'technical'>('plain');
  const [reportRefreshTick, setReportRefreshTick] = useState(0);

  const scrollToReport = () => {
    requestAnimationFrame(() => {
      setTimeout(() => {
        reportRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 90);
    });
  };

  useEffect(() => {
    setDraftG1(selectedGene1);
    setDraftG2(selectedGene2);
    setEvalTrigger((c) => c + 1);
  }, [selectedGene1, selectedGene2]);

  useEffect(() => {
    if (inspectNonce > 0) {
      selectorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [inspectNonce]);

  useEffect(() => {
    let cancelled = false;
    setPredLoading(true);
    setPredError(null);

    bspaiClient
      .predictPair(selectedGene1, selectedGene2)
      .then((data) => {
        if (!cancelled) setPrediction(data);
      })
      .catch((err) => {
        if (!cancelled) {
          setPredError(err instanceof Error ? err.message : 'Prediction failed');
          setPrediction(null);
        }
      })
      .finally(() => {
        if (!cancelled) setPredLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedGene1, selectedGene2]);

  useEffect(() => {
    let cancelled = false;
    setReportLoading(true);
    setReportError(null);

    bspaiClient
      .getReport(selectedGene1, selectedGene2, false)
      .then((data) => {
        if (!cancelled) setReport(data);
      })
      .catch((err) => {
        if (!cancelled) {
          setReportError(err instanceof Error ? err.message : 'Failed to generate report');
        }
      })
      .finally(() => {
        if (!cancelled) setReportLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedGene1, selectedGene2, reportRefreshTick]);

  const handleEvaluate = (g1: string, g2: string) => {
    const clean1 = g1.trim().toUpperCase();
    const clean2 = g2.trim().toUpperCase();
    if (!clean1 || !clean2) return;
    if (clean1 === clean2) {
      setPredError('Please pick two different target genes to form a bispecific pair.');
      return;
    }
    setPredError(null);
    setDraftG1(clean1);
    setDraftG2(clean2);
    setEvalTrigger((c) => c + 1);
    onChangePair(clean1, clean2);
    scrollToReport();
  };

  const resolvedG1 = prediction?.gene1 ?? selectedGene1;
  const resolvedG2 = prediction?.gene2 ?? selectedGene2;
  const hasAliasMapping =
    resolvedG1 !== selectedGene1.trim().toUpperCase() ||
    resolvedG2 !== selectedGene2.trim().toUpperCase();
  const reportTitlePair = hasAliasMapping
    ? `${resolvedG1} + ${resolvedG2} (${selectedGene1.trim().toUpperCase()} + ${selectedGene2.trim().toUpperCase()})`
    : `${resolvedG1} + ${resolvedG2}`;

  const handleDownloadReport = () => {
    if (!report) return;
    const stripMd = (md: string) =>
      md
        .replace(/\$\\Delta_x\s*=\s*([^$]+)\$/g, 'dx = $1')
        .replace(/\$\\Delta_y\s*=\s*([^$]+)\$/g, 'dy = $1')
        .replace(/\$([^$]+)\$/g, '$1')
        .replace(/\*\*([^*]+)\*\*/g, '$1')
        .replace(/`([^`]+)`/g, '$1')
        .replace(/^#{1,4}\s+/gm, '')
        .replace(/^>\s+/gm, '');

    const stageInfo = prediction ? formatClinicalStage(prediction) : null;

    const summaryHeader = [
      `UBHAYA-BANDHA (DUAL BINDING) — TARGET PAIR REPORT`,
      `=================================================`,
      `Target Pair:                 ${reportTitlePair}`,
      ...(prediction && stageInfo
        ? [
            `Pairwise Druggability Score: ${prediction.score.toFixed(4)}`,
            `Global Candidate Rank:       #${prediction.rank.toLocaleString()} / ${(prediction.totalPairs || totalPairs).toLocaleString()}`,
            `Mechanism Class:             ${prediction.mechanismType}`,
            ...(prediction.structuralGnn
              ? [
                  `3D Structural Fit Verdict:   ${
                    prediction.structuralGnn.bindingAssessment.verdict === 'conditional-bind'
                      ? `Partial Fit (${prediction.structuralGnn.bindingAssessment.fitPercent}% — requires linker/Fab spacing optimization)`
                      : prediction.structuralGnn.bindingAssessment.verdict === 'blocked'
                        ? `Blocked (${prediction.structuralGnn.bindingAssessment.fitPercent}% — structural/accessibility barrier)`
                        : `Full Structural Fit (${prediction.structuralGnn.bindingAssessment.fitPercent}%)`
                  }`,
                ]
              : []),
            `Clinical Progression Stage:  ${stageInfo.title} (${stageInfo.subtitle})`,
          ]
        : []),
      `Generated:                   ${new Date().toLocaleString()}`,
      ``,
      `-------------------------------------------------`,
      `READABLE ENGLISH SUMMARY`,
      `-------------------------------------------------`,
      stripMd(report.plainEnglishReport ?? report.reportMarkdown),
      ``,
      `-------------------------------------------------`,
      `FULL TECHNICAL REPORT`,
      `-------------------------------------------------`,
      stripMd(report.reportMarkdown),
      ``,
    ].join('\n');

    const blob = new Blob([summaryHeader], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `Ubhaya-Bandha_${resolvedG1}_${resolvedG2}_Report.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const stageSummary = prediction ? formatClinicalStage(prediction) : null;

  return (
    <section className="space-y-10 pb-8">
      {/* 1. Target Pair Builder Console */}
      <div
        ref={selectorRef}
        className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl p-7 sm:p-8 lg:p-10"
      >
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-10 items-center">
          {/* Left Copy */}
          <div className="lg:col-span-6 space-y-5">
            <div className="text-xs font-medium text-[#7C5333]">
              Pair explorer
            </div>
            <h1 className="text-3xl sm:text-4xl font-semibold text-[#231D18] leading-tight">
              Evaluate a bispecific{' '}
              <span className="text-[#7C5333] italic font-medium">target pair</span>
            </h1>
            <p className="text-sm sm:text-[15px] text-[#574D43] leading-relaxed">
              Select from curated targets or type any two gene symbols in our single-cell cohort to compute their Pairwise Druggability Score, open the decision report, and inspect 3D structural binding simulations below.
            </p>

            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[#7A6E61]">
              <span>Select or type any two targets</span>
              <span aria-hidden="true">·</span>
              <span>Review decision report</span>
              <span aria-hidden="true">·</span>
              <span>Inspect 3D binding below</span>
            </div>

            <div className="pt-2 space-y-2">
              <div className="text-xs font-medium text-[#6E6356]">
                Reference pairs (click to load into inputs):
              </div>
              <div className="flex flex-wrap gap-2">
                {QUICK_PAIRS.map((preset) => {
                  const isActive =
                    draftG1 === preset.g1 && draftG2 === preset.g2;
                  return (
                    <button
                      key={preset.label}
                      type="button"
                      onClick={() => {
                        setPredError(null);
                        setDraftG1(preset.g1);
                        setDraftG2(preset.g2);
                      }}
                      className={`px-3 py-1.5 text-xs font-mono rounded-lg border transition-colors cursor-pointer ${
                        isActive
                          ? 'border-[#2D241E] bg-[#2D241E] text-[#FFFEFC] font-medium'
                          : 'border-[#DFD4C3] bg-[#FAF6F0] text-[#574D43] hover:border-[#7C5333] hover:text-[#231D18]'
                      }`}
                    >
                      {preset.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Right Form Surface */}
          <div className="lg:col-span-6">
            <div className="bg-[#FAF6F0] border border-[#E6DDD0] rounded-xl p-6 space-y-4">
              <div className="flex items-center justify-between gap-2 pb-3 border-b border-[#EAE2D6]">
                <div>
                  <div className="text-sm font-semibold text-[#231D18]">
                    Target selection
                  </div>
                  <div className="text-xs text-[#7A6E61]">
                    Pick from dropdown or type any HGNC gene symbol
                  </div>
                </div>
                <button
                  type="button"
                  aria-label="Swap target genes"
                  onClick={() => {
                    setDraftG1(draftG2);
                    setDraftG2(draftG1);
                  }}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-[#574D43] bg-[#FFFEFC] hover:bg-[#EFE7DC] border border-[#DFD4C3] rounded-lg transition-colors cursor-pointer"
                >
                  <ArrowLeftRight className="w-3.5 h-3.5" />
                  Swap
                </button>
              </div>

              <GeneAutocomplete
                id="gene1-input"
                label="First Target"
                value={draftG1}
                genes={genes}
                onSelect={(g) => setDraftG1(g)}
                onSubmitCustom={(g) => handleEvaluate(g, draftG2)}
              />

              <div className="flex items-center gap-3">
                <span className="h-px flex-1 bg-[#E6DDD0]" />
                <span className="text-xs font-mono font-semibold text-[#7A6E61]">
                  +
                </span>
                <span className="h-px flex-1 bg-[#E6DDD0]" />
              </div>

              <GeneAutocomplete
                id="gene2-input"
                label="Second Target"
                value={draftG2}
                genes={genes}
                onSelect={(g) => setDraftG2(g)}
                onSubmitCustom={(g) => handleEvaluate(draftG1, g)}
              />

              <button
                type="button"
                onClick={() => handleEvaluate(draftG1, draftG2)}
                className="w-full h-11 px-5 bg-[#2D241E] hover:bg-[#1E1713] text-[#FFFEFC] text-sm font-medium rounded-lg transition-colors cursor-pointer"
              >
                Evaluate Pair &amp; View Report →
              </button>

              {predError && (
                <div className="p-3 bg-[#FDF4F0] border border-[#E5C1B5] rounded-lg text-xs text-[#853826]">
                  {predError}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 2. Target Combination Evaluation Report */}
      <div
        ref={reportRef}
        className="scroll-mt-20 space-y-5"
      >
        {/* Unified 4-Column Summary Metric Strip */}
        {prediction && !predLoading && stageSummary && (
          <div className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-[#EAE2D6]">
            <div className="p-5">
              <div className="text-xs font-medium text-[#7A6E61]">
                Pairwise Druggability Score
              </div>
              <div className="mt-2 text-3xl font-mono font-semibold text-[#7C5333] tabular-nums">
                {prediction.score.toFixed(4)}
              </div>
              <div className="mt-1.5 text-xs text-[#6E6356] font-mono">
                {prediction.gene1} + {prediction.gene2}
              </div>
            </div>

            <div className="p-5">
              <div className="text-xs font-medium text-[#7A6E61]">
                Global Candidate Rank
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-3xl font-mono font-semibold text-[#231D18] tabular-nums">
                  #{prediction.rank.toLocaleString()}
                </span>
                <span className="text-xs font-mono text-[#7A6E61] tabular-nums">
                  / {(prediction.totalPairs || totalPairs).toLocaleString()}
                </span>
              </div>
              <div className="mt-1.5 text-xs font-mono text-[#2F5A3B] font-medium">
                Top{' '}
                {Math.max(
                  0.01,
                  (prediction.rank /
                    Math.max(1, prediction.totalPairs || totalPairs)) *
                    100
                ).toFixed(2)}
                %
              </div>
            </div>

            <div className="p-5">
              <div className="text-xs font-medium text-[#7A6E61]">
                Inferred Mechanism Class
              </div>
              <div className="mt-2 text-lg font-semibold text-[#231D18]">
                {prediction.mechanismType}
              </div>
              <div className="mt-2 text-xs text-[#6E6356]">
                {prediction.gene1Info?.cell_bias ?? '—'} ·{' '}
                {prediction.gene2Info?.cell_bias ?? '—'}
              </div>
            </div>

            <div className="p-5">
              <div className="text-xs font-medium text-[#7A6E61]">
                Clinical Progression Stage
              </div>
              <div
                className={`mt-2 text-lg font-semibold ${
                  stageSummary.isClinical ? 'text-[#2F5A3B]' : 'text-[#7C5333]'
                }`}
              >
                {stageSummary.title}
              </div>
              <div className="mt-1.5 text-xs text-[#574D43] truncate" title={stageSummary.subtitle}>
                {stageSummary.subtitle}
              </div>
            </div>
          </div>
        )}

        {/* Decision Report Card */}
        <div className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl overflow-hidden">
          <div className="px-6 py-5 bg-[#FAF6F0] border-b border-[#E3D9CA] flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-[#FFFEFC] border border-[#DFD4C3] text-[#7C5333] shrink-0 mt-0.5">
                <FileText className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-[#231D18]">
                  Target Combination Report — {reportTitlePair}
                </h2>
                <p className="text-xs text-[#6E6356] mt-0.5">
                  Plain-English summary and quantitative breakdown grounded in Pairwise Druggability Score
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <div
                className="inline-flex rounded-lg border border-[#DFD4C3] bg-[#FFFEFC] p-0.5 text-xs"
                role="tablist"
                aria-label="Report format selector"
              >
                <button
                  type="button"
                  role="tab"
                  aria-selected={reportMode === 'plain'}
                  onClick={() => setReportMode('plain')}
                  className={`px-3 py-1.5 rounded-md font-medium transition-colors cursor-pointer ${
                    reportMode === 'plain'
                      ? 'bg-[#2D241E] text-[#FFFEFC]'
                      : 'text-[#665B4F] hover:text-[#231D18]'
                  }`}
                >
                  Readable English
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={reportMode === 'technical'}
                  onClick={() => setReportMode('technical')}
                  className={`px-3 py-1.5 rounded-md font-medium transition-colors cursor-pointer ${
                    reportMode === 'technical'
                      ? 'bg-[#2D241E] text-[#FFFEFC]'
                      : 'text-[#665B4F] hover:text-[#231D18]'
                  }`}
                >
                  Technical Report
                </button>
              </div>

              <button
                type="button"
                onClick={handleDownloadReport}
                disabled={!report || reportLoading}
                className="h-9 px-3.5 rounded-lg bg-[#7C5333] hover:bg-[#664226] text-xs font-medium text-[#FFFEFC] inline-flex items-center gap-1.5 transition-colors cursor-pointer disabled:opacity-50"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download Report</span>
              </button>

              <button
                type="button"
                onClick={() => setReportRefreshTick((t) => t + 1)}
                disabled={reportLoading}
                className="h-9 px-3 rounded-lg border border-[#DFD4C3] bg-[#FFFEFC] hover:bg-[#EFE7DC] text-xs font-medium text-[#574D43] inline-flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${reportLoading ? 'animate-spin' : ''}`} />
                <span>Refresh</span>
              </button>

              <button
                type="button"
                onClick={() =>
                  selectorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                }
                className="h-9 px-3 rounded-lg border border-[#DFD4C3] bg-[#FFFEFC] hover:bg-[#EFE7DC] text-xs font-medium text-[#574D43] inline-flex items-center gap-1 cursor-pointer"
              >
                <ArrowUp className="w-3.5 h-3.5" />
                <span>Change pair</span>
              </button>
            </div>
          </div>

          <div className="p-6 sm:p-8 max-w-4xl">
            {reportLoading && !report ? (
              <div className="py-10 text-center text-sm text-[#7A6E61]">
                Preparing readable and technical reports for {reportTitlePair}...
              </div>
            ) : reportError ? (
              <div className="p-4 bg-[#FDF4F0] border border-[#E5C1B5] rounded-xl text-sm text-[#853826]">
                {reportError}
              </div>
            ) : report ? (
              <div className="space-y-4">
                {reportMode === 'plain' ? (
                  <FormattedReportContent
                    markdown={
                      report.plainEnglishReport ??
                      report.reportMarkdown
                    }
                  />
                ) : (
                  <FormattedReportContent markdown={report.reportMarkdown} />
                )}
              </div>
            ) : null}
          </div>
        </div>
      </div>

      {/* 3. Interactive Animations BELOW the Report */}
      <div className="space-y-5 pt-2">
        <div className="space-y-1">
          <div className="text-xs font-medium text-[#7C5333]">
            3D structural visualizations
          </div>
          <h2 className="text-2xl font-semibold text-[#231D18]">
            Binding Mechanism &amp; 3D Structure Animations
          </h2>
          <p className="text-sm text-[#665B4F]">
            Visual walkthrough of how a bispecific antibody engages {resolvedG1} + {resolvedG2} and how their 3D folded structures fit together.
          </p>
        </div>

        {/* Live 3D Dual-Target Binding Animation */}
        <BispecificBindingAnimation
          gene1={resolvedG1}
          gene2={resolvedG2}
          bias1={prediction?.gene1Info?.cell_bias}
          bias2={prediction?.gene2Info?.cell_bias}
          mechanismType={prediction?.mechanismType}
          bindingVerdict={prediction?.structuralGnn?.bindingAssessment?.verdict}
          score={prediction?.score}
          loading={predLoading}
          evalTrigger={evalTrigger}
        />

        {/* AlphaFold 3D Protein Structure & GNN Residue-Graph Animation */}
        <AlphaFold3DViewer
          gene1={resolvedG1}
          gene2={resolvedG2}
          mechanismType={prediction?.mechanismType}
          pairGnn={prediction?.structuralGnn}
        />
      </div>
    </section>
  );
}