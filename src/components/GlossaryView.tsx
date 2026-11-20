import { ArrowUpRight, Search, X } from 'lucide-react';
import { useMemo, useState } from 'react';

type GlossaryDomain =
  | 'Targets & Pharmacology'
  | 'Single-Cell & TME'
  | '3D Structure & Graph AI'
  | 'Ranking & Evaluation';

interface GlossaryEntry {
  term: string;
  domain: GlossaryDomain;
  shortDef: string;
  detailedNote: string;
}

const GLOSSARY_ENTRIES: GlossaryEntry[] = [
  {
    term: 'Pairwise Druggability Score',
    domain: 'Ranking & Evaluation',
    shortDef:
      'The primary score produced by BSPAI to estimate how promising two biological targets are as a combined bispecific antibody pair.',
    detailedNote:
      'Higher Pairwise Druggability Scores indicate stronger alignment with clinically validated bispecific antibody combinations across tumor-vs-normal safety, single-cell co-expression, pathway overlap, and functional embeddings.',
  },
  {
    term: 'Antibody',
    domain: 'Targets & Pharmacology',
    shortDef:
      'A Y-shaped immune protein that recognizes and latches onto a specific target molecule (antigen).',
    detailedNote:
      'Natural antibodies bind one target with both arms. In modern oncology, engineered antibodies are designed to bind tumor markers or immune switches with high precision.',
  },
  {
    term: 'Antigen (Target)',
    domain: 'Targets & Pharmacology',
    shortDef:
      'A protein marker or "name tag" on the surface of a cell that an antibody can recognize and bind.',
    detailedNote:
      'Ideal therapeutic antigens are highly expressed on cancer cells or tumor-infiltrating immune cells while remaining low on healthy vital organs.',
  },
  {
    term: 'Bispecific Antibody (BsAb)',
    domain: 'Targets & Pharmacology',
    shortDef:
      'An engineered antibody with two different gripping arms capable of binding two distinct antigens at the same time.',
    detailedNote:
      'Depending on the pair, a bispecific antibody can bridge an immune killer T cell directly onto a cancer cell or block two tumor survival pathways simultaneously.',
  },
  {
    term: 'Pairwise Learning (XGBoost)',
    domain: 'Ranking & Evaluation',
    shortDef:
      'A machine-learning method that compares target combinations in pairs to learn their relative ranking.',
    detailedNote:
      'Instead of only predicting isolated yes/no labels, pairwise XGBoost compares higher-stage clinical bispecific combinations against lower-stage and exploratory pairs so top candidates rise cleanly to the top.',
  },
  {
    term: 'AUC (Area Under the ROC Curve)',
    domain: 'Ranking & Evaluation',
    shortDef:
      'A standard accuracy metric from 0.5 (random guessing) to 1.0 (perfect separation) measuring model discrimination.',
    detailedNote:
      'Evaluated across 5 stratified cross-validation folds on the Model Card page, verifying how reliably Pairwise XGBoost separates clinically validated pairs from unvalidated ones.',
  },
  {
    term: 'TAA (Tumor-Associated Antigen)',
    domain: 'Targets & Pharmacology',
    shortDef:
      'A cell-surface protein found in much higher amounts on cancer cells than on healthy normal cells.',
    detailedNote:
      'Quantified in BSPAI using tumor-minus-normal expression differences and combined into the harmonic safety score (score_safe).',
  },
  {
    term: 'Dual TAA Blockade (TAA+TAA)',
    domain: 'Targets & Pharmacology',
    shortDef:
      'A strategy targeting two growth receptors on the same tumor cell (such as EGFR + MET) to block escape routes.',
    detailedNote:
      'Assigned when both targets are enriched on epithelial/tumor cells, relying on cells that express both markers together.',
  },
  {
    term: 'Trans-Cellular Bridging (bridge-two-cells)',
    domain: 'Targets & Pharmacology',
    shortDef:
      'A bispecific mechanism where one arm holds a tumor marker and the other arm recruits a killer T cell (e.g., CD3E).',
    detailedNote:
      'By physically pulling immune effector cells right next to tumor cells, bridging bispecifics trigger localized tumor cell killing.',
  },
  {
    term: 'Immune Checkpoint & Tumor Immunotherapy',
    domain: 'Targets & Pharmacology',
    shortDef:
      'Regulatory switches (like PD-L1 and CTLA-4) that tumors use to turn off immune cells; blocking two switches restores immune attack.',
    detailedNote:
      'Pairs such as CD274 + CTLA4 and PDCD1 + CTLA4 belong to this class, helping reactivate exhausted immune cells inside tumors.',
  },
  {
    term: 'scRNA-seq (Single-Cell RNA Sequencing)',
    domain: 'Single-Cell & TME',
    shortDef:
      'A technology that measures gene expression inside individual cells rather than averaging across a whole tissue sample.',
    detailedNote:
      'Because tumors contain a mix of cancer cells, T cells, macrophages, and normal cells, single-cell data reveals exactly which cell types express each target.',
  },
  {
    term: 'Tumor Microenvironment (TME)',
    domain: 'Single-Cell & TME',
    shortDef:
      'The surrounding neighborhood of a tumor — including immune cells, regulatory Tregs, macrophages, and blood vessels.',
    detailedNote:
      'BSPAI evaluates gene expression across tumor lung, matched normal lung, lymph node, and brain metastasis compartments.',
  },
  {
    term: 'Harmonic Safety Index (score_safe)',
    domain: 'Single-Cell & TME',
    shortDef:
      'A safety metric rewarding pairs where BOTH targets are stronger in tumor tissue than in healthy normal tissue.',
    detailedNote:
      'Computed as 2·dx·dy/(dx+dy) when both tumor-over-normal deltas are positive; it penalizes combinations where either target creates healthy-organ toxicity risk.',
  },
  {
    term: 'Double-Positive & Single-Positive Ratios',
    domain: 'Single-Cell & TME',
    shortDef:
      'The share of cells expressing BOTH target genes together versus expressing AT LEAST ONE target gene.',
    detailedNote:
      'High double-positive ratios favor same-cell dual receptor blockade, while complementary single-positive patterns fit two-cell bridging designs.',
  },
  {
    term: 'Single-Cell Pearson Correlation',
    domain: 'Single-Cell & TME',
    shortDef:
      'Measures from -1 to +1 whether two target genes tend to rise and fall together inside the same cells.',
    detailedNote:
      'Computed overall and per cell compartment to separate co-regulated target pairs from mutually exclusive cell-surface markers.',
  },
  {
    term: 'Pathway Co-Occurrence (KEGG & GO)',
    domain: 'Ranking & Evaluation',
    shortDef:
      'Counts how many curated biological pathways and Gene Ontology processes include both target genes.',
    detailedNote:
      'Shared KEGG signaling pathways and GO biological processes highlight complementary or connected biological roles.',
  },
  {
    term: 'Gene2Vec Embedding Distance (vector_score)',
    domain: '3D Structure & Graph AI',
    shortDef:
      'Measures functional similarity between two genes in a learned 200-dimensional co-expression vector space.',
    detailedNote:
      'Captures broad biological relatedness between gene1 and gene2 even across tissues with sparse single-cell measurements.',
  },
  {
    term: 'AlphaFold 3D & pLDDT Confidence',
    domain: '3D Structure & Graph AI',
    shortDef:
      'Predicted 3D folded protein coordinates paired with a 0–100 confidence score (pLDDT) for every amino acid.',
    detailedNote:
      'Powers the interactive 3D structure explorer below each report so you can inspect folded domains and exposed surface regions.',
  },
  {
    term: 'GNN Structural Embedding',
    domain: '3D Structure & Graph AI',
    shortDef:
      'A Graph Neural Network that summarizes a protein’s 3D amino-acid neighborhood into a rotation-invariant vector.',
    detailedNote:
      'Connects amino-acid residues within 10 Å in 3D space to estimate pocket accessibility and dual-arm steric fit between two targets.',
  },
  {
    term: 'Ectodomain Accessibility',
    domain: '3D Structure & Graph AI',
    shortDef:
      'How cleanly a target’s binding loop sticks out on the outside of the cell membrane where an antibody arm can reach it.',
    detailedNote:
      'Targets buried across many transmembrane passes have tighter physical space, which our 3D fit check flags early.',
  },
];

const DOMAINS: Array<'All' | GlossaryDomain> = [
  'All',
  'Targets & Pharmacology',
  'Single-Cell & TME',
  '3D Structure & Graph AI',
  'Ranking & Evaluation',
];

export function GlossaryView() {
  const [selectedDomain, setSelectedDomain] = useState<'All' | GlossaryDomain>('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [activeModalEntry, setActiveModalEntry] = useState<GlossaryEntry | null>(null);

  const filteredEntries = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return GLOSSARY_ENTRIES.filter((entry) => {
      if (selectedDomain !== 'All' && entry.domain !== selectedDomain) return false;
      if (!q) return true;
      return (
        entry.term.toLowerCase().includes(q) ||
        entry.shortDef.toLowerCase().includes(q) ||
        entry.detailedNote.toLowerCase().includes(q)
      );
    });
  }, [selectedDomain, searchQuery]);

  return (
    <section className="space-y-6 pb-8">
      {/* Header */}
      <div className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl p-7 sm:p-8 space-y-5">
        <div className="space-y-1.5 max-w-2xl">
          <div className="text-xs font-medium text-[#7C5333]">
            Reference lexicon
          </div>
          <h1 className="text-2xl sm:text-3xl font-semibold text-[#231D18]">
            Research Glossary &amp; Definitions
          </h1>
          <p className="text-sm text-[#574D43] leading-relaxed">
            Key immunology, single-cell transcriptomics, and structural modeling concepts used across Ubhaya-Bandha. Select any card for a detailed methodological note.
          </p>
        </div>

        <div className="flex flex-col md:flex-row md:items-center gap-3">
          <div className="relative flex-1 max-w-md">
            <Search className="w-4 h-4 text-[#8B7F72] absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="search"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search biology or ML terms..."
              aria-label="Search glossary"
              className="w-full h-10 pl-10 pr-3.5 bg-[#FAF6F0] border border-[#DFD4C3] rounded-lg text-sm text-[#231D18] focus:outline-none focus:bg-[#FFFEFC] focus:border-[#7C5333]"
            />
          </div>

          <div className="flex flex-wrap gap-1.5">
            {DOMAINS.map((dom) => {
              const active = selectedDomain === dom;
              return (
                <button
                  key={dom}
                  type="button"
                  onClick={() => setSelectedDomain(dom)}
                  className={`px-3 py-1.5 text-xs rounded-lg border transition-colors cursor-pointer ${
                    active
                      ? 'border-[#2D241E] bg-[#2D241E] text-[#FFFEFC] font-medium'
                      : 'border-[#DFD4C3] bg-[#FAF6F0] text-[#574D43] hover:bg-[#EFE7DC]'
                  }`}
                >
                  {dom}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Glossary Cards Grid */}
      {filteredEntries.length === 0 ? (
        <div className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl p-10 text-center text-sm text-[#7A6E61]">
          No matching terms for “{searchQuery}”. Try another search or switch back to All.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {filteredEntries.map((entry) => (
            <article
              key={entry.term}
              className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl p-6 flex flex-col justify-between gap-4 hover:border-[#BFA98E] transition-colors"
            >
              <div className="space-y-2">
                <div className="text-xs font-medium text-[#7C5333]">
                  {entry.domain}
                </div>
                <h2 className="text-lg font-semibold text-[#231D18]">
                  {entry.term}
                </h2>
                <p className="text-sm text-[#574D43] leading-relaxed">
                  {entry.shortDef}
                </p>
              </div>

              <div>
                <button
                  type="button"
                  onClick={() => setActiveModalEntry(entry)}
                  className="inline-flex items-center gap-1 text-xs font-medium text-[#7C5333] hover:text-[#231D18] transition-colors cursor-pointer"
                >
                  Read detailed note
                  <ArrowUpRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </article>
          ))}
        </div>
      )}

      {/* Clean Modal for "Read more" (Reference-inspired) */}
      {activeModalEntry && (
        <div
          className="fixed inset-0 z-50 bg-[#2B231C]/45 flex items-center justify-center p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="glossary-modal-title"
          onClick={() => setActiveModalEntry(null)}
        >
          <div
            className="bg-[#FFFDF9] border border-[#DFCEB8] rounded-3xl max-w-lg w-full p-6 sm:p-8 shadow-xl space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-[11px] font-semibold tracking-[0.14em] uppercase text-[#8B5E3C]">
                  Glossary note · {activeModalEntry.domain}
                </div>
                <h2
                  id="glossary-modal-title"
                  className="mt-1 text-xl font-bold text-[#2B231C]"
                >
                  {activeModalEntry.term}
                </h2>
              </div>
              <button
                type="button"
                aria-label="Close modal"
                onClick={() => setActiveModalEntry(null)}
                className="p-1.5 rounded-lg text-[#786A5B] hover:bg-[#F3EADF] cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-sm text-[#2B231C] leading-relaxed">
              {activeModalEntry.shortDef}
            </p>

            <div className="p-4 rounded-2xl bg-[#F6EFE6] border border-[#E5D8C5] text-sm text-[#5E5245] leading-relaxed">
              {activeModalEntry.detailedNote}
            </div>

            <div className="pt-1 flex justify-end">
              <button
                type="button"
                onClick={() => setActiveModalEntry(null)}
                className="px-5 py-2 text-xs font-semibold text-[#FFFDF9] bg-[#8B5E3C] hover:bg-[#734B2E] rounded-xl transition-colors cursor-pointer"
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
