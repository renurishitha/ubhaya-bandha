# BiSpec Pairwise AI (BSPAI) — Implementation Plan

A working, demo-ready prototype of **BiSpec Pairwise AI (BSPAI)** reimplementing Zhang, Wang, and Sun (*J Cancer Res Clin Oncol* 2024, 150:237). It ranks candidate bispecific antibody target gene pairs using single-cell biological features, pairwise learning-to-rank, and retrieval-augmented analytical reports built for target-discovery research.

## User Review & Critical Decisions

> [!IMPORTANT]
> Confirmed Phase 1 choices and all 7 revision requirements are incorporated below.

- **Dual-Stack Delivery**: Because this live preview container runs Node.js 22 without Python `pip` or native ML wheels, the live interactive application runs on an Express + TypeScript ML/API backend and React + Vite frontend on port 3000, while the complete Python 3.11 (`FastAPI`, `XGBoost`, `scikit-learn`, `Scanpy`, `Gensim`, `pytest`) repository is packaged alongside it for local execution via `pip install -r requirements.txt && python scripts/run_all.py` and `uvicorn app.main:app --reload`.
- **Target Audience & Domain Posture**: Built strictly for **researchers** (computational biologists and therapeutic target-selection scientists). The main UI assumes domain fluency and omits inline biological tooltips; a dedicated **Glossary** tab and a Glossary section in the `README` provide one-sentence plain-language definitions (`bispecific antibody`, `antigen / target`, `scRNA-seq`, `tumor microenvironment`, `TAA`, `immune checkpoint`, `T cell`, `KEGG`, `GO`, `clinical phase`, `AUC`).
- **Gemini Model Configuration & Validation**: The server reads `GEMINI_MODEL` from the environment (defaulting to `gemini-4-hydrogen`) and validates at startup/invocation against an explicit allowlist of valid current `@google/genai` models (`gemini-4-hydrogen`, `gemini-4-argon`, `gemini-4-carbon`, `gemini-3.1-flash-lite`), rejecting deprecated aliases (`gemini-1.5-*`, `gemini-2.0-*`) with a clear configuration error, and falling back to a deterministic template report when `GEMINI_API_KEY` is absent.
- **Dynamic Candidate-Pair Count**: Total candidate pairs are computed dynamically from the loaded membrane-protein gene list as $N_{\text{pairs}} = \binom{G}{2} = G(G-1)/2$ (e.g., 300 genes $\rightarrow$ 44,850 pairs; 223 genes $\rightarrow$ 24,753 pairs) and displayed accurately across ranks, leaderboards, and model cards rather than hard-coding `~24,753`.
- **Visual Direction**: Clean clinical light lab dashboard with calibrated tabular monospace numerics and crisp hairline hierarchy.

---

## 1. Overview & Core Concept

- **What It Does**: Researchers select any two candidate target genes (e.g., `CD274` and `CTLA4`) and inspect:
  1. Predicted druggability score and exact global rank (`Rank k / N_pairs`) computed from the active gene catalog.
  2. Quantitative feature values across five biological dimensions and top feature-frequency importances driving the score.
  3. An interpretable RAG-grounded analytical report stamped with **dataset identifier, random seed, and model version**, with CSV and Markdown export.
- **Target Audience**: Computational biologists and therapeutic target-selection scientists evaluating bispecific antibody (BsAb) target combinations.
- **Key Value**: Addresses severe label sparsity in BsAb clinical data via pairwise learning-to-rank (`rank:pairwise`), while exposing auditable feature evidence, cross-validated benchmarks, and reproducible cross-language parity.

---

## 2. User Experience & Visual Design

### Key User Flows

1. **Predict View (Primary Workbench)**:
   - Dual searchable gene autocomplete inputs pre-seeded with `CD274` and `CTLA4`.
   - Telemetry summary strip displaying Druggability Score, Dynamic Global Rank (`#rank / computed total`), Mechanism Category, and Clinical Trial Status (including verification markers on curated clinical pairs).
   - Five-group Feature Inspector (Safety, Mechanism/Pathway, Gene Embedding, Single-Cell Activity, Expression Correlation) paired with an interactive Top-40 Feature Weight bar chart and **Export Features CSV** action.
   - RAG Report panel stamped with dataset, seed, and model version metadata, Copy Markdown button, research-aid disclaimer, and optional web-search grounding flag.
2. **Leaderboard View**:
   - Top-100 ranked target pairs rendered with `tabular-nums` alignment, mechanism filters (`All`, `Bridge Two Cells`, `TAA + TAA`, `Tumor Immunotherapy`), clinical-trial status markers, and an **Export Leaderboard CSV** button.
   - Clicking any row loads that pair directly into the Predict Workbench.
3. **Model Card View**:
   - Dataset provenance summary (active synthetic demo cohort vs. documented real GEO ingestion workflow for `GSE131907`, `GSE205013`, `GSE164690`, `GSE156625`), dynamically computed gene/pair counts, and active seed/model version.
   - Honest 5-fold cross-validation results table for all 8 models (Single Feature, Logistic Regression, Decision Tree, Random Forest, GBDT, MLP/DNN, Pointwise XGBoost, Pairwise XGBoost) alongside Zhang et al. (2024) paper reference figures.
   - Interactive multi-model ROC curve plot, 6-step incremental feature ablation chart, and explicit limitations disclosure.
4. **Glossary View**:
   - Standalone reference tab providing concise one-sentence plain-language definitions for all 11 required domain terms, separated cleanly from the researcher-focused Predict, Leaderboard, and Model Card workflows.

### Visual Identity & Theme

- **Aesthetic Direction**: Clean clinical light lab dashboard evoking precision instrumentation and empirical rigor.
- **Color Palette (60-30-10)**:
  - **60% Dominant Neutral Canvas**: Crisp white (`#FFFFFF`) and lab slate (`#F8FAFC`).
  - **30% Structural Surfaces**: Subtle slate panels (`#F1F5F9`) and 1px hairline borders (`#E2E8F0`), avoiding nested bordered cards or thick left accent bars.
  - **10% Precision Accents**: Calibrated cobalt (`#2563EB`) for primary actions and focal curves, clinical cyan (`#0891B2`) for secondary series, and semantic emerald (`#059669`) / amber (`#D97706`) paired with explicit text labels.
- **Typography & Hierarchy**: `IBM Plex Sans` for headings and analytical prose; `IBM Plex Mono` (`tabular-nums`) for all scores, ranks, feature vectors, and tables. No static pill enclosures on metadata (quiet `·` separators) and `text-wrap: balance` on display headings.
- **Top Navigation Bar**: Strict three-zone layout (`BSPAI` wordmark — `Predict`, `Leaderboard`, `Model Card`, `Glossary` links — `Export CSV` primary action), with a clear `DEMO DATA` banner/badge in the workspace toolbar whenever running on synthetic data.

---

## 3. Key Product Decisions & Trade-Offs

- **Decision 1: Executable TypeScript Engine + Complete Python Reference Repo + Parity Testing**
  - *Chosen Approach*: Run the full BSPAI mathematical pipeline (synthetic scRNA-seq generation, 5 feature groups, pairwise gradient-boosted ranking, 5-fold CV, 8-model benchmark, ablation, equal-frequency discretization, and Gemini/template RAG) in TypeScript on the live Express server, maintain the complete idiomatic Python 3.11 (`FastAPI` + `XGBoost` + `Scanpy` + `Gensim`) repository in parallel, and include a deterministic **cross-language parity test** verifying that with the same seed and input matrix, both engines produce matching feature formulas and model scores within numerical tolerance.
  - *Why*: Guarantees genuine live computation in this Node container, full local reproducibility in Python, and mathematically verified alignment between both implementations.
- **Decision 2: Documented & Tested Real GEO Ingestion Path (Python Repo)**
  - *Chosen Approach*: Implement a tested ingestion module in the Python codebase supporting `.h5ad` and 10x `.mtx` bundles placed in `data/raw/` for `GSE131907` (NSCLC), `GSE205013` (PDAC), `GSE164690` (HNSCC), and `GSE156625` (HCC), with configurable per-cell-type stratified subsampling to fit laptop RAM and automated unit tests using realistic fixture matrices.
  - *Why*: Makes transitioning from synthetic demo mode to real GEO cohorts a verified, turnkey path rather than untested boilerplate.
- **Decision 3: Environment-Configured & Validated Gemini Model (`GEMINI_MODEL`)**
  - *Chosen Approach*: Read `GEMINI_MODEL` from `.env` (default `gemini-4-hydrogen`), validate against the current supported `@google/genai` model catalog before invoking `ai.models.generateContent`, and fall back to the deterministic template engine when `GEMINI_API_KEY` is unset. Every generated report (LLM or fallback) is stamped with `dataset`, `seed`, and `model_version`.
  - *Why*: Prevents stale/deprecated model drift, keeps keys server-side, and guarantees traceable reports.
- **Decision 4: Honest Evaluation & Dynamic Pair Counting**
  - *Chosen Approach*: Dynamically compute candidate pair count $\binom{G}{2}$ from the loaded gene list and report actual 5-fold CV AUC and ablation metrics alongside Zhang et al. (2024) reference figures (`0.8929` AUC; `54.76 -> 67.06 -> 71.74 -> 73.72 -> 86.19 -> 89.29`).

---

## 4. Technical Architecture & Data Strategy

### System Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────────┐
│                   Browser UI (React + Vite + Tailwind)                    │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ Predict View │  │ Leaderboard    │  │ Model Card   │  │ Glossary    │  │
│  │ + CSV Export │  │ + CSV Export   │  │ CV/ROC/Ablat │  │ Tab Only    │  │
│  └──────┬───────┘  └───────┬────────┘  └──────┬───────┘  └──────┬──────┘  │
└─────────┼──────────────────┼──────────────────┼─────────────────┼─────────┘
          │                  │                  │                 │
          ▼                  ▼                  ▼                 ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                 Express API Server (Port 3000) & Python FastAPI Parity    │
│  POST /api/predict · GET /api/top-pairs · GET /api/features/:pair         │
│  GET /api/metrics  · POST /api/report   · GET /api/genes · CSV Export     │
└─────────┬──────────────────────────────┬────────────────────────┬─────────┘
          │                              │                        │
          ▼                              ▼                        ▼
┌──────────────────────┐    ┌────────────────────────┐   ┌──────────────────┐
│ Precomputed Pipeline │    │  RAG Retrieval & Bins  │   │  Gemini Client   │
│ - Synthetic scRNA    │    │  - Top-6 Feature Bins  │   │  GEMINI_MODEL    │
│ - Dynamic G*(G-1)/2  │    │  - Score & Global Rank │   │  (Validated ID)  │
│ - 5 Feature Groups   │    │  - Provenance Stamp    │   │  + Deterministic │
│ - Pairwise + 7 Base  │    │    (dataset/seed/ver)  │   │  Template Engine │
└──────────┬───────────┘    └────────────────────────┘   └──────────────────┘
           │
           ▼  [Verified by TS <-> Python Parity Test (shared seed)]
┌───────────────────────────────────────────────────────────────────────────┐
│   Exportable Python 3.11 Package (FastAPI · XGBoost · Scanpy · Gensim)    │
│   + Tested GEO .h5ad/.mtx Loader (GSE131907/205013/164690/156625)         │
└───────────────────────────────────────────────────────────────────────────┘
```

### Data Model & Mathematical Pipeline

1. **Data, Dynamic Pair Space & GEO Ingestion**:
   - Curated CSV of ~300 membrane-protein target genes; candidate pair space dynamically computed as $N_{\text{pairs}} = G(G-1)/2$.
   - Synthetic demo single-cell matrix across four tissue compartments (`tumor_lung`, `normal_lung`, `lymph_node`, `brain_mets`) and four cell populations (`exhausted_cd8_t`, `treg`, `macrophage_myeloid`, `epithelial`).
   - Python GEO loader (`h5ad` and `mtx`) with per-cell-type subsampling, QC filtering, log-normalization, HVG selection, PCA, clustering, and cell-type annotation.
2. **Five Feature Groups (per gene pair $x, y$)**:
   - **Safety**: $d = \text{expr}_{\text{tumor}} - \text{expr}_{\text{normal}}$; $\text{score\_safe} = (d_x \cdot d_y)/(d_x + d_y)$.
   - **Mechanism of Action**: KEGG and GO local pathway co-occurrence frequency + binary `same_pathway`.
   - **Gene Embedding**: Co-expression Gene2Vec-style vectors; Euclidean distance `vector_score` $= \sqrt{\sum_i (x_i - y_i)^2}$.
   - **Single-Cell Target Activity**: $P_{\text{gene}}$ and $P_{x,y}$ at configurable threshold $T$, yielding `double_ratio`, `sum_single_ratio`, `min_single_ratio`, `sum_single_exp`, and `min_single_exp` overall, per tissue, and per key cell type.
   - **Expression Correlation**: Pearson $r$ overall and per cell type.
3. **Pairwise Learning, Baselines & Evaluation**:
   - Clinical stage coding 1 (Approved/NDA/BLA) through 5 (Discontinued/pending), seeded with `cadonilimab/AK104`, `KN046`, `ivonescimab/AK112`, `amivantamab`, `glofitamab`, and verification flags (`needs_verification`).
   - XGBoost `rank:pairwise` (`learning_rate=0.1`, `n_estimators=100`, `max_depth=10`) plus 7 baselines evaluated under 5-fold CV (AUC binarizing stages 1–3 as positive), ROC curves, Top-40 feature weights, and 6-step incremental ablation.
4. **Discretization, CSV Export & Provenance-Stamped RAG**:
   - Top 6 features discretized into up to 5 equal-frequency bins persisted from training data; retrieved context feeds `GEMINI_MODEL` (validated env var, default `gemini-4-hydrogen`) or deterministic template fallback.
   - Every report includes a structured header stamp (`Dataset`, `Seed`, `Model Version`, `Generation Mode`), research-aid disclaimer, and non-fabrication constraint; CSV export endpoints/downloads allow researchers to export ranked pairs and feature matrices directly.

### Milestone Execution & Verification Plan

- **M1 — Repo Scaffold, Synthetic Data & GEO Loader**: Set up project layout, curated ~300-gene list, KEGG/GO local pathways, deterministic synthetic scRNA-seq generator, preprocessing, and tested Python GEO `.h5ad`/`.mtx` loader.
- **M2 — Five Feature Groups, Formula Unit Tests & TS/Python Parity**: Implement all five feature groups in both TypeScript and Python; verify exact hand-computed formulas and run the cross-language parity test (shared seed, tolerance check).
- **M3 — Labels, Pairwise Ranker, Baselines, 5-Fold CV & Ablation**: Seed clinical stage labels, train pairwise ranker + 7 baselines under 5-fold CV, save models and bin edges, generate ROC/ablation/Top-40 metrics, and verify pairwise stage ordering.
- **M4 — Discretization, Provenance-Stamped RAG & Model Validation**: Implement 5-bin quantile discretization, validated `GEMINI_MODEL` env loader, deterministic template fallback, and provenance stamp (`dataset`, `seed`, `model_version`).
- **M5 — Web App, CSV Export, API Smoke Tests & Browser Verification**: Build Express API routes and clean clinical light lab UI (Predict, Leaderboard, Model Card, Glossary), CSV exports, compile/lint checks, API smoke tests, and headless browser screenshots for `CD274+CTLA4`.
- **M6 — README, Mermaid Architecture, Glossary & Demo Walkthrough**: Write `README.md` (with Mermaid diagram, glossary, GEO ingestion guide, and parity test instructions) and deliver the 6–8 minute demo video walkthrough artifact.
