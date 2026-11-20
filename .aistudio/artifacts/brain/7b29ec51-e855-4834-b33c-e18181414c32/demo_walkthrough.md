# BiSpec Pairwise AI (BSPAI) — 6–8 Minute Demo Video Walkthrough

A timed, researcher-focused presentation guide for demonstrating **BiSpec Pairwise AI (BSPAI)** end-to-end across the live web workbench, the underlying ML pipeline, and the exportable Python codebase.

---

## Segment 1 (0:00 – 1:00) — Problem Framing & Architecture

- **Talking Points**:
  - Bispecific antibodies (BsAbs) pair two targets simultaneously, creating a combinatorial search space of $G(G-1)/2$ candidate pairs across hundreds of membrane proteins (in this catalog, **327 curated genes = 53,301 candidate pairs**, computed dynamically).
  - Standard binary classification struggles because approved/clinical BsAbs are scarce (~50+ labeled clinical pairs vs. >50,000 unlabeled candidates).
  - BSPAI adapts Zhang, Wang & Sun (*J Cancer Res Clin Oncol* 2024, 150:237) using **pairwise learning-to-rank** (`XGBRanker`, `rank:pairwise`) over five biological feature groups and **retrieval-augmented generation (RAG)** grounded in quantile-discretized model drivers.
- **On Screen**:
  - Show the top workspace telemetry ribbon highlighting the explicit **`DEMO DATA`** badge, dataset ID (`SYNTHETIC_NSCLC_GSE131907_LIKE_V1`), dynamic gene/pair count (`327 genes · 53,301 candidate pairs`), seed (`42`), and model version (`BSPAI-PairwiseXGB-v1.0.0`).

---

## Segment 2 (1:00 – 2:45) — Live Prediction & Feature Inspection (`CD274 + CTLA4`)

- **Talking Points**:
  - Query **`CD274`** (PD-L1) and **`CTLA4`** (CTLA-4) in the autocomplete selectors—the target pair behind **KN046** (Phase 3 NSCLC/PDAC, coded Stage 2).
  - Highlight the four-card readout:
    - **Druggability Score** (`5.2687`) and **Global Rank** (`#35 / 53,301`, top `0.06%`).
    - **Inferred Mechanism Class** (`tumor-immunotherapy`) and **Clinical Benchmark Status** (`Stage 2 · KN046`, marked for manual verification).
  - Scroll to the **Top-40 Feature Frequency Importances** chart and the **60-Feature Vector Inspector**, filtering across the five biological groups:
    1. **Safety**: `dx_tumor_minus_normal`, `dy_tumor_minus_normal`, `score_safe = (dx * dy) / (dx + dy)`.
    2. **Mechanism of Action**: `kegg_cooccur`, `go_cooccur`, `pathway_cooccur`, `same_pathway`.
    3. **Gene Embedding**: Gene2Vec-style co-expression Euclidean distance `vector_score`.
    4. **Target Activity**: `double_ratio`, `sum_single_ratio`, `min_single_ratio`, `sum_single_exp`, `min_single_exp` overall, across 4 tissues (`tumor_lung`, `normal_lung`, `lymph_node`, `brain_mets`), and across 4 key cell types (`exhausted_cd8_t`, `treg`, `macrophage_myeloid`, `epithelial`).
    5. **Expression Correlation**: Pearson $r$ overall and per cell population.
  - Click **Export Features CSV** to demonstrate downloadable provenance-tagged feature matrices.

---

## Segment 3 (2:45 – 4:15) — Grounded RAG Report & Provenance Stamp

- **Talking Points**:
  - Point to the **Discretized Top-6 Model Drivers** table: training-set equal-frequency quantile bin edges (`models/bin_edges.json`) map raw continuous features into human-interpretable labels (*very similar* $\rightarrow$ *very dissimilar*, *unsafe* $\rightarrow$ *high safety*, *strongly negative* $\rightarrow$ *strongly positive*).
  - Show the **Provenance Stamp** attached to every report (`Dataset`, `Mode`, `Seed`, `Model Version`, `Synthesizer Mode` — `gemini-4-hydrogen` via `@google/genai` when `GEMINI_API_KEY` is configured, or `deterministic_template` fallback when absent).
  - Highlight that the report cites exact model outputs, never invents clinical trial efficacy results, and includes a research-aid disclaimer.
  - Click **Copy Markdown** and test switching to another reference pair (`EGFR + MET`, **Amivantamab**) to show live recalculation.

---

## Segment 4 (4:15 – 5:30) — Leaderboard Filtering & CSV Export

- **Talking Points**:
  - Switch to the **Leaderboard** tab showing the top-100 globally ranked pairs.
  - Filter by mechanism class (`Bridge Two Cells`, `TAA + TAA`, `Tumor Immunotherapy`) and toggle **Clinical benchmarks only** to inspect how known Stage 1–3 drugs (`Glofitamab`, `Blinatumomab`, `Cadonilimab`, `Ivonescimab`, `Amivantamab`) rank alongside exploratory candidates like `CD3E + CD40`.
  - Click **Export Filtered CSV** (or **Export Top-100 CSV** in the top bar).

---

## Segment 5 (5:30 – 6:45) — Model Card: 5-Fold CV, ROC Curves & Ablation

- **Talking Points**:
  - Open the **Model Card** tab to inspect QC statistics (`684 / 720` cells retained after filtering 36 low-quality cells).
  - Walk through the **5-Fold Mean ROC Curves** and **8-Model Comparison Table** (Pairwise XGBoost achieving `0.8941 ± 0.0389` AUC on synthetic demo data, compared honestly alongside Pointwise XGBoost, GBDT, Random Forest, DNN MLP `256-128-64-32-1`, Logistic Regression, Decision Tree, and Single Feature `double_ratio`, with the Zhang et al. 2024 NSCLC reference column `0.8929`).
  - Show the **6-Step Incremental Feature Ablation** table and discuss the Limitations panel (synthetic demo mode, small labeled sample count, reconstructed supplementary tables).
  - Briefly open the **Glossary** tab to show the standalone plain-language reference for all 11 biological terms.

---

## Segment 6 (6:45 – 7:30) — Python Pipeline, Real GEO Ingestion & Parity Verification

- **Talking Points**:
  - Show the terminal commands in `README.md` for running the full Python stack locally:
    - `pip install -r requirements.txt && python scripts/run_all.py`
    - `uvicorn app.main:app --reload`
  - Point out `bspai/data.py:load_geo_dataset` for dropping real `.h5ad` or 10x `.mtx` files from `GSE131907`, `GSE205013`, `GSE164690`, or `GSE156625` into `data/raw/` with per-cell-type stratified subsampling.
  - Conclude by running `npm test` and `python3 -m unittest tests/test_formulas.py tests/test_geo_loader.py -v`, demonstrating that all formula tests, GEO loader tests, API smoke tests, and the **60-feature TypeScript-vs-Python parity test (`seed=42`)** pass.
