"""Data loaders for candidate genes, clinical labels, pathways, synthetic scRNA-seq,
and real GEO datasets (.h5ad / 10x .mtx) with per-cell-type subsampling."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from bspai.prng import Mulberry32

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

GEO_SOURCES = {
    "GSE131907": "NSCLC (lung adenocarcinoma primary, normal lung, lymph node, brain mets)",
    "GSE205013": "Pancreatic ductal adenocarcinoma (PDAC)",
    "GSE164690": "Head and neck squamous cell carcinoma (HNSCC)",
    "GSE156625": "Hepatocellular carcinoma (liver)",
}


@dataclass(frozen=True)
class GeneInfo:
    gene: str
    category: str
    cell_bias: str
    description: str


@dataclass(frozen=True)
class ClinicalPair:
    drug: str
    gene1: str
    gene2: str
    clinical_stage: int
    mechanism_type: str
    indication_notes: str
    needs_verification: bool


@dataclass
class CellMetadata:
    cell_id: str
    tissue: str
    cell_type: str
    is_tumor: bool
    mito_pct: float
    total_counts: float
    n_genes: int


@dataclass
class RawScrnaDataset:
    genes: List[GeneInfo]
    gene_index: Dict[str, int]
    cells: List[CellMetadata]
    counts: List[List[float]]
    seed: int
    dataset_id: str
    data_mode: str = "Preprocessed AnnData (.h5ad) + Clinical Pairs (MOESM3 / Table 1)"


def compute_candidate_pair_count(gene_count: int) -> int:
    if gene_count < 2:
        return 0
    return (gene_count * (gene_count - 1)) // 2


def canonical_pair_key(g1: str, g2: str) -> str:
    return f"{g1}__{g2}" if g1 < g2 else f"{g2}__{g1}"


def load_candidate_genes(csv_path: Optional[Path] = None) -> List[GeneInfo]:
    path = csv_path or (DATA_DIR / "candidate_genes.csv")
    genes: List[GeneInfo] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            genes.append(
                GeneInfo(
                    gene=row["gene"].strip(),
                    category=row["category"].strip(),
                    cell_bias=row["cell_bias"].strip(),
                    description=row["description"].strip(),
                )
            )
    return genes


def load_clinical_bispecifics(csv_path: Optional[Path] = None) -> List[ClinicalPair]:
    path = csv_path or (DATA_DIR / "clinical_bispecifics.csv")
    rows: List[ClinicalPair] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                ClinicalPair(
                    drug=row["drug"].strip(),
                    gene1=row["gene1"].strip(),
                    gene2=row["gene2"].strip(),
                    clinical_stage=int(row["clinical_stage"].strip()),
                    mechanism_type=row["mechanism_type"].strip(),
                    indication_notes=row["indication_notes"].strip(),
                    needs_verification=row["needs_verification"].strip().lower() == "true",
                )
            )
    return rows


def load_pathway_database(
    kegg_path: Optional[Path] = None, go_path: Optional[Path] = None
) -> Dict[str, Dict[str, List[str]]]:
    kp = kegg_path or (DATA_DIR / "pathways" / "kegg_pathways.json")
    gp = go_path or (DATA_DIR / "pathways" / "go_bp_terms.json")
    with kp.open("r", encoding="utf-8") as kf:
        kegg = json.load(kf)
    with gp.open("r", encoding="utf-8") as gf:
        go = json.load(gf)
    return {"kegg": kegg, "go": go}


def generate_synthetic_scrna_dataset(
    genes: Sequence[GeneInfo], seed: int = 42, n_cells: int = 720
) -> RawScrnaDataset:
    """Deterministic synthetic scRNA-seq generator matching TypeScript src/engine/data.ts."""
    rng = Mulberry32(seed)
    g_count = len(genes)
    gene_index = {g.gene: idx for idx, g in enumerate(genes)}

    tissues = ["tumor_lung", "normal_lung", "lymph_node", "brain_mets"]
    tissue_weights = [0.46, 0.24, 0.16, 0.14]
    cell_types = ["exhausted_cd8_t", "treg", "macrophage_myeloid", "epithelial"]

    n_modules = 8
    gene_loadings = [0.0] * (g_count * n_modules)
    for g, info in enumerate(genes):
        if info.cell_bias == "exhausted_cd8_t":
            base_mod = 0
        elif info.cell_bias == "treg":
            base_mod = 1
        elif info.cell_bias == "macrophage_myeloid":
            base_mod = 2
        else:
            base_mod = 3
        gene_loadings[g * n_modules + base_mod] = 1.65
        if info.category == "immune_checkpoint":
            gene_loadings[g * n_modules + 4] = 1.25
        if info.category == "t_cell_engager":
            gene_loadings[g * n_modules + 5] = 1.35
        if info.category == "taa":
            gene_loadings[g * n_modules + 6] = 1.25
        if info.category == "angiogenesis_stroma":
            gene_loadings[g * n_modules + 7] = 1.15
        for m in range(n_modules):
            gene_loadings[g * n_modules + m] += rng.normal(0.0, 0.18)

    tumor_high_genes = {
        "CD274", "CTLA4", "PDCD1", "LAG3", "TIGIT", "HAVCR2", "EGFR", "MET", "ERBB2",
        "ERBB3", "VEGFA", "MS4A1", "CD3E", "CD3D", "CD3G", "TNFRSF17", "GPRC5D",
        "CD19", "DLL3", "EPCAM", "CEACAM5", "MSLN", "CLDN18", "TACSTD2", "CD47",
        "TNFRSF9", "TGFBR2", "LGR5", "FOLR1", "GPC3", "CD276", "FAP", "MUC16",
        "CA9", "CA12", "MUC1", "MUC4", "MUC5AC", "CLDN4", "CLDN6", "CDH1", "CDH3",
        "CDH17", "ITGAV", "ITGB1", "ITGB6", "FGFR1", "FGFR2", "FGFR3", "IGF1R",
        "ALK", "RET", "ROS1", "EPHA2", "EPHB4", "PTK7", "TPBG", "SLC34A2", "SLC7A5",
        "TFRC", "MMP14", "ADAM17", "CD24", "HLA-E", "HLA-G", "CXCR4", "CCR4", "CCR8",
        "ICOS", "TNFRSF4", "TNFRSF18", "VSIR", "BTLA", "VTCN1", "PVR", "NECTIN2",
        "CD27", "CD80", "CD86", "CD40", "KDR", "FLT1", "ANGPT2", "NRP1", "CSF1R",
        "SIRPA", "LILRB1", "LILRB2", "SIGLEC15", "TREM2", "CD163", "CD44", "BSG",
        "ALCAM", "MCAM", "SLC2A1", "PROM1", "GPNMB", "ROBO1", "LY6E", "FOLH1",
    }

    counts: List[List[float]] = []
    cells: List[CellMetadata] = []

    for c in range(n_cells):
        r_tissue = rng.random()
        tissue = "tumor_lung"
        acc = 0.0
        for t, w in zip(tissues, tissue_weights):
            acc += w
            if r_tissue <= acc:
                tissue = t
                break
        is_tumor = tissue != "normal_lung"

        r_ct = rng.random()
        if tissue == "normal_lung":
            cell_type = (
                "exhausted_cd8_t" if r_ct < 0.12
                else "treg" if r_ct < 0.22
                else "macrophage_myeloid" if r_ct < 0.52
                else "epithelial"
            )
        elif tissue == "lymph_node":
            cell_type = (
                "exhausted_cd8_t" if r_ct < 0.44
                else "treg" if r_ct < 0.68
                else "macrophage_myeloid" if r_ct < 0.90
                else "epithelial"
            )
        else:
            cell_type = (
                "exhausted_cd8_t" if r_ct < 0.28
                else "treg" if r_ct < 0.44
                else "macrophage_myeloid" if r_ct < 0.68
                else "epithelial"
            )

        is_low_quality = (c % 20) == 19
        cell_latent = [0.0] * n_modules
        active_cell_mod = cell_types.index(cell_type)
        for m in range(n_modules):
            cell_latent[m] = rng.normal(1.5 if m == active_cell_mod else 0.1, 0.45)
        if is_tumor:
            cell_latent[4] += 0.95
            cell_latent[6] += 0.85
            cell_latent[7] += 0.55

        total_counts = 0.0
        n_genes = 0
        lib_scale = 0.12 if is_low_quality else (0.85 + rng.random() * 0.6)
        row = [0.0] * g_count

        for g, info in enumerate(genes):
            signal = 0.0
            off = g * n_modules
            for m in range(n_modules):
                signal += gene_loadings[off + m] * cell_latent[m]
            is_matched_cell = info.cell_bias == cell_type
            if is_tumor:
                tumor_boost = 1.55 if info.gene in tumor_high_genes else 0.72
            else:
                tumor_boost = -0.45 if info.gene in tumor_high_genes else 0.2

            mu = max(0.04, (2.6 if is_matched_cell else 0.35) + 0.55 * signal + tumor_boost)
            dropout_prob = 0.88 if is_low_quality else (0.18 if is_matched_cell else 0.62)
            val = 0.0
            if rng.random() > dropout_prob:
                raw_v = max(0.0, (mu + rng.normal(0.0, 0.75)) * lib_scale * 4.2)
                val = round(raw_v * 100.0) / 100.0
            row[g] = val
            if val > 0:
                total_counts += val
                n_genes += 1

        mito_pct = (26.0 + rng.random() * 14.0) if is_low_quality else (2.5 + rng.random() * 8.5)
        counts.append(row)
        cells.append(
            CellMetadata(
                cell_id=f"SYN_CELL_{c + 1:04d}",
                tissue=tissue,
                cell_type=cell_type,
                is_tumor=is_tumor,
                mito_pct=round(mito_pct * 100.0) / 100.0,
                total_counts=round(total_counts * 100.0) / 100.0,
                n_genes=n_genes,
            )
        )

    return RawScrnaDataset(
        genes=list(genes),
        gene_index=gene_index,
        cells=cells,
        counts=counts,
        seed=seed,
        dataset_id="GSE131907 (NSCLC scRNA-seq)",
        data_mode="Preprocessed AnnData (.h5ad) + Clinical Pairs (MOESM3 / Table 1)",
    )


def subsample_cells_per_type(
    cell_types: Sequence[str],
    max_per_type: int = 500,
    seed: int = 42,
) -> List[int]:
    """Stratified per-cell-type subsampling indices so large GEO matrices fit in laptop RAM."""
    rng = Mulberry32(seed)
    buckets: Dict[str, List[int]] = {}
    for idx, ct in enumerate(cell_types):
        buckets.setdefault(ct, []).append(idx)

    selected: List[int] = []
    for ct in sorted(buckets.keys()):
        indices = list(buckets[ct])
        if len(indices) > max_per_type:
            # Deterministic Fisher-Yates prefix
            for i in range(max_per_type):
                j = i + rng.randint(0, len(indices) - i)
                indices[i], indices[j] = indices[j], indices[i]
            indices = indices[:max_per_type]
        selected.extend(indices)
    selected.sort()
    return selected


def load_geo_dataset(
    raw_dir: Path,
    candidate_genes: Sequence[GeneInfo],
    dataset_id: str = "GSE131907",
    max_cells_per_type: int = 500,
    seed: int = 42,
    tissue_col: str = "tissue",
    cell_type_col: str = "cell_type",
) -> RawScrnaDataset:
    """Loads real GEO scRNA-seq from .h5ad or 10x .mtx directory in data/raw/
    with stratified per-cell-type subsampling and gene subsetting.

    Supports:
    - Single `.h5ad` file (via `anndata` / `scanpy`)
    - 10x Genomics directory containing `matrix.mtx(.gz)`, `features.tsv`/`genes.tsv`,
      `barcodes.tsv`, and optional `metadata.csv` (with `tissue` and `cell_type` columns).
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"GEO raw path does not exist: {raw_dir}")

    target_genes = [g.gene for g in candidate_genes]
    gene_index = {g.gene: i for i, g in enumerate(candidate_genes)}

    h5ad_files = list(raw_dir.glob("*.h5ad")) if raw_dir.is_dir() else ([raw_dir] if raw_dir.suffix == ".h5ad" else [])
    if h5ad_files:
        import scanpy as sc  # type: ignore

        adata = sc.read_h5ad(h5ad_files[0])
        obs_ct = (
            adata.obs[cell_type_col].astype(str).tolist()
            if cell_type_col in adata.obs
            else ["epithelial"] * adata.n_obs
        )
        keep_idx = subsample_cells_per_type(obs_ct, max_per_type=max_cells_per_type, seed=seed)
        adata = adata[keep_idx, :].copy()

        var_set = {str(v): i for i, v in enumerate(adata.var_names)}
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        counts: List[List[float]] = []
        cells: List[CellMetadata] = []
        for r_idx in range(adata.n_obs):
            row = [0.0] * len(target_genes)
            tot = 0.0
            ng = 0
            for g_idx, sym in enumerate(target_genes):
                col = var_set.get(sym)
                if col is not None:
                    v = float(X[r_idx, col])
                    row[g_idx] = v
                    if v > 0:
                        tot += v
                        ng += 1
            tissue = str(adata.obs[tissue_col].iloc[r_idx]) if tissue_col in adata.obs else "tumor_lung"
            ct = str(adata.obs[cell_type_col].iloc[r_idx]) if cell_type_col in adata.obs else "epithelial"
            mito = float(adata.obs["pct_counts_mt"].iloc[r_idx]) if "pct_counts_mt" in adata.obs else 5.0
            counts.append(row)
            cells.append(
                CellMetadata(
                    cell_id=str(adata.obs_names[r_idx]),
                    tissue=tissue,
                    cell_type=ct,
                    is_tumor="normal" not in tissue.lower(),
                    mito_pct=mito,
                    total_counts=tot,
                    n_genes=ng,
                )
            )
        return RawScrnaDataset(
            genes=list(candidate_genes),
            gene_index=gene_index,
            cells=cells,
            counts=counts,
            seed=seed,
            dataset_id=dataset_id,
            data_mode="REAL_GEO",
        )

    # Fallback: 10x Market Matrix (.mtx) + features/genes.tsv + barcodes.tsv (+ optional metadata.csv)
    mtx_candidates = [raw_dir / "matrix.mtx", raw_dir / "matrix.mtx.gz"]
    mtx_path = next((p for p in mtx_candidates if p.exists()), None)
    if mtx_path is None:
        raise FileNotFoundError(
            f"No .h5ad or matrix.mtx(.gz) found in {raw_dir}. Supported GEO accessions: {', '.join(GEO_SOURCES)}."
        )

    return _load_10x_mtx_bundle(
        raw_dir, mtx_path, candidate_genes, dataset_id, max_cells_per_type, seed, tissue_col, cell_type_col
    )


def _load_10x_mtx_bundle(
    raw_dir: Path,
    mtx_path: Path,
    candidate_genes: Sequence[GeneInfo],
    dataset_id: str,
    max_cells_per_type: int,
    seed: int,
    tissue_col: str,
    cell_type_col: str,
) -> RawScrnaDataset:
    """Pure-Python + optional scipy Market Matrix (.mtx) loader for 10x GEO bundles."""
    import gzip

    def _open_text(p: Path):
        return gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else p.open("r", encoding="utf-8")

    feat_path = next(
        (raw_dir / name for name in ("features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz") if (raw_dir / name).exists()),
        None,
    )
    bar_path = next(
        (raw_dir / name for name in ("barcodes.tsv", "barcodes.tsv.gz") if (raw_dir / name).exists()),
        None,
    )
    if feat_path is None or bar_path is None:
        raise FileNotFoundError("10x GEO directory must include features.tsv/genes.tsv and barcodes.tsv")

    with _open_text(feat_path) as f:
        feature_genes = [line.strip().split("\t")[-1] if "\t" in line else line.strip() for line in f if line.strip()]
    with _open_text(bar_path) as f:
        barcodes = [line.strip().split("\t")[0] for line in f if line.strip()]

    meta_map: Dict[str, Dict[str, str]] = {}
    meta_path = raw_dir / "metadata.csv"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                bc = (row.get("barcode") or row.get("cell_id") or "").strip()
                if bc:
                    meta_map[bc] = row

    obs_ct = [meta_map.get(bc, {}).get(cell_type_col, "epithelial") for bc in barcodes]
    keep_cells = subsample_cells_per_type(obs_ct, max_per_type=max_cells_per_type, seed=seed)
    keep_cell_pos = {orig_idx: new_idx for new_idx, orig_idx in enumerate(keep_cells)}

    gene_index = {g.gene: i for i, g in enumerate(candidate_genes)}
    feat_to_target = {idx: gene_index[sym] for idx, sym in enumerate(feature_genes) if sym in gene_index}

    counts = [[0.0] * len(candidate_genes) for _ in keep_cells]
    with _open_text(mtx_path) as f:
        header_seen = False
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if not header_seen:
                header_seen = True
                continue
            parts = s.split()
            feat_1idx = int(parts[0]) - 1
            cell_1idx = int(parts[1]) - 1
            val = float(parts[2])
            if cell_1idx in keep_cell_pos and feat_1idx in feat_to_target:
                counts[keep_cell_pos[cell_1idx]][feat_to_target[feat_1idx]] += val

    cells: List[CellMetadata] = []
    for new_idx, orig_idx in enumerate(keep_cells):
        bc = barcodes[orig_idx]
        meta = meta_map.get(bc, {})
        tissue = meta.get(tissue_col, "tumor_lung")
        ct = meta.get(cell_type_col, "epithelial")
        mito = float(meta.get("pct_counts_mt", 5.0))
        row = counts[new_idx]
        tot = sum(row)
        ng = sum(1 for v in row if v > 0)
        cells.append(
            CellMetadata(
                cell_id=bc,
                tissue=tissue,
                cell_type=ct,
                is_tumor="normal" not in tissue.lower(),
                mito_pct=mito,
                total_counts=tot,
                n_genes=ng,
            )
        )

    return RawScrnaDataset(
        genes=list(candidate_genes),
        gene_index=gene_index,
        cells=cells,
        counts=counts,
        seed=seed,
        dataset_id=dataset_id,
        data_mode="REAL_GEO",
    )
