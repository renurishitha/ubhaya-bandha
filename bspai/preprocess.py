"""Scanpy preprocessing pipeline with deterministic pure-Python parity fallback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from bspai.data import CellMetadata, GeneInfo, RawScrnaDataset, compute_candidate_pair_count


@dataclass
class PreprocessedDataset:
    genes: List[GeneInfo]
    gene_index: Dict[str, int]
    cells: List[CellMetadata]
    norm_expr: List[List[float]]
    hvg_genes: List[str]
    qc_stats: Dict[str, object]
    seed: int
    dataset_id: str
    data_mode: str


def preprocess_scrna_dataset(
    raw: RawScrnaDataset,
    min_genes: int = 40,
    max_mito_pct: float = 20.0,
    target_sum: float = 10000.0,
    n_top_hvg: int = 150,
    use_scanpy: bool = False,
) -> PreprocessedDataset:
    """Standardize formats, filter low-quality cells, normalize to target_sum + log1p,
    and identify highly variable genes (with optional full Scanpy PCA/Leiden step).
    """
    if use_scanpy:
        try:
            return _preprocess_with_scanpy(raw, min_genes, max_mito_pct, target_sum, n_top_hvg)
        except ImportError:
            pass

    kept_cells: List[CellMetadata] = []
    norm_expr: List[List[float]] = []
    g_count = len(raw.genes)

    for c_idx, cell in enumerate(raw.cells):
        if cell.n_genes >= min_genes and cell.mito_pct <= max_mito_pct and cell.total_counts > 0:
            kept_cells.append(cell)
            scale = target_sum / cell.total_counts
            row = [math.log1p(v * scale) for v in raw.counts[c_idx]]
            norm_expr.append(row)

    n_kept = len(kept_cells)
    dispersions = []
    for g in range(g_count):
        mean = sum(norm_expr[r][g] for r in range(n_kept)) / max(1, n_kept)
        var = sum((norm_expr[r][g] - mean) ** 2 for r in range(n_kept)) / max(1, n_kept - 1)
        disp = (var / mean) if mean > 1e-8 else 0.0
        dispersions.append((raw.genes[g].gene, disp))
    dispersions.sort(key=lambda x: x[1], reverse=True)
    hvg_genes = [g for g, _ in dispersions[: min(n_top_hvg, g_count)]]

    return PreprocessedDataset(
        genes=raw.genes,
        gene_index=raw.gene_index,
        cells=kept_cells,
        norm_expr=norm_expr,
        hvg_genes=hvg_genes,
        qc_stats={
            "raw_cells": len(raw.cells),
            "filtered_cells": len(raw.cells) - n_kept,
            "retained_cells": n_kept,
            "gene_count": g_count,
            "total_candidate_pairs": compute_candidate_pair_count(g_count),
        },
        seed=raw.seed,
        dataset_id=raw.dataset_id,
        data_mode=raw.data_mode,
    )


def _preprocess_with_scanpy(
    raw: RawScrnaDataset,
    min_genes: int,
    max_mito_pct: float,
    target_sum: float,
    n_top_hvg: int,
) -> PreprocessedDataset:
    import anndata as ad  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    import scanpy as sc  # type: ignore

    obs = pd.DataFrame(
        {
            "tissue": [c.tissue for c in raw.cells],
            "cell_type": [c.cell_type for c in raw.cells],
            "pct_counts_mt": [c.mito_pct for c in raw.cells],
        },
        index=[c.cell_id for c in raw.cells],
    )
    var = pd.DataFrame(index=[g.gene for g in raw.genes])
    adata = ad.AnnData(X=np.array(raw.counts, dtype=np.float32), obs=obs, var=var)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata = adata[adata.obs["pct_counts_mt"] <= max_mito_pct, :].copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_hvg, adata.n_vars))
    sc.tl.pca(adata, n_comps=min(16, adata.n_vars - 1, adata.n_obs - 1))
    sc.pp.neighbors(adata)
    try:
        sc.tl.leiden(adata)
    except Exception:
        pass

    kept_ids = set(adata.obs_names.tolist())
    kept_cells = [c for c in raw.cells if c.cell_id in kept_ids]
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    norm_expr = X.tolist()
    hvg_genes = adata.var_names[adata.var["highly_variable"]].tolist()

    return PreprocessedDataset(
        genes=raw.genes,
        gene_index=raw.gene_index,
        cells=kept_cells,
        norm_expr=norm_expr,
        hvg_genes=hvg_genes,
        qc_stats={
            "raw_cells": len(raw.cells),
            "filtered_cells": len(raw.cells) - len(kept_cells),
            "retained_cells": len(kept_cells),
            "gene_count": len(raw.genes),
            "total_candidate_pairs": compute_candidate_pair_count(len(raw.genes)),
        },
        seed=raw.seed,
        dataset_id=raw.dataset_id,
        data_mode=raw.data_mode,
    )
