"""Expression correlation features: Pearson r overall and per key cell type."""

from __future__ import annotations

import math
from typing import List, Sequence

from bspai.preprocess import PreprocessedDataset

CORR_GROUPS = (
    "overall",
    "exhausted_cd8_t",
    "treg",
    "macrophage_myeloid",
    "epithelial",
)


def compute_pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    mx = sum(x[:n]) / n
    my = sum(y[:n]) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    sx = sum((x[i] - mx) ** 2 for i in range(n))
    sy = sum((y[i] - my) ** 2 for i in range(n))
    den = math.sqrt(sx * sy)
    if den < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, num / den))


class CorrPrecomputed:
    def __init__(self, ds: PreprocessedDataset) -> None:
        g_count = len(ds.genes)
        self.group_sizes: List[int] = []
        self.unit_vectors: List[List[List[float]]] = []

        for grp in CORR_GROUPS:
            indices = [
                c
                for c, cell in enumerate(ds.cells)
                if grp == "overall" or cell.cell_type == grp
            ]
            n_grp = len(indices)
            self.group_sizes.append(n_grp)
            vecs = [[0.0] * n_grp for _ in range(g_count)]
            if n_grp > 1:
                for g in range(g_count):
                    mean = sum(ds.norm_expr[c][g] for c in indices) / n_grp
                    diffs = [ds.norm_expr[c][g] - mean for c in indices]
                    norm = math.sqrt(sum(d * d for d in diffs))
                    if norm > 1e-12:
                        vecs[g] = [d / norm for d in diffs]
            self.unit_vectors.append(vecs)

    def compute_pair_correlations(self, idx1: int, idx2: int) -> List[float]:
        out: List[float] = []
        for grp_idx, n_grp in enumerate(self.group_sizes):
            if n_grp < 2:
                out.append(0.0)
                continue
            v1 = self.unit_vectors[grp_idx][idx1]
            v2 = self.unit_vectors[grp_idx][idx2]
            dot = sum(a * b for a, b in zip(v1, v2))
            out.append(max(-1.0, min(1.0, dot)))
        return out
