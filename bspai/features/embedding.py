"""Gene2Vec-style co-expression embeddings and Euclidean distance (vector_score)."""

from __future__ import annotations

import math
from typing import List, Sequence

from bspai.preprocess import PreprocessedDataset
from bspai.prng import Mulberry32


def _normalize(v: List[float]) -> None:
    sq = sum(x * x for x in v)
    n = math.sqrt(sq)
    if n > 1e-12:
        for i in range(len(v)):
            v[i] /= n


def build_gene_embeddings(ds: PreprocessedDataset, dim: int = 16) -> List[List[float]]:
    """Deterministic spectral co-expression Gene2Vec-style embeddings matching TypeScript."""
    n_cells = len(ds.cells)
    g_count = len(ds.genes)
    rng = Mulberry32(ds.seed + 101)

    Z: List[List[float]] = []
    for g in range(g_count):
        mean = sum(ds.norm_expr[c][g] for c in range(n_cells)) / max(1, n_cells)
        vec = [ds.norm_expr[c][g] - mean for c in range(n_cells)]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-12:
            vec = [v / norm for v in vec]
        else:
            vec = [0.0] * n_cells
        Z.append(vec)

    C = [[0.0] * g_count for _ in range(g_count)]
    for i in range(g_count):
        C[i][i] = 1.0
        zi = Z[i]
        for j in range(i + 1, g_count):
            zj = Z[j]
            dot = sum(zi[c] * zj[c] for c in range(n_cells))
            val = max(0.0, dot)
            C[i][j] = val
            C[j][i] = val

    emb = [[0.0] * dim for _ in range(g_count)]
    Cwork = [list(row) for row in C]

    for d in range(dim):
        v = [rng.normal(0.0, 1.0) for _ in range(g_count)]
        _normalize(v)
        for _ in range(15):
            w = [sum(Cwork[i][j] * v[j] for j in range(g_count)) for i in range(g_count)]
            _normalize(w)
            v = w
        lam = sum(v[i] * sum(Cwork[i][j] * v[j] for j in range(g_count)) for i in range(g_count))
        scale = math.sqrt(max(0.0, lam))
        for g in range(g_count):
            emb[g][d] = v[g] * scale
        for i in range(g_count):
            lvi = lam * v[i]
            row = Cwork[i]
            for j in range(g_count):
                row[j] -= lvi * v[j]

    return emb


def compute_euclidean_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))
