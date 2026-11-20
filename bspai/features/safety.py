"""Safety feature group: tumor-vs-normal differential expression and score_safe."""

from __future__ import annotations

from typing import List

from bspai.preprocess import PreprocessedDataset


def compute_gene_tumor_deltas(ds: PreprocessedDataset) -> List[float]:
    g_count = len(ds.genes)
    tumor_sum = [0.0] * g_count
    normal_sum = [0.0] * g_count
    tumor_n = 0
    normal_n = 0

    for c_idx, cell in enumerate(ds.cells):
        row = ds.norm_expr[c_idx]
        if cell.is_tumor:
            tumor_n += 1
            for g in range(g_count):
                tumor_sum[g] += row[g]
        else:
            normal_n += 1
            for g in range(g_count):
                normal_sum[g] += row[g]

    t_den = max(1, tumor_n)
    n_den = max(1, normal_n)
    return [(tumor_sum[g] / t_den) - (normal_sum[g] / n_den) for g in range(g_count)]


def compute_safety_score(dx: float, dy: float) -> float:
    """Paper formula: score_safe = (dx * dy) / (dx + dy), guarded for non-positive deltas."""
    if dx > 0 and dy > 0:
        return (dx * dy) / (dx + dy)
    if dx <= 0 and dy <= 0:
        denom = abs(dx) + abs(dy)
        if denom < 1e-12:
            return 0.0
        return -abs(dx * dy) / denom
    return min(dx, dy)
