"""Single-cell target activity features across tissues and key cell types."""

from __future__ import annotations

from typing import Dict, List, Sequence

from bspai.preprocess import PreprocessedDataset

ACTIVITY_GROUPS = (
    "overall",
    "tumor_lung",
    "normal_lung",
    "lymph_node",
    "brain_mets",
    "exhausted_cd8_t",
    "treg",
    "macrophage_myeloid",
    "epithelial",
)


def compute_single_positive_ratio(expr: Sequence[float], threshold: float) -> float:
    if not expr:
        return 0.0
    return sum(1 for v in expr if v > threshold) / len(expr)


def compute_double_positive_ratio(
    expr1: Sequence[float], expr2: Sequence[float], threshold: float
) -> float:
    n = min(len(expr1), len(expr2))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if expr1[i] > threshold and expr2[i] > threshold) / n


class ActivityPrecomputed:
    def __init__(self, ds: PreprocessedDataset, threshold: float = 0.5) -> None:
        self.threshold = threshold
        g_count = len(ds.genes)
        self.group_cell_indices: List[List[int]] = []
        for grp in ACTIVITY_GROUPS:
            if grp == "overall":
                idx = list(range(len(ds.cells)))
            elif grp in ("tumor_lung", "normal_lung", "lymph_node", "brain_mets"):
                idx = [c for c, cell in enumerate(ds.cells) if cell.tissue == grp]
            else:
                idx = [c for c, cell in enumerate(ds.cells) if cell.cell_type == grp]
            self.group_cell_indices.append(idx)

        self.single_ratio: List[List[float]] = []
        self.mean_exp: List[List[float]] = []
        self.binary_flags: List[List[List[int]]] = []

        for cells_in_grp in self.group_cell_indices:
            n_grp = len(cells_in_grp)
            sr = [0.0] * g_count
            me = [0.0] * g_count
            bf = [[0] * n_grp for _ in range(g_count)]
            if n_grp > 0:
                for g in range(g_count):
                    pos = 0
                    total = 0.0
                    for k, c_idx in enumerate(cells_in_grp):
                        val = ds.norm_expr[c_idx][g]
                        total += val
                        if val > threshold:
                            pos += 1
                            bf[g][k] = 1
                    sr[g] = pos / n_grp
                    me[g] = total / n_grp
            self.single_ratio.append(sr)
            self.mean_exp.append(me)
            self.binary_flags.append(bf)

    def compute_pair_activity(self, idx1: int, idx2: int) -> List[float]:
        out: List[float] = []
        single_ratio_max = 0.0
        double_ratio_max = 0.0

        for grp_idx in range(len(ACTIVITY_GROUPS)):
            n_grp = len(self.group_cell_indices[grp_idx])
            p1 = self.single_ratio[grp_idx][idx1]
            p2 = self.single_ratio[grp_idx][idx2]
            e1 = self.mean_exp[grp_idx][idx1]
            e2 = self.mean_exp[grp_idx][idx2]

            dbl = 0.0
            if n_grp > 0:
                b1 = self.binary_flags[grp_idx][idx1]
                b2 = self.binary_flags[grp_idx][idx2]
                dbl = sum(1 for k in range(n_grp) if b1[k] and b2[k]) / n_grp

            out.extend([dbl, p1 + p2, min(p1, p2), e1 + e2, min(e1, e2)])
            if grp_idx > 0:
                single_ratio_max = max(single_ratio_max, p1, p2)
                double_ratio_max = max(double_ratio_max, dbl)

        out.extend([single_ratio_max, double_ratio_max])
        return out
