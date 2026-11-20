"""Unified 5-group feature engineering pipeline for BSPAI."""

from __future__ import annotations

from typing import Dict, List, Tuple

from bspai.features.activity import ACTIVITY_GROUPS, ActivityPrecomputed
from bspai.features.corr import CORR_GROUPS, CorrPrecomputed
from bspai.features.embedding import build_gene_embeddings, compute_euclidean_distance
from bspai.features.pathway import PathwayIndex, compute_mechanism_features
from bspai.features.safety import compute_gene_tumor_deltas, compute_safety_score
from bspai.preprocess import PreprocessedDataset


def get_feature_schema() -> Tuple[List[str], List[str]]:
    names: List[str] = []
    groups: List[str] = []

    def push(name: str, grp: str) -> None:
        names.append(name)
        groups.append(grp)

    push("dx_tumor_minus_normal", "safety")
    push("dy_tumor_minus_normal", "safety")
    push("score_safe", "safety")

    push("kegg_cooccur", "mechanism")
    push("go_cooccur", "mechanism")
    push("pathway_cooccur", "mechanism")
    push("same_pathway", "mechanism")

    push("vector_score", "embedding")

    for grp in ACTIVITY_GROUPS:
        push(f"double_ratio_{grp}", "activity")
        push(f"sum_single_ratio_{grp}", "activity")
        push(f"min_single_ratio_{grp}", "activity")
        push(f"sum_single_exp_{grp}", "activity")
        push(f"min_single_exp_{grp}", "activity")
    push("single_ratio_max", "activity")
    push("double_ratio_max", "activity")

    for grp in CORR_GROUPS:
        push(f"pearson_{grp}", "correlation")

    return names, groups


class FeatureContext:
    def __init__(
        self,
        ds: PreprocessedDataset,
        pathways: Dict[str, Dict[str, List[str]]],
        activity_threshold: float = 0.5,
        embedding_dim: int = 16,
    ) -> None:
        self.ds = ds
        self.deltas = compute_gene_tumor_deltas(ds)
        self.pathway_index = PathwayIndex(pathways)
        self.embeddings = build_gene_embeddings(ds, dim=embedding_dim)
        self.activity_stats = ActivityPrecomputed(ds, threshold=activity_threshold)
        self.corr_stats = CorrPrecomputed(ds)
        self.feature_names, self.feature_groups = get_feature_schema()

    def compute_pair_features(self, gene1: str, gene2: str) -> List[float]:
        idx1 = self.ds.gene_index[gene1]
        idx2 = self.ds.gene_index[gene2]
        return self.compute_pair_features_by_idx(idx1, idx2)

    def compute_pair_features_by_idx(self, idx1: int, idx2: int) -> List[float]:
        g1 = self.ds.genes[idx1].gene
        g2 = self.ds.genes[idx2].gene
        dx = self.deltas[idx1]
        dy = self.deltas[idx2]
        safe = compute_safety_score(dx, dy)
        mech = compute_mechanism_features(g1, g2, self.pathway_index)
        vec_score = compute_euclidean_distance(self.embeddings[idx1], self.embeddings[idx2])
        act = self.activity_stats.compute_pair_activity(idx1, idx2)
        corr = self.corr_stats.compute_pair_correlations(idx1, idx2)

        return [
            dx,
            dy,
            safe,
            float(mech["kegg_cooccur"]),
            float(mech["go_cooccur"]),
            float(mech["pathway_cooccur"]),
            float(mech["same_pathway"]),
            vec_score,
            *act,
            *corr,
        ]
