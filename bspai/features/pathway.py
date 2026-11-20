"""Mechanism of action features: KEGG and GO pathway co-occurrence."""

from __future__ import annotations

from typing import Dict, List, Set


class PathwayIndex:
    def __init__(self, db: Dict[str, Dict[str, List[str]]]) -> None:
        self.kegg_sets: Dict[str, Set[str]] = {}
        for pw, genes in db.get("kegg", {}).items():
            for g in genes:
                self.kegg_sets.setdefault(g, set()).add(pw)

        self.go_sets: Dict[str, Set[str]] = {}
        for term, genes in db.get("go", {}).items():
            for g in genes:
                self.go_sets.setdefault(g, set()).add(term)


def compute_mechanism_features(gene1: str, gene2: str, index: PathwayIndex) -> Dict[str, int]:
    kegg_cooccur = len(index.kegg_sets.get(gene1, set()) & index.kegg_sets.get(gene2, set()))
    go_cooccur = len(index.go_sets.get(gene1, set()) & index.go_sets.get(gene2, set()))
    pathway_cooccur = kegg_cooccur + go_cooccur
    return {
        "kegg_cooccur": kegg_cooccur,
        "go_cooccur": go_cooccur,
        "pathway_cooccur": pathway_cooccur,
        "same_pathway": 1 if pathway_cooccur > 0 else 0,
    }
