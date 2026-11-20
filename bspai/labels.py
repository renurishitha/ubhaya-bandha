"""Clinical stage label aggregation and training pair sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from bspai.data import ClinicalPair, GeneInfo, canonical_pair_key
from bspai.prng import Mulberry32, seeded_shuffle


@dataclass
class PairLabelRecord:
    pair_key: str
    best_stage: int
    relevance: int
    binary_positive: int
    drugs: List[str]
    needs_verification: bool
    mechanism_type: str


def build_clinical_label_map(
    clinical_pairs: Sequence[ClinicalPair], genes: Sequence[GeneInfo]
) -> Dict[str, PairLabelRecord]:
    gene_set = {g.gene for g in genes}
    out: Dict[str, PairLabelRecord] = {}
    for row in clinical_pairs:
        if row.gene1 not in gene_set or row.gene2 not in gene_set or row.gene1 == row.gene2:
            continue
        key = canonical_pair_key(row.gene1, row.gene2)
        if key not in out:
            out[key] = PairLabelRecord(
                pair_key=key,
                best_stage=row.clinical_stage,
                relevance=6 - row.clinical_stage,
                binary_positive=1 if row.clinical_stage <= 3 else 0,
                drugs=[row.drug],
                needs_verification=row.needs_verification,
                mechanism_type=row.mechanism_type,
            )
        else:
            rec = out[key]
            if row.clinical_stage < rec.best_stage:
                rec.best_stage = row.clinical_stage
                rec.relevance = 6 - row.clinical_stage
                rec.binary_positive = 1 if row.clinical_stage <= 3 else 0
            if row.drug not in rec.drugs:
                rec.drugs.append(row.drug)
            rec.needs_verification = rec.needs_verification or row.needs_verification
    return out


def build_training_pair_subset(
    genes: Sequence[GeneInfo],
    label_map: Dict[str, PairLabelRecord],
    seed: int = 42,
    unlabeled_sample_count: int = 160,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    rng = Mulberry32(seed + 777)
    labeled: List[Tuple[int, int]] = []
    unlabeled: List[Tuple[int, int]] = []

    g_count = len(genes)
    for i in range(g_count):
        for j in range(i + 1, g_count):
            key = canonical_pair_key(genes[i].gene, genes[j].gene)
            if key in label_map:
                labeled.append((i, j))
            else:
                unlabeled.append((i, j))

    sampled = seeded_shuffle(unlabeled, rng)[: min(unlabeled_sample_count, len(unlabeled))]
    pairs = labeled + sampled
    relevances: List[int] = []
    binary_labels: List[int] = []

    for i, j in pairs:
        rec = label_map.get(canonical_pair_key(genes[i].gene, genes[j].gene))
        if rec:
            relevances.append(rec.relevance)
            binary_labels.append(rec.binary_positive)
        else:
            relevances.append(0)
            binary_labels.append(0)

    return pairs, relevances, binary_labels
