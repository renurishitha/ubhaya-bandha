"""Equal-frequency (quantile) 5-bin discretization over top-6 features."""

from __future__ import annotations

from typing import Dict, List, Sequence


def _get_labels_for_feature(name: str, bin_count: int) -> List[str]:
    if name == "vector_score":
        base = ["very similar", "similar", "moderate similarity", "dissimilar", "very dissimilar"]
    elif name == "score_safe":
        base = ["unsafe", "low safety", "moderate safety", "good safety", "high safety"]
    elif name.startswith("pearson_"):
        base = ["strongly negative", "moderately negative", "near zero", "moderately positive", "strongly positive"]
    elif "pathway" in name or "kegg" in name or "go_" in name:
        base = ["no overlap", "minimal overlap", "moderate overlap", "high overlap", "very high overlap"]
    else:
        base = ["very low", "low", "moderate", "high", "very high"]
    if bin_count == 5:
        return base
    return [base[round((i / max(1, bin_count - 1)) * (len(base) - 1))] for i in range(bin_count)]


def fit_feature_discretizer(
    feature_names: Sequence[str],
    training_rows: Sequence[Sequence[float]],
    importances: Sequence[Dict[str, object]],
    top_k: int = 6,
    n_bins: int = 5,
) -> List[Dict[str, object]]:
    chosen: List[str] = []
    for item in importances:
        fname = str(item["feature"])
        if int(item["weight"]) > 0 and fname not in chosen:
            chosen.append(fname)
            if len(chosen) >= top_k:
                break

    for fallback in (
        "vector_score",
        "score_safe",
        "pearson_overall",
        "single_ratio_max",
        "double_ratio_max",
        "pathway_cooccur",
    ):
        if len(chosen) >= top_k:
            break
        if fallback in feature_names and fallback not in chosen:
            chosen.append(fallback)

    specs: List[Dict[str, object]] = []
    for fname in chosen:
        col = feature_names.index(fname)
        vals = sorted(row[col] for row in training_rows)
        raw_edges = [vals[0]]
        for b in range(1, n_bins):
            q_pos = (b / n_bins) * (len(vals) - 1)
            lo = int(q_pos)
            hi = min(len(vals) - 1, lo + 1)
            frac = q_pos - lo
            raw_edges.append(vals[lo] + frac * (vals[hi] - vals[lo]))
        raw_edges.append(vals[-1])

        edges = [raw_edges[0]]
        for val in raw_edges[1:]:
            if val > edges[-1] + 1e-9:
                edges.append(val)
        if len(edges) < 2:
            edges.append(edges[0] + 1e-6)

        bin_count = max(1, len(edges) - 1)
        specs.append(
            {
                "feature": fname,
                "edges": [round(e, 6) for e in edges],
                "labels": _get_labels_for_feature(fname, bin_count),
            }
        )
    return specs


def apply_discretizer(
    specs: Sequence[Dict[str, object]], feature_values: Dict[str, float]
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for spec in specs:
        fname = str(spec["feature"])
        val = float(feature_values.get(fname, 0.0))
        edges = spec["edges"]
        labels = spec["labels"]
        n_bins = len(labels)
        bin_idx = n_bins - 1
        for b in range(n_bins):
            if val <= float(edges[b + 1]) or b == n_bins - 1:
                bin_idx = b
                break
        out.append(
            {
                "feature": fname,
                "value": val,
                "bin_index": bin_idx,
                "bin_label": labels[bin_idx],
            }
        )
    return out
