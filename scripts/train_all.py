"""Train Pairwise XGBoost + 7 baselines under 5-fold CV, fit discretizer, and score all candidate pairs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from bspai import MODEL_VERSION
from bspai.data import (
    ROOT_DIR,
    canonical_pair_key,
    compute_candidate_pair_count,
    generate_synthetic_scrna_dataset,
    load_candidate_genes,
    load_pathway_database,
)
from bspai.discretize import fit_feature_discretizer
from bspai.evaluate import evaluate_models_5fold
from bspai.features import FeatureContext
from bspai.preprocess import preprocess_scrna_dataset


def main() -> None:
    load_dotenv()
    seed = int(os.getenv("BSPAI_SEED", "42"))
    proc_dir = ROOT_DIR / "data" / "processed"
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(proc_dir / "training_features.npz", allow_pickle=False)
    X_train = npz["X"]
    relevances = npz["relevances"]
    binary_labels = npz["binary_labels"]
    meta = json.loads((proc_dir / "dataset_meta.json").read_text(encoding="utf-8"))

    genes = load_candidate_genes()
    raw = generate_synthetic_scrna_dataset(genes, seed=seed, n_cells=720)
    prep = preprocess_scrna_dataset(raw, use_scanpy=False)
    ctx = FeatureContext(prep, load_pathway_database())

    eval_out = evaluate_models_5fold(
        ctx,
        X_train,
        relevances,
        binary_labels,
        seed=seed,
        plots_dir=models_dir / "plots",
    )
    full_model = eval_out["full_model"]
    full_model.save_model(str(models_dir / "pairwise_xgb.json"))

    bin_specs = fit_feature_discretizer(
        ctx.feature_names,
        X_train.tolist(),
        eval_out["top_feature_importances"],
        top_k=6,
        n_bins=5,
    )
    (models_dir / "bin_edges.json").write_text(json.dumps(bin_specs, indent=2), encoding="utf-8")

    # Score all G*(G-1)/2 pairs
    g_count = len(genes)
    all_rows = []
    pair_meta = []
    for i in range(g_count):
        for j in range(i + 1, g_count):
            all_rows.append(ctx.compute_pair_features_by_idx(i, j))
            pair_meta.append((genes[i].gene, genes[j].gene))

    scores = full_model.predict(np.asarray(all_rows, dtype=np.float32))
    ranked = [
        {
            "pair_key": canonical_pair_key(g1, g2),
            "g1": min(g1, g2),
            "g2": max(g1, g2),
            "score": round(float(sc), 6),
        }
        for (g1, g2), sc in zip(pair_meta, scores)
    ]
    ranked.sort(key=lambda d: (-d["score"], d["pair_key"]))
    for idx, item in enumerate(ranked):
        item["rank"] = idx + 1

    (models_dir / "ranked_pairs.json").write_text(json.dumps(ranked), encoding="utf-8")
    (models_dir / "evaluation_metrics.json").write_text(
        json.dumps(
            {
                "dataset_id": meta["dataset_id"],
                "seed": seed,
                "model_version": MODEL_VERSION,
                "gene_count": g_count,
                "total_candidate_pairs": compute_candidate_pair_count(g_count),
                "benchmarks": eval_out["benchmarks"],
                "ablation": eval_out["ablation"],
                "top_feature_importances": eval_out["top_feature_importances"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[BSPAI] Trained & saved Pairwise XGBoost + metrics -> {models_dir}")


if __name__ == "__main__":
    main()
