"""Build 5-group pairwise feature matrix for training pairs and candidate catalog."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from bspai.data import (
    ROOT_DIR,
    canonical_pair_key,
    generate_synthetic_scrna_dataset,
    load_candidate_genes,
    load_clinical_bispecifics,
    load_geo_dataset,
    load_pathway_database,
)
from bspai.features import FeatureContext
from bspai.labels import build_clinical_label_map, build_training_pair_subset
from bspai.preprocess import preprocess_scrna_dataset


def main() -> None:
    load_dotenv()
    seed = int(os.getenv("BSPAI_SEED", "42"))
    genes = load_candidate_genes()
    raw_dir = ROOT_DIR / "data" / "raw"

    if raw_dir.exists() and (list(raw_dir.glob("*.h5ad")) or (raw_dir / "matrix.mtx").exists() or (raw_dir / "matrix.mtx.gz").exists()):
        print(f"[BSPAI] Detected real GEO data in {raw_dir}; running GEO loader...")
        raw = load_geo_dataset(raw_dir, genes, seed=seed)
        prep = preprocess_scrna_dataset(raw, use_scanpy=True)
    else:
        raw = generate_synthetic_scrna_dataset(genes, seed=seed, n_cells=720)
        prep = preprocess_scrna_dataset(raw, use_scanpy=False)

    pathways = load_pathway_database()
    ctx = FeatureContext(prep, pathways)

    clinical = load_clinical_bispecifics()
    label_map = build_clinical_label_map(clinical, genes)
    train_pairs, relevances, binary_labels = build_training_pair_subset(genes, label_map, seed=seed)

    X_train = np.array(
        [ctx.compute_pair_features_by_idx(i, j) for i, j in train_pairs],
        dtype=np.float32,
    )
    out_dir = ROOT_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "training_features.npz",
        X=X_train,
        relevances=np.array(relevances, dtype=np.int32),
        binary_labels=np.array(binary_labels, dtype=np.int32),
        pair_keys=np.array([canonical_pair_key(genes[i].gene, genes[j].gene) for i, j in train_pairs]),
        feature_names=np.array(ctx.feature_names),
        feature_groups=np.array(ctx.feature_groups),
    )
    (out_dir / "dataset_meta.json").write_text(
        json.dumps({"dataset_id": prep.dataset_id, "data_mode": prep.data_mode, "seed": seed, "qc_stats": prep.qc_stats}, indent=2),
        encoding="utf-8",
    )
    print(f"[BSPAI] Built training matrix shape={X_train.shape} -> {out_dir / 'training_features.npz'}")


if __name__ == "__main__":
    main()
