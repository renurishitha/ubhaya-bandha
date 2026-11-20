"""Generate and save deterministic synthetic scRNA-seq dataset summary to data/processed/."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from bspai.data import (
    ROOT_DIR,
    compute_candidate_pair_count,
    generate_synthetic_scrna_dataset,
    load_candidate_genes,
)
from bspai.preprocess import preprocess_scrna_dataset


def main() -> None:
    load_dotenv()
    seed = int(os.getenv("BSPAI_SEED", "42"))
    genes = load_candidate_genes()
    raw = generate_synthetic_scrna_dataset(genes, seed=seed, n_cells=720)
    prep = preprocess_scrna_dataset(raw)

    out_dir = ROOT_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_id": prep.dataset_id,
        "seed": seed,
        "gene_count": len(genes),
        "total_candidate_pairs": compute_candidate_pair_count(len(genes)),
        "qc_stats": prep.qc_stats,
        "top_hvg_genes": prep.hvg_genes[:25],
    }
    (out_dir / "synthetic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[BSPAI] Saved synthetic dataset summary -> {out_dir / 'synthetic_summary.json'}")


if __name__ == "__main__":
    main()
