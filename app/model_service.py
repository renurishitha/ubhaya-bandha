r"""Loads trained XGBoost .joblib and runs live scRNA-seq + 3D-GNN inference."""
import joblib
import numpy as np
import pandas as pd
import scanpy as sc

try:
  from app.bispec_features import KEY_SUBTYPES, build_var_upper_map, compute_pair_features
except ImportError:
  from bispec_features import KEY_SUBTYPES, build_var_upper_map, compute_pair_features


class ModelService:
  def __init__(self, model_path: str, h5ad_path: str):
    artifact = joblib.load(model_path)
    self.model = artifact["model"]
    self.feature_cols = artifact["feature_cols"]
    self.X_ref = artifact.get("X_ref", None)

    print(f"[Model] Loaded XGBoost with {len(self.feature_cols)} features.")
    adata_raw = sc.read_h5ad(h5ad_path)
    self.adata = (
        adata_raw.raw.to_adata() if adata_raw.raw is not None else adata_raw
    )
    self.var_upper_map = build_var_upper_map(self.adata)
    print(
        f"[Data] AnnData loaded: {self.adata.n_obs} cells x"
        f" {self.adata.n_vars} genes."
    )

  def predict(
      self, gene_a: str, gene_b: str, structured_ctx: dict | None = None
  ) -> dict:
    pair = sorted([gene_a.strip().upper(), gene_b.strip().upper()])
    live_feats = compute_pair_features(
        self.adata, pair[0], pair[1], self.var_upper_map
    )

    row_dict = {col: live_feats.get(col, 0.0) for col in self.feature_cols}
    input_df = pd.DataFrame([row_dict])[self.feature_cols]

    if self.X_ref is not None:
      diffs = input_df.values[0] - self.X_ref[self.feature_cols].values
      score = float(np.mean(self.model.predict_proba(diffs)[:, 1]))
    else:
      score = float(self.model.predict_proba(input_df)[0][1])

    subtypes_profile = []
    for sub in KEY_SUBTYPES:
      k = sub.replace(":", "_").replace("+", "plus").replace(" ", "_")
      subtypes_profile.append({
          "subtype": sub,
          "corrcoef": round(float(live_feats.get(f"corrcoef_{k}", 0.0)), 4),
          "double_ratio": round(
              float(live_feats.get(f"double_ratio_{k}", 0.0)), 4
          ),
          "sum_single_exp": round(
              float(live_feats.get(f"sum_single_exp_{k}", 0.0)), 4
          ),
      })

    return {
        "target_a": pair[0],
        "target_b": pair[1],
        "model_score": round(score, 4),
        "safe_avg": round(float(live_feats.get("safe_avg", 0.0)), 5),
        "double_ratio_max": round(
            float(live_feats.get("double_ratio_max", 0.0)), 4
        ),
        "sum_single_ratio_max": round(
            float(live_feats.get("sum_single_ratio_max", 0.0)), 4
        ),
        "gnn_3d_dist": live_feats.get("gnn_3d_dist", 1.0),
        "gnn_3d_cos": live_feats.get("gnn_3d_cos", 0.0),
        "subtypes": subtypes_profile,
        "all_features": {
            k: round(float(v), 5) for k, v in live_feats.items()
        },
    }