from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from bspai.features import FeatureContext
from bspai.models import build_baselines, train_pairwise_xgb


def evaluate_models_5fold(
    ctx: FeatureContext,
    X: np.ndarray,
    relevances: Sequence[int],
    binary_labels: Sequence[int],
    seed: int = 42,
    plots_dir: Path | None = None,
) -> Dict[str, object]:
    y = np.asarray(binary_labels, dtype=int)
    rel = np.asarray(relevances, dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    double_col = ctx.feature_names.index("double_ratio_overall")

    model_scores: Dict[str, List[float]] = {
        "Pairwise XGBoost (BSPAI)": [],
        "Pointwise XGBoost": [],
        "GBDT": [],
        "Random Forest": [],
        "DNN (MLP 256-128-64-32-1)": [],
        "Logistic Regression": [],
        "Decision Tree": [],
        "Single Feature (double_ratio)": [],
    }
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    roc_grid = np.linspace(0.0, 1.0, 41)
    interp_tprs: Dict[str, List[np.ndarray]] = {k: [] for k in model_scores}

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        r_tr = rel[tr_idx]

        pw_model = train_pairwise_xgb(X_tr, r_tr, seed=seed + fold_idx)
        preds: Dict[str, np.ndarray] = {
            "Pairwise XGBoost (BSPAI)": pw_model.predict(X_va),
            "Single Feature (double_ratio)": X_va[:, double_col],
        }

        for name, clf in build_baselines(seed=seed + fold_idx).items():
            clf.fit(X_tr, y_tr)
            preds[name] = clf.predict_proba(X_va)[:, 1]

        for name, scores in preds.items():
            auc = float(roc_auc_score(y_va, scores))
            model_scores[name].append(auc)
            fpr, tpr, _ = roc_curve(y_va, scores)
            interp_tprs[name].append(np.interp(roc_grid, fpr, tpr))

    benchmarks = []
    for name, aucs in model_scores.items():
        mean_tpr = np.mean(interp_tprs[name], axis=0)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        roc_data[name] = (roc_grid, mean_tpr)
        benchmarks.append(
            {
                "model": name,
                "mean_auc": round(float(np.mean(aucs)), 4),
                "std_auc": round(float(np.std(aucs)), 4),
                "fold_aucs": [round(float(a), 4) for a in aucs],
            }
        )

    # Full model for feature-frequency ("weight") importance
    full_model = train_pairwise_xgb(X, rel, seed=seed)
    booster = full_model.get_booster()
    weights = booster.get_score(importance_type="weight")
    feat_imp = []
    for idx, fname in enumerate(ctx.feature_names):
        w = weights.get(f"f{idx}", weights.get(fname, 0.0))
        feat_imp.append({"feature": fname, "weight": int(w), "group": ctx.feature_groups[idx]})
    feat_imp.sort(key=lambda d: (-d["weight"], d["feature"]))
    top_40 = feat_imp[:40]

    # Incremental 6-step ablation
    ablation = _run_ablation(ctx, X, rel, y, skf, seed)

    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)
        _save_roc_plot(roc_data, benchmarks, plots_dir / "roc_curves.png")
        _save_top40_plot(top_40, plots_dir / "top40_importance.png")

    return {
        "benchmarks": benchmarks,
        "ablation": ablation,
        "top_feature_importances": top_40,
        "full_model": full_model,
    }


def _run_ablation(
    ctx: FeatureContext,
    X: np.ndarray,
    rel: np.ndarray,
    y: np.ndarray,
    skf: StratifiedKFold,
    seed: int,
) -> List[Dict[str, object]]:
    all_cols = list(range(len(ctx.feature_names)))
    dbl_col = ctx.feature_names.index("double_ratio_overall")
    safety_cols = [c for c, g in enumerate(ctx.feature_groups) if g == "safety"]
    mech_cols = [c for c, g in enumerate(ctx.feature_groups) if g == "mechanism"]
    emb_cols = [c for c, g in enumerate(ctx.feature_groups) if g == "embedding"]
    overall_act_cols = [
        c
        for c, n in enumerate(ctx.feature_names)
        if ctx.feature_groups[c] == "activity" and (n.endswith("_overall") or n.endswith("_max"))
    ]
    overall_corr_col = ctx.feature_names.index("pearson_overall")

    steps = [
        ("1. Double-positive only", [dbl_col], True),
        ("2. + Safety (score_safe)", [dbl_col, *safety_cols], False),
        ("3. + Mechanism (KEGG/GO)", [dbl_col, *safety_cols, *mech_cols], False),
        ("4. + Gene embedding (vector_score)", [dbl_col, *safety_cols, *mech_cols, *emb_cols], False),
        (
            "5. + Single-cell overall expression",
            sorted(set([dbl_col, *safety_cols, *mech_cols, *emb_cols, *overall_act_cols, overall_corr_col])),
            False,
        ),
        ("6. + Grouped tissue & cell-type features", all_cols, False),
    ]

    results = []
    for step_name, cols, single_only in steps:
        aucs = []
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            if single_only:
                scores = X[va_idx][:, cols[0]]
            else:
                mdl = train_pairwise_xgb(X[tr_idx][:, cols], rel[tr_idx], seed=seed + 17 + fold_idx)
                scores = mdl.predict(X[va_idx][:, cols])
            aucs.append(float(roc_auc_score(y[va_idx], scores)))
        results.append(
            {
                "step": step_name,
                "feature_count": len(cols),
                "mean_auc": round(float(np.mean(aucs)), 4),
                "std_auc": round(float(np.std(aucs)), 4),
            }
        )
    return results


def _save_roc_plot(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    benchmarks: List[Dict[str, object]],
    out_path: Path,
) -> None:
    auc_lookup = {b["model"]: b["mean_auc"] for b in benchmarks}
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, (fpr, tpr) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_lookup[name]:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("BSPAI 5-Fold Cross-Validation ROC Curves")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_top40_plot(top_40: List[Dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 9))
    names = [d["feature"] for d in reversed(top_40)]
    weights = [d["weight"] for d in reversed(top_40)]
    ax.barh(names, weights, color="#2563EB")
    ax.set_xlabel("XGBoost Split Weight (Frequency)")
    ax.set_title("Top-40 Feature Frequency Importance")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
