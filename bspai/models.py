"""Pairwise XGBoost Ranker and 7 baseline models for BSPAI."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRanker


def train_pairwise_xgb(
    X: np.ndarray,
    relevances: Sequence[int],
    seed: int = 42,
) -> XGBRanker:
    """XGBRanker with objective='rank:pairwise' and paper hyperparameters."""
    model = XGBRanker(
        objective="rank:pairwise",
        learning_rate=0.1,
        n_estimators=100,
        max_depth=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        random_state=seed,
    )
    qid = np.zeros(len(relevances), dtype=int)
    model.fit(X, np.asarray(relevances, dtype=float), qid=qid)
    return model


def build_baselines(seed: int = 42) -> Dict[str, Any]:
    return {
        "Pointwise XGBoost": XGBClassifier(
            learning_rate=0.1,
            n_estimators=100,
            max_depth=10,
            eval_metric="logloss",
            random_state=seed,
        ),
        "GBDT": GradientBoostingClassifier(
            learning_rate=0.1,
            n_estimators=100,
            max_depth=4,
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=seed,
        ),
        "DNN (MLP 256-128-64-32-1)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128, 64, 32),
                        activation="relu",
                        solver="adam",
                        max_iter=300,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(C=0.1, max_iter=1000, random_state=seed)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            min_samples_leaf=5,
            max_depth=10,
            random_state=seed,
        ),
    }
