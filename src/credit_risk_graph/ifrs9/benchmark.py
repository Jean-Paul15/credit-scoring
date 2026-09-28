from __future__ import annotations

from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk_graph.ifrs9.data import feature_lists, model_columns, target
from credit_risk_graph.ifrs9.metrics import full_metrics


def _preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), categorical),
        ]
    )


def challengers(profile: str, seed: int) -> dict[str, Pipeline]:
    """Modèles de comparaison : régression logistique (référence usuelle) et LightGBM (boîte noire)."""
    categorical, numeric = feature_lists(profile)
    lightgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, random_state=seed, verbosity=-1)
    return {
        "logistic_regression": Pipeline([("prep", _preprocessor(categorical, numeric)), ("model", LogisticRegression(max_iter=1000, C=0.5))]),
        "lightgbm": Pipeline([("prep", _preprocessor(categorical, numeric)), ("model", lightgbm)]),
    }


def benchmark(profile: str, train: pd.DataFrame, evaluation: dict[str, pd.DataFrame], seed: int) -> pd.DataFrame:
    columns = model_columns(profile)
    rows: list[dict[str, Any]] = []
    for name, pipeline in challengers(profile, seed).items():
        pipeline.fit(train[columns], target(train))
        for split, frame in evaluation.items():
            prob = pipeline.predict_proba(frame[columns])[:, 1]
            rows.append({"modele": name, "profil": profile, "split": split, **full_metrics(target(frame).to_numpy(), prob)})
    return pd.DataFrame(rows)
