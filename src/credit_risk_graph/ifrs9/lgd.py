from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from credit_risk_graph.features.schema import CATEGORICAL_FEATURES

LGD_TARGET = "current_lgd_true"
LGD_NUMERIC = [
    "collateral_coverage",
    "principal_amount",
    "interest_rate",
    "outstanding_balance",
    "current_ltv_proxy",
    "loan_age_months",
    "current_dpd",
    "macro_stress",
]
LGD_CATEGORICAL = [c for c in CATEGORICAL_FEATURES if c in {"product_type", "collateral_type", "region"}]


@dataclass
class LGDModel:
    features: list[str]
    ebm: ExplainableBoostingRegressor

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(self.ebm.predict(frame[self.features]), 0.0, 1.0)


def lgd_features(params: dict[str, Any]) -> list[str]:
    excluded = set(params["lgd"]["features_exclude"])
    return [c for c in LGD_CATEGORICAL + LGD_NUMERIC if c not in excluded]


def defaulted_exposures(frame: pd.DataFrame, dpd_threshold: int) -> pd.DataFrame:
    """LGD apprise sur les expositions en défaut (workout) : seule population où la perte est réalisée."""
    return frame.loc[frame["current_dpd"] >= dpd_threshold].reset_index(drop=True)


def fit_lgd_model(train_defaults: pd.DataFrame, params: dict[str, Any]) -> LGDModel:
    features = lgd_features(params)
    ebm = ExplainableBoostingRegressor(
        feature_names=features,
        max_bins=64,
        interactions=3,
        outer_bags=4,
        learning_rate=0.03,
        min_samples_leaf=100,
        random_state=params["seed"],
        n_jobs=params["pd_model"]["ebm"]["n_jobs"],
    )
    sample = train_defaults.sample(n=min(len(train_defaults), 100_000), random_state=params["seed"])
    ebm.fit(sample[features], sample[LGD_TARGET])
    return LGDModel(features, ebm)


def evaluate_lgd(model: LGDModel, frame: pd.DataFrame) -> dict[str, float]:
    pred = model.predict(frame)
    return {"r2": float(r2_score(frame[LGD_TARGET], pred)), "mae": float(mean_absolute_error(frame[LGD_TARGET], pred)), "mean_lgd_observed": float(frame[LGD_TARGET].mean()), "mean_lgd_predicted": float(pred.mean())}


def downturn_add_on(lgd_pred: np.ndarray, macro_stress: np.ndarray, quantile: float) -> float:
    """Écart entre la LGD moyenne en période de stress macro élevé et la LGD moyenne (>= 0)."""
    stressed = macro_stress >= np.quantile(macro_stress, quantile)
    return float(max(lgd_pred[stressed].mean() - lgd_pred.mean(), 0.0))


def apply_regulatory_floors(lgd: np.ndarray, collateral_type: pd.Series, floors: dict[str, float]) -> np.ndarray:
    """Planchers de LGD par type de garantie (vue prudentielle uniquement)."""
    floor = collateral_type.map(floors).fillna(0.0).to_numpy(dtype=float)
    return np.maximum(lgd, floor)
