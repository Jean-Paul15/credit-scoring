from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.stats import spearmanr

from credit_risk_graph.features.schema import THIRD_PARTY_FEATURES
from credit_risk_graph.ifrs9.data import model_columns, target
from credit_risk_graph.ifrs9.params import METRICS_DIR


@dataclass
class PDModel:
    """EBM (glass-box) et liste ordonnée des variables ; la PD brute est non calibrée."""

    profile: str
    features: list[str]
    ebm: ExplainableBoostingClassifier

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.ebm.predict_proba(frame[self.features])[:, 1]

    def term_contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Contribution additive (échelle logit) de chaque terme : explication locale exacte, sans SHAP."""
        scores = self.ebm.eval_terms(frame[self.features])
        return pd.DataFrame(scores, columns=self.ebm.term_names_, index=frame.index)


def verify_monotone_signs(x: pd.DataFrame, y: pd.Series, constraints: dict[str, int], min_abs_rho: float = 0.02) -> dict[str, float]:
    """Refuse une contrainte de monotonie dont le signe contredit nettement la corrélation de rang observée."""
    checked: dict[str, float] = {}
    for name, sign in constraints.items():
        if name not in x.columns:
            continue
        rho = float(spearmanr(x[name], y, nan_policy="omit").statistic)
        checked[name] = rho
        if abs(rho) > min_abs_rho and np.sign(rho) != sign:
            raise ValueError(f"Contrainte de monotonie incohérente pour {name}: signe {sign}, rho de Spearman {rho:.3f}")
    return checked


def _monotone_vector(features: list[str], constraints: dict[str, int]) -> list[int]:
    return [int(constraints.get(name, 0)) for name in features]


def assert_no_third_party_features(features: list[str]) -> None:
    """Un modèle éligible ne doit utiliser aucune variable dérivée du comportement de tiers."""
    forbidden = sorted(THIRD_PARTY_FEATURES.intersection(features))
    if forbidden:
        raise ValueError(f"Variables relatives à des tiers interdites dans un modèle éligible : {forbidden}")


def ebm_settings(params: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Paramètres de base + meilleure configuration de la recherche aléatoire (si elle a été exécutée) ou surcharge explicite."""
    settings = dict(params["pd_model"]["ebm"])
    if overrides is None:
        tuned = METRICS_DIR / "ebm_tuning.json"
        overrides = json.loads(tuned.read_text(encoding="utf-8")).get("best_params", {}) if tuned.exists() else {}
    settings.update(overrides)
    return settings


def fit_pd_model(
    profile: str, train: pd.DataFrame, params: dict[str, Any], diagnostic: bool = False, overrides: dict[str, Any] | None = None
) -> tuple[PDModel, dict[str, float]]:
    features = model_columns(profile)
    if not diagnostic:
        assert_no_third_party_features(features)
    constraints = params["pd_model"]["monotone_constraints"]
    x, y = train[features], target(train)
    signs = verify_monotone_signs(x, y, constraints)
    ebm = ExplainableBoostingClassifier(
        feature_names=features,
        monotone_constraints=_monotone_vector(features, constraints),
        random_state=params["seed"],
        **ebm_settings(params, overrides),
    )
    ebm.fit(x, y)
    return PDModel(profile, features, ebm), signs


def global_importance(model: PDModel) -> pd.DataFrame:
    table = pd.DataFrame({"terme": model.ebm.term_names_, "importance": model.ebm.term_importances()})
    table["part"] = table["importance"] / table["importance"].sum()
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def term_shape(model: PDModel, term: str) -> pd.DataFrame:
    """Fonction de forme d'un terme principal : contribution logit par intervalle de valeurs."""
    index = list(model.ebm.term_names_).index(term)
    data = model.ebm.explain_global().data(index)
    names, scores = data["names"], np.asarray(data["scores"], dtype=float)
    if len(names) == len(scores) + 1:
        return pd.DataFrame({"borne_inf": names[:-1], "borne_sup": names[1:], "contribution_logit": scores})
    return pd.DataFrame({"modalite": names[: len(scores)], "contribution_logit": scores})
