"""Étape DVC `tune_ebm` : recherche aléatoire des hyperparamètres de l'EBM, validation croisée temporelle à fenêtre croissante."""
from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.model_selection import ParameterSampler

from credit_risk_graph.ifrs9.data import load_performing, target
from credit_risk_graph.ifrs9.metrics import full_metrics
from credit_risk_graph.ifrs9.params import load_params
from credit_risk_graph.ifrs9.pd_model import fit_pd_model
from credit_risk_graph.ifrs9.stages.common import save_metrics, save_table
from credit_risk_graph.ifrs9.tracking import log_metrics, log_table, tracked_run
from credit_risk_graph.utils.logging import get_logger

LOGGER = get_logger(__name__)
PROFILE = "behavioural_macro"


def _distribution(spec: Any):
    return loguniform(*spec["loguniform"]) if isinstance(spec, dict) else spec


def sample_configurations(space: dict[str, Any], n_iter: int, seed: int) -> list[dict[str, Any]]:
    distributions = {name: _distribution(spec) for name, spec in space.items()}
    configs = list(ParameterSampler(distributions, n_iter=n_iter, random_state=seed))
    return [{k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in config.items()} for config in configs]


def evaluate_configuration(pool: pd.DataFrame, override: dict[str, Any], params: dict[str, Any]) -> dict[str, float]:
    """Moyenne sur les plis temporels : ajustement sur le passé, mesure sur une année postérieure (sans chevauchement de labels)."""
    tuning = params["pd_model"]["tuning"]
    scores = []
    for fold in tuning["folds"]:
        fit = pool[pool["as_of_date"] <= pd.Timestamp(fold["fit_until"])]
        fit = fit.sample(n=min(len(fit), tuning["fit_rows"]), random_state=params["seed"])
        valid = pool[(pool["as_of_date"] >= pd.Timestamp(fold["validate_from"])) & (pool["as_of_date"] <= pd.Timestamp(fold["validate_until"]))]
        model, _ = fit_pd_model(PROFILE, fit, params, overrides=override)
        train_auc = full_metrics(target(fit).to_numpy(), model.predict_proba(fit))["auc"]
        metrics = full_metrics(target(valid).to_numpy(), model.predict_proba(valid))
        scores.append({"auc_pli": metrics["auc"], "brier_pli": metrics["brier"], "ecart_auc": train_auc - metrics["auc"]})
    return {key: float(np.mean([s[key] for s in scores])) for key in scores[0]}


def main() -> None:
    params = load_params()
    tuning = params["pd_model"]["tuning"]
    pool = load_performing("train", params, embargo=False)
    configs = sample_configurations(tuning["space"], tuning["n_iter"], params["seed"])
    rows = []
    with tracked_run("tune_ebm"):
        mlflow.log_param("n_configurations", len(configs))
        for index, override in enumerate(configs):
            LOGGER.info("Configuration %s/%s : %s", index + 1, len(configs), override)
            result = evaluate_configuration(pool, override, params)
            rows.append({"configuration": index, **override, **result})
            log_metrics(result, f"config{index}.")
            mlflow.log_params({f"config{index}.{k}": v for k, v in override.items()})
        table = pd.DataFrame(rows).sort_values(tuning["objective"] + "_pli", ascending=False).reset_index(drop=True)
        best = configs[int(table.loc[0, "configuration"])]
        summary = {"methode": tuning["method"], "objectif": tuning["objective"], "auc_moyenne_pli_meilleure": float(table.loc[0, "auc_pli"]), "best_params": best}
        save_table(table, "ebm_tuning")
        save_metrics(summary, "ebm_tuning")
        log_table(table, "ebm_tuning")
        mlflow.log_params({f"best.{k}": v for k, v in best.items()})
        LOGGER.info("Meilleure configuration : %s (AUC moyenne des plis %.4f)", best, summary["auc_moyenne_pli_meilleure"])


if __name__ == "__main__":
    main()
