"""Étape DVC `train_pd` : EBM sur chaque profil de variables, challengers, sélection du profil champion."""
from __future__ import annotations

import mlflow
import pandas as pd

from credit_risk_graph.ifrs9.benchmark import benchmark
from credit_risk_graph.ifrs9.data import SPLITS, load_performing, sample_rows, target
from credit_risk_graph.ifrs9.metrics import full_metrics
from credit_risk_graph.ifrs9.params import load_params
from credit_risk_graph.ifrs9.pd_model import PDModel, fit_pd_model, global_importance
from credit_risk_graph.ifrs9.stages.common import save_metrics, save_model, save_table
from credit_risk_graph.ifrs9.tracking import log_json, log_metrics, log_table, tracked_run
from credit_risk_graph.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _evaluate(model: PDModel, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [{"modele": "ebm", "profil": model.profile, "split": name, **full_metrics(target(f).to_numpy(), model.predict_proba(f))} for name, f in frames.items()]
    return pd.DataFrame(rows)


def select_champion(results: pd.DataFrame, models: dict[str, PDModel], params: dict) -> str:
    """Profil le plus simple, éligible IFRS 9 (variables macro présentes), dont le Gini OOT est proche du meilleur."""
    required = set(params["pd_model"]["champion_requires"])
    oot = results[(results["split"] == "out_of_time") & (results["modele"] == "ebm")].set_index("profil")["gini"]
    eligible = [p for p in params["pd_model"]["profiles"] if required <= set(models[p].features)]
    best = max(oot[p] for p in eligible)
    tolerance = params["pd_model"]["min_gini_gain_for_complexity"]
    return next(p for p in eligible if oot[p] >= best - tolerance)


def main() -> None:
    params = load_params()
    frames = {name: load_performing(name, params) for name in SPLITS}
    train = sample_rows(frames["train"], params)
    evaluation = {name: frames[name] for name in SPLITS if name != "train"}
    results, models, importances, signs = [], {}, {}, {}
    with tracked_run("train_pd") as run:
        mlflow.log_params({"train_rows": len(train), "train_default_rate": float(target(train).mean())})
        candidates = params["pd_model"]["profiles"]
        for profile in candidates + params["pd_model"]["diagnostic_profiles"]:
            LOGGER.info("EBM profil %s", profile)
            model, signs[profile] = fit_pd_model(profile, train, params, diagnostic=profile not in candidates)
            models[profile] = model
            frame = _evaluate(model, {**evaluation, "train": train})
            results.append(frame.assign(eligible=profile in candidates))
            if profile in candidates:
                results.append(benchmark(profile, train, {**evaluation, "train": train}, params["seed"]).assign(eligible=True))
            importances[profile] = global_importance(model)
            for _, row in frame.iterrows():
                log_metrics({k: v for k, v in row.items() if isinstance(v, float)}, f"{profile}.{row['split']}.")
        table = pd.concat(results, ignore_index=True)
        champion = select_champion(table, models, params)
        mlflow.log_artifact(str(save_model(models[champion], "pd_ebm_champion")), artifact_path="model")
        save_table(table, "pd_model_comparison")
        save_table(importances[champion], "pd_ebm_importance")
        save_metrics({"champion_profile": champion, "monotone_spearman": signs, "n_features": len(models[champion].features)}, "pd_model")
        mlflow.set_tag("champion_profile", champion)
        log_table(table, "pd_model_comparison")
        log_table(importances[champion], "pd_ebm_importance")
        log_json({"champion_profile": champion, "monotone_spearman": signs[champion]}, "pd_model")
        LOGGER.info("Champion : %s (run %s)", champion, run.info.run_id)


if __name__ == "__main__":
    main()
