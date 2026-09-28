"""Étape DVC `lgd_ead` : LGD (EBM de régression sur expositions en défaut), LGD de ralentissement, contrôle de la projection d'EAD."""
from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk_graph.ifrs9.data import load_full_split as load_split
from credit_risk_graph.ifrs9.data import observation_cutoff
from credit_risk_graph.ifrs9.ead import exposure_paths
from credit_risk_graph.ifrs9.lgd import defaulted_exposures, downturn_add_on, evaluate_lgd, fit_lgd_model
from credit_risk_graph.ifrs9.params import load_params
from credit_risk_graph.ifrs9.stages.common import save_metrics, save_model, save_table
from credit_risk_graph.ifrs9.tracking import log_json, log_metrics, log_table, tracked_run


def ead_projection_error(panel: pd.DataFrame, horizon: int = 12, slippage: float = 0.0) -> pd.DataFrame:
    """Compare l'exposition contractuelle projetée à 12 mois au solde réellement observé un an plus tard (cohortes de janvier)."""
    start = panel[(panel["as_of_date"].dt.month == 1) & (panel["remaining_term_months"] > horizon)].copy()
    start["year"] = start["as_of_date"].dt.year
    projected = exposure_paths(
        start["outstanding_balance"].to_numpy(), start["interest_rate"].to_numpy(), start["installment_amount"].to_numpy(), start["remaining_term_months"].to_numpy(), horizon + 1, slippage=slippage
    )[:, horizon]
    start["projected"] = projected
    later = panel[panel["as_of_date"].dt.month == 1][["loan_id", "as_of_date", "outstanding_balance"]].copy()
    later["year"] = later["as_of_date"].dt.year - 1
    merged = start.merge(later[["loan_id", "year", "outstanding_balance"]], on=["loan_id", "year"], suffixes=("", "_realise"))
    merged["erreur_relative"] = (merged["projected"] - merged["outstanding_balance_realise"]) / merged["outstanding_balance_realise"].clip(lower=1.0)
    return merged.groupby("year").agg(prets=("loan_id", "size"), biais_relatif_moyen=("erreur_relative", "mean"), mape=("erreur_relative", lambda s: float(s.abs().mean()))).reset_index()


def main() -> None:
    params = load_params()
    dpd = params["default_definition"]["dpd_threshold"]
    train, validation = load_split("train"), load_split("validation")
    test, oot = load_split("test"), load_split("out_of_time")
    cutoff = observation_cutoff(params)
    train_def = defaulted_exposures(train, dpd)
    model = fit_lgd_model(train_def, params)
    evaluation = {name: evaluate_lgd(model, defaulted_exposures(frame, dpd)) for name, frame in {"validation": validation, "test": test, "out_of_time": oot}.items()}
    macro_all = pd.concat([train, validation, test, oot])["macro_stress"].to_numpy()
    lgd_all = model.predict(pd.concat([train_def, defaulted_exposures(validation, dpd), defaulted_exposures(test, dpd), defaulted_exposures(oot, dpd)]))
    macro_def = pd.concat([train_def, defaulted_exposures(validation, dpd), defaulted_exposures(test, dpd), defaulted_exposures(oot, dpd)])["macro_stress"].to_numpy()
    add_on = downturn_add_on(lgd_all, macro_def, params["lgd"]["downturn_quantile"])
    observed = pd.concat([validation, test, oot])
    observed = observed[observed["as_of_date"] <= cutoff]
    raw_error = ead_projection_error(observed)
    slippage = float(max(0.0, -raw_error["biais_relatif_moyen"].mean()))
    adjusted = ead_projection_error(observed, slippage=slippage)
    ead_error = raw_error.merge(adjusted, on="year", suffixes=("", "_ajuste")).drop(columns="prets_ajuste")
    summary = {"lgd_evaluation": evaluation, "downturn_add_on": add_on, "mean_lgd_defaulted": float(lgd_all.mean()), "ead_ccf": params["ead"]["ccf"], "ead_slippage_annual": slippage, "macro_stress_mean": float(macro_all.mean())}
    with tracked_run("lgd_ead"):
        save_model(model, "lgd_ebm")
        save_table(ead_error, "ead_projection_error")
        save_metrics(summary, "lgd_ead")
        for split, metrics in evaluation.items():
            log_metrics(metrics, f"lgd.{split}.")
        log_metrics({"downturn_add_on": add_on, "mean_lgd_defaulted": summary["mean_lgd_defaulted"], "ead_mape_mean": float(np.nanmean(ead_error["mape"]))}, "lgd_ead.")
        log_table(ead_error, "ead_projection_error")
        log_json(summary, "lgd_ead")


if __name__ == "__main__":
    main()
