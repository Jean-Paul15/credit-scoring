"""Étape DVC `backtest` : cohortes annuelles, tests par grade et global, stabilité du pouvoir discriminant."""
from __future__ import annotations

import pandas as pd

from credit_risk_graph.ifrs9.backtesting import annual_cohort, backtest_by_grade, backtest_portfolio, discrimination_by_year
from credit_risk_graph.ifrs9.data import SPLITS, load_performing, target
from credit_risk_graph.ifrs9.params import TABLES_DIR, load_params
from credit_risk_graph.ifrs9.rating import assign_grade
from credit_risk_graph.ifrs9.stages.common import load_model, save_metrics, save_table
from credit_risk_graph.ifrs9.tracking import log_json, log_metrics, log_table, tracked_run
from credit_risk_graph.utils.io import ensure_dir

COHORTS_FILE = "backtest_cohorts.parquet"


def build_cohorts(params: dict) -> pd.DataFrame:
    model, calibration = load_model("pd_ebm_champion"), load_model("pd_calibration")
    parts = []
    for split in SPLITS:
        cohort = annual_cohort(load_performing(split, params), params["backtesting"]["cohort_month"])
        pd_cal = calibration["calibrator"].predict(model.predict_proba(cohort))
        parts.append(
            pd.DataFrame(
                {
                    "loan_id": cohort["loan_id"].to_numpy(),
                    "as_of_date": cohort["as_of_date"].to_numpy(),
                    "split": split,
                    "pd": pd_cal,
                    "grade": assign_grade(pd_cal, calibration["grade_edges"]),
                    "y": target(cohort).to_numpy(),
                    "stage_dpd_ge_30": (cohort["current_dpd"] > 30).to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    params = load_params()
    cfg = params["backtesting"]
    cohorts = build_cohorts(params)
    ensure_dir(TABLES_DIR)
    cohorts.to_parquet(TABLES_DIR / COHORTS_FILE, index=False)
    portfolio = backtest_portfolio(cohorts, cfg)
    by_grade = backtest_by_grade(cohorts, cfg)
    discrimination = discrimination_by_year(cohorts, cfg, params["seed"])
    out_of_sample = portfolio[portfolio["split"].isin(["test", "out_of_time"])]
    summary = {
        "years_tested": int(len(portfolio)),
        "out_of_sample_years": out_of_sample["annee"].tolist(),
        "out_of_sample_red_lights": int((out_of_sample["feu"] == "rouge").sum()),
        "out_of_sample_yellow_lights": int((out_of_sample["feu"] == "jaune").sum()),
        "grade_year_red_lights_out_of_sample": int(((by_grade["split"].isin(["test", "out_of_time"])) & (by_grade["feu"] == "rouge")).sum()),
        "gini_out_of_sample_min": float(discrimination[discrimination["split"].isin(["test", "out_of_time"])]["gini"].min()),
        "hosmer_lemeshow_rejections_out_of_sample": int((out_of_sample["hl_pvalue"] < cfg["alpha"]).sum()),
    }
    with tracked_run("backtest"):
        for name, table in {"backtest_portfolio": portfolio, "backtest_by_grade": by_grade, "backtest_discrimination": discrimination}.items():
            save_table(table, name)
            log_table(table, name)
        save_metrics(summary, "backtest")
        log_metrics(summary, "backtest.")
        log_json(summary, "backtest")


if __name__ == "__main__":
    main()
