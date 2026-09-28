"""Étape DVC `ecl` : staging, scénarios macro pondérés, ECL par prêt, rapprochement avec la vue prudentielle."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from credit_risk_graph.ifrs9 import basel_view
from credit_risk_graph.ifrs9.data import load_full_split as load_split
from credit_risk_graph.ifrs9.data import observation_cutoff
from credit_risk_graph.ifrs9.lgd import apply_regulatory_floors
from credit_risk_graph.ifrs9.params import METRICS_DIR, TABLES_DIR, load_params
from credit_risk_graph.ifrs9.portfolio import ECLEngine
from credit_risk_graph.ifrs9.scenarios import build_scenarios
from credit_risk_graph.ifrs9.stages.backtest import COHORTS_FILE
from credit_risk_graph.ifrs9.stages.common import load_model, save_metrics, save_model, save_table
from credit_risk_graph.ifrs9.tracking import log_json, log_metrics, log_table, tracked_run


def _portfolio() -> pd.DataFrame:
    """Prêts ouverts au dernier arrêté observé."""
    oot = load_split("out_of_time")
    latest = oot[oot["as_of_date"] == oot["as_of_date"].max()]
    return latest[latest["outstanding_balance"] > 0].reset_index(drop=True)


def build_engine(params: dict) -> ECLEngine:
    calibration = load_model("pd_calibration")
    history = pd.concat([load_split(s) for s in ("validation", "test", "out_of_time")])["macro_stress"]
    slippage = json.loads((METRICS_DIR / "lgd_ead.json").read_text(encoding="utf-8"))["ead_slippage_annual"]
    return ECLEngine(load_model("pd_ebm_champion"), calibration["calibrator"], calibration["grade_edges"], load_model("lgd_ebm"), build_scenarios(history, params["scenarios"]), params, slippage)


def prudential_view(frame: pd.DataFrame, result_table: pd.DataFrame, lgd_paths: dict, params: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Perte attendue prudentielle indicative : PD moyenne de long terme par grade, LGD de ralentissement, planchers."""
    ttc = basel_view.through_the_cycle_pd(pd.read_parquet(TABLES_DIR / COHORTS_FILE))
    ttc_pd = result_table["grade"].map(ttc.set_index("grade")["ttc_pd"]).fillna(result_table["pd_12m"].mean()).to_numpy()
    ttc_pd = np.where(result_table["stage"].to_numpy() == 3, 1.0, ttc_pd)
    add_on = max(float(lgd_paths["adverse"][:, 0].mean()) - float(lgd_paths["baseline"][:, 0].mean()), 0.0)
    lgd = apply_regulatory_floors(lgd_paths["baseline"][:, 0] + add_on, frame["collateral_type"], params["lgd"]["floors"])
    return basel_view.prudential_expected_loss(ttc_pd, params["calibration"]["pd_floor"], lgd, result_table["ead"].to_numpy()), ttc


def main() -> None:
    params = load_params()
    engine = build_engine(params)
    frame = _portfolio()
    result = engine.run(frame)
    prudential, ttc = prudential_view(frame, result.table, result.lgd_paths, params)
    table = result.table.assign(el_prudentielle=prudential)
    by_stage = table.groupby("stage").agg(prets=("loan_id", "size"), ead=("ead", "sum"), ecl_ponderee=("ecl_ponderee", "sum"), el_prudentielle=("el_prudentielle", "sum")).reset_index()
    by_stage["couverture"] = by_stage["ecl_ponderee"] / by_stage["ead"]
    scenario_table = pd.DataFrame([{"scenario": s.name, "poids": s.weight, "choc_macro": s.shock, "ecl": float(table[f"ecl_{s.name}"].sum())} for s in engine.scenarios])
    reconciliation = basel_view.reconcile(table["ecl_ponderee"].to_numpy(), prudential, table["stage"])
    summary = {
        "as_of": str(frame["as_of_date"].max().date()),
        "loans": int(len(table)),
        "ead_total": float(table["ead"].sum()),
        "ecl_ponderee_total": float(table["ecl_ponderee"].sum()),
        "ecl_12m_if_all_stage1": float(table["ecl_12m_tous_stage1"].sum()),
        "prudential_el_total": float(prudential.sum()),
        "stage_mix": {int(k): float(v) for k, v in table["stage"].value_counts(normalize=True).sort_index().items()},
        "observation_cutoff_labels": str(observation_cutoff(params).date()),
    }
    with tracked_run("ecl"):
        table.to_parquet(TABLES_DIR / "ecl_by_loan.parquet", index=False)
        save_model(engine, "ecl_engine")
        for name, out in {"ecl_by_stage": by_stage, "ecl_by_scenario": scenario_table, "ecl_vs_prudential": reconciliation, "ttc_pd_by_grade": ttc}.items():
            save_table(out, name)
            log_table(out, name)
        save_metrics(summary, "ecl")
        log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))}, "ecl.")
        log_metrics(summary["stage_mix"], "ecl.stage_mix.")
        log_json(summary, "ecl")


if __name__ == "__main__":
    main()
