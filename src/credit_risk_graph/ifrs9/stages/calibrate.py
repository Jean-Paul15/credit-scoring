"""Étape DVC `calibrate` : calibrage de la PD sur la validation, échelle maître de grades, contrôle sur test et hors période."""
from __future__ import annotations

import pandas as pd

from credit_risk_graph.ifrs9.calibration import PDCalibrator, calibration_slope_intercept
from credit_risk_graph.ifrs9.data import load_performing, target
from credit_risk_graph.ifrs9.metrics import calibration_metrics, discrimination_metrics
from credit_risk_graph.ifrs9.params import load_params
from credit_risk_graph.ifrs9.rating import assign_grade, fit_master_scale, grade_table
from credit_risk_graph.ifrs9.stages.common import load_model, save_metrics, save_model, save_table
from credit_risk_graph.ifrs9.tracking import log_json, log_metrics, log_table, tracked_run


def main() -> None:
    params = load_params()
    cfg = params["calibration"]
    model = load_model("pd_ebm_champion")
    validation = load_performing("validation", params)
    y_val = target(validation).to_numpy()
    raw_val = model.predict_proba(validation)
    calibrators = {m: PDCalibrator(m).fit(raw_val, y_val) for m in ("platt", "isotonic")}
    chosen = calibrators[cfg["method"]]
    edges = fit_master_scale(chosen.predict(raw_val), params["rating"]["n_grades"])

    rows, grades = [], []
    with tracked_run("calibrate"):
        for split in ("validation", "test", "out_of_time"):
            frame = validation if split == "validation" else load_performing(split, params)
            y, raw = target(frame).to_numpy(), model.predict_proba(frame)
            for label, prob in {"brute": raw, "platt": calibrators["platt"].predict(raw), "isotonic": calibrators["isotonic"].predict(raw)}.items():
                metrics = {**calibration_metrics(y, prob), **calibration_slope_intercept(y, prob), **discrimination_metrics(y, prob)}
                rows.append({"split": split, "pd": label, **metrics})
                log_metrics(metrics, f"{split}.{label}.")
            grades.append(grade_table(assign_grade(chosen.predict(raw), edges), chosen.predict(raw), y).assign(split=split))
        table, grade_summary = pd.DataFrame(rows), pd.concat(grades, ignore_index=True)
        save_model({"calibrator": chosen, "grade_edges": edges, "method": cfg["method"]}, "pd_calibration")
        save_table(table, "pd_calibration_metrics")
        save_table(grade_summary, "pd_grade_table")
        save_metrics({"method": cfg["method"], "coefficients": chosen.coefficients(), "grade_edges": edges.tolist()}, "pd_calibration")
        log_table(table, "pd_calibration_metrics")
        log_table(grade_summary, "pd_grade_table")
        log_json({"method": cfg["method"], "coefficients": chosen.coefficients(), "grade_edges": edges.tolist()}, "pd_calibration")


if __name__ == "__main__":
    main()
