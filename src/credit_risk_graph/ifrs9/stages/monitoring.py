"""Étape DVC `monitoring` : dérive des données et du score (PSI/CSI) par rapport à la référence de validation."""
from __future__ import annotations

import pandas as pd

from credit_risk_graph.ifrs9.data import feature_lists, load_performing
from credit_risk_graph.ifrs9.params import load_params
from credit_risk_graph.ifrs9.stages.common import load_model, save_metrics, save_table
from credit_risk_graph.ifrs9.tracking import log_json, log_metrics, log_table, tracked_run
from credit_risk_graph.monitoring.drift import feature_psi, psi_band, psi_numeric, score_psi_over_time


def main() -> None:
    params = load_params()
    cfg = params["monitoring"]
    model, calibration = load_model("pd_ebm_champion"), load_model("pd_calibration")
    categorical, numeric = feature_lists(model.profile)
    reference = load_performing(cfg["reference_split"], params)
    reference["score"] = calibration["calibrator"].predict(model.predict_proba(reference))
    feature_tables, score_tables, headline = [], [], {}
    with tracked_run("monitoring"):
        for split in ("test", "out_of_time"):
            current = load_performing(split, params)
            current["score"] = calibration["calibrator"].predict(model.predict_proba(current))
            table = feature_psi(reference, current, numeric, categorical, cfg).assign(split=split)
            feature_tables.append(table)
            score_tables.append(score_psi_over_time(reference["score"], current, "score", cfg).assign(split=split))
            overall = psi_numeric(reference["score"], current["score"], cfg["psi_bins"])
            headline[split] = {"score_psi": overall, "score_status": psi_band(overall, cfg), "features_alert": int((table["statut"] == "alerte").sum()), "features_watch": int((table["statut"] == "à surveiller").sum()), "max_feature_psi": float(table["psi"].max())}
            log_metrics(headline[split], f"{split}.")
        features, scores = pd.concat(feature_tables, ignore_index=True), pd.concat(score_tables, ignore_index=True)
        save_table(features, "psi_features")
        save_table(scores, "psi_score_monthly")
        save_metrics(headline, "monitoring")
        log_table(features, "psi_features")
        log_table(scores, "psi_score_monthly")
        log_json(headline, "monitoring")


if __name__ == "__main__":
    main()
