"""Étape DVC `figures` : figures publiques et fiche modèle, lues dans les sorties des étapes précédentes."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from credit_risk_graph.ifrs9 import plots
from credit_risk_graph.ifrs9.eda import reliability_inputs
from credit_risk_graph.ifrs9.params import IFRS9_DIR, METRICS_DIR, TABLES_DIR, load_params
from credit_risk_graph.ifrs9.pd_model import term_shape
from credit_risk_graph.ifrs9.stages.common import figure_path, load_model
from credit_risk_graph.ifrs9.tracking import experiment_id, log_figure, tracked_run


def _read_metrics(name: str) -> dict:
    return json.loads((METRICS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(figure_path(name), dpi=130, bbox_inches="tight")
    log_figure(fig, name)
    plt.close(fig)


def export_mlflow_runs() -> None:
    """Journal lisible des exécutions MLflow (étape, date, version du code et des données, nombre de paramètres et de métriques)."""
    runs = mlflow.search_runs(experiment_ids=[experiment_id()], output_format="pandas")
    runs = runs[runs["run_id"] != mlflow.active_run().info.run_id]
    table = pd.DataFrame(
        {
            "etape": runs["tags.pipeline_stage"],
            "debut": runs["start_time"].dt.strftime("%Y-%m-%d %H:%M"),
            "statut": runs["status"],
            "version_code": runs["tags.git_commit"],
            "version_donnees": runs["tags.data_version"],
            "parametres": runs.filter(like="params.").notna().sum(axis=1),
            "metriques": runs.filter(like="metrics.").notna().sum(axis=1),
        }
    ).sort_values("debut")
    table.to_csv(IFRS9_DIR / "mlflow_runs.csv", index=False)


def write_model_card(params: dict) -> None:
    pd_model, calibration = _read_metrics("pd_model"), _read_metrics("pd_calibration")
    comparison = pd.read_csv(TABLES_DIR / "pd_model_comparison.csv")
    champion = comparison[(comparison["profil"] == pd_model["champion_profile"]) & (comparison["modele"] == "ebm")].set_index("split")
    card = {
        "modele": "Explainable Boosting Machine (PD 12 mois)",
        "profil_de_variables": pd_model["champion_profile"],
        "nombre_de_variables": pd_model["n_features"],
        "definition_du_defaut": f"retard >= {params['default_definition']['dpd_threshold']} jours",
        "variables_de_tiers": "exclues (garde-fou testé)",
        "calibrage": calibration["method"],
        "hyperparametres": _read_metrics("ebm_tuning")["best_params"],
        "performance": {split: {"auc": round(float(champion.loc[split, "auc"]), 4), "gini": round(float(champion.loc[split, "gini"]), 4)} for split in ("train", "validation", "test", "out_of_time")},
        "donnees": "synthetiques",
        "statut_reglementaire": "aligne sur IFRS 9 et Bale III (risque de credit), non valide par un superviseur",
    }
    (IFRS9_DIR / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    params = load_params()
    plots.set_style()
    model = load_model("pd_ebm_champion")
    importance = pd.read_csv(TABLES_DIR / "pd_ebm_importance.csv")
    with tracked_run("figures"):
        _save(plots.importance(importance), "ebm_importance")
        for term in importance["terme"].head(4):
            _save(plots.shape(term_shape(model, term), term), f"ebm_shape_{term.replace(' ', '_').replace('&', 'x')}")
        y, probs = reliability_inputs("out_of_time")
        _save(plots.reliability(y, probs), "calibration_out_of_time")
        _save(plots.gini_stability(pd.read_csv(TABLES_DIR / "backtest_discrimination.csv")), "gini_stability")
        _save(plots.traffic_light_grid(pd.read_csv(TABLES_DIR / "backtest_by_grade.csv")), "backtest_traffic_lights")
        psi_features, psi_score = pd.read_csv(TABLES_DIR / "psi_features.csv"), pd.read_csv(TABLES_DIR / "psi_score_monthly.csv")
        _save(plots.psi_bars(psi_features[psi_features["split"] == "out_of_time"]), "psi_features")
        _save(plots.psi_time(psi_score[psi_score["split"] == "out_of_time"]), "psi_score_monthly")
        _save(plots.ecl_by_stage(pd.read_csv(TABLES_DIR / "ecl_by_stage.csv")), "ecl_coverage_by_stage")
        write_model_card(params)
        export_mlflow_runs()


if __name__ == "__main__":
    main()
