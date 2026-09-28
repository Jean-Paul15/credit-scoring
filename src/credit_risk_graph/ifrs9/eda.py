from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk_graph.config import DATA_DIR
from credit_risk_graph.features.schema import TARGET_COLUMN
from credit_risk_graph.ifrs9.data import load_performing, target
from credit_risk_graph.ifrs9.default_definition import DefaultDefinition, definition_sensitivity
from credit_risk_graph.ifrs9.params import load_params
from credit_risk_graph.ifrs9.stages.common import load_model
from credit_risk_graph.utils.io import read_parquet


def calendar_report() -> pd.DataFrame:
    """Plages de dates de chaque table : aucune date ne doit dépasser la date du jour."""
    today = pd.Timestamp.today().normalize()
    sources = {
        "prêts (octroi)": ("loans_clean", "origination_date"),
        "performance mensuelle": ("performance_clean", "as_of_date"),
        "macro": ("macro_clean", "month"),
    }
    rows = []
    for label, (table, column) in sources.items():
        series = read_parquet(DATA_DIR / "clean" / f"{table}.parquet")[column]
        rows.append({"table": label, "première_date": series.min().date(), "dernière_date": series.max().date(), "postérieure_à_aujourd_hui": bool(series.max() > today)})
    return pd.DataFrame(rows)


def portfolio_overview() -> pd.DataFrame:
    loans = read_parquet(DATA_DIR / "clean" / "loans_clean.parquet")
    perf = read_parquet(DATA_DIR / "clean" / "performance_clean.parquet")
    return pd.DataFrame(
        {
            "emprunteurs": [read_parquet(DATA_DIR / "clean" / "borrowers_clean.parquet")["borrower_id"].nunique()],
            "prêts": [len(loans)],
            "lignes de performance mensuelle": [len(perf)],
            "encours initial total": [loans["principal_amount"].sum()],
            "taux d'intérêt moyen": [loans["interest_rate"].mean()],
            "maturité moyenne (mois)": [loans["tenor_months"].mean()],
        }
    )


def default_rate_by(dimension: str) -> pd.DataFrame:
    """Taux de défaut à 12 mois (population saine, fenêtre complète) par dimension ; `mois` = série temporelle."""
    params = load_params()
    frames = pd.concat([load_performing(s, params) for s in ("train", "validation", "test", "out_of_time")], ignore_index=True)
    key = frames["as_of_date"].dt.to_period("M").astype(str) if dimension == "mois" else frames[dimension]
    grouped = frames.groupby(key).agg(expositions=("loan_id", "size"), taux_defaut_12m=(TARGET_COLUMN, "mean"))
    return grouped.rename_axis(dimension).reset_index()


def definition_sensitivity_report() -> pd.DataFrame:
    params = load_params()
    perf = read_parquet(DATA_DIR / "clean" / "performance_clean.parquet")
    sample_loans = perf["loan_id"].drop_duplicates().sample(20_000, random_state=params["seed"])
    return definition_sensitivity(perf[perf["loan_id"].isin(sample_loans)], DefaultDefinition.from_params(params))


def reliability_inputs(split: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    params = load_params()
    model, calibration = load_model("pd_ebm_champion"), load_model("pd_calibration")
    frame = load_performing(split, params)
    raw = model.predict_proba(frame)
    return target(frame).to_numpy(), {"EBM brut": raw, "EBM calibré": calibration["calibrator"].predict(raw)}


def local_explanations(model, n: int = 3) -> pd.DataFrame:
    """Décomposition additive (logit) de n dossiers : les 5 termes les plus contributifs de chacun."""
    params = load_params()
    sample = load_performing("out_of_time", params).sample(n, random_state=params["seed"])
    contributions = model.term_contributions(sample)
    rows = []
    for idx in contributions.index:
        top = contributions.loc[idx].abs().sort_values(ascending=False).head(5).index
        rows.append({"dossier": sample.loc[idx, "loan_id"], "PD brute": float(model.predict_proba(sample.loc[[idx]])[0]), **{t: float(contributions.loc[idx, t]) for t in top}})
    return pd.DataFrame(rows)
