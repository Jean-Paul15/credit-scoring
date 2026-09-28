from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_EPS = 1e-6


def _proportions(counts: np.ndarray) -> np.ndarray:
    return np.maximum(counts / max(counts.sum(), 1), _EPS)


def psi_from_proportions(reference: np.ndarray, current: np.ndarray) -> float:
    return float(np.sum((current - reference) * np.log(current / reference)))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI sur des classes définies par les quantiles de la population de référence ; valeurs manquantes = classe dédiée."""
    ref = reference.to_numpy(dtype=float)
    cur = current.to_numpy(dtype=float)
    edges = np.unique(np.quantile(ref[~np.isnan(ref)], np.linspace(0, 1, bins + 1)[1:-1]))
    ref_counts = np.bincount(np.searchsorted(edges, ref[~np.isnan(ref)], side="right"), minlength=len(edges) + 1)
    cur_counts = np.bincount(np.searchsorted(edges, cur[~np.isnan(cur)], side="right"), minlength=len(edges) + 1)
    ref_counts = np.append(ref_counts, np.isnan(ref).sum())
    cur_counts = np.append(cur_counts, np.isnan(cur).sum())
    return psi_from_proportions(_proportions(ref_counts), _proportions(cur_counts))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    levels = sorted(set(reference.dropna().unique()) | set(current.dropna().unique()))
    ref_counts = reference.value_counts().reindex(levels, fill_value=0).to_numpy()
    cur_counts = current.value_counts().reindex(levels, fill_value=0).to_numpy()
    return psi_from_proportions(_proportions(ref_counts), _proportions(cur_counts))


def psi_band(value: float, cfg: dict[str, Any]) -> str:
    if value < cfg["psi_stable_max"]:
        return "stable"
    return "à surveiller" if value < cfg["psi_alert_min"] else "alerte"


def feature_psi(reference: pd.DataFrame, current: pd.DataFrame, numeric: list[str], categorical: list[str], cfg: dict[str, Any]) -> pd.DataFrame:
    """PSI (= CSI sur les variables) de chaque variable entre la référence et la population courante."""
    rows = [{"variable": c, "type": "numérique", "psi": psi_numeric(reference[c], current[c], cfg["psi_bins"])} for c in numeric]
    rows += [{"variable": c, "type": "catégorielle", "psi": psi_categorical(reference[c], current[c])} for c in categorical]
    table = pd.DataFrame(rows)
    table["statut"] = table["psi"].map(lambda v: psi_band(v, cfg))
    return table.sort_values("psi", ascending=False).reset_index(drop=True)


def score_psi_over_time(reference_score: pd.Series, frame: pd.DataFrame, score_col: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """PSI mensuel du score de PD par rapport à la référence."""
    rows = []
    for month, grp in frame.groupby(frame["as_of_date"].dt.to_period("M")):
        value = psi_numeric(reference_score, grp[score_col], cfg["psi_bins"])
        rows.append({"mois": str(month), "n": len(grp), "psi_score": value, "statut": psi_band(value, cfg)})
    return pd.DataFrame(rows)
