from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, binom, chi2, norm
from sklearn.metrics import roc_auc_score

from credit_risk_graph.ifrs9.metrics import ks_statistic


def annual_cohort(frame: pd.DataFrame, month: int) -> pd.DataFrame:
    """Une observation par prêt et par an : PD à la date de cohorte, défaut observé dans les 12 mois suivants."""
    return frame.loc[frame["as_of_date"].dt.month == month].copy()


def binomial_pvalue(n: int, defaults: int, mean_pd: float) -> float:
    """P(X >= d) sous H0 : PD vraie = PD moyenne prédite, défauts indépendants. Petite valeur = PD sous-estimée."""
    if defaults <= 0:
        return 1.0
    return float(binom.sf(defaults - 1, n, mean_pd))


def vasicek_pvalue(n: int, defaults: int, mean_pd: float, rho: float) -> float:
    """Test ajusté de la corrélation de défaut (modèle à un facteur, loi limite de Vasicek)."""
    rate = min(max(defaults / n, 1e-9), 1 - 1e-9)
    z = (np.sqrt(1 - rho) * norm.ppf(rate) - norm.ppf(mean_pd)) / np.sqrt(rho)
    return float(1 - norm.cdf(z))


def jeffreys_pvalue(n: int, defaults: int, mean_pd: float) -> float:
    """Probabilité a posteriori (prior de Jeffreys) que la PD vraie soit <= PD prédite. Petite valeur = sous-estimation."""
    return float(beta.cdf(mean_pd, defaults + 0.5, n - defaults + 0.5))


def hosmer_lemeshow(y: np.ndarray, prob: np.ndarray, groups: int = 10) -> tuple[float, float]:
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, groups + 1)))
    idx = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, len(edges) - 2)
    stat, used = 0.0, 0
    for g in range(len(edges) - 1):
        mask = idx == g
        n = int(mask.sum())
        if n == 0:
            continue
        expected, observed = prob[mask].sum(), y[mask].sum()
        variance = expected * (1 - prob[mask].mean())
        if variance > 0:
            stat += (observed - expected) ** 2 / variance
            used += 1
    dof = max(used - 2, 1)
    return float(stat), float(chi2.sf(stat, dof))


def traffic_light(pvalue: float, green_min: float, red_max: float) -> str:
    if pvalue >= green_min:
        return "vert"
    return "jaune" if pvalue >= red_max else "rouge"


def _row_tests(n: int, defaults: int, mean_pd: float, cfg: dict[str, Any]) -> dict[str, Any]:
    p_binom = binomial_pvalue(n, defaults, mean_pd)
    return {
        "exposures": n,
        "defaults": defaults,
        "mean_pd": mean_pd,
        "observed_dr": defaults / n,
        "p_binomial": p_binom,
        "p_vasicek": vasicek_pvalue(n, defaults, mean_pd, cfg["asset_correlation"]),
        "p_jeffreys": jeffreys_pvalue(n, defaults, mean_pd),
        "feu": traffic_light(p_binom, cfg["green_min_pvalue"], cfg["red_max_pvalue"]),
    }


def backtest_portfolio(cohorts: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Une ligne par année de cohorte : calibrage global, tests binomial / Vasicek / Jeffreys / Hosmer-Lemeshow."""
    rows = []
    for (year, split), grp in cohorts.groupby([cohorts["as_of_date"].dt.year, "split"]):
        y, p = grp["y"].to_numpy(), grp["pd"].to_numpy()
        hl_stat, hl_p = hosmer_lemeshow(y, p)
        rows.append({"annee": int(year), "split": split, **_row_tests(len(grp), int(y.sum()), float(p.mean()), cfg), "hl_stat": hl_stat, "hl_pvalue": hl_p})
    return pd.DataFrame(rows).sort_values("annee").reset_index(drop=True)


def backtest_by_grade(cohorts: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Test par grade et par année (unité réglementaire du backtesting de PD)."""
    rows = []
    for (year, split, grade), grp in cohorts.groupby([cohorts["as_of_date"].dt.year, "split", "grade"]):
        rows.append({"annee": int(year), "split": split, "grade": int(grade), **_row_tests(len(grp), int(grp["y"].sum()), float(grp["pd"].mean()), cfg)})
    return pd.DataFrame(rows).sort_values(["annee", "grade"]).reset_index(drop=True)


def bootstrap_auc_interval(y: np.ndarray, prob: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], prob[idx]))
    return float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))


def discrimination_by_year(cohorts: pd.DataFrame, cfg: dict[str, Any], seed: int) -> pd.DataFrame:
    """Pouvoir discriminant par cohorte annuelle, avec intervalle de confiance bootstrap sur l'AUC."""
    rows = []
    for (year, split), grp in cohorts.groupby([cohorts["as_of_date"].dt.year, "split"]):
        y, p = grp["y"].to_numpy(), grp["pd"].to_numpy()
        auc = float(roc_auc_score(y, p))
        low, high = bootstrap_auc_interval(y, p, cfg["bootstrap_draws"], seed)
        rows.append({"annee": int(year), "split": split, "auc": auc, "auc_ic95_bas": low, "auc_ic95_haut": high, "gini": 2 * auc - 1, "ks": ks_statistic(y, p)})
    return pd.DataFrame(rows).sort_values("annee").reset_index(drop=True)
