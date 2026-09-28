from __future__ import annotations

import numpy as np
import pandas as pd


def through_the_cycle_pd(cohorts: pd.DataFrame) -> pd.DataFrame:
    """PD moyenne de long terme par grade : moyenne des taux de défaut annuels observés (Bâle CRE36.63, approche indicative)."""
    yearly = cohorts.groupby(["grade", cohorts["as_of_date"].dt.year]).agg(dr=("y", "mean"), n=("y", "size")).reset_index()
    table = yearly.groupby("grade").agg(ttc_pd=("dr", "mean"), annees=("dr", "size"), exposures=("n", "sum")).reset_index()
    return table


def prudential_expected_loss(
    ttc_pd: np.ndarray,
    pd_floor: float,
    lgd_downturn: np.ndarray,
    ead: np.ndarray,
) -> np.ndarray:
    """Perte attendue prudentielle indicative = max(PD TTC, plancher) x LGD de ralentissement (plancher inclus) x EAD.

    Ce n'est pas un calcul d'actifs pondérés : aucune fonction de pondération IRB n'est implémentée.
    """
    return np.maximum(ttc_pd, pd_floor) * lgd_downturn * ead


def reconcile(ifrs9_ecl: np.ndarray, prudential_el: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    """Rapprochement IFRS 9 / prudentiel par groupe (stage ou grade)."""
    table = pd.DataFrame({"groupe": groups.to_numpy(), "ecl_ifrs9": ifrs9_ecl, "el_prudentielle": prudential_el})
    summary = table.groupby("groupe").sum().reset_index()
    summary["ratio_ifrs9_sur_prudentiel"] = summary["ecl_ifrs9"] / summary["el_prudentielle"].replace(0, np.nan)
    return summary
