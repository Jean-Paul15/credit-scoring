from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DefaultDefinition:
    """Définition opérationnelle du défaut : retard > seuil, éventuellement complété par un indicateur d'improbabilité de paiement."""

    dpd_threshold: int = 90
    utp_restructured_min_dpd: int = 60
    cure_months: int = 3

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "DefaultDefinition":
        block = params["default_definition"]
        return cls(block["dpd_threshold"], block["utp_restructured_min_dpd"], block["cure_months"])


def flag_default(panel: pd.DataFrame, definition: DefaultDefinition, include_utp: bool = False) -> pd.Series:
    """Indicateur de défaut instantané. `include_utp` ajoute les restructurations en retard significatif."""
    flag = panel["current_dpd"] >= definition.dpd_threshold
    if include_utp:
        flag |= (panel["restructured_flag"] == 1) & (panel["current_dpd"] >= definition.utp_restructured_min_dpd)
    return flag.astype(int)


def flag_with_cure(panel: pd.DataFrame, definition: DefaultDefinition, include_utp: bool = False) -> pd.Series:
    """Un prêt reste en défaut tant que `cure_months` mois consécutifs sains ne se sont pas écoulés."""
    ordered = panel.sort_values(["loan_id", "as_of_date"])
    raw = flag_default(ordered, definition, include_utp)
    window = definition.cure_months + 1
    cured = raw.groupby(ordered["loan_id"]).rolling(window, min_periods=1).max()
    cured = cured.reset_index(level=0, drop=True).astype(int)
    return cured.reindex(panel.index)


def definition_sensitivity(panel: pd.DataFrame, definition: DefaultDefinition) -> pd.DataFrame:
    """Taux de défaut mensuel moyen selon la variante de définition (retard seul, avec UTP, avec guérison)."""
    variants = {
        f"retard >= {definition.dpd_threshold} j": flag_default(panel, definition),
        f"retard >= {definition.dpd_threshold} j ou restructuré >= {definition.utp_restructured_min_dpd} j": flag_default(
            panel, definition, include_utp=True
        ),
        f"retard >= {definition.dpd_threshold} j avec guérison {definition.cure_months} mois": flag_with_cure(panel, definition),
    }
    rows = [{"variante": name, "taux_defaut_mensuel": float(series.mean()), "lignes_en_defaut": int(series.sum())} for name, series in variants.items()]
    return pd.DataFrame(rows)
