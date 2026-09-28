from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def assign_stage(frame: pd.DataFrame, pd_12m: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Staging IFRS 9 vectorisé.

    Stage 3 : retard >= seuil de défaut. Stage 2 : retard > 30 j (présomption réfutable §5.5.11),
    restructuration, ou hausse significative du risque de crédit (SICR) mesurée par la PD 12 mois
    courante rapportée à la PD à l'octroi (seuils relatif ET absolu, choix internes). Sinon Stage 1.
    """
    origination = frame["origination_pd"].to_numpy(dtype=float)
    ratio = pd_12m / np.maximum(origination, 1e-4)
    delta = pd_12m - origination
    sicr = (ratio >= cfg["sicr_relative_multiplier"]) & (delta >= cfg["sicr_absolute_delta"])
    dpd = frame["current_dpd"].to_numpy()
    stage2 = (dpd > cfg["stage2_dpd"]) | (frame["restructured_flag"].to_numpy() == 1) | sicr
    stage = np.where(stage2, 2, 1)
    return np.where(dpd >= cfg["stage3_dpd"], 3, stage)


def stage_migration(stage_from: np.ndarray, stage_to: np.ndarray) -> pd.DataFrame:
    """Matrice de migration entre deux dates d'arrêté (part de chaque stage d'origine)."""
    matrix = pd.crosstab(pd.Series(stage_from, name="stage_initial"), pd.Series(stage_to, name="stage_final"), normalize="index")
    return matrix.reset_index()
