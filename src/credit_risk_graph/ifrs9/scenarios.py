from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MACRO_MIN, MACRO_MAX = 0.03, 0.85  # bornes du générateur de stress macro


@dataclass(frozen=True)
class Scenario:
    name: str
    weight: float
    shock: float  # écart du stress macro par rapport au niveau courant en année 1 (0 = scénario central)


def build_scenarios(macro_history: pd.Series, cfg: dict[str, Any]) -> list[Scenario]:
    """Trois scénarios : central, défavorable, favorable. Les chocs sont lus sur la distribution historique du stress macro."""
    median = float(macro_history.median())
    shocks = {
        "baseline": 0.0,
        "adverse": float(macro_history.quantile(cfg["adverse_quantile"])) - median,
        "favourable": float(macro_history.quantile(cfg["favourable_quantile"])) - median,
    }
    weights = cfg["weights"]
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Les pondérations des scénarios doivent sommer à 1")
    return [Scenario(name, float(weights[name]), shocks[name]) for name in ("baseline", "adverse", "favourable")]


def macro_paths(current_stress: np.ndarray, scenario: Scenario, cfg: dict[str, Any]) -> np.ndarray:
    """Stress macro par prêt et par année : choc en année 1, retour linéaire vers le niveau courant."""
    years = cfg["horizon_years"]
    reversion = cfg["reversion_years"]
    shock_by_year = np.array([scenario.shock * (1.0 - min(1.0, k / reversion)) for k in range(years)])
    return np.clip(current_stress[:, None] + shock_by_year[None, :], MACRO_MIN, MACRO_MAX)


def scenario_pd_matrix(score_fn, frame: pd.DataFrame, scenario: Scenario, cfg: dict[str, Any]) -> np.ndarray:
    """PD 12 mois par prêt et par année de projection : le modèle est ré-évalué avec le stress macro du scénario."""
    paths = macro_paths(frame["macro_stress"].to_numpy(dtype=float), scenario, cfg)
    sector_gap = frame["sector_stress"].to_numpy(dtype=float) - frame["macro_stress"].to_numpy(dtype=float)
    out = np.zeros_like(paths)
    for k in range(paths.shape[1]):
        shocked = frame.copy()
        shocked["macro_stress"] = paths[:, k]
        shocked["sector_stress"] = np.clip(paths[:, k] + sector_gap, MACRO_MIN, MACRO_MAX)
        out[:, k] = score_fn(shocked)
    return out
