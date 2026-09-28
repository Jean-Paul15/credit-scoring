from __future__ import annotations

import numpy as np


def monthly_hazard(pd_12m: np.ndarray) -> np.ndarray:
    """Intensité de défaut mensuelle constante équivalente à une PD 12 mois."""
    return 1.0 - (1.0 - np.clip(pd_12m, 0.0, 1.0)) ** (1.0 / 12.0)


def marginal_pd(pd_by_year: np.ndarray, horizon_months: int) -> np.ndarray:
    """PD marginale mensuelle (défaut au mois m sachant la survie jusqu'à m-1) à partir de PD 12 mois par année de projection."""
    year_index = np.minimum(np.arange(horizon_months) // 12, pd_by_year.shape[1] - 1)
    hazard = monthly_hazard(pd_by_year[:, year_index])
    survival_before = np.concatenate([np.ones((hazard.shape[0], 1)), np.cumprod(1.0 - hazard, axis=1)[:, :-1]], axis=1)
    return survival_before * hazard


def discount_factors(annual_rate: np.ndarray, horizon_months: int) -> np.ndarray:
    """Facteurs d'actualisation au taux d'intérêt effectif du prêt (fin de mois)."""
    months = np.arange(1, horizon_months + 1)[None, :]
    return (1.0 + annual_rate[:, None] / 12.0) ** (-months)


def expected_credit_loss(
    pd_by_year: np.ndarray,
    lgd_by_year: np.ndarray,
    exposure: np.ndarray,
    annual_rate: np.ndarray,
    stage: np.ndarray,
    max_horizon: int,
) -> np.ndarray:
    """ECL par prêt pour un scénario.

    Stage 1 : perte attendue à 12 mois. Stage 2 : perte attendue sur la durée de vie résiduelle (l'exposition projetée
    s'annule à maturité). Stage 3 : PD = 1, ECL = LGD x EAD (pas d'actualisation des recouvrements dans cette version).
    """
    horizon = exposure.shape[1]
    marginal = marginal_pd(pd_by_year, horizon)
    lgd_month = lgd_by_year[:, np.minimum(np.arange(horizon) // 12, lgd_by_year.shape[1] - 1)]
    loss = marginal * lgd_month * exposure * discount_factors(annual_rate, horizon)
    twelve_month = loss[:, :12].sum(axis=1)
    lifetime = loss[:, :max_horizon].sum(axis=1)
    default_loss = lgd_by_year[:, 0] * exposure[:, 0]
    return np.where(stage == 1, twelve_month, np.where(stage == 2, lifetime, default_loss))


def probability_weighted(ecl_by_scenario: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(weights[name] * values for name, values in ecl_by_scenario.items())
