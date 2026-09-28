from __future__ import annotations

import numpy as np


def exposure_paths(
    balance: np.ndarray, annual_rate: np.ndarray, installment: np.ndarray, remaining_term: np.ndarray, horizon: int, ccf: float = 1.0, slippage: float = 0.0
) -> np.ndarray:
    """Exposition projetée en début de chaque mois (colonne 0 = solde à la date d'arrêté) d'un prêt amortissable.

    Solde suivant = Solde * (1 + r/12) - échéance, borné à 0 ; l'exposition est nulle au-delà de la maturité résiduelle.
    Les prêts du portefeuille sont des prêts à terme sans engagement hors bilan : le facteur de conversion (CCF) vaut 1.
    `slippage` (annuel) corrige l'échéancier contractuel du ralentissement d'amortissement observé (retards, paiements partiels) :
    l'exposition est majorée de slippage x mois/12, sans jamais dépasser le solde à la date d'arrêté.
    """
    paths = np.zeros((len(balance), horizon))
    current = balance.astype(float).copy()
    monthly = annual_rate / 12.0
    for month in range(horizon):
        paths[:, month] = np.where(remaining_term > month, current, 0.0) * ccf
        current = np.maximum(current * (1.0 + monthly) - installment, 0.0)
    if slippage:
        factor = 1.0 + slippage * np.arange(horizon)[None, :] / 12.0
        paths = np.minimum(paths * factor, (balance * ccf)[:, None])
    return paths


def exposure_at_default(balance: np.ndarray, ccf: float = 1.0) -> np.ndarray:
    """EAD à la date d'arrêté : solde restant dû (aucun hors-bilan)."""
    return balance.astype(float) * ccf
