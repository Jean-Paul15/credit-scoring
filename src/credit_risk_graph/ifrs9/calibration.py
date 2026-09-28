from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

Method = Literal["platt", "isotonic"]
_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


@dataclass
class PDCalibrator:
    """Recalage de la PD brute sur les taux de défaut observés (échantillon de validation)."""

    method: Method
    _model: object = None

    def fit(self, raw_pd: np.ndarray, y: np.ndarray) -> "PDCalibrator":
        if self.method == "platt":
            self._model = LogisticRegression(C=1e6, max_iter=500).fit(_logit(raw_pd).reshape(-1, 1), y)
        else:
            self._model = IsotonicRegression(y_min=_EPS, y_max=1 - _EPS, out_of_bounds="clip").fit(raw_pd, y)
        return self

    def predict(self, raw_pd: np.ndarray) -> np.ndarray:
        if self.method == "platt":
            return self._model.predict_proba(_logit(raw_pd).reshape(-1, 1))[:, 1]
        return np.asarray(self._model.predict(raw_pd), dtype=float)

    def coefficients(self) -> dict[str, float]:
        if self.method != "platt":
            return {}
        return {"pente": float(self._model.coef_[0, 0]), "ordonnee": float(self._model.intercept_[0])}


def calibration_slope_intercept(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Pente et ordonnée de la régression logistique de y sur logit(PD) : pente 1 et ordonnée 0 = calibrage parfait."""
    fit = LogisticRegression(C=1e6, max_iter=500).fit(_logit(prob).reshape(-1, 1), y)
    return {"slope": float(fit.coef_[0, 0]), "intercept": float(fit.intercept_[0])}


def apply_floor(pd_values: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(pd_values, floor)
