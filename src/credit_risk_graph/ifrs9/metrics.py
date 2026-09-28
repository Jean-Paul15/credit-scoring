from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    """ECE par déciles de PD (quantiles), pondérée par l'effectif."""
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(prob[mask].mean() - y_true[mask].mean())
    return float(total)


def calibration_in_the_large(y_true: np.ndarray, prob: np.ndarray) -> float:
    """PD moyenne prédite moins taux de défaut observé (0 = calibrage global exact)."""
    return float(np.mean(prob) - np.mean(y_true))


def discrimination_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    auc = float(roc_auc_score(y_true, prob))
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, prob),
        "pr_auc": float(average_precision_score(y_true, prob)),
    }


def calibration_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(brier_score_loss(y_true, prob)),
        "ece": expected_calibration_error(y_true, prob),
        "calibration_in_the_large": calibration_in_the_large(y_true, prob),
        "mean_pd": float(np.mean(prob)),
        "observed_default_rate": float(np.mean(y_true)),
    }


def full_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    return {**discrimination_metrics(y_true, prob), **calibration_metrics(y_true, prob)}
