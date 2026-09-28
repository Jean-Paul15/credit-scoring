from __future__ import annotations

import numpy as np
import pandas as pd


def fit_master_scale(calibrated_pd: np.ndarray, n_grades: int) -> np.ndarray:
    """Bornes de grades par quantiles de PD sur l'échantillon de référence (1 = meilleure qualité)."""
    inner = np.quantile(calibrated_pd, np.linspace(0, 1, n_grades + 1)[1:-1])
    return np.unique(inner)


def assign_grade(calibrated_pd: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, calibrated_pd, side="right") + 1


def grade_table(grade: np.ndarray, calibrated_pd: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"grade": grade, "pd": calibrated_pd, "y": y})
    table = frame.groupby("grade").agg(exposures=("y", "size"), defaults=("y", "sum"), mean_pd=("pd", "mean"), observed_dr=("y", "mean"))
    return table.reset_index()
