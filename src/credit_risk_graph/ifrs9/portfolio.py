from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from credit_risk_graph.ifrs9.calibration import PDCalibrator
from credit_risk_graph.ifrs9.ead import exposure_at_default, exposure_paths
from credit_risk_graph.ifrs9.ecl import expected_credit_loss, probability_weighted
from credit_risk_graph.ifrs9.lgd import LGDModel
from credit_risk_graph.ifrs9.pd_model import PDModel
from credit_risk_graph.ifrs9.rating import assign_grade
from credit_risk_graph.ifrs9.scenarios import Scenario, macro_paths, scenario_pd_matrix
from credit_risk_graph.ifrs9.staging import assign_stage

PD_FLOOR_MODEL = 1e-4


@dataclass
class ECLResult:
    """Sorties par prêt et trajectoires nécessaires à la vue prudentielle."""

    table: pd.DataFrame
    lgd_paths: dict[str, np.ndarray]


@dataclass
class ECLEngine:
    """Chaîne complète PD calibrée -> staging -> scénarios -> ECL, identique en batch (DVC) et en ligne (API)."""

    pd_model: PDModel
    calibrator: PDCalibrator
    grade_edges: np.ndarray
    lgd_model: LGDModel
    scenarios: list[Scenario]
    params: dict[str, Any]
    ead_slippage: float = 0.0

    def score_pd(self, frame: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.pd_model.predict_proba(frame))

    def _lgd_paths(self, frame: pd.DataFrame, macro: np.ndarray) -> np.ndarray:
        """LGD conditionnelle au défaut, par année de projection, sous le stress macro du scénario."""
        conditional = frame.copy()
        conditional["current_dpd"] = np.maximum(conditional["current_dpd"], self.params["staging"]["stage3_dpd"])
        out = np.zeros_like(macro)
        for k in range(macro.shape[1]):
            conditional["macro_stress"] = macro[:, k]
            out[:, k] = self.lgd_model.predict(conditional)
        return out

    def run(self, frame: pd.DataFrame) -> ECLResult:
        p = self.params
        pd_now = self.score_pd(frame)
        stage = assign_stage(frame, pd_now, p["staging"])
        horizon = p["ecl"]["max_horizon_months"]
        rate = frame["interest_rate"].to_numpy(dtype=float)
        exposure = exposure_paths(
            frame["outstanding_balance"].to_numpy(dtype=float), rate, frame["installment_amount"].to_numpy(dtype=float), frame["remaining_term_months"].to_numpy(), horizon, p["ead"]["ccf"], self.ead_slippage
        )
        ecl_by, pd_paths, lgd_paths = {}, {}, {}
        for sc in self.scenarios:
            pd_paths[sc.name] = np.clip(scenario_pd_matrix(self.score_pd, frame, sc, p["scenarios"]), PD_FLOOR_MODEL, 1.0)
            lgd_paths[sc.name] = self._lgd_paths(frame, macro_paths(frame["macro_stress"].to_numpy(dtype=float), sc, p["scenarios"]))
            ecl_by[sc.name] = expected_credit_loss(pd_paths[sc.name], lgd_paths[sc.name], exposure, rate, stage, horizon)
        weights = {s.name: s.weight for s in self.scenarios}
        table = pd.DataFrame(
            {
                "loan_id": frame["loan_id"].to_numpy() if "loan_id" in frame else np.arange(len(frame)),
                "stage": stage,
                "grade": assign_grade(pd_now, self.grade_edges),
                "pd_12m": pd_now,
                "ead": exposure_at_default(frame["outstanding_balance"].to_numpy(dtype=float), p["ead"]["ccf"]),
                "lgd_baseline": lgd_paths["baseline"][:, 0],
                **{f"ecl_{name}": values for name, values in ecl_by.items()},
                "ecl_ponderee": probability_weighted(ecl_by, weights),
                "ecl_12m_tous_stage1": expected_credit_loss(pd_paths["baseline"], lgd_paths["baseline"], exposure, rate, np.ones_like(stage), horizon),
            }
        )
        return ECLResult(table, lgd_paths)
