from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk_graph.config import DATA_DIR
from credit_risk_graph.utils.io import read_parquet


def load_split(name: str) -> pd.DataFrame:
    frame = read_parquet(DATA_DIR / "splits" / f"{name}.parquet")
    return augment_runtime_graph_features(frame)


def augment_runtime_graph_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "graph_relational_pressure" not in enriched.columns:
        enriched["graph_relational_pressure"] = np.clip(
            0.34 * enriched["graph_neighbor_default_rate_12m"]
            + 0.24 * enriched["graph_contagion_score_6m"]
            + 0.18 * enriched["graph_community_risk"]
            + 0.12 * enriched["graph_neighbor_community_risk_6m"]
            + 0.07 * enriched["graph_guarantor_default_rate"]
            + 0.05 * enriched["graph_employer_peer_default_rate"],
            0.0,
            1.0,
        )
    if "graph_fragility_interaction" not in enriched.columns:
        fragility_scale = np.clip(
            0.30
            + 1.10 * enriched["historical_stage2_share_6m"]
            + 0.18 * enriched["historical_missed_payments_6m"]
            + 0.55 * enriched["macro_stress"]
            + 0.20 * np.clip(enriched["current_dpd"] / 60.0, 0.0, 1.5),
            0.0,
            2.5,
        )
        enriched["graph_fragility_interaction"] = np.clip(
            enriched["graph_relational_pressure"] * fragility_scale,
            0.0,
            2.5,
        )
    if "graph_employer_stress_interaction" not in enriched.columns:
        enriched["graph_employer_stress_interaction"] = np.clip(
            enriched["graph_employer_peer_default_rate"]
            * (0.60 * enriched["macro_stress"] + 0.40 * enriched["sector_stress"]),
            0.0,
            1.0,
        )
    if "graph_guarantor_stress_interaction" not in enriched.columns:
        enriched["graph_guarantor_stress_interaction"] = np.clip(
            enriched["graph_guarantor_default_rate"] * np.clip(enriched["current_ltv_proxy"], 0.0, 2.5),
            0.0,
            2.5,
        )
    if "graph_concentration_stress" not in enriched.columns:
        enriched["graph_concentration_stress"] = np.clip(
            enriched["graph_neighbor_default_rate_6m"] * enriched["counterparty_hhi_3m"],
            0.0,
            1.0,
        )
    return enriched
