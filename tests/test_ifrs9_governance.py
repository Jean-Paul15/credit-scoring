from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk_graph.features.schema import THIRD_PARTY_FEATURES
from credit_risk_graph.ifrs9.data import model_columns
from credit_risk_graph.ifrs9.default_definition import DefaultDefinition, flag_default, flag_with_cure
from credit_risk_graph.ifrs9.params import load_params
from credit_risk_graph.ifrs9.pd_model import assert_no_third_party_features
from credit_risk_graph.ifrs9.scenarios import Scenario, build_scenarios, macro_paths
from credit_risk_graph.ifrs9.staging import assign_stage
from credit_risk_graph.monitoring.drift import psi_band, psi_categorical, psi_numeric

STAGING = {"stage2_dpd": 30, "stage3_dpd": 90, "sicr_relative_multiplier": 1.7, "sicr_absolute_delta": 0.01}
MONITORING = {"psi_stable_max": 0.10, "psi_alert_min": 0.25, "psi_bins": 10}


def _frame(dpd, restructured, origination):
    return pd.DataFrame({"current_dpd": dpd, "restructured_flag": restructured, "origination_pd": origination})


def test_stage_rules():
    frame = _frame([0, 45, 95, 0, 0], [0, 0, 0, 1, 0], [0.05, 0.05, 0.05, 0.05, 0.05])
    pd_now = np.array([0.05, 0.05, 0.05, 0.05, 0.12])
    assert assign_stage(frame, pd_now, STAGING).tolist() == [1, 2, 3, 2, 2]


def test_sicr_needs_relative_and_absolute_increase():
    frame = _frame([0, 0], [0, 0], [0.002, 0.002])
    stage = assign_stage(frame, np.array([0.005, 0.02]), STAGING)
    assert stage.tolist() == [1, 2]


def test_default_flag_and_utp():
    definition = DefaultDefinition(90, 60, 3)
    panel = pd.DataFrame({"current_dpd": [30, 65, 90], "restructured_flag": [0, 1, 0]})
    assert flag_default(panel, definition).tolist() == [0, 0, 1]
    assert flag_default(panel, definition, include_utp=True).tolist() == [0, 1, 1]


def test_cure_period_keeps_default_status():
    definition = DefaultDefinition(90, 60, cure_months=2)
    panel = pd.DataFrame(
        {
            "loan_id": ["a"] * 6,
            "as_of_date": pd.date_range("2020-01-01", periods=6, freq="MS"),
            "current_dpd": [0, 90, 0, 0, 0, 0],
            "restructured_flag": 0,
        }
    )
    assert flag_with_cure(panel, definition).tolist() == [0, 1, 1, 1, 0, 0]


def test_psi_is_zero_for_identical_and_high_for_shifted_populations():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 1, 20_000))
    assert psi_numeric(base, base) < 1e-9
    assert psi_numeric(base, pd.Series(rng.normal(0, 1, 20_000))) < 0.02
    assert psi_numeric(base, pd.Series(rng.normal(1.5, 1, 20_000))) > 0.25
    assert psi_categorical(pd.Series(list("aabb")), pd.Series(list("aaab"))) > 0


def test_psi_bands():
    assert psi_band(0.05, MONITORING) == "stable"
    assert psi_band(0.15, MONITORING) == "à surveiller"
    assert psi_band(0.30, MONITORING) == "alerte"


def test_scenarios_weights_and_reversion():
    history = pd.Series(np.linspace(0.1, 0.4, 100))
    cfg = {"weights": {"baseline": 0.5, "adverse": 0.3, "favourable": 0.2}, "adverse_quantile": 0.9, "favourable_quantile": 0.1, "reversion_years": 3, "horizon_years": 5}
    scenarios = build_scenarios(history, cfg)
    adverse = next(s for s in scenarios if s.name == "adverse")
    assert adverse.shock > 0 > next(s for s in scenarios if s.name == "favourable").shock
    path = macro_paths(np.array([0.2]), adverse, cfg)[0]
    assert path[0] > path[1] > path[2] > 0.2 and np.isclose(path[3], 0.2) and np.isclose(path[4], 0.2)
    with pytest.raises(ValueError):
        build_scenarios(history, {**cfg, "weights": {"baseline": 0.5, "adverse": 0.5, "favourable": 0.5}})
    assert Scenario("x", 1.0, 0.0).shock == 0.0


def test_eligible_profiles_use_no_third_party_variables():
    params = load_params()
    for profile in params["pd_model"]["profiles"]:
        assert_no_third_party_features(model_columns(profile))
    with pytest.raises(ValueError):
        assert_no_third_party_features(model_columns("graph_enriched"))
    assert "graph_neighbor_default_rate" in THIRD_PARTY_FEATURES and "risky_tx_share_3m" not in THIRD_PARTY_FEATURES


def test_random_search_samples_reproducible_and_typed_configurations():
    from credit_risk_graph.ifrs9.stages.tune_ebm import sample_configurations

    space = {"learning_rate": {"loguniform": [0.005, 0.05]}, "max_bins": [16, 32, 64], "interactions": [0, 3]}
    first = sample_configurations(space, 6, seed=1)
    assert first == sample_configurations(space, 6, seed=1)
    assert len(first) == 6 and all(0.005 <= c["learning_rate"] <= 0.05 for c in first)
    assert all(isinstance(c["max_bins"], int) and c["interactions"] in (0, 3) for c in first)
