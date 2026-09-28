from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk_graph.config import GenerationConfig
from credit_risk_graph.data.pipeline import _monthly_transaction_features, _performance_history_features, _split_snapshot_dataset
from credit_risk_graph.features.schema import TARGET_COLUMN
from credit_risk_graph.ifrs9 import data as ifrs9_data
from credit_risk_graph.ifrs9.data import model_columns, observation_cutoff, train_embargo_cutoff
from credit_risk_graph.ifrs9.params import load_params

# Colonnes issues du simulateur (vérité cachée ou étiquette) : jamais des variables explicatives.
TARGET_DERIVED = {
    TARGET_COLUMN,
    "current_pd_12m_true",
    "lifetime_pd_true",
    "current_lgd_true",
    "default_flag_current",
    "label_horizon_months",
    "label_is_complete_12m",
    "stage_ifrs9",
}


def test_historical_features_never_include_the_current_month():
    months = pd.date_range("2020-01-01", periods=8, freq="MS")
    panel = pd.DataFrame(
        {
            "loan_id": "L1",
            "as_of_date": months,
            "current_dpd": [0, 0, 0, 0, 0, 0, 0, 90],
            "payment_ratio": 1.0,
            "payment_state": "full",
            "restructured_flag": 0,
            "months_since_origination": range(8),
            "tenor_months": 24,
        }
    )
    history = _performance_history_features(panel)
    last = history.iloc[-1]
    assert last["current_dpd"] == 90
    assert last["historical_dpd_max_6m"] == 0
    assert last["historical_dpd_mean_3m"] == 0


def test_transaction_features_are_lagged_by_one_month():
    rows = []
    for month, count in enumerate([2, 4, 6, 100]):
        for k in range(count):
            rows.append(
                {
                    "borrower_id": "B1",
                    "tx_date": pd.Timestamp(2020, month + 1, 1) + pd.Timedelta(days=k % 20),
                    "counterparty_id": f"C{k}",
                    "amount": 100.0,
                    "direction": "in",
                    "transaction_id": f"T{month}-{k}",
                    "is_night": 0,
                    "is_high_risk_pattern": 0,
                }
            )
    monthly = _monthly_transaction_features(pd.DataFrame(rows)).sort_values("vintage_month").reset_index(drop=True)
    assert monthly.loc[3, "txn_count"] == 100
    assert monthly.loc[3, "txn_count_3m"] == pytest.approx(4.0)  # moyenne des 3 mois précédents (2, 4, 6), sans le mois courant


def test_temporal_splits_are_ordered_and_disjoint():
    config = GenerationConfig()
    dates = pd.date_range(config.start_date, config.end_date, freq="MS")
    snapshot = pd.DataFrame({"as_of_date": dates, "loan_id": [f"L{i}" for i in range(len(dates))]})
    splits = _split_snapshot_dataset(snapshot, config)
    assert sum(len(frame) for frame in splits.values()) == len(snapshot)
    assert splits["train"]["as_of_date"].max() <= pd.Timestamp(config.train_cutoff)
    assert splits["validation"]["as_of_date"].min() > pd.Timestamp(config.train_cutoff)
    assert splits["validation"]["as_of_date"].max() <= pd.Timestamp(config.validation_cutoff)
    assert splits["test"]["as_of_date"].min() > pd.Timestamp(config.validation_cutoff)
    assert splits["test"]["as_of_date"].max() <= pd.Timestamp(config.oot_cutoff)
    assert splits["out_of_time"]["as_of_date"].min() > pd.Timestamp(config.oot_cutoff)


def test_simulation_calendar_is_entirely_in_the_past():
    config = GenerationConfig()
    assert pd.Timestamp(config.end_date) < pd.Timestamp.today()


def test_embargo_and_label_cutoffs_are_ordered():
    params = load_params()
    config = GenerationConfig()
    assert train_embargo_cutoff(params) < pd.Timestamp(config.train_cutoff)
    assert observation_cutoff(params) < pd.Timestamp(config.end_date)


def test_load_performing_applies_embargo_censoring_and_exclusion_of_defaults(monkeypatch: pytest.MonkeyPatch):
    params = load_params()
    dates = pd.date_range("2016-02-01", "2024-12-01", freq="MS")
    frame = pd.DataFrame({"as_of_date": dates, "current_dpd": np.where(np.arange(len(dates)) % 5 == 0, 120, 0), "loan_id": range(len(dates))})
    monkeypatch.setattr(ifrs9_data, "load_full_split", lambda split: frame.copy())
    train = ifrs9_data.load_performing("train", params)
    validation = ifrs9_data.load_performing("validation", params)
    assert train["as_of_date"].max() <= train_embargo_cutoff(params)
    assert validation["as_of_date"].max() <= observation_cutoff(params)
    assert (train["current_dpd"] < params["population"]["performing_dpd_max"]).all()
    assert validation["as_of_date"].max() > train_embargo_cutoff(params)


def test_no_feature_profile_uses_target_derived_columns():
    params = load_params()
    for profile in params["pd_model"]["profiles"] + params["pd_model"]["diagnostic_profiles"]:
        assert not TARGET_DERIVED.intersection(model_columns(profile)), profile
