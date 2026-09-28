import pandas as pd

from credit_risk_graph.config import ARTIFACTS_DIR, DATA_DIR, GenerationConfig
from credit_risk_graph.data.pipeline import build_datasets
from credit_risk_graph.utils.io import read_json, read_parquet


def test_small_dataset_pipeline_builds_expected_artifacts() -> None:
    cfg = GenerationConfig(
        n_borrowers=250,
        n_loans=420,
        n_transactions=3000,
        end_date="2025-12-31",
        train_cutoff="2023-06-30",
        validation_cutoff="2023-12-31",
        oot_cutoff="2024-06-30",
    )
    build_datasets(cfg)
    summary = read_json(ARTIFACTS_DIR / "metadata" / "dataset_summary.json")
    assert "credit-risk-tests-" in str(DATA_DIR)
    assert "credit-risk-tests-" in str(ARTIFACTS_DIR)
    assert summary["raw_counts"]["transactions"] == 3000
    assert "split_default_counts" in summary
    train = read_parquet(DATA_DIR / "splits" / "train.parquet")
    validation = read_parquet(DATA_DIR / "splits" / "validation.parquet")
    test = read_parquet(DATA_DIR / "splits" / "test.parquet")
    oot = read_parquet(DATA_DIR / "splits" / "out_of_time.parquet")
    assert not train.empty
    assert train["as_of_date"].max() <= pd.Timestamp(cfg.train_cutoff)
    assert validation["as_of_date"].min() > pd.Timestamp(cfg.train_cutoff)
    assert test["as_of_date"].min() > pd.Timestamp(cfg.validation_cutoff)
    assert oot["as_of_date"].min() > pd.Timestamp(cfg.oot_cutoff)
    assert train["default_90dpd_12m"].isin([0, 1]).all()
    expected_graph_columns = {
        "graph_degree_3m",
        "graph_weighted_degree_6m",
        "graph_neighbor_default_rate_12m",
        "graph_temporal_weighted_degree",
        "graph_employer_peer_default_rate",
        "graph_guarantor_default_rate",
        "graph_relational_pressure",
        "graph_fragility_interaction",
    }
    assert expected_graph_columns.issubset(train.columns)
