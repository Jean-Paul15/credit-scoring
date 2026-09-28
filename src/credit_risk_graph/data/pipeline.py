from __future__ import annotations

import gc
from dataclasses import asdict

import numpy as np
import pandas as pd

from credit_risk_graph.config import ARTIFACTS_DIR, DATA_DIR, GenerationConfig
from credit_risk_graph.data.cleaning import CleanBundle, clean_synthetic_bundle
from credit_risk_graph.data.generator import SyntheticBundle, generate_synthetic_bundle
from credit_risk_graph.features.graph import build_training_graph_features
from credit_risk_graph.utils.io import read_parquet, write_json, write_parquet
from credit_risk_graph.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _load_raw_bundle() -> SyntheticBundle:
    raw_dir = DATA_DIR / "raw"
    return SyntheticBundle(
        borrowers=read_parquet(raw_dir / "borrowers.parquet"),
        employers=read_parquet(raw_dir / "employers.parquet"),
        merchants=read_parquet(raw_dir / "merchants.parquet"),
        macro=read_parquet(raw_dir / "macro.parquet"),
        loans=read_parquet(raw_dir / "loans.parquet"),
        transactions=read_parquet(raw_dir / "transactions.parquet"),
        performance=read_parquet(raw_dir / "performance.parquet"),
        data_dictionary=read_parquet(raw_dir / "data_dictionary.parquet"),
    )


def _sector_stress_column(sector: pd.Series) -> pd.Series:
    values = np.select(
        [sector.eq("Commerce"), sector.eq("Transport"), sector.eq("Agriculture")],
        ["sector_stress_commerce", "sector_stress_transport", "sector_stress_agriculture"],
        default="sector_stress_commerce",
    )
    return pd.Series(values, index=sector.index)


def _monthly_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    tx = transactions[
        [
            "borrower_id",
            "tx_date",
            "counterparty_id",
            "amount",
            "direction",
            "transaction_id",
            "is_night",
            "is_high_risk_pattern",
        ]
    ].copy()
    tx["vintage_month"] = tx["tx_date"].dt.to_period("M").dt.to_timestamp()
    tx["inbound_amount"] = np.where(tx["direction"].eq("in"), tx["amount"], 0.0)
    tx["outbound_amount"] = np.where(tx["direction"].eq("out"), tx["amount"], 0.0)
    pair_amount = (
        tx.groupby(["borrower_id", "vintage_month", "counterparty_id"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "pair_amount"})
    )
    concentration = (
        pair_amount.groupby(["borrower_id", "vintage_month"], as_index=False)["pair_amount"]
        .apply(lambda s: float(((s / s.sum()) ** 2).sum()))
        .rename(columns={"pair_amount": "counterparty_hhi"})
    )
    monthly = (
        tx.groupby(["borrower_id", "vintage_month"], as_index=False)
        .agg(
            txn_count=("transaction_id", "count"),
            txn_amount=("amount", "sum"),
            avg_tx_amount=("amount", "mean"),
            std_tx_amount=("amount", "std"),
            unique_counterparties=("counterparty_id", "nunique"),
            night_tx_share=("is_night", "mean"),
            risky_tx_share=("is_high_risk_pattern", "mean"),
            inbound_amount=("inbound_amount", "sum"),
            outbound_amount=("outbound_amount", "sum"),
        )
        .merge(concentration, on=["borrower_id", "vintage_month"], how="left")
        .sort_values(["borrower_id", "vintage_month"])
    )
    monthly["std_tx_amount"] = monthly["std_tx_amount"].fillna(0.0)
    rolling_cols = ["txn_count", "txn_amount", "avg_tx_amount", "std_tx_amount", "unique_counterparties", "night_tx_share", "risky_tx_share", "inbound_amount", "outbound_amount", "counterparty_hhi"]
    for col in rolling_cols:
        monthly[f"{col}_3m"] = (
            monthly.groupby("borrower_id")[col]
            .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
        )
    monthly["inbound_outbound_ratio_3m"] = monthly["inbound_amount_3m"] / (monthly["outbound_amount_3m"] + 1.0)
    return monthly


def _performance_history_features(performance: pd.DataFrame) -> pd.DataFrame:
    history = performance.copy()
    history = history.sort_values(["loan_id", "as_of_date"]).reset_index(drop=True)
    grouped = history.groupby("loan_id", sort=False)

    history["historical_dpd_mean_3m"] = grouped["current_dpd"].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    history["historical_dpd_max_6m"] = grouped["current_dpd"].transform(lambda s: s.shift(1).rolling(window=6, min_periods=1).max())
    history["historical_dpd_std_6m"] = grouped["current_dpd"].transform(lambda s: s.shift(1).rolling(window=6, min_periods=2).std())
    history["historical_payment_ratio_mean_3m"] = grouped["payment_ratio"].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    history["historical_payment_ratio_delta_1m"] = history["payment_ratio"] - grouped["payment_ratio"].shift(1)
    history["dpd_delta_1m"] = history["current_dpd"] - grouped["current_dpd"].shift(1)

    missed_flag = history["payment_state"].eq("miss").astype(float)
    partial_flag = history["payment_state"].eq("partial").astype(float)
    stage2_flag = ((history["current_dpd"] >= 30) | history["restructured_flag"].eq(1)).astype(float)
    history["historical_missed_payments_6m"] = missed_flag.groupby(history["loan_id"], sort=False).transform(
        lambda s: s.shift(1).rolling(window=6, min_periods=1).sum()
    )
    history["historical_partial_payments_6m"] = partial_flag.groupby(history["loan_id"], sort=False).transform(
        lambda s: s.shift(1).rolling(window=6, min_periods=1).sum()
    )
    history["historical_stage2_share_6m"] = stage2_flag.groupby(history["loan_id"], sort=False).transform(
        lambda s: s.shift(1).rolling(window=6, min_periods=1).mean()
    )
    history["historical_restructured_ever"] = grouped["restructured_flag"].transform(lambda s: s.shift(1).fillna(0).cummax())

    history["loan_age_months"] = history["months_since_origination"].astype(float)
    history["remaining_term_ratio"] = np.clip(
        (history["tenor_months"] - history["months_since_origination"]) / history["tenor_months"].clip(lower=1),
        0.0,
        1.0,
    )

    fill_zero_cols = [
        "historical_dpd_mean_3m",
        "historical_dpd_max_6m",
        "historical_dpd_std_6m",
        "historical_payment_ratio_mean_3m",
        "historical_payment_ratio_delta_1m",
        "dpd_delta_1m",
        "historical_missed_payments_6m",
        "historical_partial_payments_6m",
        "historical_stage2_share_6m",
        "historical_restructured_ever",
    ]
    history[fill_zero_cols] = history[fill_zero_cols].fillna(0.0)
    return history.drop(columns=["payment_state", "tenor_months"])


def _build_snapshot_chunks(clean: CleanBundle, config: GenerationConfig) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    performance = clean.performance[
        [
            "loan_id",
            "borrower_id",
            "as_of_date",
            "months_since_origination",
            "remaining_term_months",
            "current_dpd",
            "payment_ratio",
            "payment_state",
            "restructured_flag",
            "outstanding_balance",
            "current_pd_12m_true",
            "current_lgd_true",
            "stage_ifrs9",
            "default_90dpd_12m",
            "label_horizon_months",
            "label_is_complete_12m",
        ]
    ].copy()
    loans = clean.loans[
        [
            "loan_id",
            "borrower_id",
            "product_type",
            "collateral_type",
            "principal_amount",
            "interest_rate",
            "installment_amount",
            "collateral_coverage",
            "origination_pd",
            "base_lgd",
            "tenor_months",
            "guarantor_id",
        ]
    ].copy()
    borrowers = clean.borrowers[
        [
            "borrower_id",
            "region",
            "sector",
            "employment_type",
            "kyc_level",
            "employer_id",
            "monthly_income",
            "savings_balance",
            "debt_service_ratio",
            "digital_engagement",
            "external_score",
            "prior_loans_count",
            "months_on_job",
            "household_size",
        ]
    ].copy()
    macro = clean.macro[
        [
            "month",
            "region",
            "macro_stress",
            "sector_stress_commerce",
            "sector_stress_transport",
            "sector_stress_agriculture",
        ]
    ].copy()
    LOGGER.info("Building monthly transaction features.")
    tx_monthly = _monthly_transaction_features(clean.transactions)
    LOGGER.info("Monthly transaction features ready with %s borrower-month rows.", len(tx_monthly))
    LOGGER.info("Building performance history features.")
    performance = _performance_history_features(
        performance.merge(loans[["loan_id", "tenor_months"]], on="loan_id", how="left")
    )
    LOGGER.info("Performance history features ready with %s rows.", len(performance))

    performance["month_key"] = performance["as_of_date"].dt.to_period("M").dt.to_timestamp()
    borrower_default_rates = (
        performance.loc[performance["as_of_date"] <= pd.Timestamp(config.train_cutoff)]
        .groupby("borrower_id", as_index=False)["default_90dpd_12m"]
        .mean()
        .rename(columns={"default_90dpd_12m": "default_rate_train"})
    )
    LOGGER.info("Building graph feature bundle.")
    graph_bundle = build_training_graph_features(clean.transactions, pd.Timestamp(config.train_cutoff), borrower_default_rates)
    LOGGER.info(
        "Graph feature bundle ready with %s static rows and %s borrower-month rows.",
        len(graph_bundle.borrower_features),
        len(graph_bundle.borrower_month_features),
    )
    graph_layout = graph_bundle.sampled_layout.copy()
    graph_static = graph_bundle.borrower_features.copy()
    graph_month = graph_bundle.borrower_month_features.copy()
    employer_peer_default = (
        borrowers[["borrower_id", "employer_id"]]
        .merge(borrower_default_rates, on="borrower_id", how="left")
        .groupby("employer_id", as_index=False)["default_rate_train"]
        .mean()
        .rename(columns={"default_rate_train": "graph_employer_peer_default_rate"})
    )
    guarantor_default = borrower_default_rates.rename(
        columns={
            "borrower_id": "guarantor_id",
            "default_rate_train": "graph_guarantor_default_rate",
        }
    )
    fill_zero_cols = [
        "txn_count_3m",
        "txn_amount_3m",
        "avg_tx_amount_3m",
        "std_tx_amount_3m",
        "unique_counterparties_3m",
        "night_tx_share_3m",
        "risky_tx_share_3m",
        "inbound_amount_3m",
        "outbound_amount_3m",
        "counterparty_hhi_3m",
        "inbound_outbound_ratio_3m",
        "graph_degree",
        "graph_weighted_degree",
        "graph_pagerank",
        "graph_clustering",
        "graph_component_size",
        "graph_neighbor_default_rate",
        "graph_contagion_score",
        "graph_community_risk",
        "graph_degree_3m",
        "graph_degree_6m",
        "graph_weighted_degree_3m",
        "graph_weighted_degree_6m",
        "graph_neighbor_default_rate_6m",
        "graph_neighbor_default_rate_12m",
        "graph_contagion_score_6m",
        "graph_counterparty_hhi_6m",
        "graph_temporal_weighted_degree",
        "graph_neighbor_community_risk_6m",
        "graph_employer_peer_default_rate",
        "graph_guarantor_default_rate",
        "graph_relational_pressure",
        "graph_fragility_interaction",
        "graph_employer_stress_interaction",
        "graph_guarantor_stress_interaction",
        "graph_concentration_stress",
    ]
    selected_columns = [
        "loan_id",
        "borrower_id",
        "as_of_date",
        "month_key",
        "region",
        "sector",
        "employment_type",
        "product_type",
        "collateral_type",
        "kyc_level",
        "principal_amount",
        "interest_rate",
        "installment_amount",
        "collateral_coverage",
        "origination_pd",
        "base_lgd",
        "monthly_income",
        "savings_balance",
        "debt_service_ratio",
        "digital_engagement",
        "external_score",
        "prior_loans_count",
        "months_on_job",
        "household_size",
        "loan_age_months",
        "current_dpd",
        "historical_dpd_mean_3m",
        "historical_dpd_max_6m",
        "historical_dpd_std_6m",
        "dpd_delta_1m",
        "payment_ratio",
        "historical_payment_ratio_mean_3m",
        "historical_payment_ratio_delta_1m",
        "historical_missed_payments_6m",
        "historical_partial_payments_6m",
        "historical_stage2_share_6m",
        "restructured_flag",
        "historical_restructured_ever",
        "outstanding_balance",
        "remaining_term_months",
        "remaining_term_ratio",
        "macro_stress",
        "sector_stress",
        "txn_count_3m",
        "txn_amount_3m",
        "avg_tx_amount_3m",
        "std_tx_amount_3m",
        "unique_counterparties_3m",
        "night_tx_share_3m",
        "risky_tx_share_3m",
        "inbound_amount_3m",
        "outbound_amount_3m",
        "counterparty_hhi_3m",
        "inbound_outbound_ratio_3m",
        "graph_degree",
        "graph_weighted_degree",
        "graph_pagerank",
        "graph_clustering",
        "graph_component_size",
        "graph_neighbor_default_rate",
        "graph_contagion_score",
        "graph_community_risk",
        "graph_degree_3m",
        "graph_degree_6m",
        "graph_weighted_degree_3m",
        "graph_weighted_degree_6m",
        "graph_neighbor_default_rate_6m",
        "graph_neighbor_default_rate_12m",
        "graph_contagion_score_6m",
        "graph_counterparty_hhi_6m",
        "graph_temporal_weighted_degree",
        "graph_neighbor_community_risk_6m",
        "graph_employer_peer_default_rate",
        "graph_guarantor_default_rate",
        "graph_relational_pressure",
        "graph_fragility_interaction",
        "graph_employer_stress_interaction",
        "graph_guarantor_stress_interaction",
        "graph_concentration_stress",
        "current_ltv_proxy",
        "installment_to_income",
        "balance_to_income",
        "savings_buffer_ratio",
        "expected_lifetime_pd_rule",
        "current_pd_12m_true",
        "current_lgd_true",
        "ecl",
        "ecl_ratio",
        "stage_ifrs9",
        "ifrs9_stage_numeric",
        "default_90dpd_12m",
        "label_horizon_months",
        "label_is_complete_12m",
        "vintage_index",
    ]
    loans_static = loans.drop(columns=["tenor_months"]).copy()
    month_keys = performance["month_key"].sort_values().drop_duplicates().tolist()
    chunk_frames: list[pd.DataFrame] = []
    chunk_size_months = 6

    for idx in range(0, len(month_keys), chunk_size_months):
        month_window = month_keys[idx : idx + chunk_size_months]
        perf_chunk = performance.loc[performance["month_key"].isin(month_window)].copy()
        tx_chunk = tx_monthly.loc[tx_monthly["vintage_month"].isin(month_window)].copy()
        graph_month_chunk = graph_month.loc[graph_month["month_key"].isin(month_window)].copy()
        macro_chunk = macro.loc[macro["month"].isin(month_window)].copy()

        chunk = (
            perf_chunk.merge(loans_static, on=["loan_id", "borrower_id"], how="left", suffixes=("", "_loan"))
            .merge(borrowers, on="borrower_id", how="left", suffixes=("", "_borrower"))
            .merge(
                macro_chunk,
                left_on=["month_key", "region"],
                right_on=["month", "region"],
                how="left",
            )
            .merge(
                tx_chunk,
                left_on=["borrower_id", "month_key"],
                right_on=["borrower_id", "vintage_month"],
                how="left",
            )
            .merge(graph_static, on="borrower_id", how="left")
            .merge(graph_month_chunk, on=["borrower_id", "month_key"], how="left")
            .merge(employer_peer_default, on="employer_id", how="left")
            .merge(guarantor_default, on="guarantor_id", how="left")
        )
        LOGGER.info(
            "Snapshot chunk %s -> %s merged with %s rows.",
            month_window[0].date().isoformat(),
            month_window[-1].date().isoformat(),
            len(chunk),
        )

        stress_col = _sector_stress_column(chunk["sector"])
        stress_matrix = chunk[["sector_stress_commerce", "sector_stress_transport", "sector_stress_agriculture"]].to_numpy()
        col_idx = stress_col.map(
            {
                "sector_stress_commerce": 0,
                "sector_stress_transport": 1,
                "sector_stress_agriculture": 2,
            }
        ).to_numpy()
        chunk["sector_stress"] = stress_matrix[np.arange(len(chunk)), col_idx]
        chunk["remaining_term_years"] = np.maximum(chunk["remaining_term_months"], 1) / 12.0
        chunk["current_ltv_proxy"] = chunk["outstanding_balance"] / (
            chunk["principal_amount"] * (chunk["collateral_coverage"] + 0.15) + 1.0
        )
        chunk["installment_to_income"] = chunk["installment_amount"] / np.clip(chunk["monthly_income"], 1.0, None)
        chunk["balance_to_income"] = chunk["outstanding_balance"] / np.clip(chunk["monthly_income"], 1.0, None)
        chunk["savings_buffer_ratio"] = chunk["savings_balance"] / np.clip(chunk["outstanding_balance"], 1.0, None)
        chunk["graph_relational_pressure"] = np.clip(
            0.28 * chunk["graph_neighbor_default_rate_12m"]
            + 0.25 * chunk["graph_contagion_score_6m"]
            + 0.18 * chunk["graph_community_risk"]
            + 0.14 * chunk["graph_neighbor_community_risk_6m"]
            + 0.10 * chunk["graph_guarantor_default_rate"]
            + 0.05 * chunk["graph_employer_peer_default_rate"],
            0.0,
            1.0,
        )
        fragility_scale = np.clip(
            0.35
            + chunk["historical_stage2_share_6m"]
            + (chunk["historical_missed_payments_6m"] / 6.0)
            + 0.40 * chunk["macro_stress"],
            0.0,
            2.0,
        )
        chunk["graph_fragility_interaction"] = np.clip(chunk["graph_relational_pressure"] * fragility_scale, 0.0, 2.0)
        chunk["graph_employer_stress_interaction"] = np.clip(
            chunk["graph_employer_peer_default_rate"] * (0.55 * chunk["macro_stress"] + 0.45 * chunk["sector_stress"]),
            0.0,
            1.0,
        )
        chunk["graph_guarantor_stress_interaction"] = np.clip(
            chunk["graph_guarantor_default_rate"] * np.clip(chunk["current_ltv_proxy"], 0.0, 2.0),
            0.0,
            2.0,
        )
        chunk["graph_concentration_stress"] = np.clip(
            chunk["graph_neighbor_default_rate_6m"] * chunk["counterparty_hhi_3m"],
            0.0,
            1.0,
        )
        chunk["expected_lifetime_pd_rule"] = np.clip(
            1 - (1 - chunk["current_pd_12m_true"]) ** chunk["remaining_term_years"],
            chunk["current_pd_12m_true"],
            0.995,
        )
        chunk["ifrs9_stage_numeric"] = chunk["stage_ifrs9"].map({"Stage 1": 1, "Stage 2": 2, "Stage 3": 3}).astype(int)
        chunk["ecl"] = np.where(
            chunk["stage_ifrs9"].eq("Stage 1"),
            chunk["outstanding_balance"] * chunk["current_pd_12m_true"] * chunk["current_lgd_true"],
            chunk["outstanding_balance"] * chunk["expected_lifetime_pd_rule"] * chunk["current_lgd_true"],
        )
        chunk["ecl_ratio"] = chunk["ecl"] / np.clip(chunk["outstanding_balance"], 1.0, None)
        chunk["vintage_index"] = chunk["month_key"].rank(method="dense").astype(int)
        chunk = chunk.loc[chunk["label_is_complete_12m"] == 1].copy()

        for col in fill_zero_cols:
            chunk[col] = chunk[col].fillna(0.0)

        chunk_frames.append(
            chunk[selected_columns].sort_values(["as_of_date", "loan_id"]).reset_index(drop=True)
        )
        del perf_chunk, tx_chunk, graph_month_chunk, macro_chunk, chunk
        gc.collect()

    del performance, loans, loans_static, borrowers, macro, tx_monthly, graph_static, graph_month
    del borrower_default_rates, employer_peer_default, guarantor_default
    gc.collect()

    return chunk_frames, graph_layout


def _build_snapshot_dataset(clean: CleanBundle, config: GenerationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunk_frames, graph_layout = _build_snapshot_chunks(clean, config)
    snapshot = pd.concat(chunk_frames, ignore_index=True).sort_values(["as_of_date", "loan_id"]).reset_index(drop=True)
    return snapshot, graph_layout


def _split_snapshot_dataset(snapshot: pd.DataFrame, config: GenerationConfig) -> dict[str, pd.DataFrame]:
    train_cutoff = pd.Timestamp(config.train_cutoff)
    validation_cutoff = pd.Timestamp(config.validation_cutoff)
    oot_cutoff = pd.Timestamp(config.oot_cutoff)
    return {
        "train": snapshot.loc[snapshot["as_of_date"] <= train_cutoff].reset_index(drop=True),
        "validation": snapshot.loc[(snapshot["as_of_date"] > train_cutoff) & (snapshot["as_of_date"] <= validation_cutoff)].reset_index(drop=True),
        "test": snapshot.loc[(snapshot["as_of_date"] > validation_cutoff) & (snapshot["as_of_date"] <= oot_cutoff)].reset_index(drop=True),
        "out_of_time": snapshot.loc[snapshot["as_of_date"] > oot_cutoff].reset_index(drop=True),
    }


def build_datasets(config: GenerationConfig) -> None:
    LOGGER.info("Generating synthetic raw data.")
    bundle = generate_synthetic_bundle(config)
    raw_counts = {
        "borrowers": int(len(bundle.borrowers)),
        "loans": int(len(bundle.loans)),
        "transactions": int(len(bundle.transactions)),
        "performance_rows": int(len(bundle.performance)),
    }
    raw_dir = DATA_DIR / "raw"
    write_parquet(bundle.borrowers, raw_dir / "borrowers.parquet")
    write_parquet(bundle.employers, raw_dir / "employers.parquet")
    write_parquet(bundle.merchants, raw_dir / "merchants.parquet")
    write_parquet(bundle.macro, raw_dir / "macro.parquet")
    write_parquet(bundle.loans, raw_dir / "loans.parquet")
    write_parquet(bundle.transactions, raw_dir / "transactions.parquet")
    write_parquet(bundle.performance, raw_dir / "performance.parquet")
    write_parquet(bundle.data_dictionary, raw_dir / "data_dictionary.parquet")
    LOGGER.info("Raw data persisted to disk.")

    del bundle
    gc.collect()

    LOGGER.info("Cleaning synthetic data.")
    raw_bundle = _load_raw_bundle()
    clean = clean_synthetic_bundle(raw_bundle)
    LOGGER.info("Cleaning completed.")
    del raw_bundle
    gc.collect()
    clean_dir = DATA_DIR / "clean"
    write_parquet(clean.borrowers, clean_dir / "borrowers_clean.parquet")
    write_parquet(clean.employers, clean_dir / "employers_clean.parquet")
    write_parquet(clean.merchants, clean_dir / "merchants_clean.parquet")
    write_parquet(clean.macro, clean_dir / "macro_clean.parquet")
    write_parquet(clean.loans, clean_dir / "loans_clean.parquet")
    write_parquet(clean.transactions, clean_dir / "transactions_clean.parquet")
    write_parquet(clean.performance, clean_dir / "performance_clean.parquet")
    write_parquet(clean.quality_log, clean_dir / "quality_log.parquet")
    LOGGER.info("Clean tables persisted to disk.")

    LOGGER.info("Building enriched modeling dataset and graph layout.")
    snapshot_chunks, graph_layout = _build_snapshot_chunks(clean, config)
    snapshot_rows = sum(len(chunk) for chunk in snapshot_chunks)
    LOGGER.info("Enriched snapshot built with %s rows across %s chunks.", snapshot_rows, len(snapshot_chunks))
    enriched_dir = DATA_DIR / "enriched"
    write_parquet(graph_layout, enriched_dir / "graph_layout.parquet")
    if snapshot_rows <= 500_000:
        snapshot = pd.concat(snapshot_chunks, ignore_index=True).sort_values(["as_of_date", "loan_id"]).reset_index(drop=True)
        write_parquet(snapshot, enriched_dir / "loan_snapshots.parquet")
    else:
        full_snapshot_path = enriched_dir / "loan_snapshots.parquet"
        if full_snapshot_path.exists():
            full_snapshot_path.unlink()
        snapshot_sample = pd.concat(
            [chunk.head(min(len(chunk), 5_000)) for chunk in snapshot_chunks],
            ignore_index=True,
        ).sort_values(["as_of_date", "loan_id"]).reset_index(drop=True)
        write_parquet(snapshot_sample, enriched_dir / "loan_snapshots_sample.parquet")
    LOGGER.info("Enriched artifacts persisted to disk.")

    del clean
    gc.collect()

    split_frames_map: dict[str, list[pd.DataFrame]] = {
        "train": [],
        "validation": [],
        "test": [],
        "out_of_time": [],
    }
    stage_counts: dict[str, int] = {}
    default_sum = 0
    for chunk in snapshot_chunks:
        default_sum += int(chunk["default_90dpd_12m"].sum())
        stage_chunk = chunk["stage_ifrs9"].value_counts()
        for stage_name, count in stage_chunk.items():
            stage_counts[stage_name] = stage_counts.get(stage_name, 0) + int(count)
        chunk_splits = _split_snapshot_dataset(chunk, config)
        for name, frame in chunk_splits.items():
            if not frame.empty:
                split_frames_map[name].append(frame)

    splits = {
        name: (
            pd.concat(frames, ignore_index=True).sort_values(["as_of_date", "loan_id"]).reset_index(drop=True)
            if frames
            else pd.DataFrame(columns=snapshot_chunks[0].columns if snapshot_chunks else [])
        )
        for name, frames in split_frames_map.items()
    }
    split_dir = DATA_DIR / "splits"
    for name, frame in splits.items():
        write_parquet(frame, split_dir / f"{name}.parquet")
    LOGGER.info("Temporal splits persisted to disk.")

    dataset_summary = {
        "generation_config": asdict(config),
        "raw_counts": raw_counts,
        "snapshot_rows": int(snapshot_rows),
        "split_rows": {name: int(len(frame)) for name, frame in splits.items()},
        "split_default_counts": {name: int(frame["default_90dpd_12m"].sum()) for name, frame in splits.items()},
        "split_date_ranges": {
            name: {
                "min_as_of_date": frame["as_of_date"].min().date().isoformat(),
                "max_as_of_date": frame["as_of_date"].max().date().isoformat(),
            }
            for name, frame in splits.items()
            if not frame.empty
        },
        "default_rate_snapshot": float(default_sum / snapshot_rows) if snapshot_rows else 0.0,
        "stage_distribution": {
            stage_name: round(count / snapshot_rows, 4)
            for stage_name, count in stage_counts.items()
        } if snapshot_rows else {},
    }
    write_json(dataset_summary, ARTIFACTS_DIR / "metadata" / "dataset_summary.json")
