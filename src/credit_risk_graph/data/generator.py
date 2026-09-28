from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from credit_risk_graph.config import GenerationConfig

REGIONS = [
    "Lome",
    "Kara",
    "Sokode",
    "Atakpame",
    "Dapaong",
    "Cotonou",
    "Abidjan",
    "Ouagadougou",
]

SECTORS = [
    "Commerce",
    "Transport",
    "FonctionPublique",
    "Agriculture",
    "Services",
    "BTP",
    "Sante",
    "Education",
    "Artisanat",
]

PRODUCTS = ["PretPersonnel", "MicroPME", "AvanceSalaire", "Equipement"]
EMPLOYMENT_TYPES = ["Salarie", "Informel", "Independant", "MicroEntrepreneur"]
COLLATERAL_TYPES = ["Aucun", "Vehicule", "Stock", "Depot", "Foncier"]
CHANNELS = ["Agence", "MobileApp", "USSD", "Agent", "Web"]
TX_TYPES = ["P2P", "AchatMarchand", "Facture", "Depot", "Retrait"]
COUNTERPARTY_TYPES = ["borrower", "merchant", "utility"]


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(slots=True)
class SyntheticBundle:
    borrowers: pd.DataFrame
    employers: pd.DataFrame
    merchants: pd.DataFrame
    macro: pd.DataFrame
    loans: pd.DataFrame
    transactions: pd.DataFrame
    performance: pd.DataFrame
    data_dictionary: pd.DataFrame


def _build_macro_frame(config: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    months = pd.date_range(config.start_date, config.end_date, freq="MS")
    rows: list[dict[str, Any]] = []
    region_base = {region: idx * 0.05 for idx, region in enumerate(REGIONS)}
    for month_idx, month in enumerate(months):
        seasonal = np.sin(2 * np.pi * month.month / 12)
        for region in REGIONS:
            stress = 0.15 + 0.06 * seasonal + region_base[region] + rng.normal(0, 0.025)
            rows.append(
                {
                    "month": month,
                    "region": region,
                    "gdp_growth_proxy": 3.2 - 2.5 * stress + rng.normal(0, 0.15),
                    "inflation_proxy": 2.8 + 5.0 * stress + rng.normal(0, 0.25),
                    "unemployment_proxy": 6.0 + 7.0 * stress + rng.normal(0, 0.35),
                    "fuel_stress_proxy": 100 + 12 * stress + rng.normal(0, 1.5),
                    "macro_stress": float(np.clip(stress, 0.03, 0.85)),
                    "sector_stress_commerce": float(np.clip(stress + rng.normal(0, 0.02), 0.02, 0.9)),
                    "sector_stress_transport": float(np.clip(stress + 0.03 + rng.normal(0, 0.02), 0.02, 0.9)),
                    "sector_stress_agriculture": float(np.clip(stress + 0.04 * np.cos(month_idx / 2), 0.02, 0.9)),
                }
            )
    return pd.DataFrame(rows)


def _generate_employers(config: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    n_employers = max(900, config.n_borrowers // 18)
    employers = pd.DataFrame(
        {
            "employer_id": [f"EMP{idx:05d}" for idx in range(n_employers)],
            "region": rng.choice(REGIONS, size=n_employers, p=np.array([0.22, 0.10, 0.12, 0.10, 0.08, 0.14, 0.16, 0.08])),
            "sector": rng.choice(SECTORS, size=n_employers),
            "employer_size": rng.choice(["Micro", "PME", "GrandCompte"], size=n_employers, p=[0.48, 0.38, 0.14]),
        }
    )
    employers["employer_risk_score"] = np.clip(
        rng.normal(0.45, 0.16, size=n_employers)
        + (employers["employer_size"].eq("Micro") * 0.08).astype(float)
        + (employers["sector"].isin(["Transport", "Artisanat"]).astype(float) * 0.05),
        0.05,
        0.98,
    )
    return employers


def _generate_merchants(config: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    n_merchants = max(3_500, config.n_borrowers // 5)
    merchants = pd.DataFrame(
        {
            "merchant_id": [f"MER{idx:05d}" for idx in range(n_merchants)],
            "region": rng.choice(REGIONS, size=n_merchants),
            "merchant_sector": rng.choice(["Alimentation", "Telecom", "Transport", "Sante", "Education", "Energie"], size=n_merchants),
            "merchant_risk_score": np.clip(rng.normal(0.38, 0.14, size=n_merchants), 0.03, 0.95),
        }
    )
    return merchants


def _generate_borrowers(config: GenerationConfig, employers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    employer_ids = employers["employer_id"].to_numpy()
    borrowers = pd.DataFrame({"borrower_id": [f"BOR{idx:06d}" for idx in range(config.n_borrowers)]})
    borrowers["employment_type"] = rng.choice(EMPLOYMENT_TYPES, size=config.n_borrowers, p=[0.38, 0.24, 0.18, 0.20])
    salaried_mask = borrowers["employment_type"].eq("Salarie").to_numpy()
    employer_choice = rng.choice(employer_ids, size=config.n_borrowers)
    borrowers["employer_id"] = np.where(salaried_mask, employer_choice, "SELF_EMPLOYED")
    employer_region_map = employers.set_index("employer_id")["region"].to_dict()
    employer_sector_map = employers.set_index("employer_id")["sector"].to_dict()
    employer_risk_map = employers.set_index("employer_id")["employer_risk_score"].to_dict()
    fallback_regions = pd.Series(rng.choice(REGIONS, size=config.n_borrowers), index=borrowers.index)
    fallback_sectors = pd.Series(rng.choice(SECTORS, size=config.n_borrowers), index=borrowers.index)
    borrowers["region"] = borrowers["employer_id"].map(employer_region_map).fillna(fallback_regions)
    borrowers["sector"] = borrowers["employer_id"].map(employer_sector_map).fillna(fallback_sectors)
    fallback_employer_risk = pd.Series(
        np.clip(rng.normal(0.46, 0.10, size=config.n_borrowers), 0.08, 0.95),
        index=borrowers.index,
    )
    borrowers["employer_risk_score"] = borrowers["employer_id"].map(employer_risk_map).fillna(fallback_employer_risk)
    borrowers["age"] = np.clip(rng.normal(37, 10, size=config.n_borrowers).round(), 21, 69).astype(int)
    borrowers["months_on_job"] = np.clip(rng.gamma(shape=3.4, scale=14, size=config.n_borrowers), 1, 240).round().astype(int)
    borrowers["household_size"] = np.clip(rng.poisson(3.7, size=config.n_borrowers) + 1, 1, 10).astype(int)
    borrowers["prior_loans_count"] = np.clip(rng.poisson(1.3, size=config.n_borrowers), 0, 9).astype(int)
    borrowers["monthly_income"] = np.clip(
        rng.lognormal(mean=12.0, sigma=0.52, size=config.n_borrowers)
        * np.where(salaried_mask, 1.12, 0.88),
        85_000,
        3_500_000,
    )
    borrowers["savings_balance"] = np.clip(
        borrowers["monthly_income"].to_numpy() * rng.uniform(0.25, 2.8, size=config.n_borrowers),
        10_000,
        7_000_000,
    )
    base_risk = (
        0.22
        + 0.18 * borrowers["employment_type"].isin(["Informel", "MicroEntrepreneur"]).astype(float)
        + 0.12 * borrowers["sector"].isin(["Transport", "Agriculture", "Artisanat"]).astype(float)
        + 0.10 * (borrowers["months_on_job"] < 12).astype(float)
        + 0.05 * (borrowers["prior_loans_count"] == 0).astype(float)
        - 0.06 * (borrowers["monthly_income"] > 600_000).astype(float)
        + 0.16 * borrowers["employer_risk_score"].to_numpy()
        + rng.normal(0, 0.08, size=config.n_borrowers)
    )
    borrowers["latent_risk"] = np.clip(base_risk, 0.03, 0.95)
    borrowers["external_score"] = np.clip(
        790 - 470 * borrowers["latent_risk"] + rng.normal(0, 42, size=config.n_borrowers),
        280,
        870,
    )
    borrowers["debt_service_ratio"] = np.clip(
        0.14 + 0.32 * borrowers["latent_risk"] + 0.06 * borrowers["prior_loans_count"] + rng.normal(0, 0.05, size=config.n_borrowers),
        0.03,
        0.82,
    )
    borrowers["digital_engagement"] = np.clip(
        0.35 + 0.25 * borrowers["employment_type"].eq("Salarie").astype(float) + rng.normal(0, 0.11, size=config.n_borrowers),
        0.02,
        0.98,
    )
    borrowers["kyc_level"] = rng.choice(["KYC1", "KYC2", "KYC3"], size=config.n_borrowers, p=[0.20, 0.48, 0.32])
    return borrowers


def _generate_loans(config: GenerationConfig, borrowers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    borrower_idx = rng.integers(0, len(borrowers), size=config.n_loans)
    selected = borrowers.iloc[borrower_idx].reset_index(drop=True)
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date) - pd.offsets.MonthBegin(13)
    if end <= start:
        end = start + pd.Timedelta(days=30)
    horizon_days = max(1, (end - start).days)
    origination_offsets = np.floor(rng.beta(1.7, 1.25, size=config.n_loans) * horizon_days).astype(int)
    origination_dates = start + pd.to_timedelta(origination_offsets, unit="D")
    product = rng.choice(PRODUCTS, size=config.n_loans, p=[0.42, 0.26, 0.18, 0.14])
    tenor = np.select(
        [product == "PretPersonnel", product == "MicroPME", product == "AvanceSalaire", product == "Equipement"],
        [
            rng.integers(18, 61, size=config.n_loans),
            rng.integers(12, 49, size=config.n_loans),
            rng.integers(6, 19, size=config.n_loans),
            rng.integers(24, 61, size=config.n_loans),
        ],
        default=rng.integers(config.min_tenor_months, config.max_tenor_months + 1, size=config.n_loans),
    )
    amount_multiplier = np.select(
        [product == "PretPersonnel", product == "MicroPME", product == "AvanceSalaire", product == "Equipement"],
        [5.2, 6.8, 2.2, 7.8],
        default=4.0,
    )
    principal = np.clip(
        selected["monthly_income"].to_numpy() * amount_multiplier * rng.uniform(0.65, 1.35, size=config.n_loans),
        150_000,
        18_000_000,
    )
    collateral_type = rng.choice(COLLATERAL_TYPES, size=config.n_loans, p=[0.30, 0.18, 0.20, 0.18, 0.14])
    collateral_coverage = np.clip(
        np.select(
            [collateral_type == "Aucun", collateral_type == "Vehicule", collateral_type == "Stock", collateral_type == "Depot", collateral_type == "Foncier"],
            [
                rng.uniform(0.0, 0.05, size=config.n_loans),
                rng.uniform(0.35, 0.7, size=config.n_loans),
                rng.uniform(0.25, 0.65, size=config.n_loans),
                rng.uniform(0.55, 1.1, size=config.n_loans),
                rng.uniform(0.65, 1.25, size=config.n_loans),
            ],
            default=rng.uniform(0.1, 0.5, size=config.n_loans),
        ),
        0.0,
        1.4,
    )
    rate = np.clip(
        0.10
        + 0.07 * selected["latent_risk"].to_numpy()
        + 0.015 * np.isin(product, ["MicroPME", "AvanceSalaire"])
        + rng.normal(0, 0.01, size=config.n_loans),
        0.08,
        0.29,
    )
    monthly_rate = rate / 12.0
    annuity_factor = np.where(
        monthly_rate > 0,
        (monthly_rate * (1 + monthly_rate) ** tenor) / (((1 + monthly_rate) ** tenor) - 1),
        1.0 / np.maximum(tenor, 1),
    )
    installment = principal * annuity_factor
    origination_pd = np.clip(
        _sigmoid(
            -2.90
            + 3.70 * selected["latent_risk"].to_numpy()
            + 0.78 * selected["debt_service_ratio"].to_numpy()
            - 0.58 * collateral_coverage
            + 0.25 * np.isin(product, ["MicroPME", "AvanceSalaire"])
            + rng.normal(0, 0.18, size=config.n_loans)
        ),
        0.01,
        0.75,
    )
    base_lgd = np.clip(0.62 - 0.38 * collateral_coverage + 0.06 * (product == "MicroPME"), 0.10, 0.88)
    guarantor_pool = np.where(rng.random(config.n_loans) < 0.28, rng.choice(borrowers["borrower_id"], size=config.n_loans), "NO_GUARANTOR")
    loans = pd.DataFrame(
        {
            "loan_id": [f"LOA{idx:06d}" for idx in range(config.n_loans)],
            "borrower_id": selected["borrower_id"],
            "region": selected["region"],
            "sector": selected["sector"],
            "employment_type": selected["employment_type"],
            "product_type": product,
            "origination_date": origination_dates,
            "tenor_months": tenor.astype(int),
            "principal_amount": principal.round(2),
            "interest_rate": rate.round(4),
            "installment_amount": installment.round(2),
            "collateral_type": collateral_type,
            "collateral_coverage": collateral_coverage.round(4),
            "guarantor_id": guarantor_pool,
            "origination_pd": origination_pd.round(6),
            "base_lgd": base_lgd.round(6),
        }
    )
    return loans


def _generate_transactions(
    config: GenerationConfig,
    borrowers: pd.DataFrame,
    merchants: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    borrower_ids = borrowers["borrower_id"].to_numpy()
    merchant_ids = merchants["merchant_id"].to_numpy()
    borrower_risk_band = pd.qcut(borrowers["latent_risk"], q=6, labels=False, duplicates="drop").astype(int)
    borrower_clusters = (
        borrowers["region"].astype(str)
        + "|"
        + borrowers["sector"].astype(str)
        + "|"
        + borrower_risk_band.astype(str)
    ).to_numpy()
    borrower_regions = borrowers["region"].astype(str).to_numpy()
    cluster_pools = (
        pd.DataFrame({"borrower_id": borrower_ids, "network_cluster": borrower_clusters})
        .groupby("network_cluster")["borrower_id"]
        .apply(np.array)
        .to_dict()
    )
    region_pools = (
        pd.DataFrame({"borrower_id": borrower_ids, "region": borrower_regions})
        .groupby("region")["borrower_id"]
        .apply(np.array)
        .to_dict()
    )
    activity_weights = np.clip(
        0.4
        + borrowers["digital_engagement"].to_numpy()
        + 0.6 * (borrowers["monthly_income"].to_numpy() / borrowers["monthly_income"].median())
        + 0.25 * borrowers["prior_loans_count"].to_numpy(),
        0.05,
        None,
    )
    activity_weights = activity_weights / activity_weights.sum()
    borrower_choice = rng.choice(len(borrowers), size=config.n_transactions, p=activity_weights)
    sampled = borrowers.iloc[borrower_choice].reset_index(drop=True)
    base_dates = pd.Timestamp(config.start_date) + pd.to_timedelta(
        rng.integers(0, (pd.Timestamp(config.end_date) - pd.Timestamp(config.start_date)).days + 1, size=config.n_transactions),
        unit="D",
    )
    tx_hours = rng.integers(0, 24, size=config.n_transactions)
    tx_minutes = rng.integers(0, 60, size=config.n_transactions)
    tx_dates = base_dates + pd.to_timedelta(tx_hours, unit="h") + pd.to_timedelta(tx_minutes, unit="m")
    tx_type = rng.choice(TX_TYPES, size=config.n_transactions, p=[0.26, 0.28, 0.12, 0.18, 0.16])
    counterparty_type = rng.choice(COUNTERPARTY_TYPES, size=config.n_transactions, p=[0.33, 0.52, 0.15])
    direction = np.where(np.isin(tx_type, ["Depot"]), "in", "out")
    borrower_counterparties = rng.choice(borrower_ids, size=config.n_transactions)
    sampled_clusters = borrower_clusters[borrower_choice]
    sampled_regions = borrower_regions[borrower_choice]
    borrower_tx_idx = np.flatnonzero(counterparty_type == "borrower")
    relation_draw = rng.random(len(borrower_tx_idx))
    same_cluster_idx = borrower_tx_idx[relation_draw < 0.76]
    same_region_idx = borrower_tx_idx[(relation_draw >= 0.76) & (relation_draw < 0.93)]
    if len(same_cluster_idx):
        same_cluster_keys = sampled_clusters[same_cluster_idx]
        for cluster_name in np.unique(same_cluster_keys):
            idx = same_cluster_idx[same_cluster_keys == cluster_name]
            pool = cluster_pools.get(cluster_name, borrower_ids)
            borrower_counterparties[idx] = rng.choice(pool, size=len(idx))
    if len(same_region_idx):
        same_region_keys = sampled_regions[same_region_idx]
        for region_name in np.unique(same_region_keys):
            idx = same_region_idx[same_region_keys == region_name]
            pool = region_pools.get(region_name, borrower_ids)
            borrower_counterparties[idx] = rng.choice(pool, size=len(idx))
    merchant_counterparties = rng.choice(merchant_ids, size=config.n_transactions)
    utility_counterparties = rng.choice(["TELCO", "EAU", "ELECTRICITE", "IMPOTS"], size=config.n_transactions)
    borrower_self_mask = (borrower_counterparties == sampled["borrower_id"].to_numpy()) & (counterparty_type == "borrower")
    if borrower_self_mask.any():
        borrower_counterparties[borrower_self_mask] = rng.choice(borrower_ids, size=int(borrower_self_mask.sum()))
    counterparty_id = np.where(
        counterparty_type == "borrower",
        borrower_counterparties,
        np.where(counterparty_type == "merchant", merchant_counterparties, utility_counterparties),
    )
    sampled_income = sampled["monthly_income"].to_numpy()
    tx_amount = np.exp(
        rng.normal(
            np.log(np.clip(sampled_income / 18, 7_500, 250_000)),
            0.9 + 0.2 * np.isin(tx_type, ["Retrait", "P2P"]),
            size=config.n_transactions,
        )
    )
    tx_amount *= np.where(sampled["latent_risk"].to_numpy() > 0.62, rng.uniform(0.60, 1.75, size=config.n_transactions), rng.uniform(0.72, 1.38, size=config.n_transactions))
    channels = rng.choice(CHANNELS, size=config.n_transactions, p=[0.18, 0.17, 0.23, 0.28, 0.14])
    night_flag = ((pd.Series(tx_dates).dt.hour < 6) | (pd.Series(tx_dates).dt.hour >= 22)).to_numpy()
    tx = pd.DataFrame(
        {
            "transaction_id": [f"TRX{idx:07d}" for idx in range(config.n_transactions)],
            "tx_date": tx_dates,
            "borrower_id": sampled["borrower_id"],
            "region": sampled["region"],
            "channel": channels,
            "tx_type": tx_type,
            "direction": direction,
            "counterparty_type": counterparty_type,
            "counterparty_id": counterparty_id,
            "amount": np.clip(tx_amount, 500, 2_400_000).round(2),
            "is_night": night_flag.astype(int),
            "borrower_latent_risk": sampled["latent_risk"].round(6),
        }
    )
    tx["is_high_risk_pattern"] = (
        (
            (tx["channel"].isin(["Agent", "USSD"]))
            & (tx["tx_type"].isin(["Retrait", "P2P"]))
            & (tx["amount"] > tx["amount"].quantile(0.82))
        )
        | (
            tx["counterparty_type"].eq("borrower")
            & tx["borrower_latent_risk"].gt(0.70)
            & tx["amount"].gt(tx["amount"].quantile(0.74))
        )
    ).astype(int)
    return tx.sort_values(["tx_date", "transaction_id"]).reset_index(drop=True)


def _build_hidden_network_signals(
    borrowers: pd.DataFrame,
    merchants: pd.DataFrame,
    loans: pd.DataFrame,
    transactions: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    borrower_risk_map = borrowers.set_index("borrower_id")["latent_risk"].to_dict()
    merchant_risk_map = merchants.set_index("merchant_id")["merchant_risk_score"].to_dict()

    borrower_tx = transactions.loc[
        transactions["counterparty_type"].eq("borrower"),
        ["borrower_id", "counterparty_id", "amount", "is_high_risk_pattern"],
    ].copy()
    borrower_tx["counterparty_latent_risk"] = borrower_tx["counterparty_id"].map(borrower_risk_map).fillna(0.0)
    borrower_tx["weighted_peer_risk"] = borrower_tx["amount"] * borrower_tx["counterparty_latent_risk"]
    borrower_peer = (
        borrower_tx.groupby("borrower_id", as_index=False)
        .agg(
            peer_amount=("amount", "sum"),
            weighted_peer_risk=("weighted_peer_risk", "sum"),
            risky_flow_share=("is_high_risk_pattern", "mean"),
        )
    )
    borrower_peer["network_peer_risk"] = borrower_peer["weighted_peer_risk"] / borrower_peer["peer_amount"].clip(lower=1.0)
    borrower_pairs = (
        borrower_tx.groupby(["borrower_id", "counterparty_id"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "pair_amount"})
    )
    borrower_pairs["pair_share"] = borrower_pairs["pair_amount"] / borrower_pairs.groupby("borrower_id")["pair_amount"].transform("sum").clip(lower=1.0)
    borrower_hhi = (
        borrower_pairs.assign(pair_share_sq=lambda df: df["pair_share"] ** 2)
        .groupby("borrower_id", as_index=False)["pair_share_sq"]
        .sum()
        .rename(columns={"pair_share_sq": "network_counterparty_hhi"})
    )

    merchant_tx = transactions.loc[
        transactions["counterparty_type"].eq("merchant"),
        ["borrower_id", "counterparty_id", "amount"],
    ].copy()
    merchant_tx["merchant_risk"] = merchant_tx["counterparty_id"].map(merchant_risk_map).fillna(0.0)
    merchant_tx["weighted_merchant_risk"] = merchant_tx["amount"] * merchant_tx["merchant_risk"]
    merchant_exposure = (
        merchant_tx.groupby("borrower_id", as_index=False)
        .agg(
            merchant_amount=("amount", "sum"),
            weighted_merchant_risk=("weighted_merchant_risk", "sum"),
        )
    )
    merchant_exposure["network_merchant_risk"] = (
        merchant_exposure["weighted_merchant_risk"] / merchant_exposure["merchant_amount"].clip(lower=1.0)
    )

    employer_peer = borrowers[["borrower_id", "employer_id", "latent_risk"]].copy()
    employer_peer["employer_peer_hidden_risk"] = employer_peer.groupby("employer_id")["latent_risk"].transform("mean")
    employer_peer = employer_peer[["borrower_id", "employer_peer_hidden_risk"]]

    guarantor_signal = loans.loc[loans["guarantor_id"].ne("NO_GUARANTOR"), ["borrower_id", "guarantor_id"]].copy()
    guarantor_signal["guarantor_hidden_risk"] = guarantor_signal["guarantor_id"].map(borrower_risk_map).fillna(0.0)
    guarantor_signal = (
        guarantor_signal.groupby("borrower_id", as_index=False)["guarantor_hidden_risk"]
        .mean()
    )

    hidden = borrowers[["borrower_id"]].copy()
    hidden = hidden.merge(
        borrower_peer[["borrower_id", "network_peer_risk", "risky_flow_share"]],
        on="borrower_id",
        how="left",
    )
    hidden = hidden.merge(borrower_hhi, on="borrower_id", how="left")
    hidden = hidden.merge(
        merchant_exposure[["borrower_id", "network_merchant_risk"]],
        on="borrower_id",
        how="left",
    )
    hidden = hidden.merge(employer_peer, on="borrower_id", how="left")
    hidden = hidden.merge(guarantor_signal, on="borrower_id", how="left")
    hidden = hidden.fillna(
        {
            "network_peer_risk": 0.0,
            "risky_flow_share": 0.0,
            "network_counterparty_hhi": 0.0,
            "network_merchant_risk": 0.0,
            "employer_peer_hidden_risk": 0.0,
            "guarantor_hidden_risk": 0.0,
        }
    )
    hidden["network_hidden_stress"] = np.clip(
        0.48 * hidden["network_peer_risk"]
        + 0.18 * hidden["network_merchant_risk"]
        + 0.16 * hidden["network_counterparty_hhi"]
        + 0.12 * hidden["employer_peer_hidden_risk"]
        + 0.12 * hidden["guarantor_hidden_risk"]
        + 0.10 * hidden["risky_flow_share"]
        + rng.normal(0, 0.03, size=len(hidden)),
        0.01,
        0.98,
    )
    hidden["network_volatility"] = np.clip(
        0.04
        + 0.16 * hidden["network_counterparty_hhi"]
        + 0.08 * hidden["risky_flow_share"]
        + rng.normal(0, 0.01, size=len(hidden)),
        0.03,
        0.26,
    )
    hidden["network_phase"] = rng.uniform(-np.pi, np.pi, size=len(hidden))
    return hidden


def _simulate_performance(
    config: GenerationConfig,
    borrowers: pd.DataFrame,
    loans: pd.DataFrame,
    macro: pd.DataFrame,
    hidden_network: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    borrower_map = borrowers.set_index("borrower_id")
    macro_map = macro.set_index(["month", "region"])
    hidden_network_map = hidden_network.set_index("borrower_id")
    records: list[dict[str, Any]] = []
    for loan in loans.itertuples(index=False):
        borrower = borrower_map.loc[loan.borrower_id]
        borrower_network = hidden_network_map.loc[loan.borrower_id]
        current_dpd = 0
        outstanding = float(loan.principal_amount)
        restructured = 0
        loan_shock = float(rng.normal(0, 0.18))
        months = int(loan.tenor_months)
        active_end = min(pd.Timestamp(config.end_date), pd.Timestamp(loan.origination_date) + pd.offsets.MonthBegin(months + 1))
        payment_dates = pd.date_range(pd.Timestamp(loan.origination_date), active_end, freq="MS")
        if len(payment_dates) == 0:
            payment_dates = pd.DatetimeIndex([pd.Timestamp(loan.origination_date)])
        for step, month in enumerate(payment_dates[: months + 1]):
            macro_row = macro_map.loc[(pd.Timestamp(month).replace(day=1), loan.region)]
            borrower_risk = float(borrower["latent_risk"])
            macro_stress = float(macro_row["macro_stress"])
            income_resilience = float(np.clip(np.log1p(borrower["monthly_income"]) / np.log(1_000_000), 0.45, 1.05))
            savings_cushion = float(np.clip(np.log1p(borrower["savings_balance"]) / np.log(1_000_000), 0.15, 1.05))
            employment_resilience = float(np.clip(borrower["months_on_job"] / 120, 0.0, 1.0))
            seasonal_effect = float(0.06 * np.sin((step + 1) / 2.5) + 0.04 * np.cos((step + 1) / 4.0))
            network_cycle = float(
                borrower_network["network_volatility"] * np.sin(0.55 * (step + 1) + borrower_network["network_phase"])
            )
            network_stress = float(
                np.clip(
                    borrower_network["network_hidden_stress"] * (0.55 + 0.80 * macro_stress)
                    + network_cycle
                    + rng.normal(0, 0.02),
                    0.0,
                    1.0,
                )
            )
            transient_shock = float(rng.normal(0, 0.28))
            payment_pressure = (
                -3.05
                + 2.05 * borrower_risk
                + 0.62 * borrower["debt_service_ratio"]
                + 0.42 * macro_stress
                + 0.0019 * current_dpd
                + 0.26 * restructured
                - 0.46 * loan.collateral_coverage
                + 0.15 * (loan.product_type == "MicroPME")
                - 0.32 * income_resilience
                - 0.22 * savings_cushion
                - 0.18 * employment_resilience
                + 0.20 * borrower["employer_risk_score"]
                + 0.88 * network_stress
                + 0.24 * borrower_network["network_peer_risk"]
                + 0.16 * borrower_network["network_merchant_risk"]
                + 0.15 * borrower_network["guarantor_hidden_risk"]
                + 0.10 * borrower_network["employer_peer_hidden_risk"]
                + 0.12 * borrower_network["network_counterparty_hhi"]
                + seasonal_effect
                + loan_shock
                + transient_shock
            )
            miss_prob = float(np.clip(_sigmoid(payment_pressure), 0.01, 0.94))
            partial_prob = float(
                np.clip(
                    0.12
                    + 0.11 * borrower_risk
                    + 0.04 * macro_stress
                    + 0.015 * (current_dpd > 0)
                    + 0.08 * network_stress
                    - 0.04 * savings_cushion
                    + abs(seasonal_effect) * 0.25,
                    0.04,
                    0.34,
                )
            )
            full_prob = max(0.02, 1.0 - miss_prob - partial_prob)
            probs = np.array([full_prob, partial_prob, miss_prob], dtype=float)
            probs = probs / probs.sum()
            payment_state = rng.choice(["full", "partial", "miss"], p=probs)
            payment_ratio = 1.0 if payment_state == "full" else 0.55 if payment_state == "partial" else 0.0
            if payment_state == "full":
                current_dpd = max(0, current_dpd - 32)
            elif payment_state == "partial":
                current_dpd = min(180, current_dpd + 14)
            else:
                current_dpd = min(180, current_dpd + (34 if network_stress >= 0.28 else 30))
            if current_dpd >= 60 and rng.random() < (
                0.05
                + 0.12 * borrower_risk
                + 0.04 * macro_stress
                + 0.05 * network_stress
            ):
                restructured = 1
                current_dpd = max(30, current_dpd - 15)
            monthly_rate = float(loan.interest_rate) / 12.0
            scheduled_interest = outstanding * monthly_rate
            scheduled_principal = max(float(loan.installment_amount) - scheduled_interest, 0.10 * float(loan.installment_amount))
            principal_payment = min(outstanding, scheduled_principal * max(0.15, payment_ratio))
            outstanding = max(0.0, outstanding - principal_payment)
            current_pd_12m = float(
                np.clip(
                    _sigmoid(
                        -2.95
                        + 2.40 * borrower_risk
                        + 0.86 * macro_stress
                        + 0.0023 * current_dpd
                        + 0.28 * restructured
                        + 0.18 * (loan.product_type == "MicroPME")
                        - 0.52 * loan.collateral_coverage
                        - 0.22 * savings_cushion
                        - 0.16 * employment_resilience
                        + 0.20 * borrower["employer_risk_score"]
                        + 1.02 * network_stress
                        + 0.26 * borrower_network["network_peer_risk"]
                        + 0.18 * borrower_network["guarantor_hidden_risk"]
                        + 0.14 * borrower_network["employer_peer_hidden_risk"]
                        + 0.10 * borrower_network["network_merchant_risk"]
                        + 0.24 * max(payment_pressure, 0.0)
                        + loan_shock
                        + rng.normal(0, 0.22)
                    ),
                    0.005,
                    0.98,
                )
            )
            lifetime_pd = float(np.clip(1 - (1 - current_pd_12m) ** max(1.0, (months - step) / 12), current_pd_12m, 0.995))
            current_lgd = float(np.clip(loan.base_lgd + 0.09 * (current_dpd >= 60) + 0.08 * macro_stress, 0.08, 0.96))
            ead = float(max(outstanding, 0.0))
            records.append(
                {
                    "loan_id": loan.loan_id,
                    "borrower_id": loan.borrower_id,
                    "as_of_date": month,
                    "vintage_month": pd.Timestamp(month).to_period("M").to_timestamp(),
                    "months_since_origination": step,
                    "remaining_term_months": max(months - step, 0),
                    "current_dpd": current_dpd,
                    "payment_state": payment_state,
                    "payment_ratio": payment_ratio,
                    "restructured_flag": restructured,
                    "outstanding_balance": round(ead, 2),
                    "current_pd_12m_true": round(current_pd_12m, 6),
                    "lifetime_pd_true": round(lifetime_pd, 6),
                    "current_lgd_true": round(current_lgd, 6),
                    "default_flag_current": int(current_dpd >= 90),
                    "macro_stress": round(macro_stress, 6),
                }
            )
            if outstanding <= 0:
                break
    performance = pd.DataFrame.from_records(records).sort_values(["loan_id", "as_of_date"]).reset_index(drop=True)
    performance["stage_ifrs9"] = "Stage 1"
    sicr_mask = (
        (performance["current_dpd"] >= 30)
        | (performance["restructured_flag"] == 1)
        | (performance["current_pd_12m_true"] >= 1.7 * loans.set_index("loan_id").loc[performance["loan_id"], "origination_pd"].to_numpy())
    )
    performance.loc[sicr_mask, "stage_ifrs9"] = "Stage 2"
    performance.loc[performance["current_dpd"] >= 90, "stage_ifrs9"] = "Stage 3"
    return performance


def _add_forward_labels(performance: pd.DataFrame) -> pd.DataFrame:
    perf = performance.copy()
    perf["default_90dpd_12m"] = 0
    perf["label_horizon_months"] = 0
    perf["label_is_complete_12m"] = 0
    grouped_rows: list[pd.DataFrame] = []
    for _, grp in perf.groupby("loan_id", sort=False):
        grp = grp.sort_values("as_of_date").copy()
        defaults = grp["default_flag_current"].to_numpy(dtype=int)
        horizon = np.zeros_like(defaults)
        future_months = np.zeros_like(defaults)
        for idx in range(len(defaults)):
            future_window = defaults[idx + 1 : idx + 13]
            future_months[idx] = len(future_window)
            horizon[idx] = future_window.max() if len(future_window) else 0
        grp["default_90dpd_12m"] = horizon
        grp["label_horizon_months"] = future_months
        grp["label_is_complete_12m"] = (future_months >= 12).astype(int)
        grouped_rows.append(grp)
    return pd.concat(grouped_rows, ignore_index=True)


def _build_data_dictionary() -> pd.DataFrame:
    rows = [
        ("borrowers", "borrower_id", "Identifiant pseudonymisé de l'emprunteur"),
        ("borrowers", "latent_risk", "Risque latent synthétique servant à injecter du réalisme et de l'hétérogénéité"),
        ("loans", "origination_pd", "Proxy de PD 12 mois à l'octroi"),
        ("loans", "base_lgd", "LGD structurelle avant stress courant"),
        ("transactions", "is_high_risk_pattern", "Indicateur d'activité transactionnelle atypique"),
        ("performance", "current_dpd", "Retard courant en jours"),
        ("performance", "stage_ifrs9", "Stage IFRS 9 synthétique basé sur SICR et défaut"),
        ("performance", "default_90dpd_12m", "Défaut dans les 12 mois suivant la date d'observation"),
    ]
    return pd.DataFrame(rows, columns=["table_name", "field_name", "description"])


def generate_synthetic_bundle(config: GenerationConfig) -> SyntheticBundle:
    rng = np.random.default_rng(config.seed)
    macro = _build_macro_frame(config, rng)
    employers = _generate_employers(config, rng)
    merchants = _generate_merchants(config, rng)
    borrowers = _generate_borrowers(config, employers, rng)
    loans = _generate_loans(config, borrowers, rng)
    transactions = _generate_transactions(config, borrowers, merchants, rng)
    hidden_network = _build_hidden_network_signals(borrowers, merchants, loans, transactions, rng)
    performance = _simulate_performance(config, borrowers, loans, macro, hidden_network, rng)
    performance = _add_forward_labels(performance)
    data_dictionary = _build_data_dictionary()
    return SyntheticBundle(
        borrowers=borrowers,
        employers=employers,
        merchants=merchants,
        macro=macro,
        loans=loans,
        transactions=transactions,
        performance=performance,
        data_dictionary=data_dictionary,
    )
