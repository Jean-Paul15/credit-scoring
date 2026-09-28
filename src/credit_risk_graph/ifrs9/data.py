from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from credit_risk_graph.config import DATA_DIR, GenerationConfig
from credit_risk_graph.features.runtime_graph import load_split
from credit_risk_graph.features.schema import CATEGORICAL_FEATURES, FEATURE_PROFILES, TARGET_COLUMN
from credit_risk_graph.utils.io import read_parquet

SPLITS = ("train", "validation", "test", "out_of_time")


@lru_cache(maxsize=1)
def _contract_attributes() -> tuple[pd.DataFrame, pd.DataFrame]:
    loans = read_parquet(DATA_DIR / "clean" / "loans_clean.parquet")[["loan_id", "guarantor_id"]]
    loans = loans.assign(has_guarantor=(loans["guarantor_id"] != "NO_GUARANTOR").astype(int))[["loan_id", "has_guarantor"]]
    borrowers = read_parquet(DATA_DIR / "clean" / "borrowers_clean.parquet")[["borrower_id", "employer_risk_score"]]
    return loans, borrowers


def load_full_split(split: str) -> pd.DataFrame:
    """Découpage temporel enrichi des attributs de contrat (garant présent, note de risque de l'employeur)."""
    frame = load_split(split)
    loans, borrowers = _contract_attributes()
    return frame.merge(loans, on="loan_id", how="left").merge(borrowers, on="borrower_id", how="left")


def observation_cutoff(params: dict[str, Any]) -> pd.Timestamp:
    """Dernier `as_of_date` dont la fenêtre de défaut à 12 mois est entièrement observée (pas de censure à droite)."""
    horizon = params["population"]["label_horizon_months"]
    return pd.Timestamp(GenerationConfig().end_date) - pd.DateOffset(months=horizon)


def train_embargo_cutoff(params: dict[str, Any]) -> pd.Timestamp:
    """Dernier `as_of_date` d'apprentissage dont le label ne recouvre pas la période de validation."""
    return pd.Timestamp(GenerationConfig().train_cutoff) - pd.DateOffset(months=params["population"]["train_embargo_months"])


def load_performing(split: str, params: dict[str, Any], embargo: bool = True) -> pd.DataFrame:
    """Expositions saines à la date d'arrêté, avec fenêtre de défaut complète."""
    frame = load_full_split(split)
    performing = frame["current_dpd"] < params["population"]["performing_dpd_max"]
    observed = frame["as_of_date"] <= observation_cutoff(params)
    keep = performing & observed
    if split == "train" and embargo:
        keep &= frame["as_of_date"] <= train_embargo_cutoff(params)
    return frame.loc[keep].reset_index(drop=True)


def sample_rows(frame: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    limit = params["population"]["max_train_rows"]
    if len(frame) <= limit:
        return frame
    return frame.sample(n=limit, random_state=params["seed"]).reset_index(drop=True)


def feature_lists(profile: str) -> tuple[list[str], list[str]]:
    """(catégorielles, numériques) du profil de variables."""
    spec = FEATURE_PROFILES[profile]
    return list(spec["categorical"]), list(spec["numeric"])


def model_columns(profile: str) -> list[str]:
    categorical, numeric = feature_lists(profile)
    return categorical + numeric


def target(frame: pd.DataFrame) -> pd.Series:
    return frame[TARGET_COLUMN].astype(int)


__all__ = ["CATEGORICAL_FEATURES", "SPLITS", "TARGET_COLUMN", "feature_lists", "load_performing", "model_columns", "observation_cutoff", "sample_rows", "target"]
