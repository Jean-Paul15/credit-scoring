from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("CREDIT_RISK_DATA_DIR", str(ROOT / "data")))
ARTIFACTS_DIR = Path(os.getenv("CREDIT_RISK_ARTIFACTS_DIR", str(ROOT / "artifacts")))


@dataclass(slots=True)
class GenerationConfig:
    seed: int = 42
    n_borrowers: int = 60_000
    n_loans: int = 90_000
    n_transactions: int = 4_500_000
    start_date: str = "2016-01-01"
    end_date: str = "2024-12-31"
    train_cutoff: str = "2020-12-31"
    validation_cutoff: str = "2021-12-31"
    oot_cutoff: str = "2022-12-31"
    min_tenor_months: int = 6
    max_tenor_months: int = 60
    publication_min_borrowers: int = 50_000
    publication_min_loans: int = 80_000
    publication_min_transactions: int = 4_000_000
    publication_min_snapshot_rows: int = 800_000
    publication_min_split_rows_train: int = 500_000
    publication_min_split_rows_validation: int = 100_000
    publication_min_split_rows_test: int = 100_000
    publication_min_split_rows_oot: int = 80_000
    publication_min_defaults_train: int = 15_000
    publication_min_defaults_validation: int = 3_000
    publication_min_defaults_test: int = 3_000
    publication_min_defaults_oot: int = 2_000

    def to_dict(self) -> dict:
        return asdict(self)
