from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    try:
        frame.to_parquet(path, index=False)
    except ImportError:
        frame.to_pickle(path)


def read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError:
        return pd.read_pickle(path)


def save_joblib(payload: Any, path: Path) -> None:
    ensure_dir(path.parent)
    joblib.dump(payload, path)


def load_joblib(path: Path) -> Any:
    return joblib.load(path)
