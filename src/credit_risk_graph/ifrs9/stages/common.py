from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from credit_risk_graph.ifrs9.params import FIGURES_DIR, METRICS_DIR, MODELS_DIR, TABLES_DIR
from credit_risk_graph.utils.io import ensure_dir


def save_metrics(payload: dict[str, Any], name: str) -> Path:
    path = ensure_dir(METRICS_DIR) / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return path


def save_table(frame: pd.DataFrame, name: str) -> Path:
    path = ensure_dir(TABLES_DIR) / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def save_model(model: Any, name: str) -> Path:
    path = ensure_dir(MODELS_DIR) / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str) -> Any:
    return joblib.load(MODELS_DIR / f"{name}.joblib")


def figure_path(name: str) -> Path:
    return ensure_dir(FIGURES_DIR) / f"{name}.png"
