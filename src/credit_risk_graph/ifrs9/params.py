from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from credit_risk_graph.config import ARTIFACTS_DIR, ROOT

PARAMS_PATH = ROOT / "params.yaml"
IFRS9_DIR = ARTIFACTS_DIR / "ifrs9"
MODELS_DIR = IFRS9_DIR / "models"
METRICS_DIR = IFRS9_DIR / "metrics"
TABLES_DIR = IFRS9_DIR / "tables"
FIGURES_DIR = IFRS9_DIR / "figures"


@lru_cache(maxsize=1)
def load_params() -> dict[str, Any]:
    return yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
