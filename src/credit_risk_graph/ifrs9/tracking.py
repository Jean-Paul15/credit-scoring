from __future__ import annotations

import hashlib
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow
import pandas as pd

from credit_risk_graph.config import ROOT
from credit_risk_graph.ifrs9.params import load_params

EXPERIMENT_NAME = "ifrs9-ebm-pipeline"
TRACKING_DB = ROOT / "mlruns" / "mlflow.db"
ARTIFACT_ROOT = ROOT / "mlruns" / "artifacts"


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-commit"


def data_version() -> str:
    """Empreinte des données : hash du dvc.lock (sorties de l'étape de génération) si présent."""
    lock = ROOT / "dvc.lock"
    if not lock.exists():
        return "dvc-lock-absent"
    return hashlib.sha256(lock.read_bytes()).hexdigest()[:12]


def experiment_id() -> str:
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    existing = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if existing is not None:
        return existing.experiment_id
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_ROOT.as_uri())


def flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = str(value)[:500]
    return flat


@contextmanager
def tracked_run(stage: str, extra_params: dict[str, Any] | None = None, parent_run_id: str | None = None) -> Iterator[mlflow.ActiveRun]:
    """Un run MLflow par étape, tous les paramètres de params.yaml + version du code et des données."""
    with mlflow.start_run(experiment_id=experiment_id(), run_name=stage, nested=parent_run_id is not None) as run:
        mlflow.set_tags({"pipeline_stage": stage, "git_commit": _git_commit(), "data_version": data_version()})
        mlflow.log_params(flatten(load_params()))
        if extra_params:
            mlflow.log_params(flatten(extra_params, "run."))
        yield run


def log_metrics(metrics: dict[str, Any], prefix: str = "", step: int | None = None) -> None:
    for key, value in flatten(metrics, prefix).items():
        try:
            mlflow.log_metric(key.replace(" ", "_"), float(value), step=step)
        except (TypeError, ValueError):
            continue


def log_table(frame: pd.DataFrame, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.csv"
        frame.to_csv(path, index=False)
        mlflow.log_artifact(str(path), artifact_path="tables")


def log_json(payload: dict[str, Any], name: str) -> None:
    mlflow.log_dict(payload, f"metrics/{name}.json")


def log_figure(fig: Any, name: str) -> None:
    mlflow.log_figure(fig, f"figures/{name}.png")
