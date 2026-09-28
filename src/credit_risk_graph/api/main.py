from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from credit_risk_graph.ifrs9.params import METRICS_DIR
from credit_risk_graph.ifrs9.portfolio import ECLEngine
from credit_risk_graph.ifrs9.stages.common import load_model
from credit_risk_graph.monitoring.metrics import LATENCY_HISTOGRAM, REQUEST_COUNTER, render_metrics

# Champs indispensables au staging et au calcul de l'ECL ; les autres variables du modèle peuvent manquer (traitées comme valeurs manquantes).
REQUIRED_FIELDS = (
    "current_dpd",
    "restructured_flag",
    "origination_pd",
    "outstanding_balance",
    "interest_rate",
    "installment_amount",
    "remaining_term_months",
    "macro_stress",
    "sector_stress",
    "collateral_type",
    "product_type",
    "region",
)

app = FastAPI(title="Moteur IFRS 9 — PD par Explainable Boosting Machine", version="1.0.0")


class ExposurePayload(BaseModel):
    features: dict[str, float | int | str | None] = Field(description="Variables du dossier, nommées comme dans le modèle")


class BatchPayload(BaseModel):
    exposures: list[ExposurePayload] = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def get_engine() -> ECLEngine:
    return load_model("ecl_engine")


def _frame(payloads: list[ExposurePayload], engine: ECLEngine) -> tuple[pd.DataFrame, list[list[str]]]:
    columns = sorted(set(engine.pd_model.features) | set(engine.lgd_model.features) | set(REQUIRED_FIELDS))
    rows, missing = [], []
    for item in payloads:
        absent = [name for name in REQUIRED_FIELDS if item.features.get(name) is None]
        if absent:
            raise HTTPException(status_code=422, detail=f"Champs obligatoires manquants : {absent}")
        rows.append({name: item.features.get(name) for name in columns})
        missing.append([name for name in engine.pd_model.features if item.features.get(name) is None])
    frame = pd.DataFrame(rows, columns=columns)
    for name in columns:
        if name not in {"collateral_type", "product_type", "region", "sector", "employment_type", "kyc_level"}:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame, missing


def _explain(engine: ECLEngine, frame: pd.DataFrame, top: int = 5) -> list[list[dict[str, Any]]]:
    contributions = engine.pd_model.term_contributions(frame)
    out = []
    for idx in contributions.index:
        ranked = contributions.loc[idx].abs().sort_values(ascending=False).head(top).index
        out.append([{"terme": t, "contribution_logit": float(contributions.loc[idx, t])} for t in ranked])
    return out


def _score(payloads: list[ExposurePayload]) -> list[dict[str, Any]]:
    engine = get_engine()
    frame, missing = _frame(payloads, engine)
    result = engine.run(frame).table
    explanations = _explain(engine, frame)
    responses = []
    for i, row in result.reset_index(drop=True).iterrows():
        responses.append(
            {
                "pd_12m": float(row["pd_12m"]),
                "grade": int(row["grade"]),
                "stage": int(row["stage"]),
                "ead": float(row["ead"]),
                "lgd_baseline": float(row["lgd_baseline"]),
                "ecl_ponderee": float(row["ecl_ponderee"]),
                "ecl_par_scenario": {s.name: float(row[f"ecl_{s.name}"]) for s in engine.scenarios},
                "contributions_principales": explanations[i],
                "variables_manquantes": missing[i],
            }
        )
    return responses


def _timed(endpoint: str, payloads: list[ExposurePayload]) -> list[dict[str, Any]]:
    REQUEST_COUNTER.labels(endpoint=endpoint).inc()
    started = time.perf_counter()
    try:
        return _score(payloads)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Modèle indisponible : exécuter `dvc repro` pour produire les artefacts") from error
    finally:
        LATENCY_HISTOGRAM.labels(endpoint=endpoint).observe(time.perf_counter() - started)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score")
def score(payload: ExposurePayload) -> dict[str, Any]:
    return _timed("score", [payload])[0]


@app.post("/score_batch")
def score_batch(payload: BatchPayload) -> list[dict[str, Any]]:
    return _timed("score_batch", payload.exposures)


@app.get("/model_card")
def model_card() -> dict[str, Any]:
    engine = get_engine()
    card: dict[str, Any] = {
        "modele_pd": "Explainable Boosting Machine",
        "profil_de_variables": engine.pd_model.profile,
        "nombre_de_variables": len(engine.pd_model.features),
        "methode_de_calibrage": engine.calibrator.method,
        "scenarios": [{"nom": s.name, "poids": s.weight} for s in engine.scenarios],
        "variables_de_tiers_exclues": True,
        "avertissement": "Modèle entraîné sur données synthétiques : aligné sur IFRS 9 et Bâle III, non validé réglementairement.",
    }
    for name in ("pd_model", "backtest", "monitoring", "ecl"):
        path = METRICS_DIR / f"{name}.json"
        if path.exists():
            card[name] = json.loads(path.read_text(encoding="utf-8"))
    return card


@app.get("/metrics")
def metrics() -> Response:
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)

