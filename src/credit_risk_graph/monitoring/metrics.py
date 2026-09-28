from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNTER = Counter("credit_risk_api_requests_total", "Nombre total de requêtes de scoring.", ["endpoint"])
LATENCY_HISTOGRAM = Histogram("credit_risk_api_latency_seconds", "Latence des endpoints principaux.", ["endpoint"])


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
