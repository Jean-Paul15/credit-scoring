from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

from credit_risk_graph.api import main
from credit_risk_graph.ifrs9.calibration import PDCalibrator
from credit_risk_graph.ifrs9.lgd import LGDModel
from credit_risk_graph.ifrs9.pd_model import PDModel
from credit_risk_graph.ifrs9.portfolio import ECLEngine
from credit_risk_graph.ifrs9.scenarios import Scenario

PD_FEATURES = ["current_dpd", "macro_stress", "debt_service_ratio"]
LGD_FEATURES = ["collateral_type", "macro_stress", "current_dpd"]
PARAMS = {
    "staging": {"stage2_dpd": 30, "stage3_dpd": 90, "sicr_relative_multiplier": 1.7, "sicr_absolute_delta": 0.01},
    "ecl": {"max_horizon_months": 60},
    "ead": {"ccf": 1.0},
    "scenarios": {"horizon_years": 5, "reversion_years": 3},
}
LOAN = {
    "current_dpd": 10,
    "restructured_flag": 0,
    "origination_pd": 0.05,
    "outstanding_balance": 500_000.0,
    "interest_rate": 0.14,
    "installment_amount": 25_000.0,
    "remaining_term_months": 24,
    "macro_stress": 0.2,
    "sector_stress": 0.22,
    "collateral_type": "Aucun",
    "product_type": "PretPersonnel",
    "region": "Lome",
    "debt_service_ratio": 0.3,
}


@pytest.fixture(scope="module")
def engine() -> ECLEngine:
    rng = np.random.default_rng(0)
    n = 3000
    frame = pd.DataFrame(
        {
            "current_dpd": rng.integers(0, 60, n),
            "macro_stress": rng.uniform(0.1, 0.4, n),
            "debt_service_ratio": rng.uniform(0, 0.6, n),
            "collateral_type": rng.choice(["Aucun", "Foncier"], n),
        }
    )
    y = rng.binomial(1, 1 / (1 + np.exp(-(-3 + 0.03 * frame["current_dpd"] + 3 * frame["macro_stress"]))))
    ebm = ExplainableBoostingClassifier(feature_names=PD_FEATURES, interactions=0, outer_bags=2, max_rounds=60, random_state=0).fit(frame[PD_FEATURES], y)
    lgd = ExplainableBoostingRegressor(feature_names=LGD_FEATURES, interactions=0, outer_bags=2, max_rounds=60, random_state=0)
    lgd.fit(frame[LGD_FEATURES], 0.4 + 0.3 * (frame["collateral_type"] == "Aucun") + 0.2 * frame["macro_stress"])
    raw = ebm.predict_proba(frame[PD_FEATURES])[:, 1]
    scenarios = [Scenario("baseline", 0.5, 0.0), Scenario("adverse", 0.3, 0.1), Scenario("favourable", 0.2, -0.05)]
    return ECLEngine(PDModel("test", PD_FEATURES, ebm), PDCalibrator("platt").fit(raw, y), np.quantile(raw, [0.25, 0.5, 0.75]), LGDModel(LGD_FEATURES, lgd), scenarios, PARAMS)


@pytest.fixture()
def client(engine: ECLEngine, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(main, "get_engine", lambda: engine)
    return TestClient(main.app)


def test_health(client: TestClient):
    assert client.get("/health").json() == {"status": "ok"}


def test_score_returns_pd_stage_ecl_and_exact_contributions(client: TestClient):
    response = client.post("/score", json={"features": LOAN})
    assert response.status_code == 200
    body = response.json()
    assert 0 < body["pd_12m"] < 1
    assert body["stage"] in (1, 2, 3)
    assert body["ecl_ponderee"] >= 0
    assert set(body["ecl_par_scenario"]) == {"baseline", "adverse", "favourable"}
    assert len(body["contributions_principales"]) <= 5


def test_adverse_scenario_is_not_cheaper_than_favourable(client: TestClient):
    body = client.post("/score", json={"features": LOAN}).json()
    assert body["ecl_par_scenario"]["adverse"] >= body["ecl_par_scenario"]["favourable"]


def test_defaulted_loan_is_stage_3_with_lgd_times_ead(client: TestClient):
    body = client.post("/score", json={"features": {**LOAN, "current_dpd": 120}}).json()
    assert body["stage"] == 3
    assert body["ecl_par_scenario"]["baseline"] == pytest.approx(body["lgd_baseline"] * body["ead"], rel=1e-6)


def test_missing_required_field_is_rejected(client: TestClient):
    incomplete = {k: v for k, v in LOAN.items() if k != "current_dpd"}
    assert client.post("/score", json={"features": incomplete}).status_code == 422


def test_batch(client: TestClient):
    response = client.post("/score_batch", json={"exposures": [{"features": LOAN}, {"features": {**LOAN, "current_dpd": 45}}]})
    assert response.status_code == 200 and len(response.json()) == 2
