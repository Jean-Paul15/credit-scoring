from __future__ import annotations

import numpy as np
from scipy.stats import binom

from credit_risk_graph.ifrs9.backtesting import binomial_pvalue, hosmer_lemeshow, jeffreys_pvalue, traffic_light, vasicek_pvalue
from credit_risk_graph.ifrs9.calibration import PDCalibrator
from credit_risk_graph.ifrs9.metrics import expected_calibration_error


def test_binomial_pvalue_matches_scipy_and_flags_underestimation():
    assert np.isclose(binomial_pvalue(1000, 50, 0.05), binom.sf(49, 1000, 0.05))
    assert binomial_pvalue(1000, 90, 0.05) < 0.001
    assert binomial_pvalue(1000, 0, 0.05) == 1.0


def test_correlation_makes_the_test_less_severe():
    independent = binomial_pvalue(10_000, 550, 0.05)
    correlated = vasicek_pvalue(10_000, 550, 0.05, rho=0.05)
    assert correlated > independent


def test_jeffreys_pvalue_small_when_observed_rate_far_above_pd():
    assert jeffreys_pvalue(1000, 90, 0.05) < 0.001
    assert jeffreys_pvalue(1000, 50, 0.05) > 0.3


def test_traffic_light_thresholds():
    assert traffic_light(0.20, 0.05, 0.0001) == "vert"
    assert traffic_light(0.01, 0.05, 0.0001) == "jaune"
    assert traffic_light(0.00001, 0.05, 0.0001) == "rouge"


def test_hosmer_lemeshow_accepts_calibrated_and_rejects_biased_predictions():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.30, 50_000)
    y = rng.binomial(1, p)
    assert hosmer_lemeshow(y, p)[1] > 0.01
    assert hosmer_lemeshow(y, np.clip(p * 0.5, 0, 1))[1] < 1e-6


def test_platt_calibration_reduces_calibration_error():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.01, 0.4, 40_000)
    y = rng.binomial(1, true_p)
    overconfident = np.clip(true_p * 1.8, 0, 0.99)
    calibrated = PDCalibrator("platt").fit(overconfident, y).predict(overconfident)
    assert expected_calibration_error(y, calibrated) < expected_calibration_error(y, overconfident)
