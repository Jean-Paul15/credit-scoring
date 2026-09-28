from __future__ import annotations

import numpy as np

from credit_risk_graph.ifrs9.ead import exposure_paths
from credit_risk_graph.ifrs9.ecl import expected_credit_loss, marginal_pd, probability_weighted


def _flat_case(pd_12m: float, lgd: float, term: int, horizon: int = 60):
    pd_by_year = np.full((1, 5), pd_12m)
    lgd_by_year = np.full((1, 5), lgd)
    exposure = np.where(np.arange(horizon)[None, :] < term, 100.0, 0.0)
    return pd_by_year, lgd_by_year, exposure


def test_marginal_pd_sums_to_cumulative_default_probability():
    marginal = marginal_pd(np.full((1, 5), 0.12), 24)
    assert np.isclose(marginal[0, :12].sum(), 0.12)
    assert np.isclose(marginal[0].sum(), 1 - 0.88**2)


def test_stage1_is_twelve_month_ecl_without_discounting():
    pd_by_year, lgd_by_year, exposure = _flat_case(0.12, 0.45, term=36)
    ecl = expected_credit_loss(pd_by_year, lgd_by_year, exposure, np.zeros(1), np.array([1]), 60)
    assert np.isclose(ecl[0], 0.12 * 0.45 * 100.0)


def test_stage2_is_lifetime_ecl_over_remaining_term():
    pd_by_year, lgd_by_year, exposure = _flat_case(0.12, 0.45, term=36)
    ecl = expected_credit_loss(pd_by_year, lgd_by_year, exposure, np.zeros(1), np.array([2]), 60)
    assert np.isclose(ecl[0], (1 - 0.88**3) * 0.45 * 100.0)


def test_stage3_is_lgd_times_ead():
    pd_by_year, lgd_by_year, exposure = _flat_case(0.5, 0.6, term=12)
    ecl = expected_credit_loss(pd_by_year, lgd_by_year, exposure, np.zeros(1), np.array([3]), 60)
    assert np.isclose(ecl[0], 0.6 * 100.0)


def test_discounting_reduces_ecl():
    pd_by_year, lgd_by_year, exposure = _flat_case(0.12, 0.45, term=36)
    undiscounted = expected_credit_loss(pd_by_year, lgd_by_year, exposure, np.zeros(1), np.array([2]), 60)
    discounted = expected_credit_loss(pd_by_year, lgd_by_year, exposure, np.full(1, 0.15), np.array([2]), 60)
    assert discounted[0] < undiscounted[0]


def test_probability_weighting():
    weighted = probability_weighted({"a": np.array([10.0]), "b": np.array([20.0])}, {"a": 0.75, "b": 0.25})
    assert np.isclose(weighted[0], 12.5)


def test_exposure_paths_linear_without_interest_and_zero_after_maturity():
    paths = exposure_paths(np.array([120.0]), np.array([0.0]), np.array([10.0]), np.array([6]), 12)
    assert paths[0, 0] == 120.0
    assert np.isclose(paths[0, 5], 70.0)
    assert np.all(paths[0, 6:] == 0.0)
