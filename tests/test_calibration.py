"""Tests for arena_predictor/calibration.py."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'arena_predictor'))

import numpy as np
import pytest
from scipy import stats

from calibration import (
    GateResult,
    ShapeFit,
    compute_local_scale,
    compute_p_above,
    compute_p_beats_leader,
    compute_sigma,
    diagnose_scale_signal,
    fit_tail_shape_and_qhat,
)


RNG = np.random.default_rng(42)


def _make_correlated_residuals(n: int, slope: float = 1.0, noise: float = 0.3):
    """Generate y_nb_std positively correlated with |residuals|."""
    y_nb_std = RNG.uniform(5.0, 40.0, size=n)
    residuals = slope * y_nb_std * RNG.normal(0, 1, size=n) + RNG.normal(0, noise, size=n)
    predicted = RNG.uniform(1300, 1500, size=n)
    return y_nb_std, residuals, predicted


def test_diagnose_scale_signal_passes_on_correlated_data():
    y_nb_std, residuals, predicted = _make_correlated_residuals(127)
    result = diagnose_scale_signal(y_nb_std, residuals, predicted)
    assert result.passed is True
    assert result.spearman_all > 0.25
    assert result.decile_lift > 1.3
    assert result.n_all == 127
    assert isinstance(result.reason, str)


def test_diagnose_scale_signal_fails_on_uncorrelated_data():
    y_nb_std = RNG.uniform(5.0, 40.0, size=127)
    residuals = RNG.normal(0, 15.0, size=127)  # iid, no correlation with y_nb_std
    predicted = RNG.uniform(1300, 1500, size=127)
    result = diagnose_scale_signal(y_nb_std, residuals, predicted)
    assert result.passed is False


def test_diagnose_scale_signal_small_top_slice():
    """If |top| < 10, spearman_top is NaN and doesn't contribute to gate."""
    y_nb_std = RNG.uniform(5.0, 40.0, size=30)
    residuals = RNG.normal(0, 15.0, size=30)
    predicted = np.full(30, 1300.0)  # no models >= 1400
    result = diagnose_scale_signal(y_nb_std, residuals, predicted)
    assert np.isnan(result.spearman_top)
    assert result.n_top == 0


def test_diagnose_scale_signal_gate_rule_top_slice_wins():
    """If top slice passes but overall doesn't, gate passes."""
    y_nb_std = np.concatenate([RNG.uniform(5, 40, 100), RNG.uniform(5, 40, 27)])
    predicted = np.concatenate([np.full(100, 1300.0), np.full(27, 1450.0)])  # 27 "top" rows
    residuals = np.concatenate([
        RNG.normal(0, 15, 100),                              # uncorrelated
        2.0 * y_nb_std[100:] * RNG.normal(0, 1, 27),         # correlated in top slice
    ])
    result = diagnose_scale_signal(y_nb_std, residuals, predicted)
    assert result.n_top >= 10
    assert result.spearman_top >= 0.20
    assert result.passed is True


def test_compute_local_scale_applies_floor():
    y_nb_std = np.array([0.0, 5.0, 10.0, 20.0, 100.0])
    s_floor = 10.0
    out = compute_local_scale(y_nb_std, s_floor)
    np.testing.assert_array_equal(out, np.array([10.0, 10.0, 10.0, 20.0, 100.0]))


def test_compute_local_scale_no_floor_change_above_threshold():
    y_nb_std = np.array([15.0, 25.0, 35.0])
    s_floor = 10.0
    out = compute_local_scale(y_nb_std, s_floor)
    np.testing.assert_array_equal(out, y_nb_std)


def test_compute_local_scale_handles_nan():
    """NaN y_nb_std rows get the floor."""
    y_nb_std = np.array([np.nan, 5.0, np.nan, 20.0])
    s_floor = 10.0
    out = compute_local_scale(y_nb_std, s_floor)
    np.testing.assert_array_equal(out, np.array([10.0, 10.0, 10.0, 20.0]))


def test_fit_tail_shape_gate_passed_path():
    """With gate passed: t_df fit on r_i = e_i / s(x_i); q_hat anchors 95% coverage."""
    n = 200
    y_nb_std = RNG.uniform(10.0, 30.0, size=n)
    # residuals scaled by y_nb_std, drawn from t(df=8)
    r = stats.t.rvs(df=8, size=n, random_state=RNG)
    residuals = y_nb_std * r
    fit = fit_tail_shape_and_qhat(residuals, y_nb_std, gate_passed=True)
    assert 3.0 <= fit.t_df <= 200.0
    # t_df should be roughly near 8 (wide tolerance because n=200)
    assert 4.0 <= fit.t_df <= 25.0
    assert fit.q_hat > 0
    assert fit.s_floor > 0
    assert fit.fallback_used is False


def test_fit_tail_shape_fallback_path():
    """When gate is False: s(x) = 1, t_df fit on raw residuals."""
    residuals = RNG.normal(0, 15.0, size=150)
    y_nb_std = RNG.uniform(5.0, 40.0, size=150)  # ignored in fallback
    fit = fit_tail_shape_and_qhat(residuals, y_nb_std, gate_passed=False)
    assert fit.fallback_used is True
    assert fit.s_floor == 1.0
    # t_df fit on a near-Gaussian sample should clip near 200
    assert fit.t_df >= 10.0
    assert fit.q_hat > 0


def test_fit_tail_shape_coverage_anchor():
    """In the gate-passed limit with m=1, OOF 95% coverage should be ~95%."""
    n = 2000  # large sample to make the anchor behave
    y_nb_std = RNG.uniform(10.0, 30.0, size=n)
    r = stats.t.rvs(df=10, size=n, random_state=RNG)
    residuals = y_nb_std * r
    fit = fit_tail_shape_and_qhat(residuals, y_nb_std, gate_passed=True)
    s = compute_local_scale(y_nb_std, fit.s_floor)
    sigma = fit.q_hat * s
    t_crit = stats.t.ppf(0.975, fit.t_df)
    lower = -t_crit * sigma
    upper = t_crit * sigma
    coverage = float(np.mean((residuals >= lower) & (residuals <= upper)))
    # Expect ~95% coverage, within tolerance for n=2000
    assert 0.92 <= coverage <= 0.98


def test_compute_sigma_formula():
    """sigma(x) = m * q_hat * max(y_nb_std, s_floor)."""
    y_nb_std = np.array([0.0, 5.0, 10.0, 20.0])
    shape = ShapeFit(t_df=15.0, q_hat=1.2, s_floor=8.0, fallback_used=False)
    m = 1.08
    out = compute_sigma(y_nb_std, shape, m)
    expected = m * shape.q_hat * np.array([8.0, 8.0, 10.0, 20.0])
    np.testing.assert_allclose(out, expected)


def test_compute_sigma_fallback_shape():
    """With fallback (s_floor=1.0, y_nb effectively unused), sigma is near-constant."""
    y_nb_std = np.array([0.0, 5.0, 10.0])
    shape = ShapeFit(t_df=50.0, q_hat=20.0, s_floor=1.0, fallback_used=True)
    m = 1.0
    # In true fallback we'd also feed y_nb=ones to compute_sigma; but the
    # fallback path in predict.py is expected to pass y_nb=ones(n).
    out = compute_sigma(np.ones_like(y_nb_std), shape, m)
    np.testing.assert_allclose(out, np.full(3, m * shape.q_hat))
