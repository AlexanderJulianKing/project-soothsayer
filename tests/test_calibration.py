"""Tests for arena_predictor/calibration.py."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'arena_predictor'))

import numpy as np
import pytest

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
