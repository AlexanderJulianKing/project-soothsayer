"""Calibration for the arena ELO predictor.

Implements OOF normalized conformal-style shape fit (local scale + tail t_df
+ empirical quantile anchor) plus walk-forward level correction.

All functions are pure (no I/O, no global state). See
docs/superpowers/specs/2026-04-24-predictor-calibration-design.md for design.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import optimize
from scipy import stats


@dataclass
class GateResult:
    passed: bool
    reason: str
    spearman_all: float
    spearman_top: float
    log_log_slope: float
    log_log_r2: float
    decile_lift: float
    n_all: int
    n_top: int


@dataclass
class ShapeFit:
    t_df: float
    q_hat: float
    s_floor: float
    fallback_used: bool


def diagnose_scale_signal(
    y_nb_std_oof: np.ndarray,
    oof_residuals: np.ndarray,
    predicted_scores: np.ndarray,
    top_threshold: float = 1400.0,
) -> GateResult:
    raise NotImplementedError


def compute_local_scale(
    y_nb_std: np.ndarray,
    s_floor: float,
) -> np.ndarray:
    raise NotImplementedError


def fit_tail_shape_and_qhat(
    oof_residuals: np.ndarray,
    y_nb_std_oof: np.ndarray,
    gate_passed: bool,
) -> ShapeFit:
    raise NotImplementedError


def compute_sigma(
    y_nb_std: np.ndarray,
    shape: ShapeFit,
    m: float,
) -> np.ndarray:
    raise NotImplementedError


def compute_p_beats_leader(
    mu: np.ndarray,
    sigma: np.ndarray,
    t_df: float,
    max_leader: float,
    train_mask: np.ndarray,
) -> np.ndarray:
    raise NotImplementedError


def compute_p_above(
    mu: np.ndarray,
    sigma: np.ndarray,
    t_df: float,
    threshold: float,
) -> np.ndarray:
    """Shared t-CDF helper: P(score > threshold) under t(df, loc=mu, scale=sigma)."""
    raise NotImplementedError
