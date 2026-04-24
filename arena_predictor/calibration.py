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
    """Gate: does y_nb_std rank difficulty (|residual|) well enough to be worth using?

    Soft rule: pass if top-slice is monotone, OR overall correlation is decent AND
    decile lift > 1.3. Either metric passing is enough; both failing = fall back.
    """
    eps = 1e-6
    abs_e = np.abs(oof_residuals)

    # Filter to finite rows (drop NaN residuals from folds that never covered that row)
    finite = np.isfinite(y_nb_std_oof) & np.isfinite(abs_e) & np.isfinite(predicted_scores)
    y_nb = y_nb_std_oof[finite]
    e = abs_e[finite]
    mu = predicted_scores[finite]
    n_all = int(finite.sum())

    # --- spearman_all with NaN/zero drop ---
    mask_all = (y_nb > 0) & (e > 0)
    if mask_all.sum() >= 10:
        spearman_all = float(stats.spearmanr(y_nb[mask_all], e[mask_all]).correlation)
    else:
        spearman_all = float("nan")

    # --- spearman_top on rows with predicted >= top_threshold ---
    top_mask = mu >= top_threshold
    n_top = int(top_mask.sum())
    if n_top >= 10:
        top_y, top_e = y_nb[top_mask], e[top_mask]
        top_mask2 = (top_y > 0) & (top_e > 0)
        if top_mask2.sum() >= 10:
            spearman_top = float(stats.spearmanr(top_y[top_mask2], top_e[top_mask2]).correlation)
        else:
            spearman_top = float("nan")
    else:
        spearman_top = float("nan")

    # --- log-log linear fit ---
    if mask_all.sum() >= 10:
        log_y = np.log(y_nb[mask_all] + eps)
        log_e = np.log(e[mask_all] + eps)
        slope, intercept = np.polyfit(log_y, log_e, 1)
        pred = slope * log_y + intercept
        ss_res = np.sum((log_e - pred) ** 2)
        ss_tot = np.sum((log_e - log_e.mean()) ** 2)
        log_log_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        log_log_slope = float(slope)
    else:
        log_log_slope, log_log_r2 = float("nan"), float("nan")

    # --- decile lift: mean(|e|) top y_nb decile / bottom y_nb decile ---
    if n_all >= 20:
        order = np.argsort(y_nb)
        n_dec = max(1, n_all // 10)
        bottom_mean = float(e[order[:n_dec]].mean())
        top_mean = float(e[order[-n_dec:]].mean())
        decile_lift = float(top_mean / bottom_mean) if bottom_mean > 0 else float("nan")
    else:
        decile_lift = float("nan")

    # --- Gate ---
    passed_top = (not np.isnan(spearman_top)) and (spearman_top >= 0.20)
    passed_overall = (
        (not np.isnan(spearman_all)) and (spearman_all >= 0.25)
        and (not np.isnan(decile_lift)) and (decile_lift >= 1.3)
    )
    passed = bool(passed_top or passed_overall)

    if passed_top:
        reason = f"top-slice monotone (spearman_top={spearman_top:.3f} on n_top={n_top})"
    elif passed_overall:
        reason = (f"overall monotone (spearman_all={spearman_all:.3f}, "
                  f"decile_lift={decile_lift:.3f})")
    else:
        reason = (f"no monotonicity (spearman_top={spearman_top}, "
                  f"spearman_all={spearman_all}, decile_lift={decile_lift})")

    return GateResult(
        passed=passed,
        reason=reason,
        spearman_all=spearman_all,
        spearman_top=spearman_top,
        log_log_slope=log_log_slope,
        log_log_r2=log_log_r2,
        decile_lift=decile_lift,
        n_all=n_all,
        n_top=n_top,
    )


def compute_local_scale(
    y_nb_std: np.ndarray,
    s_floor: float,
) -> np.ndarray:
    """s(x) = max(y_nb_std(x), s_floor). NaN inputs are treated as 0 (i.e. get floored)."""
    y = np.where(np.isnan(y_nb_std), 0.0, y_nb_std)
    return np.maximum(y, s_floor)


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
