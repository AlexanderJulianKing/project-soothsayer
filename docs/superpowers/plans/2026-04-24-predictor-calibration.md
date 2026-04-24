# Predictor Calibration Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the constant-sigma grouped conformal emission in `arena_predictor/predict.py` with per-model calibrated predictive distributions, emit `p_beats_leader` (`P(score > current leader's score)`) as the primary calibrated probability output, and fit a global scale correction on walk-forward residuals.

**Architecture:** Three-stage calibration pipeline: (A) diagnostic gate on `y_nb_std` vs `|OOF residual|` monotonicity; (B) OOF normalized conformal-style shape fit (local scale `s(x) = max(y_nb_std, s_floor)`, tail `t_df`, empirical quantile anchor `q_hat`); (C) walk-forward level correction (one scalar `m`). Calibration logic lives in a new module `arena_predictor/calibration.py`. Walk-forward honest-eval artifact production lives in a new script `arena_predictor/walkforward_calibration.py`.

**Tech Stack:** Python 3.10+, numpy, pandas, scipy (`scipy.stats.t`, `scipy.optimize.minimize_scalar`, `scipy.stats.kstest`, `scipy.stats.binomtest`, `scipy.stats.spearmanr`), scikit-learn (`PLSRegression`, `PCA`, `StandardScaler`), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-24-predictor-calibration-design.md`

---

## File Structure

**New files:**
- `arena_predictor/calibration.py` — All calibration functions: gate, local scale, tail shape fit, sigma computation, probability formulas, fallback path. Pure functions only; no I/O, no global state.
- `arena_predictor/walkforward_calibration.py` — Walk-forward honest-eval script. Imports helpers from `_walkforward_honest.py` where possible; runs its own loop with nested LOO + per-step calibration fits + m-fit + diagnostics.
- `tests/test_calibration.py` — Unit tests for `calibration.py`.

**Modified files:**
- `arena_predictor/predict.py`
  - `predict_adaptive_knn` (line ~632): add `y_nb_std` to the return tuple.
  - `fit_and_predict_knn` (line ~726): persist `y_nb_std_oof` and `y_nb_std_final` vectors through the OOF + final loops; add them to the returned dict.
  - Main flow (lines ~1640–1725): replace the grouped-conformal block with calls into `calibration.py`; rename output column `num_one_prob` → `p_beats_leader`; change `sigma_hat` semantics from half-width to t-scale parameter; emit new `calibration_diagnostics.csv`.
  - Argparse block (search for `--margin`): add `--walkforward_calibration_path` flag.
- `arena_predictor/_walkforward_cv.py` (line ~175): has its own local `predict_adaptive_knn` definition; the update is needed only if we break that signature. Leave its local copy unchanged — the update is to `predict.py`'s version only. _walkforward_cv.py's imports already don't import from predict.py for this function, so no change needed.
- `arena_predictor/_walkforward_honest.py` (line ~179): updates the unpacked return of `predict_adaptive_knn` from `p, _, _` to `p, _, _, _` (one more element). No behavior change.

---

## Task 1: Scaffold calibration module + test file

**Files:**
- Create: `arena_predictor/calibration.py`
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Create empty calibration module with public API stubs**

Create `arena_predictor/calibration.py`:

```python
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
```

- [ ] **Step 2: Create test file skeleton**

Create `tests/test_calibration.py`:

```python
"""Tests for arena_predictor/calibration.py."""
from __future__ import annotations

import numpy as np
import pytest

from arena_predictor.calibration import (
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
```

- [ ] **Step 3: Verify import works**

Run: `python3 -c "from arena_predictor.calibration import diagnose_scale_signal"`
Expected: No output (import succeeds).

- [ ] **Step 4: Run the test file to verify collection**

Run: `python3 -m pytest tests/test_calibration.py --collect-only -q`
Expected: `0 tests collected` (no tests defined yet, but no import errors).

- [ ] **Step 5: Commit**

```bash
git add arena_predictor/calibration.py tests/test_calibration.py
git commit -m "Scaffold calibration module + test file"
```

---

## Task 2: Implement diagnose_scale_signal

**Files:**
- Modify: `arena_predictor/calibration.py` (replace `diagnose_scale_signal` stub)
- Modify: `tests/test_calibration.py` (add tests)

- [ ] **Step 1: Write failing test — metrics computation**

Append to `tests/test_calibration.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_calibration.py -v`
Expected: 4 tests fail with `NotImplementedError`.

- [ ] **Step 3: Implement diagnose_scale_signal**

Replace the stub in `arena_predictor/calibration.py` with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_calibration.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add arena_predictor/calibration.py tests/test_calibration.py
git commit -m "Implement diagnose_scale_signal with soft gate rule"
```

---

## Task 3: Implement local scale function

**Files:**
- Modify: `arena_predictor/calibration.py` (replace `compute_local_scale` stub)
- Modify: `tests/test_calibration.py` (add tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_calibration.py`:

```python
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
```

- [ ] **Step 2: Run tests — verify they fail with NotImplementedError**

Run: `python3 -m pytest tests/test_calibration.py::test_compute_local_scale_applies_floor -v`
Expected: FAIL with NotImplementedError.

- [ ] **Step 3: Implement compute_local_scale**

Replace the stub in `arena_predictor/calibration.py`:

```python
def compute_local_scale(
    y_nb_std: np.ndarray,
    s_floor: float,
) -> np.ndarray:
    """s(x) = max(y_nb_std(x), s_floor). NaN inputs are treated as 0 (i.e. get floored)."""
    y = np.where(np.isnan(y_nb_std), 0.0, y_nb_std)
    return np.maximum(y, s_floor)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_calibration.py -v`
Expected: All tests pass (7 total now).

- [ ] **Step 5: Commit**

```bash
git add arena_predictor/calibration.py tests/test_calibration.py
git commit -m "Implement compute_local_scale with floor + NaN handling"
```

---

## Task 4: Implement fit_tail_shape_and_qhat (non-circular)

**Files:**
- Modify: `arena_predictor/calibration.py` (replace `fit_tail_shape_and_qhat` stub)
- Modify: `tests/test_calibration.py` (add tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_calibration.py`:

```python
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
```

- [ ] **Step 2: Run failing tests**

Run: `python3 -m pytest tests/test_calibration.py -v -k "tail_shape"`
Expected: 3 tests FAIL with NotImplementedError.

- [ ] **Step 3: Implement fit_tail_shape_and_qhat**

Replace the stub:

```python
def fit_tail_shape_and_qhat(
    oof_residuals: np.ndarray,
    y_nb_std_oof: np.ndarray,
    gate_passed: bool,
) -> ShapeFit:
    """Fit tail t_df and empirical 95%-anchor q_hat on OOF residuals.

    Non-circular: t_df is fit on r_i = e_i / s(x_i) (or raw e_i in fallback),
    then q_hat anchors the empirical q95 of |r| to t_ppf(0.975, t_df).

    If gate_passed is False: s(x) = 1, so r_i = e_i.
    """
    finite = np.isfinite(oof_residuals) & np.isfinite(y_nb_std_oof)
    e = oof_residuals[finite]
    y_nb = y_nb_std_oof[finite]

    if gate_passed:
        # s_floor = p25 of y_nb_std
        s_floor = float(np.percentile(y_nb[y_nb > 0], 25)) if (y_nb > 0).any() else 1.0
        s = np.maximum(np.where(np.isnan(y_nb), 0.0, y_nb), s_floor)
        r = e / s
        fallback = False
    else:
        s_floor = 1.0
        r = e.copy()
        fallback = True

    # t_df fit: scipy.stats.t.fit with floc=0, discard scale
    try:
        t_df_fit, _loc, _scale = stats.t.fit(r, floc=0)
        t_df = float(np.clip(t_df_fit, 3.0, 200.0))
    except Exception:
        t_df = 200.0  # Gaussian fallback

    # Empirical q95 anchor: q_hat * t_ppf(0.975, t_df) = q95(|r|)
    q95 = float(np.quantile(np.abs(r), 0.95, method="linear"))
    t_crit = float(stats.t.ppf(0.975, t_df))
    q_hat = q95 / t_crit if t_crit > 0 else q95

    return ShapeFit(t_df=t_df, q_hat=q_hat, s_floor=s_floor, fallback_used=fallback)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_calibration.py -v -k "tail_shape"`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add arena_predictor/calibration.py tests/test_calibration.py
git commit -m "Implement fit_tail_shape_and_qhat with non-circular t_df + empirical anchor"
```

---

## Task 5: Implement compute_sigma

**Files:**
- Modify: `arena_predictor/calibration.py` (replace `compute_sigma` stub)
- Modify: `tests/test_calibration.py` (add tests)

- [ ] **Step 1: Write failing test**

Append to `tests/test_calibration.py`:

```python
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
```

- [ ] **Step 2: Run failing tests**

Run: `python3 -m pytest tests/test_calibration.py -v -k "compute_sigma"`
Expected: 2 tests FAIL.

- [ ] **Step 3: Implement compute_sigma**

Replace the stub:

```python
def compute_sigma(
    y_nb_std: np.ndarray,
    shape: ShapeFit,
    m: float,
) -> np.ndarray:
    """sigma(x) = m * q_hat * s(x), where s(x) = max(y_nb_std(x), s_floor).

    In the fallback path, callers should pass y_nb_std = np.ones(n) so that
    compute_local_scale returns s_floor (=1.0) everywhere, yielding a constant sigma.
    """
    s = compute_local_scale(y_nb_std, shape.s_floor)
    return m * shape.q_hat * s
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_calibration.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add arena_predictor/calibration.py tests/test_calibration.py
git commit -m "Implement compute_sigma"
```

---

## Task 6: Implement compute_p_beats_leader + compute_p_above

**Files:**
- Modify: `arena_predictor/calibration.py` (replace `compute_p_beats_leader` + `compute_p_above` stubs)
- Modify: `tests/test_calibration.py` (add tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_calibration.py`:

```python
def test_compute_p_above_formula():
    """P(score > threshold) = 1 - t_cdf((threshold - mu)/sigma, df)."""
    mu = np.array([1500.0, 1490.0])
    sigma = np.array([20.0, 20.0])
    t_df = 200.0  # effectively Gaussian
    threshold = 1500.0
    p = compute_p_above(mu, sigma, t_df, threshold)
    # mu == threshold => P(score > threshold) = 0.5
    np.testing.assert_allclose(p[0], 0.5, atol=1e-3)
    # mu 10 below threshold, sigma 20 => P(score > threshold) ≈ norm.sf(0.5)
    from scipy.stats import norm
    np.testing.assert_allclose(p[1], float(norm.sf(0.5)), atol=1e-2)


def test_compute_p_above_tight_sigma():
    """Very tight sigma: step function."""
    mu = np.array([1550.0, 1450.0])
    sigma = np.array([0.01, 0.01])
    t_df = 10.0
    threshold = 1500.0
    p = compute_p_above(mu, sigma, t_df, threshold)
    assert p[0] > 0.99  # far above threshold
    assert p[1] < 0.01  # far below threshold


def test_compute_p_beats_leader_nans_training_rows():
    mu = np.array([1505.0, 1495.0, 1480.0])
    sigma = np.array([20.0, 20.0, 20.0])
    t_df = 100.0
    max_leader = 1500.0
    train_mask = np.array([True, False, True])  # rows 0 and 2 are training
    p = compute_p_beats_leader(mu, sigma, t_df, max_leader, train_mask)
    assert np.isnan(p[0])
    assert np.isnan(p[2])
    assert not np.isnan(p[1])
    # Row 1: mu=1495, sigma=20, leader=1500 => slightly below 0.5
    assert 0.3 < p[1] < 0.5


def test_compute_p_beats_leader_empty_test_set():
    """All rows are training: returns all-NaN array of correct length."""
    mu = np.array([1500.0, 1490.0])
    sigma = np.array([20.0, 20.0])
    train_mask = np.array([True, True])
    p = compute_p_beats_leader(mu, sigma, 100.0, 1500.0, train_mask)
    assert len(p) == 2
    assert np.all(np.isnan(p))
```

- [ ] **Step 2: Run failing tests**

Run: `python3 -m pytest tests/test_calibration.py -v -k "p_above or p_beats"`
Expected: 4 tests FAIL.

- [ ] **Step 3: Implement compute_p_above + compute_p_beats_leader**

Replace the stubs:

```python
def compute_p_above(
    mu: np.ndarray,
    sigma: np.ndarray,
    t_df: float,
    threshold: float,
) -> np.ndarray:
    """P(score > threshold) under t-distribution with loc=mu, scale=sigma, df=t_df."""
    scale = np.where(sigma <= 1e-12, 1e-12, sigma)
    z = (threshold - mu) / scale
    return 1.0 - stats.t.cdf(z, df=t_df)


def compute_p_beats_leader(
    mu: np.ndarray,
    sigma: np.ndarray,
    t_df: float,
    max_leader: float,
    train_mask: np.ndarray,
) -> np.ndarray:
    """P(score > max_leader), NaN on training rows (where the event is meaningless)."""
    p = compute_p_above(mu, sigma, t_df, max_leader)
    p = p.astype(float).copy()
    p[train_mask] = np.nan
    return p
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_calibration.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add arena_predictor/calibration.py tests/test_calibration.py
git commit -m "Implement compute_p_above + compute_p_beats_leader with training-row NaN rule"
```

---

## Task 7: Add fallback-path integration test

**Files:**
- Modify: `tests/test_calibration.py`

This is a pure unit-test task — verifies that the combined functions produce correct behavior in the fallback case where the gate fails.

- [ ] **Step 1: Write failing test**

Append to `tests/test_calibration.py`:

```python
def test_fallback_path_produces_constant_sigma():
    """End-to-end fallback: gate fails -> s(x)=1 -> sigma_hat constant across rows."""
    n = 150
    y_nb_std = RNG.uniform(5.0, 40.0, size=n)
    residuals = RNG.normal(0, 15.0, size=n)  # uncorrelated with y_nb_std
    predicted = RNG.uniform(1300, 1500, size=n)

    gate = diagnose_scale_signal(y_nb_std, residuals, predicted)
    assert gate.passed is False  # precondition

    shape = fit_tail_shape_and_qhat(residuals, y_nb_std, gate_passed=gate.passed)
    assert shape.fallback_used is True
    assert shape.s_floor == 1.0

    # In fallback, caller passes y_nb_std = np.ones(n) to compute_sigma
    sigma = compute_sigma(np.ones(n), shape, m=1.0)
    # sigma should be constant
    assert np.allclose(sigma, sigma[0])
    # Approximately equal to q_hat
    assert abs(sigma[0] - shape.q_hat) < 1e-6


def test_gate_pass_path_produces_varying_sigma():
    n = 200
    y_nb_std = RNG.uniform(10.0, 30.0, size=n)
    r = stats.t.rvs(df=8, size=n, random_state=RNG)
    residuals = y_nb_std * r
    predicted = RNG.uniform(1300, 1500, size=n)

    gate = diagnose_scale_signal(y_nb_std, residuals, predicted)
    shape = fit_tail_shape_and_qhat(residuals, y_nb_std, gate_passed=gate.passed)
    sigma = compute_sigma(y_nb_std, shape, m=1.0)
    # sigma should vary with y_nb_std
    assert np.std(sigma) > 0.1 * np.mean(sigma)
```

- [ ] **Step 2: Run tests — verify they pass (this exercises existing functions)**

Run: `python3 -m pytest tests/test_calibration.py -v`
Expected: All tests pass (these tests exercise the already-implemented public API).

- [ ] **Step 3: Commit**

```bash
git add tests/test_calibration.py
git commit -m "Add end-to-end fallback-path + gate-pass integration tests"
```

---

## Task 8: Expose y_nb_std from predict_adaptive_knn

**Files:**
- Modify: `arena_predictor/predict.py` (function at line ~632)
- Modify: `arena_predictor/_walkforward_honest.py` (line ~179)

- [ ] **Step 1: Update `predict_adaptive_knn` to return y_nb_std as 4th element**

In `arena_predictor/predict.py`, modify the function signature and body. Find:

```python
    Returns (prediction, std_estimate, k_used).
    """
```

Change to:

```python
    Returns (prediction, std_estimate, k_used, y_nb_std).
    y_nb_std is the std of the neighborhood's y-values (for calibration).
    """
```

Find the return statement at the end of the function:

```python
    return p_corrected, std_est, k
```

Change to:

```python
    return p_corrected, std_est, k, float(np.std(y_nb))
```

Also update the return type hint at the top of the signature:

```python
) -> Tuple[float, float, int]:
```

Change to:

```python
) -> Tuple[float, float, int, float]:
```

- [ ] **Step 2: Update OOF caller in `fit_and_predict_knn` (line ~772)**

Find:

```python
            p, _, _ = predict_adaptive_knn(
                Xtr, ytr, Xva[vi:vi + 1],
                power_alpha=power_alpha, power_C=power_C,
                max_k=max_k, min_k=min_k, bw_pct=bw_pct,
            )
```

Change to:

```python
            p, _, _, _ = predict_adaptive_knn(
                Xtr, ytr, Xva[vi:vi + 1],
                power_alpha=power_alpha, power_C=power_C,
                max_k=max_k, min_k=min_k, bw_pct=bw_pct,
            )
```

(We'll wire in y_nb_std capture in Task 9; this unblocks compilation first.)

- [ ] **Step 3: Update final-loop caller in `fit_and_predict_knn` (line ~808)**

Find:

```python
        p, s, k = predict_adaptive_knn(
            X_train_sc, y_train, X_all_sc[i:i + 1],
            power_alpha=power_alpha, power_C=power_C,
            max_k=max_k, min_k=min_k, bw_pct=bw_pct,
        )
```

Change to:

```python
        p, s, k, _ = predict_adaptive_knn(
            X_train_sc, y_train, X_all_sc[i:i + 1],
            power_alpha=power_alpha, power_C=power_C,
            max_k=max_k, min_k=min_k, bw_pct=bw_pct,
        )
```

- [ ] **Step 4: Update caller in `_walkforward_honest.py` (line ~179)**

Find:

```python
        p, _, _ = predict_adaptive_knn(Xtr, y[:i], Xte,
                                       max_k=min(80, i), min_k=min(20, i))
```

Change to:

```python
        p, _, _, _ = predict_adaptive_knn(Xtr, y[:i], Xte,
                                          max_k=min(80, i), min_k=min(20, i))
```

- [ ] **Step 5: Smoke test — import predict.py to verify no syntax errors**

Run: `python3 -c "import sys; sys.path.insert(0, 'arena_predictor'); import predict"`
Expected: No output (success).

- [ ] **Step 6: Check other callers not missed**

Run: `grep -n "predict_adaptive_knn" arena_predictor/*.py arena_predictor/_*.py tests/*.py 2>/dev/null`
Expected: all callsites either unpack 4 values or reference the function's definition. Note: `arena_predictor/_walkforward_cv.py` has its OWN local definition of `predict_adaptive_knn` (it's a sweep script, not a production caller) — leave it alone; it doesn't import from predict.py.

- [ ] **Step 7: Commit**

```bash
git add arena_predictor/predict.py arena_predictor/_walkforward_honest.py
git commit -m "Expose y_nb_std from predict_adaptive_knn as 4th return value"
```

---

## Task 9: Persist y_nb_std through fit_and_predict_knn

**Files:**
- Modify: `arena_predictor/predict.py` (function at line ~726)

- [ ] **Step 1: Add y_nb_std accumulators to the OOF loop**

Find the OOF accumulator init block around line ~750:

```python
    oof_preds_sum = np.zeros(n_train)
    oof_counts = np.zeros(n_train)
    oof_folds = np.full(n_train, -1, dtype=int)
```

Add:

```python
    oof_preds_sum = np.zeros(n_train)
    oof_counts = np.zeros(n_train)
    oof_folds = np.full(n_train, -1, dtype=int)
    oof_y_nb_std_sum = np.zeros(n_train)  # NEW: average across folds
```

- [ ] **Step 2: Capture y_nb_std in the OOF inner loop**

Find (line ~772, already updated in Task 8):

```python
            p, _, _, _ = predict_adaptive_knn(
                Xtr, ytr, Xva[vi:vi + 1],
                power_alpha=power_alpha, power_C=power_C,
                max_k=max_k, min_k=min_k, bw_pct=bw_pct,
            )
            oof_preds_sum[va_i] += p
            oof_counts[va_i] += 1
```

Change to:

```python
            p, _, _, y_nb_std = predict_adaptive_knn(
                Xtr, ytr, Xva[vi:vi + 1],
                power_alpha=power_alpha, power_C=power_C,
                max_k=max_k, min_k=min_k, bw_pct=bw_pct,
            )
            oof_preds_sum[va_i] += p
            oof_counts[va_i] += 1
            oof_y_nb_std_sum[va_i] += y_nb_std
```

- [ ] **Step 3: Compute the per-row average y_nb_std after the OOF loop**

Find:

```python
    oof_preds = np.where(oof_counts > 0, oof_preds_sum / oof_counts, np.nan)
    oof_valid = oof_counts > 0
    oof_rmse = float(np.sqrt(np.nanmean((oof_preds[oof_valid] - y_train[oof_valid]) ** 2)))
```

Add one line:

```python
    oof_preds = np.where(oof_counts > 0, oof_preds_sum / oof_counts, np.nan)
    oof_y_nb_std = np.where(oof_counts > 0, oof_y_nb_std_sum / oof_counts, np.nan)  # NEW
    oof_valid = oof_counts > 0
    oof_rmse = float(np.sqrt(np.nanmean((oof_preds[oof_valid] - y_train[oof_valid]) ** 2)))
```

- [ ] **Step 4: Add y_nb_std accumulator for the final-fit loop**

Find the final-fit init block (line ~800):

```python
    mu = np.full(n_all, np.nan)
    std = np.full(n_all, np.nan)
    ks_used = np.zeros(n_all, dtype=int)
```

Add:

```python
    mu = np.full(n_all, np.nan)
    std = np.full(n_all, np.nan)
    ks_used = np.zeros(n_all, dtype=int)
    y_nb_std_final = np.full(n_all, np.nan)  # NEW
```

- [ ] **Step 5: Capture y_nb_std in the final-fit inner loop**

Find:

```python
        p, s, k, _ = predict_adaptive_knn(
            X_train_sc, y_train, X_all_sc[i:i + 1],
            power_alpha=power_alpha, power_C=power_C,
            max_k=max_k, min_k=min_k, bw_pct=bw_pct,
        )
        mu[i] = p
        std[i] = s
        ks_used[i] = k
```

Change to:

```python
        p, s, k, y_nb_std = predict_adaptive_knn(
            X_train_sc, y_train, X_all_sc[i:i + 1],
            power_alpha=power_alpha, power_C=power_C,
            max_k=max_k, min_k=min_k, bw_pct=bw_pct,
        )
        mu[i] = p
        std[i] = s
        ks_used[i] = k
        y_nb_std_final[i] = y_nb_std
```

- [ ] **Step 6: Add `y_nb_std_oof` and `y_nb_std_final` to returned dict**

Find the return dict at the end of `fit_and_predict_knn` (around line ~825):

```python
    return {
        "mu": mu,
        "std": std,
        "lower": lower,
        "upper": upper,
```

This is the beginning of the dict — the full dict continues. Find where it ends (search for the closing `}` of the return). Then add two entries. Example — find:

```python
    return {
        "mu": mu,
        "std": std,
        "lower": lower,
        "upper": upper,
        "oof_preds": oof_preds,
        "oof_folds": oof_folds,
        "ks_used": ks_used,
    }
```

Change to:

```python
    return {
        "mu": mu,
        "std": std,
        "lower": lower,
        "upper": upper,
        "oof_preds": oof_preds,
        "oof_folds": oof_folds,
        "ks_used": ks_used,
        "y_nb_std_oof": oof_y_nb_std,
        "y_nb_std_final": y_nb_std_final,
    }
```

(If the dict structure differs, add the two new keys before the closing brace.)

- [ ] **Step 7: Smoke test**

Run: `python3 -c "import sys; sys.path.insert(0, 'arena_predictor'); import predict"`
Expected: No output (successful import).

- [ ] **Step 8: Commit**

```bash
git add arena_predictor/predict.py
git commit -m "Persist y_nb_std through OOF and final-fit loops in fit_and_predict_knn"
```

---

## Task 10: Wire calibration module into predict.py main flow

**Files:**
- Modify: `arena_predictor/predict.py` (lines ~1640–1725, the conformal + probabilities block)

- [ ] **Step 1: Add calibration imports at top of predict.py**

Find the imports section near the top of `arena_predictor/predict.py`. Add after the existing scipy/sklearn imports:

```python
from calibration import (
    GateResult,
    ShapeFit,
    compute_p_beats_leader,
    compute_p_above,
    compute_sigma,
    diagnose_scale_signal,
    fit_tail_shape_and_qhat,
)
```

Note: predict.py is run as `python3 predict.py` from inside `arena_predictor/` (via `predict.sh`), so `from calibration import ...` resolves to the sibling module. If the repo uses `python3 -m arena_predictor.predict` anywhere, switch to `from arena_predictor.calibration import ...`; check `predict.sh` for the invocation pattern before deciding. Default: sibling import.

- [ ] **Step 2: Replace the conformal block with calibration module calls**

Find the block starting around line ~1650 (`# 9. Conformal intervals`) and ending around line ~1710 (after `top_by_margin_prob[train_idx] = np.nan`). Replace the entire block with:

```python
    # =========================================================================
    # 9. Calibration (OOF normalized conformal-style + walk-forward level scalar)
    # =========================================================================
    calibration_start = time.time()

    # -- OOF residuals (used by gate and shape fit) --
    oof_valid_mask = ~np.isnan(oof_preds)
    oof_preds_valid = oof_preds[oof_valid_mask]
    y_train_valid = y_all[~y_missing_mask][oof_valid_mask]
    oof_residuals_valid = y_train_valid - oof_preds_valid

    # -- Per-training-row y_nb_std from the OOF loop --
    y_nb_std_oof_full = knn_result["y_nb_std_oof"]  # length n_train
    y_nb_std_oof_valid = y_nb_std_oof_full[oof_valid_mask]

    # -- Predicted scores on training rows (for top-slice gate check) --
    mu_train_valid = mu[~y_missing_mask][oof_valid_mask]

    # -- (A) Diagnostic gate --
    gate = diagnose_scale_signal(
        y_nb_std_oof=y_nb_std_oof_valid,
        oof_residuals=oof_residuals_valid,
        predicted_scores=mu_train_valid,
        top_threshold=1400.0,
    )

    # -- (B) Shape fit (t_df + q_hat + s_floor), non-circular --
    shape = fit_tail_shape_and_qhat(
        oof_residuals=oof_residuals_valid,
        y_nb_std_oof=y_nb_std_oof_valid,
        gate_passed=gate.passed,
    )

    # -- (C) Walk-forward level correction: load fitted_m or default to 1.0 --
    m_scalar = 1.0
    if args.walkforward_calibration_path:
        wf_df = pd.read_csv(args.walkforward_calibration_path)
        # predict.py only reads fitted_m; walkforward_calibration.py is responsible for fitting it
        fitted_m_col = wf_df["fitted_m"].dropna()
        if len(fitted_m_col) == 0:
            print(f"WARNING: --walkforward_calibration_path given but fitted_m column is empty; using m=1.0",
                  file=sys.stderr)
        else:
            m_scalar = float(fitted_m_col.iloc[0])

    # -- Per-row sigma for the full output (train + test rows) --
    y_nb_std_all = knn_result["y_nb_std_final"]  # length n_all
    if gate.passed:
        sigma_hat = compute_sigma(y_nb_std_all, shape, m=m_scalar)
    else:
        # Fallback: s(x) = 1 everywhere, so sigma is constant across rows
        sigma_hat = compute_sigma(np.ones_like(y_nb_std_all), shape, m=m_scalar)
        print(f"WARNING: local scale gate failed (reason={gate.reason}); falling back to constant sigma × WF scalar",
              file=sys.stderr)

    # -- Intervals using t_crit --
    t_crit_95 = float(stats.t.ppf(0.975, shape.t_df))
    lower = mu - t_crit_95 * sigma_hat
    upper = mu + t_crit_95 * sigma_hat

    # -- max_leader threshold: max over rows with observed target --
    #    y_missing_mask is True for rows whose target is NaN (candidates).
    #    max_leader excludes candidates — only observed lmarena_Score counts.
    if (~y_missing_mask).any():
        max_leader = float(np.nanmax(y_all[~y_missing_mask]))
    else:
        max_leader = float("nan")

    # -- Probabilities --
    train_mask = ~y_missing_mask  # rows with observed target; p_beats_leader is meaningless for these
    p_beats_leader = compute_p_beats_leader(
        mu=mu,
        sigma=sigma_hat,
        t_df=shape.t_df,
        max_leader=max_leader,
        train_mask=train_mask,
    )
    top_by_margin_prob = compute_p_above(
        mu=mu,
        sigma=sigma_hat,
        t_df=shape.t_df,
        threshold=max_leader + args.margin,
    )
    top_by_margin_prob[train_mask] = np.nan

    calibration_end = time.time()
    print(f"calibration: {mmss(calibration_end - calibration_start)}")
    print(f"  gate={'PASS' if gate.passed else 'FAIL'} ({gate.reason})")
    print(f"  q_hat={shape.q_hat:.3f}  t_df={shape.t_df:.1f}  s_floor={shape.s_floor:.3f}  m={m_scalar:.3f}")
    print(f"  max_leader={max_leader:.1f}  sigma_hat range=[{np.nanmin(sigma_hat):.2f}, {np.nanmax(sigma_hat):.2f}]")
```

(The local names `std`, `lower`, `upper`, `t_df`, `num_one_prob` are all replaced by the new equivalents — do not leave the old names around.)

- [ ] **Step 3: Ensure `scipy.stats` is imported at the top**

Grep for the existing scipy import pattern. If `from scipy import stats` is not already present at the module top, add it; the new code block uses `stats.t.ppf`.

Run: `grep -n "from scipy" arena_predictor/predict.py | head -5`

If `from scipy import stats` is missing, add it near the other scipy imports (probably near the `from scipy.stats import norm` line if that exists). If only `from scipy.stats import t as t_dist` etc. exist, the cleanest change is to add `from scipy import stats` and keep the others.

- [ ] **Step 4: Remove old helpers that are now unused**

Find and delete (or leave if still referenced elsewhere — check first with grep):
- `prob_above_threshold` (line ~413) — superseded by `compute_p_above`
- `compute_grouped_conformal_intervals` (line ~460) — superseded entirely
- `_detect_suite_missing_fracs` (line ~429) — only used by grouped conformal; if no other refs, delete

Run: `grep -n "prob_above_threshold\|compute_grouped_conformal_intervals\|_detect_suite_missing_fracs" arena_predictor/predict.py`

For each that's only called inside its own definition or by the (now-deleted) grouped-conformal block, delete the definition. If any are referenced elsewhere (shouldn't be — they're internal), leave and flag in the commit message.

- [ ] **Step 5: Smoke-run predict.sh to verify no runtime errors**

Run: `bash predict.sh 2>&1 | tail -50`

Expected: script completes without errors. Last lines should include `calibration: Xs` timing and `gate=PASS/FAIL ...`. If it crashes, fix iteratively.

- [ ] **Step 6: Commit**

```bash
git add arena_predictor/predict.py
git commit -m "Wire calibration module into predict.py main flow"
```

---

## Task 11: Update predictions_best_model.csv emission

**Files:**
- Modify: `arena_predictor/predict.py` (lines ~1712–1725, the CSV emission block)

- [ ] **Step 1: Update the `pred_df` construction**

Find the block:

```python
    # -- predictions_best_model.csv --
    pred_df = pd.DataFrame({
        ID_COL: imputed_df[ID_COL].values,
        "predicted_score": mu,
        "actual_score": y_orig.values,
        "sigma_hat": conformal["sigma_hat"],
        "lower_bound": lower,
        "upper_bound": upper,
        "num_one_prob": num_one_prob,
        "top_by_margin_prob": top_by_margin_prob,
    })
    pred_df = pred_df.sort_values("predicted_score", ascending=False)
    pred_df.to_csv(os.path.join(out_dir, "predictions_best_model.csv"), index=False)
```

Change to:

```python
    # -- predictions_best_model.csv --
    # NOTE: sigma_hat is the t-distribution scale parameter, NOT a 95% half-width.
    # Consumers who need half-widths must multiply by t_crit_95 (= stats.t.ppf(0.975, t_df)).
    pred_df = pd.DataFrame({
        ID_COL: imputed_df[ID_COL].values,
        "predicted_score": mu,
        "actual_score": y_orig.values,
        "sigma_hat": sigma_hat,
        "lower_bound": lower,
        "upper_bound": upper,
        "p_beats_leader": p_beats_leader,
        "top_by_margin_prob": top_by_margin_prob,
    })
    pred_df = pred_df.sort_values("predicted_score", ascending=False)
    pred_df.to_csv(os.path.join(out_dir, "predictions_best_model.csv"), index=False)
```

- [ ] **Step 2: Update the metadata notes about column semantics**

Find the metadata block (around line 1797):

```python
        "notes": [
            "Safety-filled any residual NaNs/Infs in features post-imputation with column medians, then 0 if still NaN.",
            "num_one_prob compares to current max observed lmarena_Score among training rows.",
            f"top_by_margin_prob = P(score > max_observed + {args.margin}), only for models without actual scores.",
            "predictions_best_model.csv is sorted by predicted_score descending (highest on top).",
        ]
```

Change to:

```python
        "notes": [
            "Safety-filled any residual NaNs/Infs in features post-imputation with column medians, then 0 if still NaN.",
            "sigma_hat is the t-distribution SCALE parameter, not a half-width. For 95% intervals use mu ± t_crit_95 * sigma_hat.",
            "p_beats_leader = P(score > max_leader) where max_leader = max(observed lmarena_Score); NaN on training rows.",
            f"top_by_margin_prob = P(score > max_leader + {args.margin}), NaN on training rows.",
            "predictions_best_model.csv is sorted by predicted_score descending (highest on top).",
        ]
```

- [ ] **Step 3: Smoke run**

Run: `bash predict.sh 2>&1 | tail -30`

Then: `head -5 arena_predictor/analysis_output/$(ls -t arena_predictor/analysis_output | head -1)/predictions_best_model.csv`

Expected: columns include `model_name, predicted_score, actual_score, sigma_hat, lower_bound, upper_bound, p_beats_leader, top_by_margin_prob`. `sigma_hat` varies across rows (if gate passed) or is constant (if gate failed).

- [ ] **Step 4: Commit**

```bash
git add arena_predictor/predict.py
git commit -m "Emit p_beats_leader + per-model sigma_hat in predictions_best_model.csv"
```

---

## Task 12: Emit calibration_diagnostics.csv

**Files:**
- Modify: `arena_predictor/predict.py` (add new CSV emission after predictions_best_model.csv)

- [ ] **Step 1: Add diagnostics output right after predictions_best_model.csv is written**

Find the line (from Task 11):

```python
    pred_df.to_csv(os.path.join(out_dir, "predictions_best_model.csv"), index=False)
```

Insert immediately after:

```python

    # -- calibration_diagnostics.csv --
    # Compute OOF-level coverage and PIT for diagnostics
    oof_sigma_valid = sigma_hat[~y_missing_mask][oof_valid_mask]
    # Use OOF-level sigma without m correction for self-check (m is only appropriate for WF)
    oof_sigma_nom = oof_sigma_valid / m_scalar if m_scalar != 0 else oof_sigma_valid
    oof_residuals_for_diag = oof_residuals_valid
    oof_z = oof_residuals_for_diag / np.where(oof_sigma_nom < 1e-12, 1e-12, oof_sigma_nom)
    # PIT
    try:
        oof_u = stats.t.cdf(oof_z, df=shape.t_df)
        oof_pit_ks_pvalue = float(stats.kstest(oof_u, "uniform").pvalue)
    except Exception:
        oof_pit_ks_pvalue = float("nan")
    # 95% coverage
    oof_covered_95 = (oof_residuals_for_diag >= -t_crit_95 * oof_sigma_nom) & \
                     (oof_residuals_for_diag <= t_crit_95 * oof_sigma_nom)
    oof_coverage_95 = float(oof_covered_95.mean())

    diag_row = {
        "gate_pass": bool(gate.passed),
        "gate_reason": gate.reason,
        "spearman_all": gate.spearman_all,
        "spearman_top": gate.spearman_top,
        "log_log_slope": gate.log_log_slope,
        "log_log_r2": gate.log_log_r2,
        "decile_lift": gate.decile_lift,
        "n_oof": gate.n_all,
        "n_top_oof": gate.n_top,
        "q_hat": shape.q_hat,
        "t_df": shape.t_df,
        "s_floor": shape.s_floor,
        "m": m_scalar,
        "fallback_used": bool(shape.fallback_used),
        "oof_coverage_95": oof_coverage_95,
        "oof_pit_ks_pvalue": oof_pit_ks_pvalue,
        "max_leader": max_leader,
        "t_crit_95": t_crit_95,
    }
    # WF-specific diagnostics land in walkforward_calibration_diagnostics.csv
    # (emitted by walkforward_calibration.py). predict.py emits only OOF + shape here.
    pd.DataFrame([diag_row]).to_csv(
        os.path.join(out_dir, "calibration_diagnostics.csv"), index=False
    )

    # PIT self-check warning
    if not np.isnan(oof_pit_ks_pvalue) and oof_pit_ks_pvalue < 0.01:
        print(f"WARNING: OOF PIT KS p-value = {oof_pit_ks_pvalue:.4f} < 0.01; predictive distribution is not uniform on OOF",
              file=sys.stderr)
```

- [ ] **Step 2: Smoke run**

Run: `bash predict.sh 2>&1 | tail -30 && ls arena_predictor/analysis_output/$(ls -t arena_predictor/analysis_output | head -1)/`

Expected: `calibration_diagnostics.csv` exists in the output directory. Run `cat` on it to confirm sensible values.

- [ ] **Step 3: Commit**

```bash
git add arena_predictor/predict.py
git commit -m "Emit calibration_diagnostics.csv with gate metrics + OOF coverage + PIT"
```

---

## Task 13: Add --walkforward_calibration_path CLI arg

**Files:**
- Modify: `arena_predictor/predict.py` (argparse block)

- [ ] **Step 1: Find the argparse block**

Run: `grep -n "add_argument.*--margin" arena_predictor/predict.py`

- [ ] **Step 2: Add the new argument next to --margin**

Near the `--margin` argument, add:

```python
    parser.add_argument(
        "--walkforward_calibration_path",
        type=str,
        default=None,
        help="Path to walkforward_calibration.py's wf_residuals.csv. "
             "If provided, predict.py reads the 'fitted_m' column and applies it "
             "to sigma_hat. Fitting m itself happens inside walkforward_calibration.py, "
             "not here. If None, m=1.0 (sigma reflects OOF level only).",
    )
```

- [ ] **Step 3: Smoke test argparse**

Run: `python3 arena_predictor/predict.py --help 2>&1 | grep -A1 walkforward_calibration_path`
Expected: The new argument shows up with the help text.

- [ ] **Step 4: Verify the arg flows into the calibration block**

The block in Task 10 already references `args.walkforward_calibration_path`. Rerun `bash predict.sh` without the flag; it should use `m=1.0`.

Run: `bash predict.sh 2>&1 | grep "m=" | head -3`
Expected: `m=1.000` in the log.

- [ ] **Step 5: Commit**

```bash
git add arena_predictor/predict.py
git commit -m "Add --walkforward_calibration_path flag to feed fitted m into predict.py"
```

---

## Task 14: Scaffold walkforward_calibration.py

**Files:**
- Create: `arena_predictor/walkforward_calibration.py`

- [ ] **Step 1: Create the scaffold**

Create `arena_predictor/walkforward_calibration.py`:

```python
"""Walk-forward honest-eval + m-fit for predictor calibration.

Extends arena_predictor/_walkforward_honest.py by (a) running a nested LOO inside
each WF step to produce per-step OOF residuals and y_nb_std, (b) fitting per-step
gate + t_df + q_hat + s_floor on that prefix, (c) fitting a global scalar m across
steps, and (d) emitting diagnostics (PIT, coverage, Brier, log-loss) + wf_residuals.csv
for downstream consumption by predict.py via --walkforward_calibration_path.

See docs/superpowers/specs/2026-04-24-predictor-calibration-design.md for design.
"""
from __future__ import annotations

import itertools
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Make predict.py and calibration.py importable
sys.path.insert(0, str(Path(__file__).parent))

from _walkforward_honest import build_pooled_embeddings  # noqa: E402
from calibration import (  # noqa: E402
    compute_local_scale,
    compute_p_above,
    compute_sigma,
    diagnose_scale_signal,
    fit_tail_shape_and_qhat,
)
from predict import ID_COL, TARGET, ALT_TARGET, predict_adaptive_knn, run_imputation  # noqa: E402


OUT_DIR = Path(__file__).parent / "analysis_output" / "walkforward_calibration"


def main():
    """Entry point. Body is implemented in Task 15 (WF loop) and extended in Task 16 (m-fit + diagnostics)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] output dir: {OUT_DIR}", flush=True)
    print("[scaffold] main() body not yet implemented; see Task 15.", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

Run: `python3 arena_predictor/walkforward_calibration.py`
Expected: prints `[setup] output dir: .../walkforward_calibration`, exits cleanly.

- [ ] **Step 3: Commit**

```bash
git add arena_predictor/walkforward_calibration.py
git commit -m "Scaffold walkforward_calibration.py"
```

---

## Task 15: Implement WF loop with nested LOO + per-step calibration fits

**Files:**
- Modify: `arena_predictor/walkforward_calibration.py`

- [ ] **Step 1: Add data loading block (copy structure from _walkforward_honest.py)**

Replace the `main()` function body in `walkforward_calibration.py` with:

```python
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] output dir: {OUT_DIR}", flush=True)

    CSV = Path(__file__).parent.parent / "benchmark_combiner" / "benchmarks" / "clean_combined_all_benches_with_sem_v4_d32.csv"
    DATES = Path(__file__).parent.parent / "benchmark_combiner" / "benchmarks" / "openbench_release_dates.csv"

    src = pd.read_csv(CSV)
    dates = pd.read_csv(DATES).rename(columns={"Model": "model_name", "Release_Date": "release_date"})
    dates["release_date"] = pd.to_datetime(dates["release_date"], errors="coerce")

    pooled = build_pooled_embeddings()

    src = src.merge(dates[["model_name", "release_date"]], on="model_name", how="left")
    src = src.merge(pooled, on="model_name", how="left")
    mask = (
        src["lmarena_Score"].notna()
        & src["release_date"].notna()
        & src["all_slots_present"].fillna(False)
    )
    src = src[mask].sort_values("release_date").reset_index(drop=True)
    n = len(src)
    print(f"[pool] {n} models with target + date + all embedding slots", flush=True)

    pooled_cols = [f"p{i:04d}" for i in range(5 * 384)]
    drop_cols = (
        set(pooled_cols)
        | {"model_name", "release_date", "all_slots_present", TARGET, ALT_TARGET}
    )
    sem_cols_csv = [c for c in src.columns if c.startswith("sem_")]
    drop_cols |= set(sem_cols_csv)
    feature_cols = [
        c for c in src.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(src[c])
    ]
    print(f"[setup] imputer feature cols: {len(feature_cols)} (sem_f* dropped, rebuilt per step)", flush=True)

    y = src["lmarena_Score"].values.astype(float)
    P = src[pooled_cols].values.astype(float)

    # Production imputer settings (from predict.sh defaults; keep in sync with _walkforward_honest.py)
    imp_kwargs = dict(
        passes=14, alpha=0.9361, verbose=0,
        use_feature_selector=True, selector_tau=0.9012,
        selector_k_max=37, gp_selector_k_max=28,
        imputer_n_jobs=-1,
        categorical_threshold=0, force_categorical_cols=[],
        tolerance_percentile=91.1553, tolerance_relaxation_factor=1.2704, tolerance_multiplier=5.8849,
        tier_quantiles=None,
        calibrate_tolerances=False, calibration_target_rmse_ratio=0.6266,
        calibration_n_rounds=3, calibration_holdout_frac=0.2, recalibrate_every_n_passes=5,
        imputer_type="model_bank", confidence_threshold=0.4,
        coherence_lambda=8.0, coherence_shape="exp", coherence_gate="fixed",
        iterative_coherence=False,
        predictor_selection="loo_forward",
        use_svd_predictors=False, n_expansion_passes=1, max_confident_extras=1,
    )

    n_init = int(n * 0.80)
    n_wf = n - n_init
    print(f"[start] walk-forward from oldest {n_init} -> newest {n_wf}", flush=True)

    # Collect per-step results
    records = []
    t_start = time.time()

    for i in range(n_init, n):
        t0 = time.time()
        # ----- Fit imputer + PCA + PLS + predictor on prefix [0..i] -----
        sub = src.iloc[:i + 1].copy()
        imp_df, imputer = run_imputation(
            sub[[ID_COL] + feature_cols],
            **imp_kwargs,
        )
        svd = imputer.svd_row_factors_.reset_index(drop=True)
        traj = imputer.trajectory_features_.reset_index(drop=True)

        feat = imp_df.copy()
        for c in svd.columns:
            feat[c] = svd[c].values
            feat[f"{c}_sq"] = svd[c].values ** 2
        svd_cols = list(svd.columns)
        for a, b in itertools.combinations(range(min(4, len(svd_cols))), 2):
            feat[f"{svd_cols[a]}x{svd_cols[b]}"] = svd[svd_cols[a]].values * svd[svd_cols[b]].values
        for c in traj.columns:
            feat[c] = traj[c].values

        static_cols = [
            c for c in feat.columns
            if c != ID_COL and not c.startswith("style_") and not c.startswith("tone_")
        ]
        X_static = feat[static_cols].values.astype(float)
        med = np.nanmedian(X_static, axis=0)
        inds = np.where(np.isnan(X_static))
        X_static[inds] = np.take(med, inds[1])
        X_static = np.nan_to_num(X_static)

        n_comp_pca = min(32, i - 1, P.shape[1])
        pca = PCA(n_components=n_comp_pca, random_state=42).fit(P[:i])
        sem_tr = pca.transform(P[:i])
        sem_te = pca.transform(P[i:i + 1])

        Xtr_raw = np.hstack([X_static[:i], sem_tr])
        Xte_raw = np.hstack([X_static[i:i + 1], sem_te])

        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr_raw)
        Xte = sc.transform(Xte_raw)
        pls = PLSRegression(n_components=min(3, Xtr.shape[1], i - 1)).fit(Xtr, y[:i])
        Xtr = np.hstack([Xtr, pls.transform(Xtr)])
        Xte = np.hstack([Xte, pls.transform(Xte)])

        # ----- Point prediction for the test row -----
        p, _, _, y_nb_std_t = predict_adaptive_knn(
            Xtr, y[:i], Xte,
            max_k=min(80, i), min_k=min(20, i),
        )

        # ----- Nested LOO over prefix [0..i-1] for OOF residuals + y_nb_std_oof_t -----
        prefix_oof_preds = np.zeros(i)
        prefix_oof_y_nb_std = np.zeros(i)
        for j in range(i):
            mask_j = np.ones(i, dtype=bool)
            mask_j[j] = False
            Xtr_j = Xtr[mask_j]
            ytr_j = y[:i][mask_j]
            Xte_j = Xtr[j:j + 1]
            p_j, _, _, y_nb_std_j = predict_adaptive_knn(
                Xtr_j, ytr_j, Xte_j,
                max_k=min(80, i - 1), min_k=min(20, i - 1),
            )
            prefix_oof_preds[j] = p_j
            prefix_oof_y_nb_std[j] = y_nb_std_j
        prefix_oof_residuals = y[:i] - prefix_oof_preds

        # ----- Per-step calibration fits -----
        gate_t = diagnose_scale_signal(
            y_nb_std_oof=prefix_oof_y_nb_std,
            oof_residuals=prefix_oof_residuals,
            predicted_scores=prefix_oof_preds,
            top_threshold=1400.0,
        )
        shape_t = fit_tail_shape_and_qhat(
            oof_residuals=prefix_oof_residuals,
            y_nb_std_oof=prefix_oof_y_nb_std,
            gate_passed=gate_t.passed,
        )
        # sigma for the test row, at m=1 (m is fit globally later)
        if gate_t.passed:
            sigma_oof_t = float(compute_sigma(
                np.array([y_nb_std_t]), shape_t, m=1.0
            )[0])
        else:
            sigma_oof_t = float(compute_sigma(
                np.array([1.0]), shape_t, m=1.0
            )[0])

        # Stepwise threshold: max of observed targets in prefix [0..i-1]
        max_leader_t = float(np.max(y[:i]))

        record = {
            "step": i - n_init,
            "model_name": src.iloc[i]["model_name"],
            "release_date": src.iloc[i]["release_date"],
            "mu_t": float(p),
            "sigma_oof_t": sigma_oof_t,
            "y_t": float(y[i]),
            "max_leader_t": max_leader_t,
            "t_df_t": shape_t.t_df,
            "q_hat_t": shape_t.q_hat,
            "s_floor_t": shape_t.s_floor,
            "gate_pass_t": bool(gate_t.passed),
            "y_nb_std_t": float(y_nb_std_t),
            "err_t": float(p - y[i]),
        }
        records.append(record)

        dt = time.time() - t0
        eta = (time.time() - t_start) / (len(records)) * (n_wf - len(records))
        print(
            f"[step {i - n_init + 1}/{n_wf}] {src.iloc[i]['model_name']!r} "
            f"actual={y[i]:.0f} pred={p:.1f} sigma_oof={sigma_oof_t:.2f} "
            f"gate={'PASS' if gate_t.passed else 'FAIL'} ({dt:.1f}s, eta {eta/60:.1f}m)",
            flush=True,
        )

    wf_df = pd.DataFrame(records)
    wf_df.to_csv(OUT_DIR / "wf_residuals.csv", index=False)
    print(f"\n[done] WF loop finished in {(time.time() - t_start)/60:.1f}m", flush=True)
    print(f"[out] {OUT_DIR / 'wf_residuals.csv'}", flush=True)
    return wf_df
```

- [ ] **Step 2: Smoke test on a single step**

Run: `python3 arena_predictor/walkforward_calibration.py 2>&1 | head -40`

Expected: script runs, prints per-step progress; may take >10m to complete on full data due to nested LOO. If running out of time, confirm the first 2-3 steps emit sensible lines, then ctrl-C and move on to Task 16 (fitting m can happen after the loop writes to disk; full run can wait until the end-to-end smoke test in Task 17).

- [ ] **Step 3: Commit**

```bash
git add arena_predictor/walkforward_calibration.py
git commit -m "Implement WF loop with nested LOO + per-step calibration fits"
```

---

## Task 16: Add m-fit + WF diagnostics

**Files:**
- Modify: `arena_predictor/walkforward_calibration.py`

- [ ] **Step 1: Add m-fit + diagnostics functions above `main()`**

In `arena_predictor/walkforward_calibration.py`, add the following helper functions between the imports and `main()`:

```python
def fit_m(
    z: np.ndarray,
    t_df_t: np.ndarray,
    bounds: tuple = (0.5, 3.0),
) -> float:
    """Fit scalar m by MLE on per-step z = (y_t - mu_t) / sigma_oof_t with per-step t_df_t.

    Minimizes: -sum(t.logpdf(z_t / m, df=t_df_t)) + len(z) * log(m)
    """
    z = np.asarray(z, dtype=float)
    t_df_t = np.asarray(t_df_t, dtype=float)
    finite = np.isfinite(z) & np.isfinite(t_df_t)
    z, t_df_t = z[finite], t_df_t[finite]

    def neg_log_lik(m: float) -> float:
        if m <= 0:
            return 1e12
        return -float(np.sum(stats.t.logpdf(z / m, df=t_df_t))) + len(z) * float(np.log(m))

    result = optimize.minimize_scalar(
        neg_log_lik, bounds=bounds, method="bounded", options={"xatol": 1e-4}
    )
    m_fit = float(result.x)
    # Warn if at boundary
    if abs(m_fit - bounds[0]) < 1e-3 or abs(m_fit - bounds[1]) < 1e-3:
        print(f"WARNING: fitted m={m_fit:.3f} is at boundary {bounds}", file=sys.stderr)
    return m_fit


def compute_wf_diagnostics(
    wf_df: pd.DataFrame,
    m: float,
    top_threshold: float = 1400.0,
) -> dict:
    """Compute PIT, coverage, Brier, log-loss on WF residuals with fitted m applied.

    Emits both overall and top-slice (mu_t >= top_threshold) variants.
    """
    diag = {"fitted_m": m}
    rows_all = wf_df.copy()
    rows_all["sigma_t"] = m * rows_all["sigma_oof_t"]
    rows_all["z"] = (rows_all["y_t"] - rows_all["mu_t"]) / np.where(
        rows_all["sigma_t"] < 1e-12, 1e-12, rows_all["sigma_t"]
    )

    def _slice_metrics(df: pd.DataFrame, prefix: str) -> dict:
        n = len(df)
        if n < 1:
            return {f"{prefix}n": 0}
        out = {f"{prefix}n": int(n)}
        # PIT
        try:
            u = stats.t.cdf(df["z"].values, df=df["t_df_t"].values)
            out[f"{prefix}pit_ks_pvalue"] = float(stats.kstest(u, "uniform").pvalue)
        except Exception:
            out[f"{prefix}pit_ks_pvalue"] = float("nan")
        # Coverage at 50/80/95%
        for alpha in (0.50, 0.80, 0.95):
            pct = int(alpha * 100)
            t_crit = stats.t.ppf((1 + alpha) / 2, df["t_df_t"].values)
            lo = df["mu_t"].values - t_crit * df["sigma_t"].values
            hi = df["mu_t"].values + t_crit * df["sigma_t"].values
            covered = (df["y_t"].values >= lo) & (df["y_t"].values <= hi)
            n_cov = int(covered.sum())
            out[f"{prefix}coverage_{pct}"] = float(n_cov / n) if n > 0 else float("nan")
            if n > 0:
                ci = stats.binomtest(n_cov, n).proportion_ci(confidence_level=0.90, method="exact")
                out[f"{prefix}coverage_{pct}_ci_lo"] = float(ci.low)
                out[f"{prefix}coverage_{pct}_ci_hi"] = float(ci.high)
        # Brier + log-loss for stepwise event y_t > max_leader_t
        p_event = 1.0 - stats.t.cdf(
            (df["max_leader_t"].values - df["mu_t"].values)
            / np.where(df["sigma_t"].values < 1e-12, 1e-12, df["sigma_t"].values),
            df=df["t_df_t"].values,
        )
        y_event = (df["y_t"].values > df["max_leader_t"].values).astype(float)
        eps = 1e-6
        p_clip = np.clip(p_event, eps, 1 - eps)
        out[f"{prefix}brier"] = float(np.mean((p_clip - y_event) ** 2))
        # Log-loss: degenerate if all same class
        if len(np.unique(y_event)) < 2:
            out[f"{prefix}log_loss"] = float("nan")
        else:
            out[f"{prefix}log_loss"] = float(
                -np.mean(y_event * np.log(p_clip) + (1 - y_event) * np.log(1 - p_clip))
            )
        return out

    diag.update(_slice_metrics(rows_all, prefix="wf_"))
    top_df = rows_all[rows_all["mu_t"] >= top_threshold].copy()
    diag.update(_slice_metrics(top_df, prefix="wf_top_"))

    return diag
```

- [ ] **Step 2: Wire m-fit + diagnostics into `main()`**

At the end of the `main()` function body, replace:

```python
    wf_df = pd.DataFrame(records)
    wf_df.to_csv(OUT_DIR / "wf_residuals.csv", index=False)
    print(f"\n[done] WF loop finished in {(time.time() - t_start)/60:.1f}m", flush=True)
    print(f"[out] {OUT_DIR / 'wf_residuals.csv'}", flush=True)
    return wf_df
```

with:

```python
    wf_df = pd.DataFrame(records)

    # Fit scalar m across all steps using per-step t_df_t
    z = (wf_df["y_t"].values - wf_df["mu_t"].values) / np.where(
        wf_df["sigma_oof_t"].values < 1e-12, 1e-12, wf_df["sigma_oof_t"].values
    )
    m_fit = fit_m(z=z, t_df_t=wf_df["t_df_t"].values)
    wf_df["fitted_m"] = m_fit  # same value on every row; predict.py reads row 0

    wf_df.to_csv(OUT_DIR / "wf_residuals.csv", index=False)
    print(f"\n[m-fit] fitted_m = {m_fit:.4f}", flush=True)

    # Compute + emit diagnostics
    diag = compute_wf_diagnostics(wf_df, m=m_fit)
    pd.DataFrame([diag]).to_csv(OUT_DIR / "walkforward_calibration_diagnostics.csv", index=False)

    print(f"[done] WF loop + m-fit + diagnostics finished in {(time.time() - t_start)/60:.1f}m", flush=True)
    print(f"[out] {OUT_DIR / 'wf_residuals.csv'}", flush=True)
    print(f"[out] {OUT_DIR / 'walkforward_calibration_diagnostics.csv'}", flush=True)
    print(f"\n=== Diagnostics ===", flush=True)
    for k, v in diag.items():
        print(f"  {k}: {v}", flush=True)

    return wf_df
```

- [ ] **Step 3: Smoke test**

Run: `python3 arena_predictor/walkforward_calibration.py 2>&1 | tail -30`

Expected: full run completes (may take 30-60m). Final output shows `fitted_m = X.XXX` and diagnostics block. Both CSVs written to `arena_predictor/analysis_output/walkforward_calibration/`.

If run is impractically long to let finish in one go, can run in background via: `nohup python3 arena_predictor/walkforward_calibration.py > /tmp/wf_calib.log 2>&1 &` then check `tail -f /tmp/wf_calib.log`.

- [ ] **Step 4: Commit**

```bash
git add arena_predictor/walkforward_calibration.py
git commit -m "Implement m-fit + WF diagnostics (PIT, coverage, Brier, log-loss)"
```

---

## Task 17: End-to-end smoke test + comparison vs old output

**Files:**
- None (verification task)

- [ ] **Step 1: Run the two-run bootstrap**

```bash
# Run 1: predict.sh without fitted m
bash predict.sh 2>&1 | tail -20
```

Note the output dir. Save reference:

```bash
RUN1=$(ls -t arena_predictor/analysis_output/ | grep '^output_' | head -1)
echo "Run 1 output dir: $RUN1"
```

- [ ] **Step 2: Run walkforward_calibration.py**

```bash
python3 arena_predictor/walkforward_calibration.py 2>&1 | tail -30
```

Expected: `fitted_m = X.XXX` around 1.05-1.15 (prior was 14.69/13.61 ≈ 1.08 from RMSE ratio, though the z-score MLE may differ). If `fitted_m` is at the [0.5, 3.0] boundary, inspect diagnostics before continuing.

- [ ] **Step 3: Run 2 — predict.sh with fitted_m**

`predict.sh` already passes extra args through via `"$@"`, so:

```bash
WF_FILE=arena_predictor/analysis_output/walkforward_calibration/wf_residuals.csv
bash predict.sh --walkforward_calibration_path "../$WF_FILE" 2>&1 | tail -20
```

Note the `../` prefix: `predict.sh` `cd`s into `arena_predictor/` before running `predict.py`, so the path must be relative to that directory. If this is awkward, use an absolute path:

```bash
bash predict.sh --walkforward_calibration_path "$(pwd)/arena_predictor/analysis_output/walkforward_calibration/wf_residuals.csv"
```

- [ ] **Step 4: Inspect output CSV**

```bash
RUN2=$(ls -t arena_predictor/analysis_output/ | grep '^output_' | head -1)
head -5 arena_predictor/analysis_output/$RUN2/predictions_best_model.csv
cat arena_predictor/analysis_output/$RUN2/calibration_diagnostics.csv
```

Expected:
- `predictions_best_model.csv` columns: `model_name, predicted_score, actual_score, sigma_hat, lower_bound, upper_bound, p_beats_leader, top_by_margin_prob`
- `sigma_hat` varies across rows (if gate passed) or is constant (if gate failed)
- `p_beats_leader` is NaN on rows with observed actuals, a probability in [0,1] on candidate rows
- `calibration_diagnostics.csv` has a single row with `gate_pass`, `spearman_top`, `q_hat`, `t_df`, `m`, `oof_coverage_95`, `oof_pit_ks_pvalue`, etc.

- [ ] **Step 5: Regression smell test**

```bash
python3 - <<'PY'
import pandas as pd
new = pd.read_csv(f"arena_predictor/analysis_output/$RUN2/predictions_best_model.csv")
print(new[new['model_name'].str.contains('Opus 4.7 Thinking', case=False, na=False)])
PY
```

(Substitute `$RUN2` with the actual directory name.)

Verify:
- Opus 4.7 Thinking has `sigma_hat` in roughly 15-40 range (was 21.81 before)
- `p_beats_leader` is in [0, 1]
- Neither is a NaN for a candidate row

A nonsense result (`sigma_hat > 50`, `p_beats_leader > 0.95` or `< 0.05` without reason) means something is wrong — diagnose before committing.

- [ ] **Step 6: Inspect WF diagnostics**

```bash
cat arena_predictor/analysis_output/walkforward_calibration/walkforward_calibration_diagnostics.csv | python3 -c "import sys, csv; r = list(csv.DictReader(sys.stdin))[0]; [print(f'{k}: {v}') for k, v in r.items()]"
```

Expected:
- `wf_pit_ks_pvalue` > 0.01 (PIT not obviously non-uniform)
- `wf_coverage_95` in [0.85, 1.0] (exact binomial CI should contain 0.95 for n≈25)
- `wf_brier` and `wf_log_loss` are finite numbers
- Top-slice metrics (`wf_top_*`) emit values or NaN for degenerate slices
- `fitted_m` between 0.9 and 1.5

- [ ] **Step 7: No commit for this task — verification-only. If all checks pass, the feature is ready. If any fail, diagnose before declaring done.**

(Optionally: commit the WF output files if you want them tracked, but they're analysis artifacts and may be better left untracked. Check `.gitignore` first — if `analysis_output/` is already ignored, leave them.)

---

## Self-Review Notes

**Spec coverage:**
- Component 1 (`compute_oof_nb_std`) → Tasks 8-9
- Component 2 (`diagnose_scale_signal`) → Task 2
- Component 3 (local scale) → Task 3
- Component 4 (tail shape + q_hat, non-circular) → Task 4
- Component 5 (walk-forward m-fit with per-step t_df_t, strict nested LOO) → Tasks 15-16
- Component 6 (threshold definitions) → Task 10 (batch) + Task 15 (stepwise)
- Component 7 (output columns with explicit interval formula) → Task 11
- Component 8 (`calibration_diagnostics.csv`) → Task 12 (+ Task 16 for WF diagnostics)
- Component 9 (fallback path) → Task 7 (integration test) + Task 10 (wiring)
- Data flow + two-run bootstrap → Task 13 (CLI arg) + Task 17 (end-to-end)
- Numerical details (PIT, coverage CIs, degenerate-slice handling, m optimization with per-step t_df) → Tasks 2, 4, 16

**Known limitation:** walkforward_calibration.py's nested LOO (Task 15) is O(n^2) per step and O(n^3) total. For n=127, that's ~10^6 `predict_adaptive_knn` calls — 30-90 minutes total runtime is expected. The spec explicitly accepts this cost.
