#!/usr/bin/env python3
"""
Arena ELO Predictor — KNN Pipeline.

Predicts lmarena_Score (style-controlled Chatbot Arena ELO) from benchmark data
using adaptive KNN + kernel Ridge + jackknife variance inflation (R²=0.927).

Pipeline:
    1. Load benchmark CSV, identify numeric columns
    2. Low-variance filtering
    3. Impute missing values (ModelBankImputer default, SpecializedColumnImputer option)
    4. Build feature matrix (imputed cols + SVD factors + trajectory features)
    5. Adaptive KNN prediction with OOF cross-validation
    6. Grouped conformal calibration of prediction intervals
    7. Save predictions, OOF, metadata, dependency graph
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import json
import re
import itertools
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Union

import time
import hashlib

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from zoneinfo import ZoneInfo
from datetime import datetime

# joblib kept available for future parallel extensions
from joblib import Parallel, delayed  # noqa: F401

# ==============================================================================
# LOCAL IMPORTS
# ==============================================================================

try:
    from column_imputer import SpecializedColumnImputer, ModelBankImputer
except Exception as e:
    print("ERROR: Could not import column_imputer.py. Make sure the file is alongside this script.", file=sys.stderr)
    raise

# ML stack
from sklearn.model_selection import KFold, GroupKFold  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.linear_model import Ridge  # type: ignore
from sklearn.neighbors import NearestNeighbors  # type: ignore

# Optional: trueskill import guard (kept for compatibility)
try:
    import trueskill  # type: ignore  # noqa: F401
except ImportError:
    pass

# ==============================================================================
# CONSTANTS
# ==============================================================================

SEED = 42
np.random.seed(SEED)

TARGET = "lmarena_Score"  # style-controlled Arena ELO
ALT_TARGET = "lmsys_Score"  # raw Arena ELO — excluded as leakage
ID_COL = "model_name"

TARGETS = {TARGET, ALT_TARGET}  # exclude both to avoid leakage
EXCLUDE = TARGETS | {ID_COL}

DENSE_THRESHOLD = 0.508  # original threshold for dense-only CV evaluation
COMPLETENESS_WEIGHT_POWER = 0  # 0 = disabled (all weights 1.0)

# KNN prediction hyperparameters
KNN_DIST_MULT = 2.0   # include neighbors within dist_mult × nearest distance
KNN_MAX_K = 80
KNN_MIN_K = 20
KNN_BW_PCT = 0.3      # kernel bandwidth at this percentile of neighbor distances
KNN_VI_CLIP = (1.0, 1.5)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def _perturb_imputed_matrix(
    imputed_df: pd.DataFrame,
    sigma2_matrix: pd.DataFrame,
    feature_cols: list,
    seed: int,
) -> pd.DataFrame:
    """Sample a perturbed imputation by adding N(0, σ) noise to imputed cells.

    Only cells that were originally missing (σ² > 0 and finite) get perturbed.
    Observed cells (σ² == 0) are left unchanged.
    """
    rng = np.random.RandomState(seed)
    perturbed = imputed_df.copy()
    for col in feature_cols:
        if col not in sigma2_matrix.columns:
            continue
        s2 = sigma2_matrix[col].values
        mask = np.isfinite(s2) & (s2 > 0)
        if not mask.any():
            continue
        noise = rng.normal(0, np.sqrt(s2[mask]))
        perturbed.loc[perturbed.index[mask], col] = (
            perturbed.loc[perturbed.index[mask], col].values + noise
        )
    return perturbed


PARALLELISM_CFG = {
    "max_workers": None,
    "cv_n_jobs": 1,
    "model_n_jobs": 1,
    "selector_n_jobs": -1,
    "tree_model_n_jobs": -1,
    "tree_selector_n_jobs": -1,
    "imputer_n_jobs": -1,
}


def _cap_job_count(requested: int, cap: Optional[int]) -> int:
    if cap is None:
        return requested
    if requested <= 0:
        return cap
    return max(1, min(requested, cap))


def _configure_parallelism(args) -> None:
    global PARALLELISM_CFG
    max_workers = int(getattr(args, "max_workers", 0))
    if max_workers <= 0:
        max_workers = None

    selector_n_jobs = _cap_job_count(int(args.selector_n_jobs), max_workers)

    cv_requested = max(1, int(args.cv_n_jobs))
    if max_workers is None:
        cv_jobs = cv_requested
        model_jobs_cap = None
    else:
        cv_jobs = _cap_job_count(cv_requested, max_workers)
        model_jobs_cap = max(1, max_workers // cv_jobs) if cv_jobs else 1

    model_requested = max(1, int(getattr(args, "model_n_jobs", 1)))
    model_jobs = _cap_job_count(model_requested, model_jobs_cap) if model_jobs_cap else model_requested

    if max_workers is None:
        per_model_threads = -1
    else:
        groups = max(1, cv_jobs * model_jobs)
        per_model_threads = max(1, max_workers // groups) if groups else max_workers

    tree_n_jobs = per_model_threads if max_workers is not None else -1
    if tree_n_jobs == 0:
        tree_n_jobs = 1

    imputer_requested = int(getattr(args, "imputer_n_jobs", -1))
    if max_workers is None:
        imputer_jobs = -1 if imputer_requested <= 0 else imputer_requested
    else:
        imputer_jobs = _cap_job_count(imputer_requested, max_workers)

    PARALLELISM_CFG = {
        "max_workers": max_workers,
        "cv_n_jobs": cv_jobs,
        "model_n_jobs": model_jobs,
        "selector_n_jobs": selector_n_jobs,
        "tree_model_n_jobs": tree_n_jobs,
        "tree_selector_n_jobs": tree_n_jobs,
        "imputer_n_jobs": imputer_jobs,
    }
    print(PARALLELISM_CFG)

    args.selector_n_jobs = selector_n_jobs
    args.imputer_n_jobs = imputer_jobs
    args.cv_n_jobs = cv_jobs
    args.max_workers = max_workers or 0


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
    ]


def _is_integer_like_series(s: pd.Series, tol: float = 1e-6) -> bool:
    """True when all finite, non-null values are (near) integers."""
    s_nonnull = s.dropna()
    if s_nonnull.empty:
        return False
    vals = s_nonnull.to_numpy()
    finite = np.isfinite(vals)
    vals = vals[finite]
    if vals.size == 0:
        return False
    return np.all(np.abs(vals - np.round(vals)) <= tol)


def _find_numeric_categoricals(
    df: pd.DataFrame,
    cols: list[str],
    max_unique: int = 10,
    tol: float = 1e-6,
) -> list[str]:
    """Identify low-cardinality integer-like numeric columns to treat as categorical."""
    cats = []
    for col in cols:
        s = df[col]
        n_unique = s.nunique(dropna=True)
        if n_unique == 0 or n_unique > max_unique:
            continue
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s) or _is_integer_like_series(s, tol):
            cats.append(col)
    return cats


def _coerce_discrete_columns_to_int(df: pd.DataFrame, cols: list[str]) -> None:
    """Cast identified categorical-like numeric columns to nullable integer codes."""
    for col in cols:
        s = df[col]
        rounded = np.round(s)
        df[col] = pd.to_numeric(rounded, errors="coerce").astype("Int64")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_pst_timestamp() -> str:
    tz = ZoneInfo("America/Los_Angeles")
    return datetime.now(tz).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import mean_squared_error  # type: ignore
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mmss(delta_secs: float) -> str:
    m, s = divmod(delta_secs, 60)
    return f"{int(m)}m {s:05.2f}s"  # e.g., 0m 09.37s, 2m 03.04s


# ==============================================================================
# IMPUTATION
# ==============================================================================

def run_imputation(
    df: pd.DataFrame,
    passes: int = 14,
    alpha: float = 0.08,
    verbose: int = 1,
    use_feature_selector: bool = True,
    selector_tau: float = 0.8,
    selector_k_max: int = 30,
    gp_selector_k_max: int = 10,
    imputer_n_jobs: int = -1,
    categorical_threshold: int = 10,
    force_categorical_cols: Optional[List[str]] = None,
    tolerance_percentile: float = 95.0,
    tolerance_relaxation_factor: float = 1.3,
    tolerance_multiplier: float = 3.0,
    tier_quantiles: Optional[List[float]] = None,
    # v7.2: Per-column tolerance calibration
    calibrate_tolerances: bool = False,
    calibration_target_rmse_ratio: float = 0.5,
    calibration_n_rounds: int = 3,
    calibration_holdout_frac: float = 0.2,
    recalibrate_every_n_passes: int = 0,
    # v7.3: Imputer type selection
    imputer_type: str = "specialized",
    confidence_threshold: float = 0.4,
    coherence_lambda: float = 1.0,
    coherence_shape: str = "linear",
    coherence_gate: str = "fixed",
    iterative_coherence: bool = False,
    eb_parent: bool = False,
    use_svd_predictors: bool = False,
    n_expansion_passes: int = 1,
    max_confident_extras: int = 1,
):
    """Run the column imputer on benchmark data.

    This is a convenience wrapper that:
    1. Extracts numeric columns (excluding ID and target)
    2. Configures the selected imputer with appropriate hyperparameters
    3. Runs fit_transform and returns the imputed DataFrame

    Returns:
        Tuple of (imputed_df, imputer) where:
        - imputed_df: DataFrame with ID_COL and imputed numeric columns
        - imputer: Fitted imputer instance (SpecializedColumnImputer or ModelBankImputer)
    """
    numeric_cols = [c for c in df.columns
                    if c not in (ID_COL, TARGET, ALT_TARGET)
                    and pd.api.types.is_numeric_dtype(df[c])]
    X = df[numeric_cols].copy()
    missing_counts = X.isna().sum()
    total_missing = int(missing_counts.sum())

    if imputer_n_jobs is None or imputer_n_jobs <= 0:
        n_jobs_fit = -1
    else:
        n_jobs_fit = max(1, int(imputer_n_jobs))

    if imputer_type == "model_bank":
        imputer = ModelBankImputer(
            alpha=alpha,
            seed=SEED,
            verbose=verbose,
            n_jobs=n_jobs_fit,
            selector_k_max=selector_k_max,
            categorical_threshold=categorical_threshold,
            force_categorical_cols=force_categorical_cols,
            skew_threshold=2.0,
            confidence_threshold=confidence_threshold,
            coherence_lambda=coherence_lambda,
            coherence_shape=coherence_shape,
            coherence_gate=coherence_gate,
            iterative_coherence=iterative_coherence,
            eb_parent=eb_parent,
            use_svd_predictors=use_svd_predictors,
            n_expansion_passes=n_expansion_passes,
            max_confident_extras=max_confident_extras,
        )
    else:
        imputer = SpecializedColumnImputer(
            passes=passes,
            alpha=alpha,
            seed=SEED,
            verbose=verbose,
            n_jobs=n_jobs_fit,
            use_feature_selector=use_feature_selector,
            selector_tau=selector_tau,
            selector_k_max=selector_k_max,
            gp_selector_k_max=gp_selector_k_max,
            tier_quantiles=tier_quantiles,
            categorical_threshold=categorical_threshold,
            force_categorical_cols=force_categorical_cols,
            tolerance_percentile=tolerance_percentile,
            tolerance_relaxation_factor=tolerance_relaxation_factor,
            tolerance_multiplier=tolerance_multiplier,
            calibrate_tolerances=calibrate_tolerances,
            calibration_target_rmse_ratio=calibration_target_rmse_ratio,
            calibration_n_rounds=calibration_n_rounds,
            calibration_holdout_frac=calibration_holdout_frac,
            recalibrate_every_n_passes=recalibrate_every_n_passes,
        )
    X_imp = imputer.fit_transform(X)
    writes_by_col: Dict[str, int] = {}
    for entry in getattr(imputer, "logs_", []):
        col = entry.get("col")
        if col is not None:
            writes_by_col[col] = writes_by_col.get(col, 0) + 1
    median_fallback_total = 0
    for col, n_missing in missing_counts.items():
        median_fallback_total += max(0, int(n_missing) - int(writes_by_col.get(col, 0)))
    print(f"Median fallback fills: {median_fallback_total} of {total_missing} initial missing")

    imputed_df = df[[ID_COL]].join(X_imp)
    return imputed_df, imputer


def evaluate_imputation(imputer: SpecializedColumnImputer, orig_df: pd.DataFrame):

    numeric_cols = [c for c in orig_df.columns if c not in (ID_COL, TARGET, ALT_TARGET) and pd.api.types.is_numeric_dtype(orig_df[c])]
    X = orig_df[numeric_cols].copy()
    per_cell, per_col, by_bin = imputer.evaluate_quality_oof(X_df=X, n_splits=5, n_rounds=1, frac=1.0)

    # Handle case where evaluate_quality returns empty DataFrames (stub implementation)
    if per_cell.empty or "col" not in per_cell.columns:
        return per_cell, per_col, by_bin

    if "err" in per_cell.columns:
        pass
    elif "abs_err" in per_cell.columns and {"y_pred", "y_true"}.issubset(per_cell.columns):
        per_cell["err"] = per_cell["y_pred"] - per_cell["y_true"]
    else:
        per_cell["err"] = np.nan
    err_std = per_cell.groupby("col")["err"].std(ddof=1).rename("err_std").reset_index()
    per_col = per_col.merge(err_std, on="col", how="left")
    return per_cell, per_col, by_bin


# ==============================================================================
# CONFORMAL INTERVALS
# ==============================================================================

def prob_above_threshold(mu: np.ndarray, std: np.ndarray, threshold: float,
                         t_df: Optional[float] = None) -> np.ndarray:
    """P(score > threshold) using t-distribution when t_df is provided, else Gaussian."""
    scale = np.where(std <= 1e-12, 1e-12, std)
    z = (threshold - mu) / scale
    try:
        if t_df is not None and t_df > 0:
            from scipy.stats import t as t_dist  # type: ignore
            return 1.0 - t_dist.cdf(z, t_df)
        from scipy.stats import norm  # type: ignore
        return 1.0 - norm.cdf(z)
    except Exception:
        return 0.5 * (1.0 - np.erf(z / np.sqrt(2.0)))


def _detect_suite_missing_fracs(
    pre_imputation_missing: np.ndarray,
    feature_names: List[str],
) -> np.ndarray:
    """Compute per-suite missing fraction for each model.

    Groups features by prefix (e.g., 'livebench_', 'style_', 'aa_eval_')
    and returns the average suite-level missing fraction across all suites.
    A model missing all livebench columns scores 1.0 for that suite.
    """
    # Group column indices by prefix
    suite_indices: Dict[str, List[int]] = defaultdict(list)
    for i, name in enumerate(feature_names):
        prefix = name.split("_")[0] if "_" in name else name
        suite_indices[prefix].append(i)

    # Filter to suites with at least 2 columns (single-column "suites" aren't informative)
    multi_suites = {k: v for k, v in suite_indices.items() if len(v) >= 2}
    if not multi_suites:
        return np.zeros(pre_imputation_missing.shape[0])

    missing_float = pre_imputation_missing.astype(float)
    suite_missing_fracs = []
    for suite_name, col_indices in multi_suites.items():
        suite_cols = missing_float[:, col_indices]
        suite_missing_fracs.append(suite_cols.mean(axis=1))

    # Average across suites: how many suites is this model largely missing?
    return np.mean(np.column_stack(suite_missing_fracs), axis=1)


def compute_grouped_conformal_intervals(
    mu: np.ndarray,
    oof_preds: np.ndarray,
    y_train: np.ndarray,
    train_idx: np.ndarray,
    pre_imputation_missing: np.ndarray,
    feature_names: Optional[List[str]] = None,
    target_coverage: float = 0.95,
    top_threshold: float = 1400.0,
    min_group_size: int = 10,
    high_missing_frac: float = 0.35,
) -> dict:
    """Compute prediction intervals using coarse group-based conformal calibration.

    Instead of learning a 6-feature heteroscedastic scale model (which overfits
    at n~120), this assigns each model to a small number of predeclared groups
    and uses empirical OOF residual quantiles per group.

    Groups (hierarchical, backs off to parent if cell < min_group_size):
      1. predicted_top (mu >= threshold) vs predicted_rest
      2. Optionally split each by high vs low missingness

    Args:
        mu: Point predictions for ALL models, shape (n_all,).
        oof_preds: OOF predictions for training rows with valid OOF, shape (n_valid,).
        y_train: Actual target values for valid training rows, shape (n_valid,).
        train_idx: Global indices of valid training rows, shape (n_valid,).
        pre_imputation_missing: Boolean missing mask, shape (n_all, n_features).
        feature_names: Column names for pre_imputation_missing.
        target_coverage: Desired marginal coverage (default 0.95).
        top_threshold: ELO threshold for "top" group.
        min_group_size: Minimum group size; back off to parent if smaller.
        high_missing_frac: Threshold for "high missingness" split.

    Returns:
        Dict matching compute_normalized_conformal_intervals interface.
    """
    n_all = len(mu)
    n_valid = len(y_train)

    # --- Compute group assignments ---
    raw_missing_frac = pre_imputation_missing.astype(float).mean(axis=1)

    # Suite-level missingness
    suite_missing_frac = _detect_suite_missing_fracs(
        pre_imputation_missing,
        feature_names if feature_names is not None else [f"col_{i}" for i in range(pre_imputation_missing.shape[1])],
    )

    # OOF residuals
    oof_residuals = np.abs(y_train - oof_preds)

    # Group training rows by predicted score (use OOF preds to avoid leakage)
    is_top_train = oof_preds >= top_threshold
    is_high_missing_train = raw_missing_frac[train_idx] >= high_missing_frac

    # Build groups with hierarchical fallback
    groups = {}  # group_name -> (train_mask, all_mask)

    # Try 4-way split first
    candidates = {
        "top_low_miss": (is_top_train & ~is_high_missing_train,
                         (mu >= top_threshold) & (raw_missing_frac < high_missing_frac)),
        "top_high_miss": (is_top_train & is_high_missing_train,
                          (mu >= top_threshold) & (raw_missing_frac >= high_missing_frac)),
        "rest_low_miss": (~is_top_train & ~is_high_missing_train,
                          (mu < top_threshold) & (raw_missing_frac < high_missing_frac)),
        "rest_high_miss": (~is_top_train & is_high_missing_train,
                           (mu < top_threshold) & (raw_missing_frac >= high_missing_frac)),
    }

    # Back off small groups to parent
    parent_map = {
        "top_low_miss": "top", "top_high_miss": "top",
        "rest_low_miss": "rest", "rest_high_miss": "rest",
    }
    parent_candidates = {
        "top": (is_top_train, mu >= top_threshold),
        "rest": (~is_top_train, mu < top_threshold),
    }

    # Determine which groups are viable
    for name, (train_mask, all_mask) in candidates.items():
        if int(train_mask.sum()) >= min_group_size:
            groups[name] = (train_mask, all_mask)
        else:
            # Merge into parent
            parent = parent_map[name]
            if parent not in groups:
                p_train, p_all = parent_candidates[parent]
                groups[parent] = (p_train, p_all)

    # If any parent is also too small, fall back to global
    final_groups = {}
    for name, (train_mask, all_mask) in groups.items():
        if int(train_mask.sum()) >= min_group_size:
            final_groups[name] = (train_mask, all_mask)

    if not final_groups:
        # Global fallback
        final_groups["global"] = (np.ones(n_valid, dtype=bool), np.ones(n_all, dtype=bool))

    # --- Compute per-group quantiles ---
    pct = target_coverage * 100
    group_q = {}
    for name, (train_mask, _) in final_groups.items():
        resids = oof_residuals[train_mask]
        group_q[name] = float(np.percentile(resids, pct))

    # --- Assign sigma_hat (= halfwidth) for all models ---
    sigma_hat = np.full(n_all, np.nan)
    group_labels = np.full(n_all, "", dtype=object)

    for name, (_, all_mask) in final_groups.items():
        sigma_hat[all_mask & np.isnan(sigma_hat)] = group_q[name]
        group_labels[all_mask] = name

    # Any unassigned models get the global worst-case
    global_q = float(np.percentile(oof_residuals, pct))
    still_nan = np.isnan(sigma_hat)
    sigma_hat[still_nan] = global_q
    group_labels[still_nan] = "global_fallback"

    # --- Build intervals ---
    lower = mu - sigma_hat
    upper = mu + sigma_hat
    std_new = sigma_hat / 1.96

    # --- OOF coverage check (using OOF preds, not refit mu) ---
    oof_lower = oof_preds - sigma_hat[train_idx]
    oof_upper = oof_preds + sigma_hat[train_idx]
    oof_coverage = float(np.mean((y_train >= oof_lower) & (y_train <= oof_upper)))

    sigma_cv = float(np.std(sigma_hat) / np.mean(sigma_hat)) if np.mean(sigma_hat) > 0 else 0.0

    # --- Diagnostics ---
    scale_model_coef = {"method": "grouped_conformal"}
    for name, (train_mask, _) in final_groups.items():
        scale_model_coef[f"group_{name}_q{pct:.0f}"] = group_q[name]
        scale_model_coef[f"group_{name}_n_train"] = int(train_mask.sum())
    scale_model_coef["intercept"] = 0.0  # compat with diagnostics CSV

    print(f"  Grouped conformal ({len(final_groups)} groups, coverage target={target_coverage:.0%}):")
    for name in sorted(final_groups.keys()):
        train_mask, all_mask = final_groups[name]
        print(f"    {name}: n_train={int(train_mask.sum())}, n_all={int(all_mask.sum())}, "
              f"halfwidth=\u00b1{group_q[name]:.1f}")

    return {
        "std": std_new,
        "lower": lower,
        "upper": upper,
        "sigma_hat": sigma_hat,
        "q_hat": 1.0,  # halfwidth IS sigma_hat (no separate q_hat)
        "sigma_floor": 0.0,
        "oof_sigma": sigma_hat[train_idx],
        "sigma_cv": sigma_cv,
        "oof_coverage": oof_coverage,
        "scale_model_coef": scale_model_coef,
        "uncertainty_features": pd.DataFrame({
            "raw_missing_frac": raw_missing_frac,
            "suite_missing_frac": suite_missing_frac,
            "group": group_labels,
            "sigma_hat": sigma_hat,
        }),
    }


# ==============================================================================
# KNN PREDICTION
# ==============================================================================

def predict_adaptive_knn(
    Xtr: np.ndarray,
    y_tr: np.ndarray,
    Xte: np.ndarray,
    dist_mult: float = KNN_DIST_MULT,
    max_k: int = KNN_MAX_K,
    min_k: int = KNN_MIN_K,
    bw_pct: float = KNN_BW_PCT,
    vi_clip: tuple = KNN_VI_CLIP,
) -> Tuple[float, float, int]:
    """Predict a single test point using adaptive KNN + kernel Ridge + jackknife VI.

    Returns (prediction, std_estimate, k_used).
    """
    nn = NearestNeighbors(n_neighbors=max_k)
    nn.fit(Xtr)
    dists, idx = nn.kneighbors(Xte)
    d, ix = dists[0], idx[0]

    # Adaptive k: include all within dist_mult × nearest distance
    max_dist = d[0] * dist_mult
    k = max(min_k, min(int((d <= max_dist).sum()), max_k))

    y_nb = y_tr[ix[:k]]
    Xtr_nb = Xtr[ix[:k]]

    # Gaussian kernel weights
    bw_idx = max(1, int(k * bw_pct))
    bw = d[bw_idx] if bw_idx < k else d[k - 1]
    bw = max(bw, 1e-6)
    w = np.exp(-0.5 * (d[:k] / bw) ** 2)

    # Adaptive alpha = score spread of neighbors
    alpha = max(10.0, float(np.std(y_nb)))

    # Fit weighted Ridge
    mdl = Ridge(alpha=alpha)
    mdl.fit(Xtr_nb, y_nb, sample_weight=w)
    p = float(mdl.predict(Xte)[0])

    # Jackknife variance inflation
    mu_nb = float(np.mean(y_nb))
    jack_preds = np.zeros(k)
    for j in range(k):
        mj = np.ones(k, dtype=bool)
        mj[j] = False
        mdl_j = Ridge(alpha=alpha)
        mdl_j.fit(Xtr_nb[mj], y_nb[mj], sample_weight=w[mj])
        jack_preds[j] = mdl_j.predict(Xtr_nb[j:j + 1])[0]

    yc = y_nb - mu_nb
    pc = jack_preds - mu_nb
    denom = float(np.dot(pc, pc))
    if denom > 1e-8:
        b = float(np.clip(np.dot(yc, pc) / denom, vi_clip[0], vi_clip[1]))
    else:
        b = 1.0

    p_corrected = mu_nb + b * (p - mu_nb)

    # Rough std estimate from jackknife residuals
    jack_resid = y_nb - jack_preds
    std_est = float(np.std(jack_resid))

    return p_corrected, std_est, k


def fit_and_predict_knn(
    X_all: np.ndarray,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    cv_repeats: int = 10,
    cv_seed: int = SEED,
    dist_mult: float = KNN_DIST_MULT,
    max_k: int = KNN_MAX_K,
    min_k: int = KNN_MIN_K,
    bw_pct: float = KNN_BW_PCT,
) -> Dict[str, np.ndarray]:
    """Fit adaptive KNN on training data and predict all rows.

    Returns dict with keys: mu, std, lower, upper, oof_preds, oof_folds, ks_used.
    """
    n_all = len(y_all)
    n_train = len(train_idx)
    y_train = y_all[train_idx]

    # --- OOF predictions via CV splits ---
    if cv_splits is None:
        cv_splits = _build_repeated_splits(n_train, 5, cv_repeats, cv_seed)

    oof_preds_sum = np.zeros(n_train)
    oof_counts = np.zeros(n_train)
    oof_folds = np.full(n_train, -1, dtype=int)

    for fold_idx, (tr, va) in enumerate(cv_splits):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_all[train_idx[tr]])
        Xva = sc.transform(X_all[train_idx[va]])
        ytr = y_train[tr]

        for vi, va_i in enumerate(va):
            p, _, _ = predict_adaptive_knn(
                Xtr, ytr, Xva[vi:vi + 1],
                dist_mult=dist_mult, max_k=max_k, min_k=min_k, bw_pct=bw_pct,
            )
            oof_preds_sum[va_i] += p
            oof_counts[va_i] += 1
            oof_folds[va_i] = fold_idx % 5

    oof_preds = np.where(oof_counts > 0, oof_preds_sum / oof_counts, np.nan)
    oof_valid = oof_counts > 0
    oof_rmse = float(np.sqrt(np.nanmean((oof_preds[oof_valid] - y_train[oof_valid]) ** 2)))
    print(f"  KNN OOF RMSE: {oof_rmse:.2f} ({int(oof_valid.sum())}/{n_train} valid)")

    # --- Final predictions for all rows ---
    sc_final = StandardScaler()
    X_train_sc = sc_final.fit_transform(X_all[train_idx])
    X_all_sc = sc_final.transform(X_all)

    mu = np.full(n_all, np.nan)
    std = np.full(n_all, np.nan)
    ks_used = np.zeros(n_all, dtype=int)

    for i in pred_idx:
        p, s, k = predict_adaptive_knn(
            X_train_sc, y_train, X_all_sc[i:i + 1],
            dist_mult=dist_mult, max_k=max_k, min_k=min_k, bw_pct=bw_pct,
        )
        mu[i] = p
        std[i] = s
        ks_used[i] = k

    lower = mu - 1.96 * std
    upper = mu + 1.96 * std

    avg_k = float(np.mean(ks_used[pred_idx]))
    print(f"  KNN final predictions: {len(pred_idx)} models, avg k={avg_k:.0f}")

    return {
        "mu": mu,
        "std": std,
        "lower": lower,
        "upper": upper,
        "oof_preds": oof_preds,
        "oof_folds": oof_folds,
        "ks_used": ks_used,
    }


# ==============================================================================
# CV SPLIT HELPERS
# ==============================================================================

def _normalize_cv_repeats(repeats: int) -> int:
    try:
        reps = int(repeats)
    except Exception:
        reps = 1
    return max(1, reps)


def _extract_model_groups(model_names: pd.Series) -> np.ndarray:
    """Classify model names into provider groups for GroupKFold."""
    _PREFIX_MAP = [
        (["claude", "anthropic"], "Anthropic"),
        (["gpt", "o3", "o4", "chatgpt", "openai"], "OpenAI"),
        (["gemini", "gemma"], "Google"),
        (["qwen", "qwq"], "Alibaba"),
        (["deepseek"], "DeepSeek"),
        (["llama"], "Meta"),
        (["grok"], "xAI"),
        (["mistral", "mixtral", "codestral", "pixtral"], "Mistral"),
        (["command", "aya"], "Cohere"),
        (["phi"], "Microsoft"),
        (["nova"], "Amazon"),
        (["jamba"], "AI21"),
        (["reka"], "Reka"),
        (["yi"], "01.AI"),
    ]
    groups = []
    for name in model_names.astype(str).str.lower():
        matched = "Other"
        for prefixes, label in _PREFIX_MAP:
            if any(name.startswith(p) for p in prefixes):
                matched = label
                break
        groups.append(matched)
    return np.array(groups)


def _build_group_splits(
    n_rows: int,
    n_splits: int,
    group_labels: np.ndarray,
    repeats: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build CV splits using GroupKFold (groups stay together).

    For repeats > 1, permute the group->fold mapping to create
    different fold assignments while keeping groups intact.
    """
    reps = _normalize_cv_repeats(repeats)
    base_seed = int(seed) if seed is not None else SEED
    unique_groups = np.unique(group_labels)
    n_groups = len(unique_groups)
    effective_splits = min(n_splits, n_groups)
    if effective_splits < 2:
        raise ValueError(
            f"GroupKFold needs >= 2 groups but found {n_groups}. "
            "Consider merging small groups or using standard KFold."
        )

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    X_dummy = np.arange(n_rows)
    for rep in range(reps):
        rng = np.random.RandomState(base_seed + rep)
        # Permute group labels so each repeat has a different fold assignment
        perm = {g: rng.randint(0, 10**9) for g in unique_groups}
        permuted_labels = np.array([perm[g] for g in group_labels])
        gkf = GroupKFold(n_splits=effective_splits)
        splits.extend(
            (np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
            for tr, va in gkf.split(X_dummy, groups=permuted_labels)
        )
    return splits


def _build_repeated_splits(
    n_rows: int,
    n_splits: int,
    repeats: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_splits <= 1:
        return []
    reps = _normalize_cv_repeats(repeats)
    base_seed = int(seed) if seed is not None else SEED
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for rep in range(reps):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=base_seed + rep)
        splits.extend(
            (np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
            for tr, va in kf.split(np.arange(n_rows))
        )
    return splits


def _load_saved_splits(path: str, n_rows: int, n_splits: int, repeats: int):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("n_rows") != n_rows or payload.get("n_splits") != n_splits:
            return None
        saved_repeats = _normalize_cv_repeats(payload.get("repeats", 1))
        if saved_repeats != _normalize_cv_repeats(repeats):
            return None
        raw = payload.get("splits", [])
        splits = []
        for item in raw:
            tr = np.asarray(item.get("train", []), dtype=int)
            va = np.asarray(item.get("val", []), dtype=int)
            if tr.size == 0 or va.size == 0:
                continue
            splits.append((tr, va))
        if len(splits) == n_splits * _normalize_cv_repeats(repeats):
            return splits
    except Exception as exc:
        print(f"WARNING: failed to load CV splits from {path}: {exc}", file=sys.stderr)
    return None


def _save_splits(
    path: str,
    n_rows: int,
    n_splits: int,
    repeats: int,
    seed: int,
    splits: List[Tuple[np.ndarray, np.ndarray]],
):
    try:
        base = os.path.dirname(path)
        if base:
            os.makedirs(base, exist_ok=True)
        payload = {
            "n_rows": n_rows,
            "n_splits": n_splits,
            "repeats": _normalize_cv_repeats(repeats),
            "seed": seed,
            "splits": [
                {"train": tr.tolist(), "val": va.tolist()} for tr, va in splits
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception as exc:
        print(f"WARNING: failed to save CV splits to {path}: {exc}", file=sys.stderr)


def get_or_create_splits(
    n_rows: int,
    n_splits: int,
    path: Optional[str] = None,
    repeats: int = 1,
    seed: int = SEED,
) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    if n_splits <= 1:
        return None
    if path:
        loaded = _load_saved_splits(path, n_rows, n_splits, repeats)
        if loaded is not None:
            return loaded

    splits = _build_repeated_splits(n_rows, n_splits, repeats, seed)

    if path:
        _save_splits(path, n_rows, n_splits, repeats, seed, splits)
    return splits


# ==============================================================================
# IMPUTER VARIANCE CONTRIBUTIONS (for dependency graph)
# ==============================================================================

def compute_imputer_variance_contributions(
    imputer,
    X_df: pd.DataFrame,
    threshold: float = 0.01,
) -> Dict[str, set]:
    """Compute important predictors for each imputer model based on variance contribution.

    For each column model in the imputer, computes importance of predictors:
    - For linear models: uses coefficient-based variance contribution
    - For non-linear models: uses squared correlation (R^2) as importance proxy

    Args:
        imputer: Fitted SpecializedColumnImputer with models_ attribute.
        X_df: DataFrame with feature columns used for imputation.
        threshold: Minimum relative importance (default 1%).

    Returns:
        Dict mapping column name -> set of important predictor names.
    """
    important_predictors: Dict[str, set] = {}

    if imputer is None or not hasattr(imputer, "models_"):
        return important_predictors

    for col, model in imputer.models_.items():
        feature_names = getattr(model, "feature_names", [])
        if not feature_names:
            important_predictors[col] = set()
            continue

        # Get available features for this model
        available_cols = [f for f in feature_names if f in X_df.columns]
        if not available_cols or col not in X_df.columns:
            important_predictors[col] = set(feature_names)
            continue

        try:
            # Check if this is a linear model with coefficients
            inner_model = getattr(model, "model", None)
            scaler = getattr(model, "scaler", None)

            if inner_model is not None and hasattr(inner_model, "coef_"):
                # Linear model: use coefficient-based importance
                X_subset = X_df[available_cols].values
                coef = inner_model.coef_

                if len(coef) == len(available_cols):
                    feature_std = np.nanstd(X_subset, axis=0)
                    feature_std = np.where(feature_std == 0, 1e-10, feature_std)
                    contrib = np.abs(coef) * feature_std

                    total_contrib = contrib.sum()
                    if total_contrib > 0:
                        rel_contrib = contrib / total_contrib
                    else:
                        rel_contrib = np.ones(len(coef)) / len(coef)

                    important = set()
                    for i, feat in enumerate(available_cols):
                        if rel_contrib[i] >= threshold:
                            important.add(feat)
                    important_predictors[col] = important
                    continue

            # Non-linear model (GP, categorical): use correlation-based importance
            y = X_df[col].values
            valid_mask = ~np.isnan(y)

            if valid_mask.sum() < 10:
                important_predictors[col] = set(feature_names)
                continue

            r_squared = []
            for feat in available_cols:
                x = X_df[feat].values
                pair_valid = valid_mask & ~np.isnan(x)
                if pair_valid.sum() < 5:
                    r_squared.append(0.0)
                    continue
                x_valid = x[pair_valid]
                y_valid = y[pair_valid]
                x_mean = np.mean(x_valid)
                y_mean = np.mean(y_valid)
                cov = np.mean((x_valid - x_mean) * (y_valid - y_mean))
                std_x = np.std(x_valid)
                std_y = np.std(y_valid)
                if std_x > 0 and std_y > 0:
                    r = cov / (std_x * std_y)
                    r_squared.append(r ** 2)
                else:
                    r_squared.append(0.0)

            r_squared = np.array(r_squared)
            total_r2 = r_squared.sum()

            if total_r2 > 0:
                rel_importance = r_squared / total_r2
            else:
                rel_importance = np.ones(len(r_squared)) / len(r_squared)

            important = set()
            for i, feat in enumerate(available_cols):
                if rel_importance[i] >= threshold:
                    important.add(feat)

            important_predictors[col] = important

        except Exception:
            # On any error, keep all predictors
            important_predictors[col] = set(feature_names)

    return important_predictors


# ==============================================================================
# FEATURE COMPONENT PARSING (for dependency graph)
# ==============================================================================

def _feature_components(name: str) -> List[str]:
    """
    Break a interaction feature name (e.g., 'A*B' or 'C^2') into the base columns
    that generated it. Returns [name] when no decomposition is needed.

    Smart parsing: only splits on '*' if result looks like valid feature names
    (not fragments like "4" or "Fast" from "Grok 4 Fast").
    """
    s = str(name).strip()
    if "*" in s:
        parts = [p.strip() for p in s.split("*") if p.strip()]
        # Only treat as interaction if exactly 2 parts AND both look like valid features
        if len(parts) == 2:
            valid = all(
                len(p) > 3 and not p.replace(".", "").replace("-", "").isdigit()
                for p in parts
            )
            if valid:
                return parts
        return [s]
    if "^" in s:
        base = s.split("^", 1)[0].strip()
        return [base] if base else [s]
    return [s]


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Main entry point for the KNN prediction pipeline.

    Orchestrates:
    1. Argument parsing (imputation + KNN + output flags)
    2. Parallelism configuration
    3. Data loading and numeric column identification
    4. Low-variance filtering
    5. Imputation (ModelBankImputer via run_imputation, with caching)
    6. Feature matrix construction (imputed cols + SVD factors + trajectory)
    7. KNN prediction (fit_and_predict_knn)
    8. Grouped conformal interval calibration
    9. Output saving (predictions CSV, OOF CSV, metadata JSON, dependency graph)
    """
    ap = argparse.ArgumentParser(description="KNN Arena ELO Predictor")
    # -- Data I/O --
    ap.add_argument("--csv_path", type=str, default="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv")
    ap.add_argument("--output_root", type=str, default="analysis_output")
    # -- Imputer config --
    ap.add_argument("--passes", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=0.9361)
    ap.add_argument("--imputer_type", type=str, default="model_bank",
                    choices=["specialized", "model_bank"],
                    help="Imputer algorithm: 'model_bank' (default) or 'specialized' (SVD warm-start).")
    ap.add_argument("--confidence_threshold", type=float, default=0.4)
    ap.add_argument("--coherence_lambda", type=float, default=1.0)
    ap.add_argument("--coherence_shape", type=str, default="exp",
                    choices=["linear", "squared", "power3", "exp", "step"])
    ap.add_argument("--eb_parent", action="store_true",
                    help="Enable empirical-Bayes parent shrinkage in imputation.")
    # -- Tolerance config --
    ap.add_argument("--tolerance_percentile", type=float, default=91.1553)
    ap.add_argument("--tolerance_relaxation_factor", type=float, default=1.2704)
    ap.add_argument("--tolerance_multiplier", type=float, default=5.8849)
    ap.add_argument("--calibrate_tolerances", action="store_true")
    ap.add_argument("--calibration_target_rmse_ratio", type=float, default=0.6266)
    ap.add_argument("--calibration_n_rounds", type=int, default=3)
    ap.add_argument("--calibration_holdout_frac", type=float, default=0.2)
    ap.add_argument("--recalibrate_every_n_passes", type=int, default=5)
    # -- Feature selector (imputer) --
    ap.add_argument("--no_feature_selector", dest="use_feature_selector", action="store_false")
    ap.set_defaults(use_feature_selector=True)
    ap.add_argument("--selector_tau", type=float, default=0.9012)
    ap.add_argument("--selector_k_max", type=int, default=37)
    ap.add_argument("--gp_selector_k_max", type=int, default=28)
    # -- Parallelism --
    ap.add_argument("--imputer_n_jobs", type=int, default=-1)
    ap.add_argument("--max_workers", type=int, default=0)
    ap.add_argument("--cv_n_jobs", type=int, default=1)
    ap.add_argument("--selector_n_jobs", type=int, default=-2)
    # -- CV config --
    ap.add_argument("--cv_repeats_outer", type=int, default=None)
    ap.add_argument("--cv_seed", type=int, default=SEED)
    ap.add_argument("--outer_cv", type=int, default=5)
    ap.add_argument("--selector_cv", type=int, default=5)
    ap.add_argument("--cv_splits_path", type=str, default="")
    ap.add_argument("--cv_repeats", type=int, default=1)
    ap.add_argument("--group_cv", action="store_true",
                    help="Use GroupKFold by model provider (diagnostic).")
    # -- KNN config --
    ap.add_argument("--knn_predict", action="store_true", default=True,
                    help="Use adaptive KNN prediction (default: True).")
    ap.add_argument("--knn_dist_mult", type=float, default=2.0)
    ap.add_argument("--knn_max_k", type=int, default=80)
    ap.add_argument("--knn_min_k", type=int, default=20)
    ap.add_argument("--knn_bw_pct", type=float, default=0.3)
    # -- Misc --
    ap.add_argument("--margin", type=float, default=20.0,
                    help="Margin for 'top_by_margin_prob' column (default 20 points).")
    ap.add_argument("--categorical_threshold", type=int, default=0)
    ap.add_argument("--forced_categorical_cols", type=str, default="")
    ap.add_argument("--tier_quantiles", type=str, default="0.33,0.67")
    ap.add_argument("--exclude_models", type=str, default="",
                    help="Comma-separated model names to exclude from the dataset.")
    ap.add_argument("--svd_in_alt", dest="svd_in_features", action="store_true", default=True,
                    help="Add SVD factors from imputer to feature matrix (default: True).")
    ap.add_argument("--no_svd_in_alt", dest="svd_in_features", action="store_false",
                    help="Exclude SVD factors from feature matrix.")

    args = ap.parse_args()

    # Resolve CV repeats
    args.cv_repeats = max(1, int(args.cv_repeats))
    if args.cv_repeats_outer is None:
        args.cv_repeats_outer = args.cv_repeats
    else:
        args.cv_repeats_outer = max(1, int(args.cv_repeats_outer))
    args.cv_seed = int(args.cv_seed)

    # Need model_n_jobs for _configure_parallelism
    args.model_n_jobs = 1

    _configure_parallelism(args)

    # -- Parse tier quantiles --
    tier_quantiles: Optional[List[float]] = None
    parsed_tiers: List[float] = []
    for part in str(getattr(args, "tier_quantiles", "")).split(","):
        val = part.strip()
        if not val:
            continue
        try:
            num = float(val)
        except ValueError:
            continue
        if 0 <= num <= 1:
            parsed_tiers.append(num)
    if parsed_tiers:
        tier_quantiles = sorted(parsed_tiers)

    start = time.time()

    stamp = now_pst_timestamp()
    out_dir = os.path.join(args.output_root, f"output_{stamp}")
    ensure_dir(out_dir)

    # =========================================================================
    # 1. Load CSV
    # =========================================================================
    if not os.path.exists(args.csv_path):
        print(f"ERROR: CSV not found at {args.csv_path}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.csv_path)
    if args.exclude_models:
        exclude_set = {m.strip() for m in args.exclude_models.split(",") if m.strip()}
        n_before = len(df)
        df = df[~df[ID_COL].isin(exclude_set)].reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"Excluded {n_dropped} model(s): {exclude_set}")
    if ID_COL not in df.columns:
        raise ValueError(f"CSV must contain '{ID_COL}' column.")

    numeric_cols = [c for c in df.columns if c != ID_COL and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = get_feature_cols(df)
    categorical_numeric_cols = _find_numeric_categoricals(df, feature_cols, max_unique=10)
    if categorical_numeric_cols:
        print(f"Detected {len(categorical_numeric_cols)} low-cardinality integer-like column(s): {categorical_numeric_cols}")
        _coerce_discrete_columns_to_int(df, categorical_numeric_cols)

    # =========================================================================
    # 2. Low-variance filtering
    # =========================================================================
    def _is_low_variance(col: pd.Series, dominance_thresh: float = 0.7, min_std: float = 1e-8, min_minority: int = 2) -> bool:
        s = col.dropna()
        if len(s) <= 1:
            return True
        if s.std(ddof=0) < min_std:
            return True
        vc = s.value_counts(normalize=True)
        if vc.empty:
            return True
        minority_count = len(s) - int(s.value_counts().iloc[0])
        if vc.iloc[0] >= dominance_thresh or minority_count < min_minority:
            return True
        return False

    low_var_cols = [c for c in feature_cols if _is_low_variance(df[c])]
    if low_var_cols:
        print(f"Dropping {len(low_var_cols)} low-variance/constant feature column(s): {low_var_cols}")
        df = df.drop(columns=low_var_cols)
        feature_cols = [c for c in feature_cols if c not in low_var_cols]
        categorical_numeric_cols = [c for c in categorical_numeric_cols if c in feature_cols]

    if TARGET not in numeric_cols:
        raise ValueError(f"CSV must contain numeric target column '{TARGET}'. Found numeric: {numeric_cols}")

    y_orig = df[TARGET].copy()
    y_missing_mask = y_orig.isna().values
    y_all = pd.to_numeric(df[TARGET], errors="coerce").to_numpy()

    missing_count_by_col = df[feature_cols].isna().sum().to_dict()

    load_end = time.time()
    print(f"load data: {mmss(load_end - start)}")

    # =========================================================================
    # 3. Imputation (with caching)
    # =========================================================================
    imputer: Optional[object] = None
    imputer_predictors_map: Dict[str, List[str]] = {}
    imputer_important_predictors: Dict[str, set] = {}

    cache_dir = os.path.join(args.output_root, "_cache")
    ensure_dir(cache_dir)
    csv_hash = _sha256_file(args.csv_path)
    if args.exclude_models:
        import hashlib as _hl
        excl_hash = _hl.sha256(args.exclude_models.encode()).hexdigest()[:12]
        csv_hash = csv_hash[:52] + excl_hash
    tier_key = "none" if tier_quantiles is None else "_".join(f"{q:.3f}" for q in tier_quantiles)

    if args.imputer_type == "model_bank":
        coh_lam = getattr(args, 'coherence_lambda', 1.0)
        coh_shape = getattr(args, 'coherence_shape', 'linear')
        eb = int(getattr(args, 'eb_parent', False))
        imp_key = (
            f"imputed_modelbank_{csv_hash}_alpha{args.alpha:.6f}_"
            f"skmax{args.selector_k_max}_conf{args.confidence_threshold:.2f}_"
            f"coh{coh_lam:.2f}{coh_shape[0]}_gfixed_ic0_svdp0_exp1x1_"
            f"eb{eb}_catthr{args.categorical_threshold}_catovr{len(categorical_numeric_cols)}_skt3.0.csv"
        )
    else:
        imp_key = (
            f"imputed_full_{csv_hash}_passes{args.passes}_alpha{args.alpha:.6f}_"
            f"sel{int(args.use_feature_selector)}_st{args.selector_tau:.3f}_"
            f"skmax{args.selector_k_max}_imnj{PARALLELISM_CFG['imputer_n_jobs']}_"
            f"tolp{args.tolerance_percentile:.1f}_tolr{args.tolerance_relaxation_factor:.2f}_"
            f"tolm{args.tolerance_multiplier:.2f}_tier{tier_key}_"
            f"catthr{args.categorical_threshold}_catovr{len(categorical_numeric_cols)}_skt2.0.csv"
        )

    cache_pkl = os.path.join(cache_dir, imp_key.replace(".csv", ".pkl"))
    cache_meta = os.path.join(cache_dir, imp_key + ".meta.json")
    cache_csv = os.path.join(cache_dir, imp_key)

    imputed_path = os.path.join(out_dir, "imputed_full.csv")
    _cache_hit = False
    if os.path.exists(cache_pkl):
        import pickle as _pkl
        with open(cache_pkl, "rb") as fh:
            _cached = _pkl.load(fh)
        imputed_df = _cached["imputed_df"]

        class _CachedImputer:
            pass
        imputer = _CachedImputer()
        imputer.svd_row_factors_ = _cached.get("svd_row_factors")
        imputer.trajectory_features_ = _cached.get("trajectory_features")
        imputer.sigma2_matrix_ = _cached.get("sigma2_matrix")
        _cache_hit = True
    elif os.path.exists(cache_csv):
        imputed_df = pd.read_csv(cache_csv)

        class _CachedImputer:
            pass
        svd_cache = cache_csv + ".svd_factors.csv"
        traj_cache = cache_csv + ".trajectory.csv"
        imputer = _CachedImputer()
        imputer.svd_row_factors_ = pd.read_csv(svd_cache, index_col=0) if os.path.exists(svd_cache) else None
        imputer.trajectory_features_ = pd.read_csv(traj_cache, index_col=0) if os.path.exists(traj_cache) else None
        _cache_hit = True

    if _cache_hit:
        if os.path.exists(cache_meta):
            try:
                with open(cache_meta, "r", encoding="utf-8") as fh:
                    cache_payload = json.load(fh)
                raw_map = cache_payload.get("predictors_map", {})
                if isinstance(raw_map, dict):
                    imputer_predictors_map = {
                        str(col): [str(dep) for dep in deps] if isinstance(deps, list) else []
                        for col, deps in raw_map.items()
                    }
                raw_important = cache_payload.get("important_predictors", {})
                if isinstance(raw_important, dict):
                    imputer_important_predictors = {
                        str(col): set(deps) if isinstance(deps, list) else set()
                        for col, deps in raw_important.items()
                    }
            except Exception as exc:
                print(f"WARNING: failed to load imputer metadata cache ({exc}).", file=sys.stderr)
        already_done = True
        print(f"  Imputation cache hit: {imp_key}")
    else:
        imputed_df, imputer = run_imputation(
            df[[ID_COL] + feature_cols],
            passes=args.passes,
            alpha=args.alpha,
            verbose=0,
            use_feature_selector=args.use_feature_selector,
            selector_tau=args.selector_tau,
            selector_k_max=args.selector_k_max,
            gp_selector_k_max=args.gp_selector_k_max,
            imputer_n_jobs=PARALLELISM_CFG["imputer_n_jobs"],
            categorical_threshold=args.categorical_threshold,
            force_categorical_cols=categorical_numeric_cols,
            tolerance_percentile=args.tolerance_percentile,
            tolerance_relaxation_factor=args.tolerance_relaxation_factor,
            tolerance_multiplier=args.tolerance_multiplier,
            tier_quantiles=tier_quantiles,
            calibrate_tolerances=args.calibrate_tolerances,
            calibration_target_rmse_ratio=args.calibration_target_rmse_ratio,
            calibration_n_rounds=args.calibration_n_rounds,
            calibration_holdout_frac=args.calibration_holdout_frac,
            recalibrate_every_n_passes=args.recalibrate_every_n_passes,
            imputer_type=args.imputer_type,
            confidence_threshold=args.confidence_threshold,
            coherence_lambda=getattr(args, 'coherence_lambda', 1.0),
            coherence_shape=getattr(args, 'coherence_shape', 'linear'),
            eb_parent=getattr(args, 'eb_parent', False),
        )
        # Save imputation cache as pickle for exact float round-trip
        import pickle as _pkl
        _cache_payload_pkl = {
            "imputed_df": imputed_df,
            "svd_row_factors": getattr(imputer, 'svd_row_factors_', None),
            "trajectory_features": getattr(imputer, 'trajectory_features_', None),
            "sigma2_matrix": getattr(imputer, 'sigma2_matrix_', None),
        }
        with open(cache_pkl, "wb") as fh:
            _pkl.dump(_cache_payload_pkl, fh, protocol=4)
        imputer_predictors_map = {
            col: sorted(set(deps))
            for col, deps in getattr(imputer, "predictors_map_", {}).items()
        }
        imputer_important_predictors = compute_imputer_variance_contributions(
            imputer, imputed_df, threshold=0.01
        )
        with open(cache_meta, "w", encoding="utf-8") as fh:
            json.dump({
                "predictors_map": imputer_predictors_map,
                "important_predictors": {col: sorted(deps) for col, deps in imputer_important_predictors.items()},
            }, fh, indent=2)
        already_done = False

    # Always write the copy for this run
    imputed_df.to_csv(imputed_path, index=False)
    impute_end = time.time()
    print(f"impute:    {mmss(impute_end - load_end)}")

    # =========================================================================
    # 4. Imputation quality evaluation
    # =========================================================================
    quality_files = [
        "imputation_quality_per_cell.csv",
        "imputation_quality_per_column.csv",
        "imputation_quality_by_extrapolation_bin.csv",
    ]
    if not already_done:
        X_eval = df[feature_cols]
        per_cell, per_col, by_bin = evaluate_imputation(imputer, X_eval)
        per_cell.to_csv(os.path.join(out_dir, "imputation_quality_per_cell.csv"), index=False)
        per_col.to_csv(os.path.join(out_dir, "imputation_quality_per_column.csv"), index=False)
        by_bin.to_csv(os.path.join(out_dir, "imputation_quality_by_extrapolation_bin.csv"), index=False)
        try:
            imp_importance = imputer.get_imputation_importance()
            if not imp_importance.empty:
                imp_importance.to_csv(os.path.join(out_dir, "imputation_importance.csv"), index=False)
        except Exception:
            pass
        for qf in quality_files:
            src = os.path.join(out_dir, qf)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(cache_dir, imp_key + "." + qf))
    else:
        for qf in quality_files:
            cached_qf = os.path.join(cache_dir, imp_key + "." + qf)
            if os.path.exists(cached_qf):
                shutil.copy(cached_qf, os.path.join(out_dir, qf))

    # =========================================================================
    # 5. Build feature matrix
    # =========================================================================
    safe_features = imputed_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = safe_features.median(axis=0, skipna=True)
    safe_features = safe_features.fillna(medians)
    still_nan_cols = safe_features.columns[safe_features.isna().any()].tolist()
    if still_nan_cols:
        safe_features[still_nan_cols] = safe_features[still_nan_cols].fillna(0.0)
    if safe_features.isna().any().any():
        safe_features = safe_features.fillna(0.0)

    # Add SVD factors from imputer if available
    if args.svd_in_features and imputer is not None and hasattr(imputer, 'svd_row_factors_') and imputer.svd_row_factors_ is not None:
        svd_factors = imputer.svd_row_factors_
        svd_col_names = list(svd_factors.columns)
        for col in svd_col_names:
            if col not in safe_features.columns:
                safe_features[col] = svd_factors[col].values
                safe_features[f"{col}_sq"] = svd_factors[col].values ** 2
        n_interact = min(4, len(svd_col_names))
        for ci_idx, cj_idx in itertools.combinations(range(n_interact), 2):
            ci = svd_col_names[ci_idx]
            cj = svd_col_names[cj_idx]
            key = f"{ci}x{cj}"
            if key not in safe_features.columns:
                safe_features[key] = svd_factors[ci].values * svd_factors[cj].values

    # Add trajectory features from imputer if available
    if imputer is not None and hasattr(imputer, 'trajectory_features_') and imputer.trajectory_features_ is not None:
        for col in imputer.trajectory_features_.columns:
            if col not in safe_features.columns:
                safe_features[col] = imputer.trajectory_features_[col].values

    # Exclude target columns from feature matrix
    knn_feature_cols = [c for c in safe_features.columns
                        if c != ALT_TARGET and c != TARGET]
    knn_X = safe_features[knn_feature_cols].values
    print(f"  KNN features: {len(knn_feature_cols)} (including SVD/traj)")

    # =========================================================================
    # 6. Identify train/pred indices
    # =========================================================================
    train_idx = np.where(~y_missing_mask)[0]
    pred_idx = np.arange(len(y_all))  # predict ALL models

    if len(train_idx) < 3:
        raise ValueError(f"Not enough rows with observed {TARGET} to train models.")

    # =========================================================================
    # 7. Build CV splits
    # =========================================================================
    preprocess_end = time.time()
    print(f"preprocessing:    {mmss(preprocess_end - impute_end)}")

    target_cv_splits = None
    if args.group_cv:
        train_model_names = imputed_df[ID_COL].values[~y_missing_mask]
        group_labels_cv = _extract_model_groups(pd.Series(train_model_names))
        group_counts = dict(pd.Series(group_labels_cv).value_counts())
        small_groups = {g for g, c in group_counts.items() if c < 4 and g != "Other"}
        if small_groups:
            group_labels_cv = np.array([
                "Other" if g in small_groups else g for g in group_labels_cv
            ])
        print(f"GroupKFold: {len(set(group_labels_cv))} groups")
        target_cv_splits = _build_group_splits(
            len(train_idx), args.outer_cv, group_labels_cv,
            args.cv_repeats_outer, args.cv_seed)
    elif args.cv_splits_path:
        target_cv_splits = get_or_create_splits(
            len(train_idx),
            args.outer_cv,
            args.cv_splits_path,
            repeats=args.cv_repeats_outer,
            seed=args.cv_seed,
        )

    # =========================================================================
    # 8. KNN prediction
    # =========================================================================
    print("Using adaptive KNN prediction pipeline")
    knn_result = fit_and_predict_knn(
        knn_X,
        y_all,
        train_idx,
        pred_idx,
        cv_splits=target_cv_splits,
        cv_repeats=args.cv_repeats_outer,
        cv_seed=args.cv_seed,
        dist_mult=args.knn_dist_mult,
        max_k=args.knn_max_k,
        min_k=args.knn_min_k,
        bw_pct=args.knn_bw_pct,
    )
    mu = knn_result["mu"]
    std = knn_result["std"]
    lower = knn_result["lower"]
    upper = knn_result["upper"]
    oof_preds = knn_result["oof_preds"]
    oof_folds = knn_result["oof_folds"]

    train_end = time.time()
    print(f"train:     {mmss(train_end - preprocess_end)}")

    # =========================================================================
    # 9. Conformal intervals
    # =========================================================================
    conformal_start = time.time()

    pre_imputation_missing = df[feature_cols].isna().values

    oof_valid_mask = ~np.isnan(oof_preds)
    oof_preds_valid = oof_preds[oof_valid_mask]
    y_train_valid = y_all[~y_missing_mask][oof_valid_mask]
    train_idx_all = np.where(~y_missing_mask)[0]
    train_idx_valid = train_idx_all[oof_valid_mask]

    conformal = compute_grouped_conformal_intervals(
        mu=mu,
        oof_preds=oof_preds_valid,
        y_train=y_train_valid,
        train_idx=train_idx_valid,
        pre_imputation_missing=pre_imputation_missing,
        feature_names=list(feature_cols),
        target_coverage=0.95,
        top_threshold=1400.0,
        min_group_size=10,
    )

    # Overwrite intervals with conformal calibration
    std = conformal["std"]
    lower = conformal["lower"]
    upper = conformal["upper"]

    # Fit t-distribution df from final calibrated OOF residuals
    q_hat = conformal["q_hat"]
    sigma_hat_arr = conformal["sigma_hat"]
    cal_sigma_final = q_hat * sigma_hat_arr[train_idx_valid]
    oof_residuals = y_all[train_idx_valid] - oof_preds_valid
    oof_valid = ~np.isnan(oof_residuals)
    z_final = oof_residuals[oof_valid] / np.where(cal_sigma_final[oof_valid] < 1e-12, 1e-12, cal_sigma_final[oof_valid])
    t_df = None
    try:
        from scipy.stats import t as t_dist  # type: ignore
        t_df_fit, _t_loc, _t_scale = t_dist.fit(z_final)
        t_df = float(np.clip(t_df_fit, 3.0, 200.0))
        t_crit = float(t_dist.ppf(0.975, t_df))
        std = q_hat * sigma_hat_arr / t_crit
    except Exception:
        t_df = None

    conformal_end = time.time()
    print(f"conformal: {mmss(conformal_end - conformal_start)}")
    t_df_str = f"t_df={t_df:.1f}" if t_df is not None else "t_df=None (Gaussian fallback)"
    print(f"  q_hat={conformal['q_hat']:.3f}  sigma_floor={conformal['sigma_floor']:.3f}  "
          f"sigma_cv={conformal['sigma_cv']:.1%}  oof_coverage={conformal['oof_coverage']:.1%}  {t_df_str}")

    # =========================================================================
    # 10. Compute probabilities
    # =========================================================================
    max_observed = float(np.nanmax(y_all[train_idx]))
    num_one_prob = prob_above_threshold(mu, std, threshold=max_observed, t_df=t_df)
    top_by_margin_prob = prob_above_threshold(mu, std, threshold=max_observed + args.margin, t_df=t_df)
    top_by_margin_prob[train_idx] = np.nan

    # =========================================================================
    # 11. Save outputs
    # =========================================================================

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

    # -- oof_predictions.csv --
    train_names = imputed_df[ID_COL].values[~y_missing_mask]
    oof_df = pd.DataFrame({
        "model_name": train_names,
        "actual_score": y_all[~y_missing_mask],
        "oof_predicted_score": oof_preds,
        "fold": oof_folds,
    })
    oof_df = oof_df.dropna(subset=["oof_predicted_score"])
    oof_df.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    print(f"OOF predictions: {len(oof_df)} rows saved")

    # -- model_eval_rmse.csv --
    oof_rmse_val = float(np.sqrt(np.nanmean((oof_preds[oof_valid_mask] - y_all[train_idx][oof_valid_mask]) ** 2)))
    eval_df = pd.DataFrame([{"model": "KNN", "rmse_mean": oof_rmse_val, "rmse_std": 0.0}])
    eval_df.to_csv(os.path.join(out_dir, "model_eval_rmse.csv"), index=False)

    # -- conformal_diagnostics.csv --
    diag_row = {
        "q_hat": conformal["q_hat"],
        "sigma_floor": conformal["sigma_floor"],
        "sigma_cv": conformal["sigma_cv"],
        "oof_coverage": conformal["oof_coverage"],
        "t_df": t_df,
    }
    for k, v in conformal["scale_model_coef"].items():
        diag_row[f"scale_model_coef_{k}"] = v
    pd.DataFrame([diag_row]).to_csv(os.path.join(out_dir, "conformal_diagnostics.csv"), index=False)

    # -- conformal_uncertainty_features.csv --
    uf = conformal["uncertainty_features"].copy()
    uf.insert(0, ID_COL, imputed_df[ID_COL].values)
    uf["is_train"] = False
    uf.loc[train_idx, "is_train"] = True
    uf.to_csv(os.path.join(out_dir, "conformal_uncertainty_features.csv"), index=False)

    # -- run_config.json --
    run_config = {
        "csv_path": args.csv_path,
        "passes": int(args.passes),
        "alpha": float(args.alpha),
        "imputer_type": args.imputer_type,
        "use_feature_selector": bool(args.use_feature_selector),
        "selector_tau": float(args.selector_tau),
        "selector_k_max": int(args.selector_k_max),
        "imputer_n_jobs": int(args.imputer_n_jobs),
        "categorical_threshold": int(args.categorical_threshold),
        "tolerance_percentile": float(args.tolerance_percentile),
        "tolerance_relaxation_factor": float(args.tolerance_relaxation_factor),
        "tolerance_multiplier": float(args.tolerance_multiplier),
        "tier_quantiles": tier_quantiles or [],
        "cv_repeats_outer": int(args.cv_repeats_outer),
        "cv_seed": int(args.cv_seed),
        "outer_cv": int(args.outer_cv),
        "max_workers": int(args.max_workers),
        "cv_splits_path": args.cv_splits_path,
        "group_cv": bool(args.group_cv),
        "eb_parent": bool(getattr(args, 'eb_parent', False)),
        "coherence_lambda": float(getattr(args, 'coherence_lambda', 1.0)),
        "coherence_shape": str(getattr(args, 'coherence_shape', 'exp')),
        "confidence_threshold": float(args.confidence_threshold),
        "knn_predict": True,
        "knn_dist_mult": float(args.knn_dist_mult),
        "knn_max_k": int(args.knn_max_k),
        "knn_min_k": int(args.knn_min_k),
        "knn_bw_pct": float(args.knn_bw_pct),
        "svd_in_features": bool(args.svd_in_features),
        "used_cache": bool(_cache_hit),
        "cache_key": imp_key,
        "output_dir": out_dir,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)

    # -- metadata.json --
    from sklearn.metrics import mean_squared_error  # type: ignore
    oof_rmse = float(np.sqrt(mean_squared_error(y_train_valid, oof_preds_valid)))
    rng = np.random.RandomState(SEED)
    residuals_sq = (y_train_valid - oof_preds_valid) ** 2
    n_boot = len(residuals_sq)
    boot_rmses = np.array([np.sqrt(np.mean(residuals_sq[rng.randint(0, n_boot, size=n_boot)])) for _ in range(2000)])
    ci_lo, ci_hi = np.percentile(boot_rmses, [2.5, 97.5])

    meta = {
        "timestamp": stamp,
        "timezone": "America/Los_Angeles",
        "csv_path": os.path.abspath(args.csv_path),
        "n_rows": int(df.shape[0]),
        "n_features_numeric": int(len(numeric_cols)),
        "n_knn_features": len(knn_feature_cols),
        "target": TARGET,
        "best_model": "KNN",
        "oof_rmse": round(oof_rmse, 4),
        "oof_rmse_ci_lo": round(ci_lo, 4),
        "oof_rmse_ci_hi": round(ci_hi, 4),
        "cv_repeats_outer": int(args.cv_repeats_outer),
        "notes": [
            "Safety-filled any residual NaNs/Infs in features post-imputation with column medians, then 0 if still NaN.",
            "num_one_prob compares to current max observed lmarena_Score among training rows.",
            f"top_by_margin_prob = P(score > max_observed + {args.margin}), only for models without actual scores.",
            "predictions_best_model.csv is sorted by predicted_score descending (highest on top).",
        ]
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # =========================================================================
    # 12. Column dependency graph
    # =========================================================================
    dependency_graph: Dict[str, set] = defaultdict(set)

    def register(source: str, targets: List[str]) -> None:
        if not source:
            return
        dep_set = dependency_graph.setdefault(source, set())
        for tgt in targets:
            if tgt and tgt != source:
                dep_set.add(tgt)

    root_node = "__target_model__"
    register(root_node, knn_feature_cols)

    # Register feature components (interactions -> base features)
    for feat in list(dependency_graph.get(root_node, [])):
        dependency_graph.setdefault(feat, set())
        components = _feature_components(feat)
        if components != [feat]:
            register(feat, components)
            for comp in components:
                dependency_graph[root_node].add(comp)

    def _col_has_missing(col: str) -> bool:
        return missing_count_by_col.get(col, 0) > 0

    imputation_targets = {
        col for col in imputer_predictors_map.keys()
        if _col_has_missing(col)
    }

    _get_suite = None
    try:
        from column_imputer import get_benchmark_suite
        _get_suite = get_benchmark_suite
    except Exception:
        pass

    for col, preds in imputer_predictors_map.items():
        if not _col_has_missing(col):
            continue
        dependency_graph.setdefault(col, set())
        important_preds = imputer_important_predictors.get(col)
        if important_preds is not None:
            filtered_preds = [p for p in (preds or []) if p in important_preds]
        else:
            filtered_preds = list(preds or [])
        # Drop same-suite links
        if _get_suite is not None:
            col_suite = _get_suite(col)
            filtered_preds = [
                p for p in filtered_preds
                if col_suite == '_other' or _get_suite(p) != col_suite
            ]
        register(col, filtered_preds)

    # Ensure every dependency node exists
    for deps in list(dependency_graph.values()):
        for dep in deps:
            dependency_graph.setdefault(dep, set())
    base_feature_set = set(feature_cols)
    for col in base_feature_set:
        dependency_graph.setdefault(col, set())

    _transform_suffix = re.compile(r"(.+)_([A-Za-z0-9]+)~$")

    def _is_missingness_flag_name(col: str) -> bool:
        return str(col).endswith("__was_missing")

    def _transform_base_name(col: str) -> Optional[str]:
        if _is_missingness_flag_name(col):
            return None
        m = _transform_suffix.match(str(col))
        if not m:
            return None
        return m.group(1)

    transform_base_map: Dict[str, str] = {}
    for col in base_feature_set:
        base = _transform_base_name(col)
        if base:
            transform_base_map[col] = base

    reachable_depth: Dict[str, int] = {}
    queue = deque([(root_node, 0)])
    while queue:
        node, depth = queue.popleft()
        if node in reachable_depth:
            continue
        reachable_depth[node] = depth
        for dep in sorted(dependency_graph.get(node, [])):
            if dep not in reachable_depth:
                queue.append((dep, depth + 1))

    reachable_nodes = set(reachable_depth.keys())
    transforms_used_by_base: Dict[str, set] = defaultdict(set)
    for tcol, base in transform_base_map.items():
        if tcol in reachable_nodes and base in base_feature_set and not _is_missingness_flag_name(base):
            transforms_used_by_base[base].add(tcol)

    contributing_cols = sorted([c for c in base_feature_set if c in reachable_depth])
    dead_weight_cols = sorted([c for c in base_feature_set if c not in reachable_depth and c not in transforms_used_by_base])

    dep_graph_path = os.path.join(out_dir, "column_dependency_graph.json")
    dep_summary_path = os.path.join(out_dir, "column_dependency_summary.json")
    dep_table_path = os.path.join(out_dir, "column_dependency_summary.csv")
    degrees_path = os.path.join(out_dir, "column_degrees_of_separation.csv")

    with open(dep_graph_path, "w", encoding="utf-8") as fh:
        json.dump({node: sorted(deps) for node, deps in dependency_graph.items()}, fh, indent=2)

    dep_summary_payload = {
        "all_model_features": knn_feature_cols,
        "contributing_columns": contributing_cols,
        "dead_weight_columns": dead_weight_cols,
        "total_base_columns": len(feature_cols),
        "contributing_count": len(contributing_cols),
        "dead_weight_count": len(dead_weight_cols),
    }
    with open(dep_summary_path, "w", encoding="utf-8") as fh:
        json.dump(dep_summary_payload, fh, indent=2)

    degrees_rows = []
    for col in sorted(dependency_graph.keys()):
        degrees_rows.append({
            "column": col,
            "min_hops_from_target": reachable_depth.get(col, ""),
        })
    pd.DataFrame(degrees_rows).to_csv(degrees_path, index=False)

    used_feature_set = set(knn_feature_cols)
    rows = []
    for col in sorted(dependency_graph.keys()):
        deps_sorted = sorted(dependency_graph[col])
        rows.append({
            "column": col,
            "reachable_from_target": col in reachable_depth,
            "min_hops_from_target": reachable_depth.get(col, ""),
            "direct_dependencies": ";".join(deps_sorted),
            "is_final_feature": col in used_feature_set,
            "is_imputation_target": col in imputation_targets,
            "is_base_feature": col in base_feature_set,
        })
    pd.DataFrame(rows).to_csv(dep_table_path, index=False)
    print("column dependency outputs:")
    print(f"  graph:   {dep_graph_path}")
    print(f"  summary: {dep_summary_path}")
    print(f"  table:   {dep_table_path}")
    print(f"  degrees: {degrees_path}")

    # =========================================================================
    # 13. Summary
    # =========================================================================
    print(f"OOF RMSE:  {oof_rmse:.2f}  (95% CI: {ci_lo:.2f} \u2013 {ci_hi:.2f})")
    print(f"Done. Results saved to: {out_dir}")
    end = time.time()
    print(f"total:     {mmss(end - start)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
