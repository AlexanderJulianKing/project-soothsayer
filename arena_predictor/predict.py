#!/usr/bin/env python3
"""
Arena ELO Predictor — KNN Pipeline.

Predicts lmarena_Score (style-controlled Chatbot Arena ELO) from benchmark data
using ModelBankImputer plus adaptive KNN (sublinear power cutoff), optional
fold-internal PLS hybrid features, local kernel Ridge, and grouped conformal
prediction intervals.

Pipeline:
    1. Load benchmark CSV, identify numeric columns
    2. Low-variance filtering
    3. Impute missing values (ModelBankImputer default)
    4. Build feature matrix (imputed cols + SVD factors + trajectory features)
    5. Optionally drop style_/tone_ columns before KNN distance calculation
    6. Adaptive KNN prediction with OOF cross-validation
    7. Optionally append fold-internal PLS components to the KNN feature space
    8. Grouped conformal calibration of prediction intervals
    9. Save predictions, OOF, metadata, run config, dependency graph
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
from typing import Dict, List, Tuple, Optional

import time
import hashlib

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from zoneinfo import ZoneInfo
from datetime import datetime

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

# scipy stats (used by calibration block)
from scipy import stats  # type: ignore

# Calibration module
from calibration import (
    GateResult,
    ShapeFit,
    compute_p_beats_leader,
    compute_p_above,
    compute_sigma,
    diagnose_scale_signal,
    fit_tail_shape_and_qhat,
)  # type: ignore

# ==============================================================================
# CONSTANTS
# ==============================================================================

SEED = 42
np.random.seed(SEED)

TARGET = "lmarena_Score"  # style-controlled Arena ELO (default; override with --target)
ALT_TARGET = "lmsys_Score"  # raw Arena ELO — excluded from features as leakage
ID_COL = "model_name"

TARGETS = {TARGET, ALT_TARGET}  # both always excluded from features regardless of which is active
EXCLUDE = TARGETS | {ID_COL}

DENSE_THRESHOLD = 0.508  # original threshold for dense-only CV evaluation

# KNN prediction hyperparameters
KNN_POWER_ALPHA = 0.7  # exponent for sublinear distance cutoff (1.0 = linear)
KNN_POWER_C = 3.0      # coefficient for distance cutoff: max_dist = d0^alpha * C
KNN_MAX_K = 80
KNN_MIN_K = 20
KNN_BW_PCT = 0.15      # kernel bandwidth at this percentile of neighbor distances
KNN_VI_CLIP = (1.0, 1.5)

# Diagnostic sidecar: if set to a list, predict_adaptive_knn appends dicts with
# per-call (b_raw, b_clipped, k, mu_nb, p, p_corrected). Used for one-shot
# analyses (e.g. jackknife compression distribution). Off by default.
_JACKKNIFE_LOG = None

# Learned distance + local feature selection defaults
KNN_LOCAL_CORR_K = 0     # 0 = disabled (use all features); >0 = select top-K per neighborhood
KNN_RESID_WEIGHT = 0.3   # weight for residual-based distance reweighting (0 = plain correlation weighting)
KNN_LEARNED_DIST = False  # enable feature-weighted distance metric


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
    predictor_selection: str = "corr",
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
            predictor_selection=predictor_selection,
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
# KNN PREDICTION
# ==============================================================================

def predict_adaptive_knn(
    Xtr: np.ndarray,
    y_tr: np.ndarray,
    Xte: np.ndarray,
    power_alpha: float = KNN_POWER_ALPHA,
    power_C: float = KNN_POWER_C,
    max_k: int = KNN_MAX_K,
    min_k: int = KNN_MIN_K,
    bw_pct: float = KNN_BW_PCT,
    vi_clip: tuple = KNN_VI_CLIP,
) -> Tuple[float, float, int, float]:
    """Predict a single test point using adaptive KNN + kernel Ridge + jackknife VI.

    Uses a sublinear power cutoff for neighborhood selection:
        max_dist = d_nearest^power_alpha * power_C
    This gives tighter neighborhoods in dense regions (top models) while keeping
    adequate coverage in sparse regions, naturally adapting to feature sign flips.

    Returns (prediction, std_estimate, k_used, y_nb_std).
    y_nb_std is the std of the neighborhood's y-values (for calibration).
    """
    nn = NearestNeighbors(n_neighbors=max_k)
    nn.fit(Xtr)
    dists, idx = nn.kneighbors(Xte)
    d, ix = dists[0], idx[0]

    # Sublinear power cutoff: tighter neighborhoods for dense regions
    d0 = max(d[0], 1e-6)
    max_dist = d0 ** power_alpha * power_C
    k = max(min_k, min(int((d <= max_dist).sum()), max_k))

    y_nb = y_tr[ix[:k]]
    Xtr_nb = Xtr[ix[:k]]  # original (unweighted) features for regression

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
        b_raw = float(np.dot(yc, pc) / denom)
        b = float(np.clip(b_raw, vi_clip[0], vi_clip[1]))
    else:
        b_raw = 1.0
        b = 1.0

    p_corrected = mu_nb + b * (p - mu_nb)

    if _JACKKNIFE_LOG is not None:
        _JACKKNIFE_LOG.append({
            'b_raw': b_raw,
            'b_clipped': b,
            'k': int(k),
            'mu_nb': mu_nb,
            'y_nb_min': float(np.min(y_nb)),
            'y_nb_max': float(np.max(y_nb)),
            'y_nb_std': float(np.std(y_nb)),
            'p_pre': p,
            'p_corrected': p_corrected,
            'd0': float(d[0]),
            'coef': mdl.coef_.copy(),
            'intercept': float(mdl.intercept_),
            'x_test': Xte[0].copy(),
            'x_nb_mean': Xtr_nb.mean(axis=0),
        })

    # Rough std estimate from jackknife residuals
    jack_resid = y_nb - jack_preds
    std_est = float(np.std(jack_resid))

    return p_corrected, std_est, k, float(np.std(y_nb))


def fit_and_predict_knn(
    X_all: np.ndarray,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    cv_repeats: int = 10,
    cv_seed: int = SEED,
    power_alpha: float = KNN_POWER_ALPHA,
    power_C: float = KNN_POWER_C,
    max_k: int = KNN_MAX_K,
    min_k: int = KNN_MIN_K,
    bw_pct: float = KNN_BW_PCT,
    pls_hybrid_k: int = 0,
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
    oof_y_nb_std_sum = np.zeros(n_train)  # NEW: average across folds

    for fold_idx, (tr, va) in enumerate(cv_splits):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_all[train_idx[tr]])
        Xva = sc.transform(X_all[train_idx[va]])
        ytr = y_train[tr]

        # PLS hybrid: fit PLS on training fold only, concatenate components to features
        if pls_hybrid_k > 0:
            from sklearn.cross_decomposition import PLSRegression as _PLS
            n_comp = min(pls_hybrid_k, Xtr.shape[1], Xtr.shape[0] - 1)
            pls_f = _PLS(n_components=n_comp).fit(Xtr, ytr)
            Xtr = np.hstack([Xtr, pls_f.transform(Xtr)])
            Xva = np.hstack([Xva, pls_f.transform(Xva)])

        for vi, va_i in enumerate(va):
            p, _, _, y_nb_std = predict_adaptive_knn(
                Xtr, ytr, Xva[vi:vi + 1],
                power_alpha=power_alpha, power_C=power_C,
                max_k=max_k, min_k=min_k, bw_pct=bw_pct,
            )
            oof_preds_sum[va_i] += p
            oof_counts[va_i] += 1
            oof_y_nb_std_sum[va_i] += y_nb_std
            oof_folds[va_i] = fold_idx % 5
            if _JACKKNIFE_LOG is not None and _JACKKNIFE_LOG:
                _JACKKNIFE_LOG[-1]['is_oof'] = True
                _JACKKNIFE_LOG[-1]['train_row_idx'] = int(va_i)
                _JACKKNIFE_LOG[-1]['fold'] = int(fold_idx)

    oof_preds = np.where(oof_counts > 0, oof_preds_sum / oof_counts, np.nan)
    oof_y_nb_std = np.where(oof_counts > 0, oof_y_nb_std_sum / oof_counts, np.nan)  # NEW
    oof_valid = oof_counts > 0
    oof_rmse = float(np.sqrt(np.nanmean((oof_preds[oof_valid] - y_train[oof_valid]) ** 2)))
    print(f"  KNN OOF RMSE: {oof_rmse:.2f} ({int(oof_valid.sum())}/{n_train} valid)")

    # --- Final predictions for all rows ---
    sc_final = StandardScaler()
    X_train_sc = sc_final.fit_transform(X_all[train_idx])
    X_all_sc = sc_final.transform(X_all)

    # PLS hybrid: fit on full training set for final predictions
    if pls_hybrid_k > 0:
        from sklearn.cross_decomposition import PLSRegression as _PLS
        n_comp = min(pls_hybrid_k, X_train_sc.shape[1], X_train_sc.shape[0] - 1)
        pls_final = _PLS(n_components=n_comp).fit(X_train_sc, y_train)
        X_train_sc = np.hstack([X_train_sc, pls_final.transform(X_train_sc)])
        X_all_sc = np.hstack([X_all_sc, pls_final.transform(X_all_sc)])

    mu = np.full(n_all, np.nan)
    std = np.full(n_all, np.nan)
    ks_used = np.zeros(n_all, dtype=int)
    y_nb_std_final = np.full(n_all, np.nan)  # NEW

    for i in pred_idx:
        p, s, k, y_nb_std = predict_adaptive_knn(
            X_train_sc, y_train, X_all_sc[i:i + 1],
            power_alpha=power_alpha, power_C=power_C,
            max_k=max_k, min_k=min_k, bw_pct=bw_pct,
        )
        mu[i] = p
        std[i] = s
        ks_used[i] = k
        y_nb_std_final[i] = y_nb_std
        if _JACKKNIFE_LOG is not None and _JACKKNIFE_LOG:
            _JACKKNIFE_LOG[-1]['is_final'] = True
            _JACKKNIFE_LOG[-1]['row_idx'] = int(i)

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
        "y_nb_std_oof": oof_y_nb_std,
        "y_nb_std_final": y_nb_std_final,
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
        7. Optional style_/tone_ removal before KNN distance calculation
        8. KNN prediction (fit_and_predict_knn, with optional fold-internal PLS hybrid)
        9. Grouped conformal interval calibration
       10. Output saving (predictions CSV, OOF CSV, metadata JSON, run config, dependency graph)
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
    ap.add_argument("--predictor_selection", type=str, default="corr",
                    choices=["corr", "loo_forward"],
                    help="Predictor selection method: 'corr' (default CLI behavior, |corr|*sqrt(n)) or "
                         "'loo_forward' (greedy forward selection by leave-one-out RMSE).")
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
    ap.add_argument("--selector_cv", type=int, default=5,
                    help="CV folds for imputer feature selection.")
    ap.add_argument("--cv_splits_path", type=str, default="")
    ap.add_argument("--cv_repeats", type=int, default=1)
    ap.add_argument("--group_cv", action="store_true",
                    help="Use GroupKFold by model provider (diagnostic).")
    # -- KNN config --
    ap.add_argument("--knn_predict", action="store_true", default=True,  # always on, kept for CLI compat
                    help="Use adaptive KNN prediction (default: True).")
    ap.add_argument("--knn_power_alpha", type=float, default=0.7,
                    help="Exponent for sublinear distance cutoff (default 0.7, 1.0=linear).")
    ap.add_argument("--knn_power_c", type=float, default=3.0,
                    help="Coefficient for distance cutoff: max_dist = d0^alpha * C (default 3.0).")
    ap.add_argument("--knn_max_k", type=int, default=80)
    ap.add_argument("--knn_min_k", type=int, default=20)
    ap.add_argument("--knn_bw_pct", type=float, default=0.15)
    ap.add_argument("--drop_style_tone", action="store_true",
                    help="Remove style_/tone_ columns from KNN feature set (imputer can still use them).")
    ap.add_argument("--pls_hybrid_k", type=int, default=0,
                    help="Append K PLS components (fit per-fold on train) to KNN features (0=disabled).")
    # -- Misc --
    ap.add_argument("--margin", type=float, default=20.0,
                    help="Margin for 'top_by_margin_prob' column (default 20 points).")
    ap.add_argument(
        "--walkforward_calibration_path",
        type=str,
        default=None,
        help="Path to walkforward_calibration.py's wf_residuals.csv. "
             "If provided, predict.py reads the 'fitted_m' column and applies it "
             "to sigma_hat. Fitting m itself happens inside walkforward_calibration.py, "
             "not here. If None, m=1.0 (sigma reflects OOF level only).",
    )
    ap.add_argument("--categorical_threshold", type=int, default=0)
    ap.add_argument("--forced_categorical_cols", type=str, default="")
    ap.add_argument("--tier_quantiles", type=str, default="0.33,0.67")
    ap.add_argument("--exclude_models", type=str, default="",
                    help="Comma-separated model names to exclude from the dataset.")
    ap.add_argument("--svd_in_features", dest="svd_in_features", action="store_true", default=True,
                    help="Add SVD factors from imputer to feature matrix (default: on).")
    ap.add_argument("--no_svd_in_features", dest="svd_in_features", action="store_false",
                    help="Exclude SVD factors from feature matrix.")
    ap.add_argument("--target", type=str, default="lmarena_Score",
                    choices=["lmarena_Score", "lmsys_Score"],
                    help="Arena target to predict. lmarena_Score = style-controlled (shipped default); "
                         "lmsys_Score = raw Arena ELO. The other is kept in TARGETS and excluded from features.")

    args = ap.parse_args()

    global TARGET, ALT_TARGET
    TARGET = args.target
    ALT_TARGET = "lmsys_Score" if TARGET == "lmarena_Score" else "lmarena_Score"

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
        imp_key = (
            f"imputed_modelbank_{csv_hash}_alpha{args.alpha:.6f}_"
            f"skmax{args.selector_k_max}_conf{args.confidence_threshold:.2f}_"
            f"coh{coh_lam:.2f}{coh_shape[0]}_gfixed_ic0_svdp0_exp1x1_"
            f"catthr{args.categorical_threshold}_catovr{len(categorical_numeric_cols)}_skt3.0.csv"
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
            predictor_selection=getattr(args, 'predictor_selection', 'corr'),
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
    if getattr(args, 'drop_style_tone', False):
        before = len(knn_feature_cols)
        # 2026-05-11 experiment: drop only style_*, keep tone_* in KNN.
        # The 2026-04-18 ablation tested style+tone together; tone_* is only
        # 4 columns (already-summary judge TrueSkill values), and one of them
        # (tone_*confidence*) is the load-bearing signal for benchmaxxed
        # models like Phi-4 (0.0 percentile). PLS-3 smooths it away. Test
        # whether keeping tone_* in fixes Phi-4 without harming overall RMSE.
        knn_feature_cols = [c for c in knn_feature_cols
                            if not c.startswith('style_')]
        print(f"  drop_style (keep tone): {before} -> {len(knn_feature_cols)} features")
    knn_X = safe_features[knn_feature_cols].values
    globals()['_LAST_KNN_FEATURE_COLS'] = list(knn_feature_cols)
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

    n_folds = args.outer_cv if args.outer_cv else 5
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
            len(train_idx), n_folds, group_labels_cv,
            args.cv_repeats_outer, args.cv_seed)
    elif args.cv_splits_path:
        target_cv_splits = get_or_create_splits(
            len(train_idx),
            n_folds,
            args.cv_splits_path,
            repeats=args.cv_repeats_outer,
            seed=args.cv_seed,
        )
    else:
        target_cv_splits = _build_repeated_splits(
            len(train_idx), n_folds,
            args.cv_repeats_outer, args.cv_seed)

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
        power_alpha=args.knn_power_alpha,
        power_C=args.knn_power_c,
        max_k=args.knn_max_k,
        min_k=args.knn_min_k,
        bw_pct=args.knn_bw_pct,
        pls_hybrid_k=args.pls_hybrid_k,
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
    # 9. Calibration (OOF normalized conformal-style + walk-forward level scalar)
    # =========================================================================
    calibration_start = time.time()

    pre_imputation_missing = df[feature_cols].isna().values

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

    # =========================================================================
    # 10. (probabilities computed in section 9 above)
    # =========================================================================

    # =========================================================================
    # 11. Save outputs
    # =========================================================================

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

    # -- run_config.json --
    run_config = dict(vars(args))
    run_config.update({
        "tier_quantiles_parsed": tier_quantiles or [],
        "used_cache": bool(_cache_hit),
        "cache_key": imp_key,
        "output_dir": out_dir,
    })
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
            "sigma_hat is the t-distribution SCALE parameter, not a half-width. For 95% intervals use mu ± t_crit_95 * sigma_hat.",
            "p_beats_leader = P(score > max_leader) where max_leader = max(observed lmarena_Score); NaN on training rows.",
            f"top_by_margin_prob = P(score > max_leader + {args.margin}), NaN on training rows.",
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
