#!/usr/bin/env python3
"""
LMSYS Benchmark Score Predictor Pipeline (v7.2.1).

v7.2 Additions:
    - gp_selector_k_max: Separate feature cap for GP models (mRMR selection)
    - Per-column tolerance calibration: Calibrate tolerances based on masked evaluation
    - Periodic recalibration: Optionally recalibrate tolerances every N passes
v7.2.1 Additions:
    - Repeated CV via --cv_repeats with per-stage overrides for sweeps

This is the main orchestrator for predicting LMSYS/Chatbot Arena ELO scores from
other benchmark data. It coordinates imputation, feature selection, model training,
and evaluation in a comprehensive end-to-end pipeline.

Pipeline Overview:
    1. Data Loading: Load transformed benchmark data with missing values
    2. Imputation: Use SpecializedColumnImputer to fill missing benchmark scores
    3. Feature Selection: Tree-based ranking (LightGBM/XGBoost) with 1-SE rule
    4. ALT Target Handling: Optional imputation of alternative targets via OOF stacking
    5. Model Comparison: Evaluate BayesianRidge and ARDRegression
    6. Prediction: Generate final predictions with calibrated uncertainty intervals
    7. Dependency Analysis: Generate column dependency graphs (filtered by importance)

Key Components:
    Feature Selection:
        - rank_features_tree(): Rank features by LightGBM/XGBoost gain
        - choose_k_by_cv_from_rank(): Select k using 1-SE rule
        - select_features_tree(): Full pipeline with collinearity pruning

    ALT Target Imputation:
        - impute_alt_for_all(): Out-of-fold stacking for alternative targets
        - _fit_alt_model_on_rows(): Fit ALT model with spline calibration
        - _generate_oof_alt_predictions(): Prevent leakage via OOF

    Model Specifications:
        - ModelSpec: Dataclass defining model configuration
        - build_model_specs(): Returns BayesianRidge and ARDRegression specs

    Cross-Validation:
        - cross_val_rmse_for_model(): Standard CV evaluation
        - cross_val_rmse_with_alt(): CV with inner-loop ALT imputation

    Probability Columns (v6 additions):
        - num_one_prob: P(score > max_observed_training_score)
        - top_by_margin_prob: P(score > max_observed + margin), only for models
          without actual scores. Computed analytically from Bayesian posterior.

    v7 Additions:
        - ARDRegression for per-feature relevance determination with proper
          uncertainty quantification (includes observation noise in std)

    v7.1 Additions:
        - Calibrated uncertainty scaling: Scales model std by a factor computed
          from CV residuals to achieve proper 95% coverage
        - Variance contribution analysis: Computes (beta * std(X))^2 / var(pred)
          for each feature, grouped by base feature for polynomial terms
        - Dependency graph filtering: Only shows features with >= 1% variance
          contribution in both target and ALT models
        - New outputs: best_model_variance_contributions.csv,
          alt_model_variance_contributions.csv

Configuration:
    Via argparse CLI arguments:
    --data: Input CSV file path
    --target: Target column name (default: lmsys_Score)
    --max-features: Maximum features to select
    --n-folds: CV folds
    --use-alt: Enable ALT target imputation
    --alt-cols: Alternative target columns
    --cache-dir: Imputation cache directory
    --force-reimpute: Ignore cache
    --margin: Margin for top_by_margin_prob column (default: 20.0 points)

Example Usage:
    $ python predict.py --data clean_combined.csv --target LMSYS --max-features 20
    $ python predict.py --data clean_combined.csv --margin 30  # custom margin

    Or programmatically:
    >>> from predict import main
    >>> main(['--data', 'data.csv', '--target', 'LMSYS'])

Dependencies:
    Required: numpy, pandas, scikit-learn, joblib
    Optional: lightgbm, xgboost, catboost
    Local: SpecializedColumnImputer
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import json
import re
import itertools
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import json

import time


import numpy as np # type: ignore
import pandas as pd # type: ignore

from zoneinfo import ZoneInfo
from datetime import datetime
import hashlib

from joblib import Parallel, delayed


# ==============================================================================
# LOCAL IMPORTS
# ==============================================================================

try:
    from column_imputer import SpecializedColumnImputer
except Exception as e:
    print("ERROR: Could not import column_imputer.py. Make sure the file is alongside this script.", file=sys.stderr)
    raise

# ML stack
from sklearn.model_selection import KFold # type: ignore
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
from sklearn.linear_model import BayesianRidge, ElasticNetCV, Ridge, RidgeCV, LinearRegression, ARDRegression # type: ignore
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor # type: ignore
from sklearn.inspection import permutation_importance # type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor # type: ignore
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, WhiteKernel, ConstantKernel # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore


SEED = 42
np.random.seed(SEED)

TARGET = "lmsys_Score"
ALT_TARGET = "lmarena_Score"
ID_COL = "model_name"


TARGETS = {TARGET, ALT_TARGET}  # exclude both to avoid leakage
EXCLUDE = TARGETS | {ID_COL}

DENSE_THRESHOLD = 0.508  # original threshold for dense-only CV evaluation
COMPLETENESS_WEIGHT_POWER = 0  # 0 = disabled (all weights 1.0); 2 = quadratic weighting


PARALLELISM_CFG = {
    "max_workers": None,
    "cv_n_jobs": 1,
    "model_n_jobs": 1,
    "selector_n_jobs": -1,
    "tree_model_n_jobs": -1,
    "tree_selector_n_jobs": -1,
    "imputer_n_jobs": -1,
}


# ---- TREE-BASED FEATURE SELECTION (LightGBM / XGBoost) ----


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

    model_requested = max(1, int(args.model_n_jobs))
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
    args.model_n_jobs = model_jobs
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

# Try both; use what's available
try:
    import lightgbm as lgb# type: ignore
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False
try:
    from xgboost import XGBRegressor# type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False



def _rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def _neg_rmse(y_true, y_pred) -> float:
    # Higher is better (less negative) for our selection logic
    return -_rmse(y_true, y_pred)

def _get_lgb_params():
    return dict(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        min_child_samples=20,
        subsample=1.0,            # No subsampling for deterministic feature importance
        subsample_freq=1,
        colsample_bytree=1.0,     # No column sampling for deterministic feature importance
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=PARALLELISM_CFG["tree_selector_n_jobs"],
        random_state=42,
        verbosity=-1
    )

def _get_xgb_params():
    return dict(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=1.0,            # No subsampling for deterministic feature importance
        colsample_bytree=1.0,     # No column sampling for deterministic feature importance
        min_child_weight=1.0,
        reg_alpha=0.1,
        reg_lambda=0.1,
        tree_method="hist",
        n_jobs=PARALLELISM_CFG["tree_selector_n_jobs"],
        random_state=42
    )

def _fit_lgb(X: pd.DataFrame, y: pd.Series):
    model = lgb.LGBMRegressor(**_get_lgb_params())
    model.fit(X, y)
    return model

def _fit_xgb(X: pd.DataFrame, y: pd.Series):
    model = XGBRegressor(**_get_xgb_params())
    model.fit(X, y, verbose=False)
    return model

def _gain_importance_from_lgb(model) -> pd.Series:
    booster = model.booster_
    gains = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    s = pd.Series(gains, index=names, dtype="float64")
    return s

def _gain_importance_from_xgb(model, cols: List[str]) -> pd.Series:
    booster = model.get_booster()
    fmap = booster.feature_names
    if fmap is None:
        # Fallback: map f{idx} -> col
        gain_dict = booster.get_score(importance_type="gain")
        out = pd.Series(0.0, index=cols)
        for k, v in gain_dict.items():
            # keys like 'f12' -> 12
            try:
                i = int(k[1:])
                out.iloc[i] = v
            except Exception:
                pass
        return out
    else:
        gain_dict = booster.get_score(importance_type="gain")
        # gain_dict keys should already be column names
        out = pd.Series(0.0, index=cols)
        for k, v in gain_dict.items():
            if k in out.index:
                out.loc[k] = v
        return out

def rank_features_tree(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "lgbm",
    cv: int = 5,
    cv_repeats: int = 1,
    random_state: int = 42
) -> pd.DataFrame:
    """Rank features by tree-based gain importance using cross-validation.

    This function trains gradient boosting models (LightGBM or XGBoost) on
    multiple CV folds and aggregates the feature importance scores. Using
    CV-averaged gains is more robust than single-model importance.

    Args:
        X: Feature matrix as DataFrame.
        y: Target series.
        model_type: 'lgbm' for LightGBM or 'xgb' for XGBoost. Falls back to
            available library if requested one is not installed.
        cv: Number of cross-validation folds. Defaults to 5.
        cv_repeats: Number of CV repeats (new seed per repeat). Defaults to 1.
        random_state: Random seed for reproducibility. Defaults to 42.

    Returns:
        DataFrame with columns ['feature', 'mean_gain', 'std_gain'] sorted by
        mean_gain in descending order. Higher gain = more important.

    Raises:
        ImportError: If neither LightGBM nor XGBoost is installed.

    Algorithm:
        1. For each CV fold, fit a tree model on training data
        2. Extract gain-based feature importance from the model
        3. Align importance scores to feature names
        4. Compute mean and std across folds
        5. Return sorted DataFrame

    Example:
        >>> ranking_df = rank_features_tree(X, y, model_type='lgbm', cv=5)
        >>> top_features = ranking_df['feature'].head(20).tolist()
    """
    assert model_type in {"lgbm", "xgb"}, "model_type must be 'lgbm' or 'xgb'"
    if model_type == "lgbm" and not _HAS_LGB and _HAS_XGB:
        model_type = "xgb"
    if model_type == "xgb" and not _HAS_XGB and _HAS_LGB:
        model_type = "lgbm"
    if model_type == "lgbm" and not _HAS_LGB:
        raise ImportError("LightGBM is not installed.")
    if model_type == "xgb" and not _HAS_XGB:
        raise ImportError("XGBoost is not installed.")

    # ensure DataFrame
    X = pd.DataFrame(X).copy()
    y = pd.Series(y).copy()

    splits = _build_repeated_splits(len(X), cv, cv_repeats, random_state)
    if not splits:
        raise ValueError("cv must be >= 2 for cross-validation.")

    all_gains = []
    for tr_idx, _ in splits:
        Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
        if model_type == "lgbm":
            m = _fit_lgb(Xtr, ytr)
            gains = _gain_importance_from_lgb(m)
        else:
            m = _fit_xgb(Xtr, ytr)
            gains = _gain_importance_from_xgb(m, X.columns.tolist())
        # align to X columns (fill 0 for unseen)
        gains = gains.reindex(X.columns, fill_value=0.0)
        all_gains.append(gains.values)

    G = np.vstack(all_gains)  # cv x features
    mean_gain = G.mean(axis=0)
    std_gain = G.std(axis=0)

    df = pd.DataFrame({
        "feature": X.columns,
        "mean_gain": mean_gain,
        "std_gain": std_gain
    }).sort_values("mean_gain", ascending=False).reset_index(drop=True)
    return df

def choose_k_by_cv_from_rank(
    X: pd.DataFrame,
    y: pd.Series,
    ranking: List[str],
    model_type: str = "lgbm",
    k_grid: List[Union[int, str]] = (10, 20, 50, 100, 200, "all"),
    cv: int = 5,
    cv_repeats: int = 1,
    random_state: int = 42,
) -> Tuple[List[str], Dict]:
    """Select optimal number of features using the 1-SE rule on CV performance.

    Given a ranked list of features, this function evaluates different subset
    sizes via cross-validation and selects the smallest subset whose performance
    is within one standard error of the best performance (1-SE rule).

    The 1-SE rule provides regularization by preferring simpler models when
    performance differences are within statistical noise.

    Args:
        X: Feature matrix as DataFrame.
        y: Target series.
        ranking: List of feature names sorted by importance (most important first).
        model_type: 'lgbm' or 'xgb' for the evaluation model. Defaults to 'lgbm'.
        k_grid: List of k values to evaluate. Can include integers or 'all'.
            Defaults to (10, 20, 50, 100, 200, 'all').
        cv: Number of cross-validation folds. Defaults to 5.
        cv_repeats: Number of CV repeats (new seed per repeat). Defaults to 1.
        random_state: Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple of (chosen_cols, diagnostics) where:
        - chosen_cols: List of selected feature names
        - diagnostics: Dict containing:
            - k_grid: Evaluated k values
            - cv_means: Mean CV scores (negative RMSE) for each k
            - cv_stds: Std of CV scores for each k
            - best_idx: Index of best performing k
            - best_mean: Best mean score
            - best_se: Standard error at best k
            - chosen_idx: Index of chosen k (smallest within 1-SE)
            - chosen_K: Number of features selected

    Algorithm:
        1. For each k in k_grid, evaluate top-k features via CV
        2. Find k with best mean performance
        3. Compute 1-SE threshold: best_mean - (best_std / sqrt(n_folds_total))
        4. Select smallest k whose mean is >= threshold
    """
    # clean grid
    k_grid = [len(ranking) if (k == "all" or (isinstance(k, str) and k.lower() == "all")) else int(k)
              for k in k_grid]
    k_grid = sorted(set([k for k in k_grid if k > 0 and k <= len(ranking)]))

    splits = _build_repeated_splits(len(X), cv, cv_repeats, random_state)
    if not splits:
        raise ValueError("cv must be >= 2 for cross-validation.")
    n_folds_total = len(splits)

    def _fit_predict_score(cols: List[str]) -> float:
        scores = []
        for tr, va in splits:
            Xtr, ytr = X.iloc[tr][cols], y.iloc[tr]
            Xva, yva = X.iloc[va][cols], y.iloc[va]
            if model_type == "lgbm":
                m = _fit_lgb(Xtr, ytr)
            else:
                m = _fit_xgb(Xtr, ytr)
            pred = m.predict(Xva)
            scores.append(_neg_rmse(yva, pred))
        return float(np.mean(scores)), float(np.std(scores))

    means, stds = [], []
    for K in k_grid:
        cols = ranking[:K]
        m, s = _fit_predict_score(cols)
        means.append(m); stds.append(s)

    means = np.array(means); stds = np.array(stds)
    best_idx = int(np.argmax(means))
    se = stds[best_idx] / np.sqrt(n_folds_total)
    within = np.where(means >= means[best_idx] - se)[0]
    # pick smallest K among those within 1-SE
    chosen_idx = int(within[np.argmin([k_grid[i] for i in within])])
    chosen_K = int(k_grid[chosen_idx])
    chosen_cols = ranking[:chosen_K]
    diags = {
        "k_grid": k_grid,
        "cv_means": means.tolist(),
        "cv_stds": stds.tolist(),
        "best_idx": int(best_idx),
        "best_mean": float(means[best_idx]),
        "best_se": float(se),
        "chosen_idx": int(chosen_idx),
        "chosen_K": chosen_K,
        "cv_repeats": int(_normalize_cv_repeats(cv_repeats)),
        "n_folds_total": int(n_folds_total),
    }
    return chosen_cols, diags

def _ensure_frame_series(X, y):
    # Make X a DataFrame with stable column names
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    # Make y a Series aligned to X
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index, name="target")
    return X, y




# ==============================================================================
# POLYNOMIAL INTERACTION EXPANSION
# ==============================================================================

def expand_poly_interactions(
    X: pd.DataFrame,
    include_squares: bool = False,
    limit: int = 0,
    preset_core: Optional[List[str]] = None,
    return_core: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
    """Generate degree-2 polynomial interaction features.

    Creates pairwise interaction features (A*B) and optionally squared terms (A^2)
    from input features. When many features exist, interactions can be limited to
    the highest-variance columns to control dimensionality.

    Args:
        X: Input feature DataFrame.
        include_squares: If True, include squared terms (A^2). Defaults to False.
        limit: If > 0, only generate interactions among the top-variance `limit`
            columns. Other columns are kept as main effects only. Defaults to 0
            (no limit).
        preset_core: If provided, use these columns as the "core" for interactions
            instead of selecting by variance. Useful for test-time consistency.
        return_core: If True, return tuple (DataFrame, core_columns) so the same
            core can be applied to test data. Defaults to False.

    Returns:
        If return_core=False: DataFrame with original columns plus interactions.
        If return_core=True: Tuple of (DataFrame, list of core column names).

    Feature Naming:
        - Interactions: 'A*B' (original names joined by *)
        - Squares: 'A^2' (original name with ^2 suffix)

    Example:
        >>> X_train_poly, core = expand_poly_interactions(
        ...     X_train, include_squares=True, limit=10, return_core=True
        ... )
        >>> X_test_poly = expand_poly_interactions(
        ...     X_test, include_squares=True, limit=10, preset_core=core
        ... )
    """
    if X.empty:
        return X.copy()
    cols_all = X.columns.tolist()
    if preset_core is not None:
        core = [c for c in preset_core if c in cols_all]
        if not core:
            preset_core = None  # fallback to default logic if nothing matched
    if preset_core is None:
        if limit and limit < len(cols_all):
            var = X.var(ddof=0).sort_values(ascending=False)
            core = var.index[:limit].tolist()
        else:
            core = cols_all

    poly = PolynomialFeatures(
        degree=2,
        include_bias=False,
        interaction_only=not include_squares,
    )
    Z = poly.fit_transform(X[core].values)
    names = poly.get_feature_names_out(core)
    # sanitize: "A B" -> "A:B"
    names = [n.replace(" ", "*") for n in names]
    Zdf = pd.DataFrame(Z, index=X.index, columns=names)

    if limit and limit < len(cols_all):
        # keep main effects for the non-core columns (no interactions)
        rest = [c for c in cols_all if c not in core]
        Zdf = pd.concat([X[rest], Zdf], axis=1)
    if return_core:
        return Zdf, core
    return Zdf





def _select_target_cols_for_train(X_tr: pd.DataFrame, y_tr: np.ndarray, cfg: dict) -> List[str]:
    """Select columns for the target model using only TRAIN rows."""
    if not cfg.get("enabled", False):
        return list(X_tr.columns)
    y_numeric = pd.to_numeric(pd.Series(y_tr), errors="coerce").to_numpy()
    mask = ~np.isnan(y_numeric)
    if mask.sum() < 3:
        return list(X_tr.columns)

    sel_cols, _, _ = select_features_tree(
        X_tr.loc[mask],
        pd.Series(y_numeric[mask]),
        model_type=cfg["model_type"],
        mode=cfg["mode"],
        k_grid=cfg["k_grid"],
        cv=cfg["cv"],
        cv_repeats=cfg.get("cv_repeats", 1),
        random_state=cfg.get("cv_seed", SEED),
        drop_collinear=True,
        r2_threshold=0.95,
        min_variance=1e-12,
    )
    return sel_cols if len(sel_cols) else list(X_tr.columns)


def _alt_cv_report(
    X_no_alt_all: pd.DataFrame,
    alt_series: pd.Series,
    cfg: dict,
    out_dir: str,
    repeats: int = 1,
    seed: int = SEED,
    X_base_for_search: Optional[pd.DataFrame] = None,
):
    """
    Measures ALT imputation quality via **nested** CV: interaction pairs are
    selected per outer fold (on training rows only) so the reported RMSE is
    unbiased w.r.t. interaction selection.

    Args:
        X_no_alt_all: Feature DataFrame for production model (may include poly features).
        alt_series: ALT target values.
        cfg: ALT selector config.
        out_dir: Output directory.
        repeats: Number of CV repeats for outer folds.
        seed: Random seed.
        X_base_for_search: Base feature DataFrame (pre-poly) for inner greedy
            interaction search. If None, uses X_no_alt_all (which may include
            poly features — not recommended).

    Writes:
      - alt_imputation_cv_folds.csv (fold RMSE)
      - alt_imputation_cv_metrics.json ({rmse, sd, rmse_over_sd})
      - alt_selected_features_per_fold.csv (comma-joined base features)
      - alt_selected_features_frequency.csv (counts across folds)
      - alt_selected_interactions_per_fold.csv (interaction pairs per fold)

    Returns:
        metrics dict with keys: rmse, sd, rmse_over_sd
    """
    y = pd.to_numeric(alt_series, errors="coerce")
    mask_known = ~y.isna()
    X = pd.DataFrame(X_no_alt_all).loc[mask_known].reset_index(drop=True)
    y = y.loc[mask_known].reset_index(drop=True)

    # Base features for inner greedy search (exclude poly features)
    if X_base_for_search is not None:
        X_search = pd.DataFrame(X_base_for_search).loc[mask_known].reset_index(drop=True)
    else:
        X_search = X

    splits = _build_repeated_splits(len(X), int(cfg["cv"]), int(repeats), int(seed))
    fold_rows, feat_rows, int_pair_rows = [], [], []
    oof_actual, oof_pred = np.full(len(y), np.nan), np.full(len(y), np.nan)
    total_sse = 0.0
    total_n = 0
    for k, (tr, va) in enumerate(splits):
        # --- Nested inner loop: select interaction pairs on train rows only ---
        # Use base features (pre-poly) to match global greedy search space
        X_tr_search = X_search.iloc[tr]
        y_tr = y.iloc[tr]
        fold_pairs, _ = _greedy_select_alt_interactions(
            X_tr_search, y_tr,
            n_pca=ALT_PCA_N_COMPONENTS,
            n_folds=min(int(cfg["cv"]), len(tr)),
            n_repeats=1,   # single repeat inside nested CV for speed
            seed=seed + k,
            n_runs=1,      # no consensus inside nested CV
            consensus_min=1,
            verbose=False,
            n_jobs=1,      # already inside nested CV, no further forking
        )
        int_pair_rows.append({
            "fold": k,
            "n_pairs": len(fold_pairs),
            "pairs": ";".join(f"{a}*{b}" for a, b in fold_pairs),
        })

        # --- Fit production model with per-fold interaction pairs ---
        fit = _fit_alt_model_on_rows(X, y, cfg, tr, interaction_pairs=fold_pairs)
        pred_all = fit["pred_all"]
        pred_va = pred_all.iloc[va].to_numpy()

        # kNN residual smoothing: build inner OOF residual bank, then correct
        if ALT_KNN_ALPHA > 0 and len(tr) >= 15:
            X_tr_slice = X.iloc[tr]
            y_tr_slice = y.iloc[tr]
            r_bank = _build_alt_inner_residual_bank(
                X_tr_slice, y_tr_slice, cfg, seed=SEED + k,
            )
            rhat = _apply_alt_knn_correction(X, tr, va, r_bank)
            pred_va = pred_va + ALT_KNN_ALPHA * rhat

        yva = y.iloc[va].to_numpy()

        feat_rows.append({"fold": k, "features": ",".join(fit["selected_features"])})

        oof_actual[va] = yva
        oof_pred[va] = pred_va

        fold_rmse = float(_rmse(yva, pred_va))
        fold_rows.append({"fold": k, "rmse": fold_rmse, "n_val": int(len(va))})
        total_sse += float(np.sum((yva - pred_va) ** 2))
        total_n += int(len(va))

    rmse = float(np.sqrt(total_sse / total_n)) if total_n else float("nan")
    sd = float(np.std(y, ddof=1)) if len(y) > 1 else float("nan")

    # Bootstrap 95% CI on ALT RMSE
    valid = ~np.isnan(oof_actual)
    resid_sq = (oof_actual[valid] - oof_pred[valid]) ** 2
    rng = np.random.RandomState(seed)
    n_boot = len(resid_sq)
    boot_rmses = np.array([np.sqrt(np.mean(resid_sq[rng.randint(0, n_boot, size=n_boot)])) for _ in range(2000)])
    alt_ci_lo, alt_ci_hi = float(np.percentile(boot_rmses, 2.5)), float(np.percentile(boot_rmses, 97.5))

    metrics = {
        "rmse": rmse, "sd": sd, "rmse_over_sd": (rmse / sd) if sd and sd > 0 else float("nan"),
        "ci_lo": round(alt_ci_lo, 4), "ci_hi": round(alt_ci_hi, 4),
    }

    # Save per-row OOF predictions
    oof_df = pd.DataFrame({"actual": oof_actual[valid], "predicted": oof_pred[valid]})
    oof_df.to_csv(os.path.join(out_dir, "alt_imputation_cv_oof.csv"), index=False)

    # write outputs
    pd.DataFrame(fold_rows).to_csv(os.path.join(out_dir, "alt_imputation_cv_folds.csv"), index=False)
    with open(os.path.join(out_dir, "alt_imputation_cv_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    per_fold = pd.DataFrame(feat_rows)
    per_fold.to_csv(os.path.join(out_dir, "alt_selected_features_per_fold.csv"), index=False)

    # interaction pairs per fold
    pd.DataFrame(int_pair_rows).to_csv(
        os.path.join(out_dir, "alt_selected_interactions_per_fold.csv"), index=False)

    # frequency table
    from collections import Counter
    counts = Counter([c for row in feat_rows for c in row["features"].split(",") if c])
    freq_df = pd.DataFrame({"feature": list(counts.keys()), "count": list(counts.values())}) \
                .sort_values(["count", "feature"], ascending=[False, True])
    freq_df.to_csv(os.path.join(out_dir, "alt_selected_features_frequency.csv"), index=False)

    return metrics


ALT_PCA_N_COMPONENTS = 10  # PCA(10)->BR; 10 slightly outperforms 12 in OOF CV (17.79 vs 18.14)

# Post-PCA interactions: cross-terms appended AFTER PCA dimensionality reduction.
# PCA(10) captures linear signal from all base features; these raw (scaled)
# interaction columns are concatenated with the 10 PCs so BayesianRidge can
# exploit nonlinear formatting × capability patterns that PCA compresses away.
# Selected at runtime by _greedy_select_alt_interactions() (with caching).
ALT_POST_PCA_INTERACTIONS: Optional[List[Tuple[str, str]]] = None  # Set by greedy search at runtime

# kNN residual smoothing: corrects base predictions using cross-fitted OOF
# residuals from nearest neighbors.  Captures local structure the global
# linear model misses (e.g., model families that cluster in benchmark space).
ALT_KNN_K = 2        # very local neighborhood (k=2 beats k=3 in OOF CV: 17.82 vs 18.14)
ALT_KNN_P = 2        # inverse-square distance weighting
ALT_KNN_ALPHA = 0.3  # conservative shrinkage (only 30% of correction applied)

# Module-level SVD factors for regime model access (set in main())
_MODULE_SVD_ROW_FACTORS = None


def _compute_alt_interaction_features(X_df: pd.DataFrame, scaler=None, fit=False, pairs=None):
    """Compute post-PCA interaction features from raw benchmark columns.

    Multiplies pairs from the given list (or the global ALT_POST_PCA_INTERACTIONS),
    skipping any where a constituent column is missing, then optionally fits/applies
    a StandardScaler so they're on the same scale as PCA components.

    Args:
        X_df: Raw benchmark DataFrame (pre-PCA, pre-scaling).
        scaler: A fitted StandardScaler for transform-only mode.
        fit: If True, fits a new StandardScaler on the interactions.
        pairs: Explicit list of (col_a, col_b) tuples. If None, reads from
            global ALT_POST_PCA_INTERACTIONS.

    Returns:
        (interaction_array, interaction_names, fitted_scaler)
        interaction_array: np.ndarray of shape (n_samples, n_interactions)
        interaction_names: list of str names like "colA*colB"
        fitted_scaler: the StandardScaler (newly fit or passed-through)
    """
    interaction_pairs = pairs if pairs is not None else (ALT_POST_PCA_INTERACTIONS or [])
    computed = []
    names = []
    for col_a, col_b in interaction_pairs:
        if col_a in X_df.columns and col_b in X_df.columns:
            computed.append(X_df[col_a].values * X_df[col_b].values)
            names.append(f"{col_a}*{col_b}")
    if not computed:
        empty = np.empty((len(X_df), 0))
        return empty, names, scaler
    raw = np.column_stack(computed)
    if fit:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(raw)
    elif scaler is not None:
        scaled = scaler.transform(raw)
    else:
        scaled = raw
    return scaled, names, scaler


def _single_greedy_run(
    X_df: pd.DataFrame,
    y_alt: np.ndarray,
    cols: List[str],
    n_comp: int,
    n_folds: int,
    n_repeats: int,
    seed: int,
    prescreen_top_k: int,
    max_pairs: int,
    min_improvement: float,
    verbose: bool = False,
) -> List[Tuple[str, str]]:
    """Single greedy forward search with fixed CV splits.

    Returns list of selected (col_a, col_b) pairs.
    """
    X = X_df.reset_index(drop=True)
    y = y_alt
    n_samples, n_cols = X.shape
    n_reps = max(1, int(n_repeats))

    # Build repeated K-fold splits — fixed for entire run
    folds = []
    for rep in range(n_reps):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + rep)
        folds.extend(list(kf.split(X)))
    total_folds = len(folds)

    # Pre-compute PCA per fold
    fold_pca_train = {}
    fold_pca_val = {}
    fold_train_idx = {}
    fold_val_idx = {}
    for fi, (tr_idx, va_idx) in enumerate(folds):
        scaler = StandardScaler()
        pca = PCA(n_components=n_comp)
        fold_pca_train[fi] = pca.fit_transform(scaler.fit_transform(X.iloc[tr_idx].values))
        fold_pca_val[fi] = pca.transform(scaler.transform(X.iloc[va_idx].values))
        fold_train_idx[fi] = tr_idx
        fold_val_idx[fi] = va_idx

    # PCA-only baseline + residual correlation pre-screen
    oof_preds_sum = np.zeros(n_samples)
    oof_counts = np.zeros(n_samples)
    for fi in range(total_folds):
        br = BayesianRidge(compute_score=False)
        br.fit(fold_pca_train[fi], y[fold_train_idx[fi]])
        preds = br.predict(fold_pca_val[fi])
        oof_preds_sum[fold_val_idx[fi]] += preds
        oof_counts[fold_val_idx[fi]] += 1

    oof_preds_baseline = oof_preds_sum / np.maximum(oof_counts, 1)
    baseline_rmse = float(np.sqrt(mean_squared_error(y, oof_preds_baseline)))
    residuals = y - oof_preds_baseline

    all_candidates = list(itertools.combinations(range(n_cols), 2))
    X_vals = X.values

    def _screen_top_k(residuals, excluded_set):
        """Rank all candidate pairs by |corr| with residuals, return top-K."""
        scores = []
        r = residuals - residuals.mean()
        r_ss = (r ** 2).sum()
        for i, j in all_candidates:
            pair = (cols[i], cols[j])
            if pair in excluded_set:
                scores.append(0.0)
                continue
            product = X_vals[:, i] * X_vals[:, j]
            std_p = np.std(product)
            if std_p < 1e-12:
                scores.append(0.0)
                continue
            p = product - product.mean()
            denom = np.sqrt((p ** 2).sum() * r_ss)
            corr = np.abs((p * r).sum() / denom) if denom > 1e-12 else 0.0
            scores.append(corr if np.isfinite(corr) else 0.0)
        top_idx = np.argsort(scores)[::-1][:prescreen_top_k]
        return [(cols[all_candidates[idx][0]], cols[all_candidates[idx][1]])
                for idx in top_idx]

    # Initial screen against PCA-only residuals
    screened = _screen_top_k(residuals, set())

    # Greedy forward selection with fixed splits
    selected: List[Tuple[str, str]] = []
    current_rmse = baseline_rmse
    current_oof = oof_preds_baseline.copy()

    pair_values = {}
    for pair in screened:
        col_a, col_b = pair
        pair_values[pair] = X[col_a].values * X[col_b].values

    for step in range(1, max_pairs + 1):
        remaining = [p for p in screened if p not in set(selected)]
        best_rmse = current_rmse
        best_pair = None
        best_oof = None

        for pair in remaining:
            trial_pairs = selected + [pair]
            oof_preds_s = np.zeros(n_samples)
            oof_cnts = np.zeros(n_samples)

            for fi in range(total_folds):
                tr_idx = fold_train_idx[fi]
                va_idx = fold_val_idx[fi]
                int_cols_tr = [pair_values[p][tr_idx] for p in trial_pairs]
                int_cols_va = [pair_values[p][va_idx] for p in trial_pairs]
                int_tr = np.column_stack(int_cols_tr)
                int_va = np.column_stack(int_cols_va)
                int_scaler = StandardScaler()
                int_tr_scaled = int_scaler.fit_transform(int_tr)
                int_va_scaled = int_scaler.transform(int_va)
                X_tr_combined = np.hstack([fold_pca_train[fi], int_tr_scaled])
                X_va_combined = np.hstack([fold_pca_val[fi], int_va_scaled])
                br = BayesianRidge(compute_score=False)
                br.fit(X_tr_combined, y[tr_idx])
                preds = br.predict(X_va_combined)
                oof_preds_s[va_idx] += preds
                oof_cnts[va_idx] += 1

            oof_avg = oof_preds_s / np.maximum(oof_cnts, 1)
            trial_rmse = float(np.sqrt(mean_squared_error(y, oof_avg)))
            if trial_rmse < best_rmse:
                best_rmse = trial_rmse
                best_pair = pair
                best_oof = oof_avg

        improvement = current_rmse - best_rmse
        if best_pair is None or improvement < min_improvement:
            break

        selected.append(best_pair)
        current_rmse = best_rmse
        current_oof = best_oof

        # Re-screen candidates against updated residuals
        residuals = y - current_oof
        screened = _screen_top_k(residuals, set(selected))
        # Cache any new pair values
        for pair in screened:
            if pair not in pair_values:
                pair_values[pair] = X[pair[0]].values * X[pair[1]].values

    return selected


def _greedy_select_alt_interactions(
    X_df: pd.DataFrame,
    y_alt: pd.Series,
    n_pca: int = ALT_PCA_N_COMPONENTS,
    n_folds: int = 5,
    n_repeats: int = 5,
    seed: int = SEED,
    prescreen_top_k: int = 50,
    max_pairs: int = 20,
    min_improvement: float = 0.05,
    n_runs: int = 5,
    consensus_min: int = 3,
    verbose: bool = True,
    n_jobs: int = -1,
) -> Tuple[List[Tuple[str, str]], pd.DataFrame]:
    """Consensus greedy forward search for post-PCA interaction terms.

    Runs ``n_runs`` independent greedy searches, each with different CV splits,
    then keeps only pairs selected in >= ``consensus_min`` runs.  This prevents
    any single set of fold boundaries from dominating the selection while
    maintaining a coherent greedy path within each run.

    Args:
        X_df: Feature DataFrame for rows with known ALT values (no ALT column).
        y_alt: Known ALT target values (aligned with X_df).
        n_pca: Number of PCA components.
        n_folds: CV folds for OOF evaluation.
        n_repeats: Number of repeated K-fold shuffles per run.
        seed: Base random seed.
        prescreen_top_k: Number of candidates to keep after correlation screen.
        max_pairs: Maximum interaction pairs per run.
        min_improvement: Minimum RMSE improvement to continue adding pairs.
        n_runs: Number of independent greedy runs.
        consensus_min: Minimum appearances across runs to keep a pair.
        verbose: Print progress messages.

    Returns:
        (selected_pairs, search_log_df) where search_log_df has columns:
        pair_a, pair_b, frequency, avg_step.
    """
    X = X_df.reset_index(drop=True)
    y = pd.to_numeric(y_alt, errors="coerce").reset_index(drop=True).values
    cols = list(X.columns)
    n_samples, n_cols = X.shape
    n_comp = min(n_pca, n_samples - 1, n_cols)
    n_comp = max(1, n_comp)

    if verbose:
        n_cand = n_cols * (n_cols - 1) // 2
        print(f"ALT interactions: pre-screening {n_cand} candidates → top {prescreen_top_k}")
        print(f"ALT interactions: consensus search ({n_runs} runs, keep pairs in ≥{consensus_min})")

    # Run independent greedy searches in parallel
    eff_jobs = min(n_runs, os.cpu_count() or 4) if n_jobs == -1 else min(n_runs, max(1, n_jobs))
    all_runs: List[List[Tuple[str, str]]] = Parallel(
        n_jobs=eff_jobs, prefer="processes"
    )(
        delayed(_single_greedy_run)(
            X_df, y, cols, n_comp,
            n_folds, n_repeats, seed + run * 1000,
            prescreen_top_k, max_pairs, min_improvement,
        )
        for run in range(n_runs)
    )
    if verbose:
        for run, selected in enumerate(all_runs):
            print(f"  run {run + 1}: selected {len(selected)} pairs")

    # Vote: count appearances and track step positions
    pair_counts: Counter = Counter()
    pair_steps: dict = defaultdict(list)
    for selected in all_runs:
        for step_idx, pair in enumerate(selected):
            pair_counts[pair] += 1
            pair_steps[pair].append(step_idx + 1)

    # Keep pairs appearing in >= consensus_min runs, sorted by frequency
    # then by average step (earlier = more important)
    consensus = [
        pair for pair, count in pair_counts.items()
        if count >= consensus_min
    ]
    consensus.sort(key=lambda p: (-pair_counts[p], np.mean(pair_steps[p])))

    if verbose:
        print(f"ALT interactions: consensus — {len(consensus)} pairs (≥{consensus_min}/{n_runs} runs)")
        for pair in consensus:
            freq = pair_counts[pair]
            avg_step = np.mean(pair_steps[pair])
            print(f"    {pair[0]}*{pair[1]}  {freq}/{n_runs} runs (avg step {avg_step:.1f})")

    # Build log DataFrame compatible with downstream consumers
    search_log = []
    for i, pair in enumerate(consensus):
        search_log.append({
            "step": i + 1,
            "pair_a": pair[0],
            "pair_b": pair[1],
            "frequency": pair_counts[pair],
            "avg_step": round(float(np.mean(pair_steps[pair])), 2),
            "rmse_after": float("nan"),
            "improvement": float("nan"),
        })
    log_df = pd.DataFrame(search_log) if search_log else pd.DataFrame(
        columns=["step", "pair_a", "pair_b", "frequency", "avg_step", "rmse_after", "improvement"])

    return consensus, log_df


def _build_alt_inner_residual_bank(
    X_no_alt: pd.DataFrame,
    y_alt: pd.Series,
    cfg: dict,
    n_inner_splits: int = 5,
    seed: int = SEED,
) -> np.ndarray:
    """Build cross-fitted OOF residual bank for kNN smoothing.

    Runs inner CV on the given data (all rows must have known y) and returns
    OOF residuals — one per row, computed only from held-out predictions.
    """
    X = X_no_alt.reset_index(drop=True)
    y = pd.to_numeric(y_alt, errors="coerce").reset_index(drop=True)
    n = len(X)
    r_bank = np.zeros(n)
    if n < n_inner_splits * 2:
        return r_bank
    inner_kf = KFold(n_splits=n_inner_splits, shuffle=True, random_state=seed)
    for itr, iva in inner_kf.split(X):
        inner_fit = _fit_alt_model_on_rows(X, y, cfg, itr)
        r_bank[iva] = y.iloc[iva].to_numpy() - inner_fit["pred_all"].iloc[iva].to_numpy()
    return r_bank


def _apply_alt_knn_correction(
    X_no_alt: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    r_bank: np.ndarray,
    k: int = ALT_KNN_K,
    p: int = ALT_KNN_P,
) -> np.ndarray:
    """Compute kNN residual correction for validation rows.

    Uses PCA-transformed base features (no interactions) for distance to
    avoid curse of dimensionality.  Interactions are not needed for kNN —
    we just want distance in benchmark space.
    Returns correction vector (same length as val_idx) — caller multiplies by alpha.
    """
    scaler = StandardScaler()
    n_comp = min(ALT_PCA_N_COMPONENTS, X_no_alt.shape[1], len(train_idx) - 1)
    pca = PCA(n_components=max(1, n_comp))

    X_tr_pca = pca.fit_transform(scaler.fit_transform(X_no_alt.iloc[train_idx].values))
    X_va_pca = pca.transform(scaler.transform(X_no_alt.iloc[val_idx].values))

    n_neighbors = min(k, len(train_idx))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_tr_pca)
    D, I = nn.kneighbors(X_va_pca)
    W = 1.0 / (D + 1e-6) ** p
    W = W / W.sum(axis=1, keepdims=True)
    rhat = (W * r_bank[I]).sum(axis=1)
    return rhat



def _summarize_bayes_model(model, feature_names: List[str],
                           alt_result: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """Summarize BayesianRidge coefficients projected back to original feature space.

    Handles three cases:
    1. Post-PCA architecture (alt_result dict with pca/pca_scaler/int_scaler):
       - For PCA components: project coefficients back via pca.components_.T
       - For interaction features: report coefficients directly (already interpretable)
    2. Pipeline with PCA: project back via components_.T
    3. Pipeline with Scaler only: un-standardize coefficients
    """
    # --- Case 1: Post-PCA architecture (plain BayesianRidge + alt_result info) ---
    if alt_result is not None and not isinstance(model, Pipeline):
        br = model
        if not hasattr(br, "coef_"):
            return None
        pca_obj = alt_result.get("pca")
        pca_scaler_obj = alt_result.get("pca_scaler")
        int_names = alt_result.get("interaction_names", [])
        if pca_obj is None or pca_scaler_obj is None:
            return None
        # Dimensionality guard
        if pca_obj.components_.shape[1] != len(feature_names):
            return None
        coef_all = np.asarray(br.coef_).ravel()
        n_pca = pca_obj.n_components_
        combined_feat_names = alt_result.get("combined_feature_names", [])

        # Detect layout: count PC-nonlinear features between PCs and interactions
        n_pc_nl = sum(1 for n in combined_feat_names if n.startswith("PC") and ("^2" in n or "*PC" in n))
        coef_pca = coef_all[:n_pca]
        coef_pc_nl = coef_all[n_pca:n_pca + n_pc_nl]
        coef_int = coef_all[n_pca + n_pc_nl:]

        # Project PCA coefficients back to original (scaled) feature space
        coef_scaled = pca_obj.components_.T @ coef_pca  # (n_base_features,)
        scale = getattr(pca_scaler_obj, "scale_", np.ones(len(feature_names)))
        scale = np.where(scale == 0, 1.0, scale)
        coef_orig_base = coef_scaled / scale

        rows = []
        for i, feat in enumerate(feature_names):
            rows.append({
                "feature": feat,
                "coef_original": coef_orig_base[i],
                "coef_standardized": coef_scaled[i],
                "source": "PCA-projected",
            })
        # PC nonlinear features (squares, cross-interactions)
        pc_nl_names_list = [n for n in combined_feat_names if n.startswith("PC") and ("^2" in n or "*PC" in n)]
        for j, nlname in enumerate(pc_nl_names_list):
            if j < len(coef_pc_nl):
                rows.append({
                    "feature": nlname,
                    "coef_original": coef_pc_nl[j],
                    "coef_standardized": coef_pc_nl[j],
                    "source": "PCA-nonlinear",
                })
        # Interaction features: coefficients are on scaled interaction space
        int_scaler_obj = alt_result.get("int_scaler")
        for j, iname in enumerate(int_names):
            if j < len(coef_int):
                int_scale = 1.0
                if int_scaler_obj is not None and hasattr(int_scaler_obj, "scale_"):
                    int_scale = int_scaler_obj.scale_[j] if j < len(int_scaler_obj.scale_) else 1.0
                    if int_scale == 0:
                        int_scale = 1.0
                rows.append({
                    "feature": iname,
                    "coef_original": coef_int[j] / int_scale,
                    "coef_standardized": coef_int[j],
                    "source": "interaction",
                })
        df = pd.DataFrame(rows)
        df["abs_coef_original"] = df["coef_original"].abs()
        df["abs_coef_standardized"] = df["coef_standardized"].abs()
        df = df.sort_values("abs_coef_standardized", ascending=False).reset_index(drop=True)
        intercept_row = pd.DataFrame({
            "feature": ["__intercept__ (PCA+interactions)"],
            "coef_original": [float(br.intercept_)],
            "coef_standardized": [np.nan],
            "abs_coef_original": [abs(float(br.intercept_))],
            "abs_coef_standardized": [np.nan],
            "source": [""],
        })
        return pd.concat([intercept_row, df], ignore_index=True)

    # --- Case 2 & 3: Pipeline-based model (backward compat) ---
    if not isinstance(model, Pipeline):
        return None
    if "br" not in model.named_steps:
        return None

    br = model.named_steps["br"]

    # PCA pipeline: report PCA-space coefficients + original-space effective weights
    if "pca" in model.named_steps:
        pca_step: PCA = model.named_steps["pca"]
        scaler_step: StandardScaler = model.named_steps["scaler"]
        coef_pca = np.asarray(br.coef_).ravel()
        coef_scaled = pca_step.components_.T @ coef_pca
        scale = getattr(scaler_step, "scale_", np.ones(len(feature_names)))
        scale = np.where(scale == 0, 1.0, scale)
        coef_orig = coef_scaled / scale
        df = pd.DataFrame({
            "feature": feature_names,
            "coef_original": coef_orig,
            "coef_standardized": coef_scaled,
        })
        df["abs_coef_original"] = df["coef_original"].abs()
        df["abs_coef_standardized"] = df["coef_standardized"].abs()
        df = df.sort_values("abs_coef_standardized", ascending=False).reset_index(drop=True)
        intercept_row = pd.DataFrame({
            "feature": ["__intercept__ (via PCA)"],
            "coef_original": [float(br.intercept_)],
            "coef_standardized": [np.nan],
            "abs_coef_original": [abs(float(br.intercept_))],
            "abs_coef_standardized": [np.nan],
        })
        return pd.concat([intercept_row, df], ignore_index=True)

    # Non-PCA pipeline: original behavior
    if "scaler" not in model.named_steps:
        return None
    scaler_step = model.named_steps["scaler"]
    coef_std = np.asarray(br.coef_).ravel()
    if coef_std.size != len(feature_names):
        return None
    scale = getattr(scaler_step, "scale_", None)
    mean = getattr(scaler_step, "mean_", None)
    if scale is None or mean is None:
        return None
    scale = np.where(scale == 0, 1.0, scale)
    coef_orig = coef_std / scale
    intercept_orig = float(br.intercept_ - np.sum(coef_std * mean / scale))
    df = pd.DataFrame({
        "feature": feature_names,
        "coef_original": coef_orig,
        "coef_standardized": coef_std,
    })
    df["abs_coef_original"] = df["coef_original"].abs()
    df["abs_coef_standardized"] = df["coef_standardized"].abs()
    df = df.sort_values("abs_coef_standardized", ascending=False).reset_index(drop=True)
    intercept_row = pd.DataFrame({
        "feature": ["__intercept__"],
        "coef_original": [intercept_orig],
        "coef_standardized": [np.nan],
        "abs_coef_original": [abs(intercept_orig)],
        "abs_coef_standardized": [np.nan],
    })
    return pd.concat([intercept_row, df], ignore_index=True)


def _fit_alt_model_on_rows(
    X_no_alt_all: pd.DataFrame,
    alt_series: pd.Series,
    cfg: dict,
    train_idx: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    interaction_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Union[pd.Series, List[str], float, Optional[object]]]:
    """Fit the ALT model on specified rows and predict for all rows.

    Architecture: PCA(10) on all base features captures linear signal, then
    post-PCA interaction columns (raw scaled products) are appended.
    BayesianRidge regresses on the concatenated [PCs | interactions] matrix.

    Args:
        X_no_alt_all: Feature DataFrame WITHOUT the ALT column.
        alt_series: Original ALT target values (may contain NaN).
        cfg: Configuration dict (feature selection settings — not used for
            PCA path but kept for API compat).
        train_idx: Indices to use for training. If None, uses all rows.
        sample_weight: Optional per-row weights.
        interaction_pairs: Explicit list of (col_a, col_b) interaction pairs.
            If None, falls back to global ALT_POST_PCA_INTERACTIONS.

    Returns:
        Dict with keys:
        - pred_all: Series of predictions for all rows
        - selected_features: List of base feature names fed to PCA
        - fallback_value: Mean ALT value (used if model fails)
        - n_known: Number of training rows with known ALT
        - fitted_model: Fitted BayesianRidge instance (or None)
        - calibration_summary: Info about spline calibration (or None)
        - pca_scaler: Fitted StandardScaler for base features
        - pca: Fitted PCA transform
        - int_scaler: Fitted StandardScaler for interaction columns
        - interaction_names: List of interaction column names
        - combined_feature_names: List of all feature names [PC1..PCn, int1..intN]
    """
    X_no_alt_all = pd.DataFrame(X_no_alt_all).copy()
    alt_numeric = pd.to_numeric(pd.Series(alt_series), errors="coerce")
    alt_numeric = alt_numeric.reindex(X_no_alt_all.index)

    n_total = len(X_no_alt_all)
    if train_idx is None:
        train_idx = np.arange(n_total)
    train_idx = np.asarray(train_idx, dtype=int)
    train_idx = train_idx[(train_idx >= 0) & (train_idx < n_total)]

    train_mask = pd.Series(False, index=X_no_alt_all.index)
    if len(train_idx):
        train_mask.iloc[train_idx] = True
    known_mask = alt_numeric.notna()
    train_known_mask = train_mask & known_mask

    # Separate SVD factor columns from base features — SVD factors bypass PCA
    svd_cols = [c for c in X_no_alt_all.columns if c.startswith("_svd_f")]
    selected = [c for c in X_no_alt_all.columns if c not in svd_cols]

    n_known = int(train_known_mask.sum())
    fallback = float(np.nanmean(alt_numeric.loc[train_known_mask])) if n_known else float("nan")
    fitted_model: Optional[object] = None
    calibration_summary: Optional[dict] = None
    pca_scaler_out = None
    pca_out = None
    int_scaler_out = None
    interaction_names_out: List[str] = []
    combined_names: List[str] = []

    min_samples = max(3, ALT_PCA_N_COMPONENTS + 1)  # PCA needs n > n_components
    if n_known < min_samples or len(selected) == 0 or not np.isfinite(fallback):
        if not np.isfinite(fallback):
            fallback = float(np.nanmean(alt_numeric.to_numpy()))
            if not np.isfinite(fallback):
                fallback = 0.0
        pred_all = pd.Series(fallback, index=X_no_alt_all.index, dtype="float64")
    else:
        X_train_base = X_no_alt_all.loc[train_known_mask, selected].values

        # 1. Fit StandardScaler + PCA on base features only
        n_pca = min(ALT_PCA_N_COMPONENTS, X_train_base.shape[0] - 1, X_train_base.shape[1])
        n_pca = max(1, n_pca)
        pca_scaler = StandardScaler(with_mean=True, with_std=True)
        pca = PCA(n_components=n_pca)
        X_train_scaled = pca_scaler.fit_transform(X_train_base)
        X_train_pca = pca.fit_transform(X_train_scaled)

        # 1b. PC nonlinear features (squares and/or cross-interactions)
        _PC_SQUARES = True    # PC1², PC2², ... (10 features)
        _PC_CROSS = False     # PC1×PC2, PC1×PC3, ... (45 features) — hurts at n=111
        pc_nl_names = []
        pc_nl_train_cols = []
        if _PC_SQUARES:
            for i in range(n_pca):
                pc_nl_train_cols.append(X_train_pca[:, i] ** 2)
                pc_nl_names.append(f"PC{i+1}^2")
        if _PC_CROSS:
            for i, j in itertools.combinations(range(n_pca), 2):
                pc_nl_train_cols.append(X_train_pca[:, i] * X_train_pca[:, j])
                pc_nl_names.append(f"PC{i+1}*PC{j+1}")
        if pc_nl_train_cols:
            pc_nl_train_raw = np.column_stack(pc_nl_train_cols)
            pc_nl_scaler = StandardScaler()
            pc_nl_train = pc_nl_scaler.fit_transform(pc_nl_train_raw)
        else:
            pc_nl_train = np.empty((len(X_train_pca), 0))
            pc_nl_scaler = None

        # 1c. Direct SVD factors (bypass PCA, get own coefficients)
        svd_direct_names = []
        if svd_cols:
            svd_train_raw = X_no_alt_all.loc[train_known_mask, svd_cols].values
            svd_direct_scaler = StandardScaler()
            svd_train_scaled = svd_direct_scaler.fit_transform(svd_train_raw)
            svd_direct_names = [c for c in svd_cols]
        else:
            svd_train_scaled = np.empty((int(train_known_mask.sum()), 0))
            svd_direct_scaler = None

        # 2. Compute interaction columns from raw (pre-PCA) DataFrame — fit scaler
        int_train, int_names, int_scaler = _compute_alt_interaction_features(
            X_no_alt_all.loc[train_known_mask], scaler=None, fit=True,
            pairs=interaction_pairs,
        )

        # 3. Concatenate [PCA | PC-nonlinear | SVD-direct | scaled interactions]
        blocks_train = [X_train_pca]
        if pc_nl_train.shape[1]:
            blocks_train.append(pc_nl_train)
        if svd_train_scaled.shape[1]:
            blocks_train.append(svd_train_scaled)
        if int_train.shape[1]:
            blocks_train.append(int_train)
        X_train_combined = np.hstack(blocks_train)
        pc_names = [f"PC{i+1}" for i in range(n_pca)]
        combined_names = pc_names + pc_nl_names + svd_direct_names + int_names

        # 4. Fit plain BayesianRidge on the combined matrix
        br = BayesianRidge(compute_score=False)
        y_fit = alt_numeric.loc[train_known_mask].to_numpy()
        fit_kwargs = {}
        if sample_weight is not None:
            w_fit = sample_weight[train_known_mask.values] if hasattr(train_known_mask, 'values') else sample_weight[train_known_mask]
            fit_kwargs["sample_weight"] = w_fit
        br.fit(X_train_combined, y_fit, **fit_kwargs)

        # 5. Predict for all rows
        X_all_scaled = pca_scaler.transform(X_no_alt_all[selected].values)
        X_all_pca = pca.transform(X_all_scaled)
        # PC nonlinear features for all rows
        if pc_nl_scaler is not None:
            pc_nl_all_cols = []
            if _PC_SQUARES:
                for i in range(n_pca):
                    pc_nl_all_cols.append(X_all_pca[:, i] ** 2)
            if _PC_CROSS:
                for i, j in itertools.combinations(range(n_pca), 2):
                    pc_nl_all_cols.append(X_all_pca[:, i] * X_all_pca[:, j])
            pc_nl_all = pc_nl_scaler.transform(np.column_stack(pc_nl_all_cols))
        else:
            pc_nl_all = np.empty((len(X_all_pca), 0))
        # SVD direct features for all rows
        if svd_direct_scaler is not None and svd_cols:
            svd_all_scaled = svd_direct_scaler.transform(X_no_alt_all[svd_cols].values)
        else:
            svd_all_scaled = np.empty((len(X_all_pca), 0))
        int_all, _, _ = _compute_alt_interaction_features(
            X_no_alt_all, scaler=int_scaler, fit=False,
            pairs=interaction_pairs,
        )
        blocks_all = [X_all_pca]
        if pc_nl_all.shape[1]:
            blocks_all.append(pc_nl_all)
        if svd_all_scaled.shape[1]:
            blocks_all.append(svd_all_scaled)
        if int_all.shape[1]:
            blocks_all.append(int_all)
        X_all_combined = np.hstack(blocks_all)
        pred_all = pd.Series(
            br.predict(X_all_combined),
            index=X_no_alt_all.index,
            dtype="float64",
        )

        # X8: Residual additive head — soft regime model with two Ridge heads
        # blended by sigmoid gate on SVD factor 1 (capability proxy)
        if _MODULE_SVD_ROW_FACTORS is not None:
            stage1_train_pred = br.predict(X_train_combined)
            residuals = y_fit - stage1_train_pred
            # Select top-k raw features most correlated with residuals
            raw_train = X_no_alt_all.loc[train_known_mask, selected].values
            res_corrs = np.array([abs(np.corrcoef(raw_train[:, j], residuals)[0, 1])
                                  for j in range(raw_train.shape[1])])
            res_corrs = np.nan_to_num(res_corrs)
            top_k_res = min(7, len(res_corrs))
            top_idx_res = np.argsort(res_corrs)[-top_k_res:]
            raw_top = raw_train[:, top_idx_res]
            from sklearn.linear_model import Ridge as _Ridge
            # Soft regime: two Ridge heads blended by sigmoid on SVD factor 1
            _svd = _MODULE_SVD_ROW_FACTORS
            svd_f1_all = _svd.iloc[:, 0].reindex(X_no_alt_all.index).values
            gate_train = svd_f1_all[train_known_mask.values]
            gate_all = svd_f1_all
            gate_mean = np.mean(gate_train)
            gate_std = np.std(gate_train) + 1e-8
            gate_train_z = 3.0 * (gate_train - gate_mean) / gate_std
            gate_all_z = 3.0 * (gate_all - gate_mean) / gate_std
            sig_train = 1.0 / (1.0 + np.exp(-gate_train_z))
            sig_all = 1.0 / (1.0 + np.exp(-gate_all_z))
            res_scaler = StandardScaler()
            raw_top_scaled = res_scaler.fit_transform(raw_top)
            # High-capability head
            res_ridge_hi = _Ridge(alpha=1.0)
            res_ridge_hi.fit(raw_top_scaled, residuals, sample_weight=sig_train)
            # Low-capability head
            res_ridge_lo = _Ridge(alpha=1.0)
            res_ridge_lo.fit(raw_top_scaled, residuals, sample_weight=1.0 - sig_train)
            raw_all = X_no_alt_all[selected].values[:, top_idx_res]
            raw_all_scaled = res_scaler.transform(raw_all)
            corr_hi = res_ridge_hi.predict(raw_all_scaled)
            corr_lo = res_ridge_lo.predict(raw_all_scaled)
            res_correction = sig_all * corr_hi + (1.0 - sig_all) * corr_lo
            pred_all = pd.Series(
                pred_all.values + res_correction,
                index=pred_all.index, dtype="float64")

        fitted_model = br
        pca_scaler_out = pca_scaler
        pca_out = pca
        int_scaler_out = int_scaler
        interaction_names_out = int_names

        # 6. Spline calibration
        calib = _fit_alt_residual_spline(pred_all.loc[train_known_mask], alt_numeric.loc[train_known_mask])
        if calib is not None:
            correction = _apply_alt_residual_spline(calib, pred_all)
            pred_all = pd.Series(pred_all.values + correction, index=pred_all.index, dtype="float64")
            calibration_summary = {
                "type": "spline",
                "n_points": calib["n_points"],
                "n_knots": calib["n_knots"],
            }

    return {
        "pred_all": pred_all.astype(float),
        "selected_features": selected,
        "fallback_value": float(fallback),
        "n_known": n_known,
        "fitted_model": fitted_model,
        "calibration_summary": calibration_summary,
        "pca_scaler": pca_scaler_out,
        "pca": pca_out,
        "int_scaler": int_scaler_out,
        "interaction_names": interaction_names_out,
        "combined_feature_names": combined_names,
    }


def impute_alt_for_all(
    X_no_alt_all: pd.DataFrame,
    alt_series: pd.Series,
    cfg: dict,
) -> Dict[str, Union[pd.Series, List[str], float]]:
    """Impute missing ALT values using a model trained on observed values.

    This is the main entry point for ALT imputation. It trains a model on
    rows where ALT is observed, then fills in missing values with predictions.

    Unlike _fit_alt_model_on_rows, this function:
    1. Uses ALL observed ALT values for training (no train_idx restriction)
    2. Returns a "filled" series that blends observed and imputed values

    Args:
        X_no_alt_all: Feature DataFrame WITHOUT the ALT column.
        alt_series: Original ALT target values (with NaN for missing).
        cfg: Feature selection configuration dict.

    Returns:
        Dict with keys:
        - filled: Series with observed values kept, missing values imputed
        - predicted: Full predictions (even for rows with observed ALT)
        - selected_features: Features used by the model
        - fallback_value: Mean ALT (fallback for edge cases)
        - n_known: Number of rows with observed ALT
        - fitted_model: Fitted model instance
        - calibration_summary: Spline calibration info

    Example:
        >>> result = impute_alt_for_all(X, alt_series, cfg)
        >>> alt_filled = result['filled']  # Use for downstream modeling
    """
    X_no_alt_all = pd.DataFrame(X_no_alt_all).copy()
    fit = _fit_alt_model_on_rows(X_no_alt_all, pd.Series(alt_series), cfg)
    alt_numeric = pd.to_numeric(pd.Series(alt_series), errors="coerce").reindex(X_no_alt_all.index)

    filled = alt_numeric.copy()
    known_mask = alt_numeric.notna()
    missing_mask = ~known_mask
    pred_all = fit["pred_all"]
    if missing_mask.any():
        filled.loc[missing_mask] = pred_all.loc[missing_mask]

    return {
        "filled": filled.astype(float),
        "predicted": pred_all.astype(float),
        "selected_features": fit["selected_features"],
        "fallback_value": fit["fallback_value"],
        "n_known": fit["n_known"],
        "fitted_model": fit["fitted_model"],
        "calibration_summary": fit["calibration_summary"],
        "pca_scaler": fit.get("pca_scaler"),
        "pca": fit.get("pca"),
        "int_scaler": fit.get("int_scaler"),
        "interaction_names": fit.get("interaction_names", []),
        "combined_feature_names": fit.get("combined_feature_names", []),
    }






def select_features_tree(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgb",
    mode: Union[str, int] = "auto",           # "auto" -> 1-SE; int -> keep top-K; "none" -> no selection
    k_grid: List[Union[int, str]] = (10, 20, 50, 100, 200, "all"),
    cv: int = 5,
    cv_repeats: int = 1,
    random_state: int = 42,
    # --- new args ---
    drop_collinear: bool = True,
    r2_threshold: float = 0.95,
    min_variance: float = 1e-12,
) -> Tuple[List[str], pd.DataFrame, Dict]:
    """Full feature selection pipeline using tree-based importance ranking.

    This is the main feature selection entry point, combining:
    1. Tree-based importance ranking (LightGBM/XGBoost gain)
    2. Collinearity pruning (drop highly correlated features)
    3. Optimal k selection via 1-SE rule (if mode='auto')

    Args:
        X: Feature matrix as DataFrame.
        y: Target series.
        model_type: 'lgbm' or 'xgb' for ranking. Defaults to 'xgb'.
        mode: Selection mode:
            - 'auto': Use 1-SE rule to find optimal k
            - int: Keep exactly top-k features
            - 'none': Keep all features after collinearity pruning
        k_grid: K values to evaluate when mode='auto'. Defaults to standard grid.
        cv: Cross-validation folds. Defaults to 5.
        cv_repeats: Number of CV repeats (new seed per repeat). Defaults to 1.
        random_state: Random seed. Defaults to 42.
        drop_collinear: If True, prune highly correlated features. Defaults to True.
        r2_threshold: R² threshold for collinearity pruning (0.95 = drop if
            corr >= 0.975). Defaults to 0.95.
        min_variance: Minimum variance to consider a feature. Defaults to 1e-12.

    Returns:
        Tuple of (selected_cols, ranking_df, diagnostics) where:
        - selected_cols: List of selected feature names
        - ranking_df: DataFrame with columns [feature, mean_gain, std_gain]
        - diagnostics: Dict with selection details including:
            - chosen_K: Number of features selected
            - collinearity: Info about dropped features
            - CV traces (when mode='auto')

    Algorithm:
        1. Rank all features by tree importance (CV-averaged gain)
        2. Optionally prune features with R² >= threshold to higher-ranked ones
        3. Select final subset based on mode (auto/top-k/none)
    """
    Xdf = pd.DataFrame(X).reset_index(drop=True)
    yser = (y.iloc[:,0] if isinstance(y, pd.DataFrame) and y.shape[1]==1 else pd.Series(y)).reset_index(drop=True)

    # numeric y and mask of rows with known labels
    ynum = pd.to_numeric(yser, errors="coerce")
    row_mask = ynum.notna().to_numpy()
    Xtr = Xdf.iloc[row_mask].copy()
    ytr = ynum.iloc[row_mask].to_numpy()


    # X, y = _ensure_frame_series(X, y)
    # # Use only rows with target present
    # mask = y.notna()
    # Xtr = pd.DataFrame(X)[mask].copy()
    # ytr = pd.Series(y)[mask].copy()

    ranking_df = rank_features_tree(
        Xtr,
        ytr,
        model_type=model_type,
        cv=cv,
        cv_repeats=cv_repeats,
        random_state=random_state,
    )
    ranked_cols = ranking_df["feature"].tolist()

    # ---- prune collinear features by importance on TRAIN data ----
    collinear_info = {"r2_threshold": r2_threshold, "dropped_map": {}, "n_dropped": 0}
    if drop_collinear and len(ranked_cols) > 1:
        pruned_cols, dropped_map = _prune_correlated_by_importance(
            Xtr, ranked_cols, r2_threshold=r2_threshold, min_variance=min_variance
        )
        collinear_info["dropped_map"] = dropped_map
        collinear_info["n_dropped"] = sum(len(v) for v in dropped_map.values())
        ranked_cols = pruned_cols

    # ---- selection modes ----
    if mode == "none":
        diags = {
            "chosen_K": len(ranked_cols),
            "reason": "mode=none",
            "collinearity": collinear_info,
            "cv_repeats": int(_normalize_cv_repeats(cv_repeats)),
        }
        return ranked_cols, ranking_df, diags

    if isinstance(mode, int):
        K = max(1, min(int(mode), len(ranked_cols)))
        diags = {
            "chosen_K": K,
            "reason": "top-K",
            "collinearity": collinear_info,
            "cv_repeats": int(_normalize_cv_repeats(cv_repeats)),
        }
        return ranked_cols[:K], ranking_df, diags

    # auto (1-SE rule) over the pruned ranking
    chosen_cols, diags = choose_k_by_cv_from_rank(
        Xtr,
        pd.Series(ytr),
        ranked_cols,
        model_type=model_type,
        k_grid=list(k_grid),
        cv=cv,
        cv_repeats=cv_repeats,
        random_state=random_state,
    )
    diags = dict(diags)  # ensure mutable
    diags["collinearity"] = collinear_info
    return chosen_cols, ranking_df, diags


def _prune_correlated_by_importance(
    X_train: pd.DataFrame,
    ranked_features: List[str],
    r2_threshold: float = 0.95,
    min_variance: float = 1e-12,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Prune highly correlated features, keeping the most important one.

    Given a list of features ranked by importance, this function identifies
    groups of highly correlated features (R² >= threshold) and keeps only
    the highest-ranked member of each group.

    This prevents multicollinearity issues in downstream models and reduces
    redundancy without losing predictive information.

    Args:
        X_train: Training data for computing correlations.
        ranked_features: Features sorted by importance (most important first).
        r2_threshold: R² threshold for pruning. Features with R² >= threshold
            to any higher-ranked feature are dropped. Defaults to 0.95.
        min_variance: Minimum variance required to consider a feature.
            Features with variance <= min_variance are ignored. Defaults to 1e-12.

    Returns:
        Tuple of (kept_features, dropped_map) where:
        - kept_features: List of retained feature names (in importance order)
        - dropped_map: Dict mapping kept features to list of features they
            "absorbed" (features dropped due to high correlation with them)

    Algorithm:
        1. Compute correlation matrix on training data
        2. Convert to R² (square of Pearson correlation)
        3. Iterate through ranked features:
            a. If feature not already dropped, keep it
            b. Mark all lower-ranked features with R² >= threshold as dropped

    Example:
        >>> kept, dropped = _prune_correlated_by_importance(X_train, ranking)
        >>> print(f"Kept {len(kept)} features, dropped {sum(len(v) for v in dropped.values())}")
    """
    # Guard rails
    r2_threshold = float(r2_threshold)
    if not (0.0 < r2_threshold < 1.0):
        raise ValueError("r2_threshold must be in (0, 1).")

    # Restrict to ranked, numeric, non-constant columns
    cols = [c for c in ranked_features if c in X_train.columns]
    Xn = X_train[cols].select_dtypes(include=[np.number]).copy()
    if Xn.shape[1] <= 1:
        return [c for c in ranked_features if c in Xn.columns], {}

    variances = Xn.var(axis=0, ddof=0)
    Xn = Xn.loc[:, variances > min_variance]
    if Xn.shape[1] <= 1:
        # If everything was constant, just return what remains in order
        kept = [c for c in ranked_features if c in Xn.columns]
        return kept, {}

    # Pearson |r| -> R^2
    corr = Xn.corr().abs().fillna(0.0)
    r2 = corr ** 2
    # ignore self-corr
    np.fill_diagonal(r2.values, 0.0)

    kept: List[str] = []
    dropped = set()
    dropped_map: Dict[str, List[str]] = {}

    for f in ranked_features:
        if f not in Xn.columns or f in dropped:
            continue
        # Check correlation with ALL already-kept features (bidirectional)
        dominated = False
        for kept_feat in kept:
            if r2.loc[f, kept_feat] >= r2_threshold:
                dominated = True
                break
        if dominated:
            dropped.add(f)
            continue
        kept.append(f)
        # Drop any LOWER-ranked features with high R^2 to f
        high = r2.index[(r2.loc[f] >= r2_threshold)].tolist()
        if high:
            dropped_map[f] = high
            dropped.update(high)

    return kept, dropped_map
# ---- END TREE-BASED FEATURE SELECTION ----

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_pst_timestamp() -> str:
    from zoneinfo import ZoneInfo
    from datetime import datetime
    tz = ZoneInfo("America/Los_Angeles")
    return datetime.now(tz).strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import mean_squared_error # type: ignore
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def prediction_intervals_from_trees(model, X, z: float = 1.96):
    if not hasattr(model, "estimators_"):
        raise ValueError("Model has no estimators_ attribute for tree-based intervals.")
    all_preds = np.stack([est.predict(X) for est in model.estimators_], axis=0)
    mu = all_preds.mean(axis=0)
    std = all_preds.std(axis=0, ddof=1)
    lower = mu - z * std
    upper = mu + z * std
    return mu, std, lower, upper

def bayesian_ridge_predict_with_sigma(br: BayesianRidge, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Predict with posterior uncertainty for BayesianRidge."""
    if not hasattr(br, "sigma_"):
        raise ValueError("Model not fitted or missing sigma_.")
    mu = br.predict(X)
    Sigma = br.sigma_
    noise_var = 1.0 / float(br.alpha_)
    XS = X @ Sigma
    var = np.einsum("ij,ij->i", XS, X) + noise_var
    std = np.sqrt(np.maximum(var, 1e-12))
    return mu, std


def compute_uncertainty_calibration_factor(
    spec: "ModelSpec",
    X: np.ndarray,
    y: np.ndarray,
    target_coverage: float = 0.95,
    n_splits: int = 5,
    cv_repeats: int = 1,
    seed: int = SEED,
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Compute calibration factor to scale model uncertainty for proper coverage.

    Uses cross-validation to collect (residual, std) pairs and finds the factor k
    such that |residual| < k * std achieves target_coverage.

    This preserves relative uncertainty (extrapolation points remain more uncertain)
    while ensuring calibrated coverage.

    Args:
        spec: ModelSpec defining the model to evaluate.
        X: Feature matrix of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        target_coverage: Desired coverage probability (default: 0.95).
        n_splits: Number of CV folds.
        cv_repeats: Number of CV repeats (new seed per repeat). Defaults to 1.
        seed: Base seed for CV splits. Defaults to SEED.
        splits: Optional pre-computed splits to reuse.

    Returns:
        Calibration factor k. Multiply model std by k for calibrated intervals.
    """
    normalized_residuals = []

    if splits is None:
        splits = _build_repeated_splits(len(X), n_splits, cv_repeats, seed)
    if not splits:
        return 1.0

    for tr_idx, va_idx in splits:
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        w_tr = sample_weight[tr_idx] if sample_weight is not None else None

        model = spec.build(X_tr, y_tr, w_tr)
        preds = model.predict(X_va)

        # Get model's uncertainty estimate
        std_va = np.full(len(y_va), np.nan)

        if isinstance(model, Pipeline):
            if "br" in model.named_steps:
                scaler = model.named_steps["scaler"]
                br = model.named_steps["br"]
                X_va_scaled = scaler.transform(X_va)
                _, std_va = bayesian_ridge_predict_with_sigma(br, X_va_scaled)
            elif "ard" in model.named_steps:
                scaler = model.named_steps["scaler"]
                ard = model.named_steps["ard"]
                X_va_scaled = scaler.transform(X_va)
                _, std_param = ard.predict(X_va_scaled, return_std=True)
                noise_var = 1.0 / ard.alpha_
                std_va = np.sqrt(std_param**2 + noise_var)

        if np.isnan(std_va).all():
            # Fallback: use constant std = 1 (calibration factor becomes raw residual quantile)
            std_va = np.ones(len(y_va))

        # Compute |residual| / std
        abs_residuals = np.abs(y_va - preds)
        # Avoid division by zero
        std_va = np.maximum(std_va, 1e-8)
        normalized_residuals.extend(abs_residuals / std_va)

    normalized_residuals = np.array(normalized_residuals)
    # Find k such that |residual| / std < k covers target_coverage
    calibration_factor = float(np.percentile(normalized_residuals, target_coverage * 100))
    # Ensure reasonable bounds
    calibration_factor = max(0.5, min(calibration_factor, 5.0))
    return calibration_factor


def hgb_quantile_preds(X_train, y_train, X, q_low=0.05, q_high=0.95, seed=SEED):
    hgb_low = HistGradientBoostingRegressor(loss="quantile", quantile=q_low, random_state=seed)
    hgb_high = HistGradientBoostingRegressor(loss="quantile", quantile=q_high, random_state=seed)
    hgb_mean = HistGradientBoostingRegressor(loss="squared_error", random_state=seed)
    hgb_low.fit(X_train, y_train)
    hgb_high.fit(X_train, y_train)
    hgb_mean.fit(X_train, y_train)
    lower = hgb_low.predict(X)
    upper = hgb_high.predict(X)
    mu = hgb_mean.predict(X)
    std = (upper - lower) / (2.0 * 1.96)
    return mu, std, lower, upper

def prob_above_threshold(mu: np.ndarray, std: np.ndarray, threshold: float) -> np.ndarray:
    try:
        from scipy.stats import norm # type: ignore
        z = (threshold - mu) / np.where(std <= 1e-12, 1e-12, std)
        return 1.0 - norm.cdf(z) 
    except Exception:
        z = (threshold - mu) / np.where(std <= 1e-12, 1e-12, std)
        return 0.5 * (1.0 - np.erf(z / np.sqrt(2.0)))

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
):
    """Run the specialized column imputer on benchmark data.

    This is a convenience wrapper around SpecializedColumnImputer that:
    1. Extracts numeric columns (excluding ID and target)
    2. Configures the imputer with appropriate hyperparameters
    3. Runs fit_transform and returns the imputed DataFrame

    Args:
        df: Input DataFrame with ID_COL and numeric benchmark columns.
        passes: Maximum imputation iterations. Defaults to 14.
        alpha: Significance level for prediction intervals. Defaults to 0.08.
        verbose: Verbosity level (0=silent, 1=progress, 2=debug). Defaults to 1.
        use_feature_selector: If True, use per-column feature selection.
            Defaults to True.
        selector_tau: Maximum allowed correlation with already-selected predictors
            (used by SpecializedColumnImputer). Defaults to 0.8.
        selector_k_max: Maximum predictors per column for linear models. Defaults to 30.
        gp_selector_k_max: Maximum predictors for GP models (mRMR selection). Defaults to 10.
        imputer_n_jobs: Parallel workers for imputation. Defaults to -1 (all cores).
        categorical_threshold: Max distinct numeric values to auto-treat as categorical.
        force_categorical_cols: Explicit columns to impute as categorical.
        tolerance_percentile: Percentile of training uncertainties used to set initial
            tolerance per column.
        tolerance_relaxation_factor: Multiplicative relaxation when a round writes no
            cells.
        tolerance_multiplier: Multiplier on the initial tolerance to account for
            higher uncertainty in missing rows.
        tier_quantiles: Quantiles that define easy/medium/hard imputation tiers.
        calibrate_tolerances: If True, run per-column tolerance calibration.
        calibration_target_rmse_ratio: Target RMSE/SD ratio for calibration.
        calibration_n_rounds: Monte Carlo rounds for calibration.
        calibration_holdout_frac: Fraction of known values to hold out for calibration.
        recalibrate_every_n_passes: Recalibrate every N passes (0 = only at start).

    Returns:
        Tuple of (imputed_df, imputer) where:
        - imputed_df: DataFrame with ID_COL and imputed numeric columns
        - imputer: Fitted SpecializedColumnImputer instance
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
    imputer = SpecializedColumnImputer(
        passes=passes,  # Number of rounds over all tiers
        alpha=alpha,
        seed=SEED,
        verbose=verbose,
        n_jobs=n_jobs_fit,  # Simplified: single n_jobs parameter
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
        # v7.2: Per-column tolerance calibration
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
    elif "abs_err" in per_cell.columns and {"y_pred","y_true"}.issubset(per_cell.columns):
        per_cell["err"] = per_cell["y_pred"] - per_cell["y_true"]
    else:
        per_cell["err"] = np.nan
    err_std = per_cell.groupby("col")["err"].std(ddof=1).rename("err_std").reset_index()
    per_col = per_col.merge(err_std, on="col", how="left")
    return per_cell, per_col, by_bin

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
        # Valid features are typically longer than 3 chars and not pure numbers
        if len(parts) == 2:
            valid = all(
                len(p) > 3 and not p.replace(".", "").replace("-", "").isdigit()
                for p in parts
            )
            if valid:
                return parts
        # Otherwise, it's a name containing '*', not an interaction
        return [s]
    if "^" in s:
        base = s.split("^", 1)[0].strip()
        return [base] if base else [s]
    return [s]


def _fit_alt_residual_spline(
    base_pred: pd.Series,
    actual: pd.Series,
    min_points: int = 15,
    max_knots: int = 6,
) -> Optional[dict]:
    """
    Fits a spline-based calibration model on residuals vs. base predictions.
    Returns dict with fitted transformer/regressor or None if insufficient data.
    """
    finite_base = pd.Series(np.isfinite(base_pred.to_numpy()), index=base_pred.index)
    finite_actual = pd.Series(np.isfinite(actual.to_numpy()), index=actual.index)
    mask = base_pred.notna() & actual.notna() & finite_base & finite_actual
    n = int(mask.sum())
    if n < min_points:
        return None
    x = base_pred.loc[mask].to_numpy().reshape(-1, 1)
    residual = (actual.loc[mask] - base_pred.loc[mask]).to_numpy()
    n_knots = max(3, min(max_knots, n // 3))
    transformer = SplineTransformer(
        n_knots=n_knots,
        degree=3,
        include_bias=True,
    )
    Xt = transformer.fit_transform(x)
    reg = LinearRegression()
    reg.fit(Xt, residual)
    return {
        "transformer": transformer,
        "reg": reg,
        "n_points": n,
        "n_knots": n_knots,
    }


def _apply_alt_residual_spline(calib: dict, preds: pd.Series) -> np.ndarray:
    transformer: SplineTransformer = calib["transformer"]
    reg: LinearRegression = calib["reg"]
    X = preds.to_numpy().reshape(-1, 1)
    Xt = transformer.transform(X)
    return reg.predict(Xt)


# ==============================================================================
# MODEL SPECIFICATIONS
# ==============================================================================

@dataclass
class ModelSpec:
    """Specification for a regression model to evaluate.

    A ModelSpec encapsulates everything needed to train and identify a model:
    - A name for reporting
    - A build function that takes (X_train, y_train) and returns a fitted model
    - A kind label for grouping (linear, tree, blend, etc.)

    Attributes:
        name: Human-readable model name (e.g., "BayesianRidge").
        build: Callable that takes (X_train, y_train) and returns fitted model.
            The returned model must have a predict(X) method.
        kind: Model category for grouping ("linear", "tree", "blend", etc.).
    """
    name: str
    build: callable
    kind: str


def build_model_specs() -> List[ModelSpec]:
    """Build the list of model specifications to evaluate.

    Returns:
        List containing ModelSpecs for BayesianRidge, ARDRegression, and
        GaussianProcessRegressor for cross-validation comparison.
    """
    specs: List[ModelSpec] = []

    # BayesianRidge - robust linear baseline with uncertainty
    def fit_bayes(Xtr, ytr, wtr=None):
        pipe = Pipeline([("scaler", StandardScaler()), ("br", BayesianRidge(compute_score=True, max_iter=10000))])
        if wtr is not None:
            pipe.fit(Xtr, ytr, br__sample_weight=wtr)
        else:
            pipe.fit(Xtr, ytr)
        return pipe
    specs.append(ModelSpec("BayesianRidge", fit_bayes, "linear"))

    # ARDRegression - Bayesian with per-feature relevance determination
    def fit_ard(Xtr, ytr, wtr=None):
        pipe = Pipeline([("scaler", StandardScaler()), ("ard", ARDRegression(max_iter=10000))])
        pipe.fit(Xtr, ytr)  # ARDRegression does not support sample_weight
        return pipe
    specs.append(ModelSpec("ARDRegression", fit_ard, "linear"))

    return specs

def cross_val_rmse_for_model(
    spec: ModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_jobs: int = 1,
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    cv_repeats: int = 1,
    seed: int = SEED,
    sample_weight: Optional[np.ndarray] = None,
    dense_mask: Optional[np.ndarray] = None,
):
    """Evaluate a model specification using cross-validated RMSE.

    Performs K-fold cross-validation to estimate out-of-sample RMSE for a
    given model specification. Supports both sequential and parallel execution.

    Args:
        spec: ModelSpec defining the model to evaluate (contains build function).
        X: Feature matrix of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        n_splits: Number of CV folds. Defaults to 5.
        n_jobs: Parallel workers for fold evaluation. 1 = sequential. Defaults to 1.
        splits: Pre-computed (train_idx, val_idx) tuples. If None, generates new
            KFold splits. Useful for ensuring consistent splits across models.
        cv_repeats: Number of CV repeats (new seed per repeat). Defaults to 1.
        seed: Base seed for CV splits. Defaults to SEED.

    Returns:
        Tuple of (mean_rmse, std_rmse) across folds.

    Example:
        >>> spec = ModelSpec("BayesianRidge", fit_bayes, "linear")
        >>> mean_rmse, std_rmse = cross_val_rmse_for_model(spec, X_train, y_train)
        >>> print(f"RMSE: {mean_rmse:.3f} ± {std_rmse:.3f}")
    """
    if splits is None:
        splits = _build_repeated_splits(len(X), n_splits, cv_repeats, seed)
    if not splits:
        raise ValueError("n_splits must be >= 2 for cross-validation.")

    def _eval_fold(tr, va):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        wtr = sample_weight[tr] if sample_weight is not None else None
        model = spec.build(Xtr, ytr, wtr)
        pred = model.predict(Xva)
        if dense_mask is not None:
            dense_va = dense_mask[va]
            if dense_va.any():
                return rmse(yva[dense_va], pred[dense_va])
            return float("nan")
        return rmse(yva, pred)

    if n_jobs == 1:
        rmses = [_eval_fold(tr, va) for tr, va in splits]
    else:
        rmses = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_eval_fold)(tr, va) for tr, va in splits
        )
    rmses = np.asarray(rmses, dtype=float)
    return float(np.mean(rmses)), float(np.std(rmses, ddof=1))

def calibrate_prediction_intervals(
    spec: ModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    target_coverage: float = 0.95,
    n_splits: int = 5,
    cv_repeats: int = 1,
    seed: int = SEED,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Compute conformal calibration factor for prediction intervals.

    Uses conformal prediction to determine the residual quantile needed to achieve
    target coverage. This provides honest prediction intervals based on actual
    held-out errors rather than model assumptions.

    Args:
        spec: ModelSpec defining the model to evaluate.
        X: Feature matrix of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        target_coverage: Desired coverage probability (default: 0.95 for 95% intervals).
        n_splits: Number of CV folds for collecting residuals.
        cv_repeats: Number of CV repeats (new seed per repeat). Defaults to 1.
        seed: Base seed for CV splits. Defaults to SEED.

    Returns:
        Calibration quantile: the absolute residual threshold for target coverage.

    Example:
        >>> cal_quantile = calibrate_prediction_intervals(spec, X, y, target_coverage=0.95)
        >>> # For new predictions: intervals = (pred - cal_quantile, pred + cal_quantile)
    """
    residuals = []

    splits = _build_repeated_splits(len(X), n_splits, cv_repeats, seed)
    if not splits:
        return float("nan")

    for tr_idx, va_idx in splits:
        w_tr = sample_weight[tr_idx] if sample_weight is not None else None
        model = spec.build(X[tr_idx], y[tr_idx], w_tr)
        preds = model.predict(X[va_idx])
        residuals.extend(np.abs(y[va_idx] - preds))

    residuals = np.array(residuals)
    # Conformal quantile: find the residual threshold that covers target_coverage
    calibration_quantile = np.percentile(residuals, target_coverage * 100)
    return calibration_quantile

def compute_coverage(y_true: np.ndarray, y_pred: np.ndarray, intervals: Tuple[np.ndarray, np.ndarray]) -> float:
    """Check what fraction of true values fall within predicted intervals.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        intervals: Tuple of (lower_bounds, upper_bounds).

    Returns:
        Coverage fraction (0 to 1).

    Example:
        >>> lower = predictions - calibration_quantile
        >>> upper = predictions + calibration_quantile
        >>> coverage = compute_coverage(y_test, predictions, (lower, upper))
        >>> print(f"Actual coverage: {coverage:.1%}")
    """
    lower, upper = intervals
    covered = (y_true >= lower) & (y_true <= upper)
    return np.mean(covered)


def _fit_positive_ridge(X_tr, y_tr, X_va, alphas=None):
    """Fit Ridge(positive=True) with manual alpha selection via LOO approximation.

    Since RidgeCV doesn't support positive=True, we loop over alphas and pick
    the one with lowest training MSE (with regularization acting as implicit CV).
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 20)
    best_alpha = alphas[len(alphas) // 2]
    best_loss = np.inf
    for a in alphas:
        r = Ridge(alpha=a, positive=True)
        r.fit(X_tr, y_tr)
        loss = np.mean((y_tr - r.predict(X_tr)) ** 2) + a * np.sum(r.coef_ ** 2)
        if loss < best_loss:
            best_loss = loss
            best_alpha = a
    final = Ridge(alpha=best_alpha, positive=True)
    final.fit(X_tr, y_tr)
    return final, final.predict(X_va)


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


def compute_normalized_conformal_intervals(
    mu: np.ndarray,
    oof_preds: np.ndarray,
    y_train: np.ndarray,
    X_features: np.ndarray,
    train_idx: np.ndarray,
    pre_imputation_missing: np.ndarray,
    feature_gains: np.ndarray,
    feature_names: Optional[List[str]] = None,
    target_coverage: float = 0.95,
    n_splits: int = 5,
    k_neighbors: int = 7,
    n_pca_components: int = 10,
    sigma_floor_percentile: float = 5.0,
    seed: int = 42,
) -> dict:
    """Compute heteroscedastic prediction intervals via normalized conformal prediction.

    Builds a scale model from mask-native uncertainty features (raw missing
    fraction, clean importance-weighted missing, suite-level missing, distance
    metrics) with monotone (positive-only) Ridge coefficients so that more
    missing data always implies wider intervals.

    Args:
        mu: Point predictions for ALL models (train + pred), shape (n_all,).
        oof_preds: OOF predictions for training rows (may contain NaN), shape (n_train,).
        y_train: Actual target values for training rows, shape (n_train,).
        X_features: Feature matrix for ALL models (post-imputation), shape (n_all, p).
        train_idx: Indices (into the n_all arrays) of training rows.
        pre_imputation_missing: Boolean missing mask BEFORE imputation, shape (n_all, n_raw_features).
        feature_gains: Importance (gain) per raw feature column, shape (n_raw_features,).
        feature_names: Column names corresponding to pre_imputation_missing columns.
        target_coverage: Desired marginal coverage (default 0.95).
        n_splits: Folds for cross-fitting the scale model.
        k_neighbors: Neighbors for k-NN distance feature.
        n_pca_components: Max PCA components for distance features.
        sigma_floor_percentile: Percentile for flooring sigma_hat.
        seed: Random seed.

    Returns:
        Dict with keys: std, lower, upper, sigma_hat, q_hat, sigma_floor,
        oof_sigma, sigma_cv, oof_coverage, scale_model_coef, uncertainty_features.
    """
    n_all = len(mu)
    n_train = len(y_train)
    rng = np.random.RandomState(seed)

    # --- a. Mask-native uncertainty features ---
    # (i) Raw unweighted missing fraction (no gain contamination)
    raw_missing_frac = pre_imputation_missing.astype(float).mean(axis=1)

    # (ii) Importance-weighted missing fraction with target/ALT gains zeroed out
    clean_gains = feature_gains.copy()
    if feature_names is not None:
        for i, name in enumerate(feature_names):
            if name in TARGETS:
                clean_gains[i] = 0.0
    log_gains = np.log1p(clean_gains)
    total_log_gain = log_gains.sum()
    if total_log_gain < 1e-12:
        total_log_gain = 1.0
    imp_weighted_missing_clean = (pre_imputation_missing.astype(float) @ log_gains) / total_log_gain

    # (iii) Suite-level missing fraction (detects entire benchmark suites missing)
    suite_missing_frac = _detect_suite_missing_fracs(
        pre_imputation_missing,
        feature_names if feature_names is not None else [f"col_{i}" for i in range(pre_imputation_missing.shape[1])],
    )

    # --- b. PCA distance features (fit on training rows, transform all) ---
    n_comp = min(n_pca_components, X_features.shape[1], len(train_idx) - 1)
    n_comp = max(1, n_comp)
    pca_scaler = StandardScaler()
    pca = PCA(n_components=n_comp)
    train_pca = pca.fit_transform(pca_scaler.fit_transform(X_features[train_idx]))
    all_pca = pca.transform(pca_scaler.transform(X_features))

    # Standardize PCA components by training std
    train_std = np.std(train_pca, axis=0)
    train_std[train_std < 1e-12] = 1.0
    train_pca_z = train_pca / train_std
    all_pca_z = all_pca / train_std

    # Mahalanobis distance from training centroid
    centroid = np.mean(train_pca_z, axis=0)
    mahal_dist = np.sqrt(np.sum((all_pca_z - centroid) ** 2, axis=1))

    # k-NN distance
    k_actual = min(k_neighbors, len(train_idx) - 1)
    k_actual = max(1, k_actual)
    nn = NearestNeighbors(n_neighbors=k_actual)
    nn.fit(train_pca_z)
    knn_dists, _ = nn.kneighbors(all_pca_z)
    knn_dist = np.mean(knn_dists, axis=1)

    # --- c. Cross-fit the scale model (on training rows only) ---
    # OOF residuals for training rows
    oof_valid_mask = ~np.isnan(oof_preds)
    residuals_all_train = np.abs(y_train - oof_preds)

    # Build uncertainty features for all models
    # 5 features: raw_missing_frac, imp_weighted_missing_clean, suite_missing_frac, mahal_dist, knn_dist
    U_all = np.column_stack([
        raw_missing_frac, imp_weighted_missing_clean, suite_missing_frac,
        mahal_dist, knn_dist,
    ])

    # Cross-fit: predict log-residuals for each training row OOF
    # Uses Ridge(positive=True) so more missing/distant → wider intervals (monotone constraint)
    oof_sigma = np.full(n_train, np.nan)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rng.randint(0, 2**31))
    train_positions = np.arange(n_train)  # positions within training subset
    valid_positions = train_positions[oof_valid_mask]

    for fold_tr, fold_va in kf.split(valid_positions):
        fold_tr_pos = valid_positions[fold_tr]
        fold_va_pos = valid_positions[fold_va]

        # Get the global indices for this fold's training subset
        fold_tr_global = train_idx[fold_tr_pos]

        # Re-fit Scaler + PCA + kNN on this fold's training subset
        fold_scaler = StandardScaler()
        fold_n_comp = min(n_pca_components, X_features.shape[1], len(fold_tr_global) - 1)
        fold_n_comp = max(1, fold_n_comp)
        fold_pca = PCA(n_components=fold_n_comp)
        fold_train_pca = fold_pca.fit_transform(fold_scaler.fit_transform(X_features[fold_tr_global]))

        fold_train_std = np.std(fold_train_pca, axis=0)
        fold_train_std[fold_train_std < 1e-12] = 1.0
        fold_train_pca_z = fold_train_pca / fold_train_std

        # Transform fold-val rows
        fold_va_global = train_idx[fold_va_pos]
        fold_va_pca = fold_pca.transform(fold_scaler.transform(X_features[fold_va_global]))
        fold_va_pca_z = fold_va_pca / fold_train_std

        fold_centroid = np.mean(fold_train_pca_z, axis=0)
        fold_mahal_tr = np.sqrt(np.sum((fold_train_pca_z - fold_centroid) ** 2, axis=1))
        fold_mahal_va = np.sqrt(np.sum((fold_va_pca_z - fold_centroid) ** 2, axis=1))

        fold_k = min(k_neighbors, len(fold_tr_global) - 1)
        fold_k = max(1, fold_k)
        fold_nn = NearestNeighbors(n_neighbors=fold_k)
        fold_nn.fit(fold_train_pca_z)
        fold_knn_tr, _ = fold_nn.kneighbors(fold_train_pca_z)
        fold_knn_va, _ = fold_nn.kneighbors(fold_va_pca_z)

        # Build uncertainty features for this fold
        U_fold_tr = np.column_stack([
            raw_missing_frac[fold_tr_global],
            imp_weighted_missing_clean[fold_tr_global],
            suite_missing_frac[fold_tr_global],
            fold_mahal_tr,
            np.mean(fold_knn_tr, axis=1),
        ])
        U_fold_va = np.column_stack([
            raw_missing_frac[fold_va_global],
            imp_weighted_missing_clean[fold_va_global],
            suite_missing_frac[fold_va_global],
            fold_mahal_va,
            np.mean(fold_knn_va, axis=1),
        ])

        # Fit Ridge(positive=True) on log1p(|residual|) — monotone constraint
        log_r_tr = np.log1p(residuals_all_train[fold_tr_pos])
        _, fold_preds = _fit_positive_ridge(U_fold_tr, log_r_tr, U_fold_va)
        oof_sigma[fold_va_pos] = fold_preds

    # Convert from log space
    oof_sigma_valid = np.expm1(oof_sigma[oof_valid_mask])

    # --- d. Floor and calibrate ---
    sigma_floor = max(np.percentile(oof_sigma_valid, sigma_floor_percentile), 1e-6)
    oof_sigma_valid = np.maximum(oof_sigma_valid, sigma_floor)

    conformity_scores = residuals_all_train[oof_valid_mask] / oof_sigma_valid
    n_valid = len(conformity_scores)
    q_level = min(np.ceil((n_valid + 1) * target_coverage) / n_valid, 1.0)
    q_hat = float(np.percentile(conformity_scores, q_level * 100))

    # OOF coverage check
    oof_lower = mu[train_idx][oof_valid_mask] - q_hat * oof_sigma_valid
    oof_upper = mu[train_idx][oof_valid_mask] + q_hat * oof_sigma_valid
    oof_coverage = float(np.mean((y_train[oof_valid_mask] >= oof_lower) & (y_train[oof_valid_mask] <= oof_upper)))

    # --- e. Fit final featurizer + scale model on all valid training rows ---
    log_r_all_valid = np.log1p(residuals_all_train[oof_valid_mask])
    U_train_valid = U_all[train_idx[oof_valid_mask]]
    final_ridge, _ = _fit_positive_ridge(U_train_valid, log_r_all_valid, U_train_valid)

    # --- f. Produce intervals for all models ---
    sigma_hat = np.expm1(final_ridge.predict(U_all))
    sigma_hat = np.maximum(sigma_hat, sigma_floor)

    lower = mu - q_hat * sigma_hat
    upper = mu + q_hat * sigma_hat
    std_new = q_hat * sigma_hat / 1.96  # equivalent std for prob_above_threshold

    sigma_cv = float(np.std(sigma_hat) / np.mean(sigma_hat)) if np.mean(sigma_hat) > 0 else 0.0

    # Scale model coefficients
    coef_names = [
        "raw_missing_frac", "imp_weighted_missing_clean", "suite_missing_frac",
        "mahal_dist", "knn_dist",
    ]
    scale_model_coef = dict(zip(coef_names, final_ridge.coef_.tolist()))
    scale_model_coef["intercept"] = float(final_ridge.intercept_)

    return {
        "std": std_new,
        "lower": lower,
        "upper": upper,
        "sigma_hat": sigma_hat,
        "q_hat": q_hat,
        "sigma_floor": sigma_floor,
        "oof_sigma": oof_sigma_valid,
        "sigma_cv": sigma_cv,
        "oof_coverage": oof_coverage,
        "scale_model_coef": scale_model_coef,
        "uncertainty_features": pd.DataFrame({
            "raw_missing_frac": raw_missing_frac,
            "imp_weighted_missing_clean": imp_weighted_missing_clean,
            "suite_missing_frac": suite_missing_frac,
            "mahal_dist": mahal_dist,
            "knn_dist": knn_dist,
            "sigma_hat": sigma_hat,
        }),
    }


def cross_val_rmse_multi_seed(
    spec: ModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    seeds: List[int] = None,
    n_splits: int = 5,
) -> Tuple[float, float]:
    """Run CV with multiple seeds and report mean±std of RMSE estimates.

    Evaluates model performance across different random train/test splits to
    assess stability and reduce dependence on a single random split.

    Args:
        spec: ModelSpec defining the model to evaluate.
        X: Feature matrix of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        seeds: List of random seeds to use. Defaults to [42, 123, 456, 789, 1000].
        n_splits: Number of CV folds per seed. Defaults to 5.

    Returns:
        Tuple of (mean_rmse, std_rmse) across all seeds.

    Example:
        >>> mean, std = cross_val_rmse_multi_seed(spec, X, y)
        >>> print(f"RMSE: {mean:.2f} ± {std:.2f} (across {len(seeds)} seeds)")
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1000]

    results = []
    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(kf.split(X))
        mean_rmse, _ = cross_val_rmse_for_model(spec, X, y, n_splits=n_splits, splits=splits)
        results.append(mean_rmse)

    return float(np.mean(results)), float(np.std(results, ddof=1))

def fit_and_predict_all(
    specs: List[ModelSpec],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_all: np.ndarray,
    cv_n_jobs: int = 1,
    model_n_jobs: int = 1,
    cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    cv_repeats: int = 1,
    cv_seed: int = SEED,
    sample_weight: Optional[np.ndarray] = None,
):
    def _fit_single(spec: ModelSpec) -> Tuple[str, Dict[str, np.ndarray]]:
        # Compute calibration factor for uncertainty scaling
        calibration_factor = 1.0
        if spec.name in ("BayesianRidge", "ARDRegression"):
            try:
                calibration_factor = compute_uncertainty_calibration_factor(
                    spec,
                    X_train,
                    y_train,
                    target_coverage=0.95,
                    n_splits=5,
                    cv_repeats=cv_repeats,
                    seed=cv_seed,
                    splits=cv_splits,
                    sample_weight=sample_weight,
                )
            except Exception:
                calibration_factor = 1.0

        model = spec.build(X_train, y_train, sample_weight)
        mu = model.predict(X_all)
        std = np.full_like(mu, fill_value=np.nan, dtype=float)
        lower = np.full_like(mu, fill_value=np.nan, dtype=float)
        upper = np.full_like(mu, fill_value=np.nan, dtype=float)

        if spec.name == "BayesianRidge":
            if isinstance(model, Pipeline) and "br" in model.named_steps:
                X_all_scaled = model.named_steps["scaler"].transform(X_all)
                br = model.named_steps["br"]
                mu_br, std_br = bayesian_ridge_predict_with_sigma(br, X_all_scaled)
                mu = mu_br
                std = std_br * calibration_factor  # Apply calibration
                lower = mu - 1.96 * std
                upper = mu + 1.96 * std
        elif spec.name == "ARDRegression":
            if isinstance(model, Pipeline) and "ard" in model.named_steps:
                X_all_scaled = model.named_steps["scaler"].transform(X_all)
                ard = model.named_steps["ard"]
                # predict(return_std=True) gives parameter uncertainty only
                # Must add observation noise for total predictive uncertainty
                mu, std_param = ard.predict(X_all_scaled, return_std=True)
                noise_var = 1.0 / ard.alpha_  # observation noise variance
                std = np.sqrt(std_param**2 + noise_var) * calibration_factor  # Apply calibration
                lower = mu - 1.96 * std
                upper = mu + 1.96 * std
        elif spec.kind == "tree":
            try:
                mu2, std2, lower2, upper2 = prediction_intervals_from_trees(model, X_all, z=1.96)
                std, lower, upper = std2, lower2, upper2
            except Exception:
                pass
        elif spec.name == "HistGradientBoosting":
            try:
                mu_h, std_h, low_h, up_h = hgb_quantile_preds(
                    X_train, y_train, X_all, 0.05, 0.95, seed=SEED
                )
                mu, std, lower, upper = mu_h, std_h, low_h, up_h
            except Exception:
                pass

        if np.isnan(std).all():
            mean_rmse, _ = cross_val_rmse_for_model(
                spec,
                X_train,
                y_train,
                n_splits=5,
                n_jobs=cv_n_jobs,
                splits=cv_splits,
                cv_repeats=cv_repeats,
                seed=cv_seed,
            )
            std = np.full_like(mu, mean_rmse)
            lower = mu - 1.96 * std
            upper = mu + 1.96 * std

        return spec.name, {"mu": mu, "std": std, "lower": lower, "upper": upper, "fitted": model, "calibration_factor": calibration_factor}

    if model_n_jobs == 1:
        items = [_fit_single(spec) for spec in specs]
    else:
        items = Parallel(n_jobs=model_n_jobs, prefer="processes")(
            delayed(_fit_single)(spec) for spec in specs
        )

    return {name: payload for name, payload in items}

def fit_and_predict_all_with_alt(
    specs: List[ModelSpec],
    X_df: pd.DataFrame,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    alt_col: str,
    cv_n_jobs: int = 1,
    model_n_jobs: int = 1,
    selector_cfg: Optional[dict] = None,
    poly_cfg: Optional[dict] = None,
    cv_base_features: Optional[pd.DataFrame] = None,
    cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    cv_repeats: int = 1,
    cv_seed: int = SEED,
    oof_repeats: int = 1,
    sample_weight: Optional[np.ndarray] = None,
):
    """Fit all models and generate predictions with proper ALT handling.

    This function simulates deployment conditions where the ALT target must
    be imputed for rows missing the main target. It uses OOF stacking to
    prevent leakage.

    The key insight is that at deployment time:
    - Training rows had known targets, so we used OOF ALT predictions during training
    - Inference rows have no known target, so we use full-fit ALT predictions

    Args:
        specs: List of ModelSpec instances to evaluate.
        X_df: Full feature DataFrame including alt_col.
        y_all: Full target array (may have NaN for inference rows).
        train_idx: Indices of training rows (with known target).
        pred_idx: Indices for prediction (typically all rows).
        alt_col: Name of the ALT target column.
        cv_n_jobs: Parallel workers for CV. Defaults to 1.
        model_n_jobs: Parallel workers for model fitting. Defaults to 1.
        selector_cfg: Feature selection config. Defaults to None.
        poly_cfg: Polynomial interaction config. Defaults to None.
        cv_base_features: Base features for CV (before ALT augmentation).
        cv_splits: Pre-computed CV splits. Defaults to None.
        oof_repeats: Repeats for inner ALT OOF CV. Defaults to 1.

    Returns:
        Dict mapping model name to dict with keys:
        - mu: Point predictions
        - std: Uncertainty estimates
        - lower: Lower confidence bound
        - upper: Upper confidence bound
        - fitted: Fitted model instance

    Algorithm:
        1. Generate OOF ALT predictions for training rows (prevents leakage)
        2. Train ALT model on all training rows (for inference predictions)
        3. Prepare training data with OOF ALT values
        4. Prepare inference data with full-fit ALT predictions
        5. Fit and predict with each model spec
    """
    if alt_col not in X_df.columns:
        X_train = X_df.values[train_idx]
        y_train = y_all[train_idx]
        X_all = X_df.values
        w_train = sample_weight[train_idx] if sample_weight is not None else None
        return fit_and_predict_all(
            specs,
            X_train,
            y_train,
            X_all,
            cv_n_jobs=cv_n_jobs,
            model_n_jobs=model_n_jobs,
            cv_splits=cv_splits,
            cv_repeats=cv_repeats,
            cv_seed=cv_seed,
            sample_weight=w_train,
        )

    alt_feature_names = [c for c in X_df.columns if c != alt_col]
    X_no_alt_df = X_df[alt_feature_names].copy()
    alt_numeric = pd.to_numeric(X_df[alt_col], errors="coerce")

    # 1. Generate OOF predictions for the TRAINING rows
    # These will be used to train the final model
    X_train_df = X_df.iloc[train_idx]
    y_alt_train = alt_numeric.iloc[train_idx]
    
    # We need a robust OOF generator
    w_train = sample_weight[train_idx] if sample_weight is not None else None
    oof_preds_train = _generate_oof_alt_predictions(
        X_train_df,
        y_alt_train,
        ALT_SELECTOR_CFG,
        n_splits=5,
        repeats=oof_repeats,
        seed=cv_seed,
        sample_weight=w_train,
    )

    # 2. Train Alt Model on ALL training rows (for inference on test set)
    alt_fit_full = _fit_alt_model_on_rows(X_no_alt_df, alt_numeric, ALT_SELECTOR_CFG, train_idx,
                                          sample_weight=sample_weight)
    alt_pred_all = alt_fit_full["pred_all"]

    # 2b. kNN residual correction for inference (non-training) rows
    if ALT_KNN_ALPHA > 0:
        known_train_mask = alt_numeric.iloc[train_idx].notna()
        known_train_positions = np.where(known_train_mask)[0]
        known_train_global = train_idx[known_train_positions]
        all_idx = np.arange(len(X_no_alt_df))
        infer_idx = np.setdiff1d(all_idx, train_idx)
        if len(known_train_global) >= 15 and len(infer_idx) > 0:
            # Use OOF residuals from step 1 as the residual bank
            oof_residuals = (
                alt_numeric.iloc[train_idx].loc[known_train_mask].to_numpy()
                - oof_preds_train.loc[known_train_mask].to_numpy()
            )
            rhat_infer = _apply_alt_knn_correction(
                X_no_alt_df, known_train_global, infer_idx, oof_residuals,
            )
            alt_pred_all.iloc[infer_idx] += ALT_KNN_ALPHA * rhat_infer

    # 3. Prepare Training Data for Final Model (using OOF)
    # We replace the Alt column with OOF predictions
    X_train_final_df = X_train_df.copy()
    X_train_final_df[alt_col] = oof_preds_train.values
    
    X_train = X_train_final_df.values
    y_train = y_all[train_idx]

    # 4. Prepare Inference Data (using Full-Fit Preds)
    # For the final prediction, we want to simulate the test time scenario.
    # At test time, we predict Alt using the model trained on Train.
    # So we use alt_pred_all for everyone.
    X_all_aug_df = X_df.copy()
    X_all_aug_df[alt_col] = alt_pred_all.values
    X_all_aug = X_all_aug_df.values

    def _fit_single(spec: ModelSpec) -> Tuple[str, Dict[str, np.ndarray]]:
        # Compute calibration factor for uncertainty scaling
        calibration_factor = 1.0
        if spec.name in ("BayesianRidge", "ARDRegression"):
            try:
                calibration_factor = compute_uncertainty_calibration_factor(
                    spec,
                    X_train,
                    y_train,
                    target_coverage=0.95,
                    n_splits=5,
                    cv_repeats=cv_repeats,
                    seed=cv_seed,
                    splits=cv_splits,
                    sample_weight=w_train,
                )
            except Exception:
                calibration_factor = 1.0

        model = spec.build(X_train, y_train, w_train)
        mu = model.predict(X_all_aug)

        std = np.full_like(mu, np.nan)
        lower = np.full_like(mu, np.nan)
        upper = np.full_like(mu, np.nan)

        if spec.name == "BayesianRidge":
            if isinstance(model, Pipeline) and "br" in model.named_steps:
                X_all_scaled = model.named_steps["scaler"].transform(X_all_aug)
                br = model.named_steps["br"]
                mu_br, std_br = bayesian_ridge_predict_with_sigma(br, X_all_scaled)
                mu = mu_br
                std = std_br * calibration_factor  # Apply calibration
                lower = mu - 1.96 * std
                upper = mu + 1.96 * std
        elif spec.name == "ARDRegression":
            if isinstance(model, Pipeline) and "ard" in model.named_steps:
                X_all_scaled = model.named_steps["scaler"].transform(X_all_aug)
                ard = model.named_steps["ard"]
                # predict(return_std=True) gives parameter uncertainty only
                # Must add observation noise for total predictive uncertainty
                mu, std_param = ard.predict(X_all_scaled, return_std=True)
                noise_var = 1.0 / ard.alpha_  # observation noise variance
                std = np.sqrt(std_param**2 + noise_var) * calibration_factor  # Apply calibration
                lower = mu - 1.96 * std
                upper = mu + 1.96 * std

        if np.isnan(std).all():
            base_df_for_cv = cv_base_features if cv_base_features is not None else X_df
            X_train_df_for_cv = base_df_for_cv.iloc[train_idx]
            y_train_for_cv = y_all[train_idx]
            mean_rmse, _ = cross_val_rmse_with_alt(
                spec,
                X_train_df_for_cv,
                y_train_for_cv,
                alt_col,
                n_splits=5,
                n_jobs=cv_n_jobs,
                selector_cfg=selector_cfg,
                poly_cfg=poly_cfg,
                splits=cv_splits,
                cv_repeats=cv_repeats,
                seed=cv_seed,
                oof_repeats=oof_repeats,
            )
            std = np.full_like(mu, mean_rmse)
            lower = mu - 1.96 * std
            upper = mu + 1.96 * std

        return spec.name, {"mu": mu, "std": std, "lower": lower, "upper": upper, "fitted": model, "calibration_factor": calibration_factor}

    if model_n_jobs == 1:
        items = [_fit_single(spec) for spec in specs]
    else:
        items = Parallel(n_jobs=model_n_jobs, prefer="processes")(
            delayed(_fit_single)(spec) for spec in specs
        )

    return {name: payload for name, payload in items}


def _generate_oof_alt_predictions(
    X_df: pd.DataFrame,
    y_alt: pd.Series,
    cfg: dict,
    n_splits: int = 5,
    repeats: int = 1,
    seed: int = SEED,
    sample_weight: Optional[np.ndarray] = None,
) -> pd.Series:
    """Generate out-of-fold predictions for the ALT target to prevent leakage.

    When using an imputed ALT column as a feature for predicting the main target,
    we must avoid leakage: the ALT model shouldn't see the same rows during
    training that will use its predictions for final model training.

    This function implements OOF stacking:
    1. Split data into K folds
    2. For each fold, train ALT model on K-1 folds, predict on held-out fold
    3. Concatenate held-out predictions to get full OOF predictions

    Args:
        X_df: Feature DataFrame (may contain ALT_TARGET column which is ignored).
        y_alt: Original ALT target values (with missing values).
        cfg: Configuration dict for ALT feature selection.
        n_splits: Number of CV folds. Defaults to 5.
        repeats: Number of CV repeats (new seed per repeat). Defaults to 1.
        seed: Random seed. Defaults to SEED.

    Returns:
        Series of OOF predictions, same index as X_df.

    Note:
        For rows where ALT was observed, OOF predictions may differ from
        observed values. For rows where ALT was missing, OOF predictions
        provide the imputed value without leakage.
        When repeats > 1, OOF predictions are averaged across repeats.
    """
    # Safety check for small data
    if len(X_df) < n_splits * 2:
        # Fallback to single fit if too small for CV
        fit = _fit_alt_model_on_rows(X_df.drop(columns=[ALT_TARGET], errors="ignore"), y_alt, cfg,
                                     sample_weight=sample_weight)
        return fit["pred_all"]

    splits = _build_repeated_splits(len(X_df), n_splits, repeats, seed)
    if not splits:
        fit = _fit_alt_model_on_rows(X_df.drop(columns=[ALT_TARGET], errors="ignore"), y_alt, cfg,
                                     sample_weight=sample_weight)
        return fit["pred_all"]

    oof_sum = pd.Series(0.0, index=X_df.index, dtype=float)
    oof_count = pd.Series(0, index=X_df.index, dtype=int)

    X_no_alt = X_df.drop(columns=[ALT_TARGET], errors="ignore")
    y_alt_numeric = pd.to_numeric(y_alt, errors="coerce")

    for fold_k, (tr, va) in enumerate(splits):
        # Train Alt Model on tr
        # We pass the global X_no_alt but specify train_idx=tr
        # This function handles the training on 'tr' and prediction on everything
        fit = _fit_alt_model_on_rows(X_no_alt, y_alt_numeric, cfg, train_idx=tr,
                                     sample_weight=sample_weight)

        # We only care about the predictions for 'va'
        pred_all = fit["pred_all"]
        va_idx = X_df.index[va]
        pred_va = pred_all.loc[va_idx].values.copy()

        # kNN residual smoothing on known-ALT training rows
        if ALT_KNN_ALPHA > 0:
            y_tr = y_alt_numeric.iloc[tr]
            known_in_tr = np.where(y_tr.notna())[0]
            known_tr_global = tr[known_in_tr]  # indices into X_no_alt
            if len(known_in_tr) >= 15:
                # Build inner OOF residual bank on known-ALT training rows
                X_known = X_no_alt.iloc[known_tr_global].reset_index(drop=True)
                y_known = y_alt_numeric.iloc[known_tr_global].reset_index(drop=True)
                r_bank = _build_alt_inner_residual_bank(
                    X_known, y_known, cfg, seed=seed + fold_k,
                )
                # kNN correction: neighbors from known-ALT training rows only
                rhat = _apply_alt_knn_correction(
                    X_no_alt, known_tr_global, va, r_bank,
                )
                pred_va = pred_va + ALT_KNN_ALPHA * rhat

        # Assign to OOF
        oof_sum.loc[va_idx] += pred_va
        oof_count.loc[va_idx] += 1

    oof_preds = oof_sum / oof_count.replace(0, np.nan)
    if oof_preds.isna().any():
        fallback_fit = _fit_alt_model_on_rows(X_no_alt, y_alt_numeric, cfg,
                                              sample_weight=sample_weight)
        oof_preds = oof_preds.fillna(fallback_fit["pred_all"])
    return oof_preds


def cross_val_rmse_with_alt(
    spec: ModelSpec,
    X_df: pd.DataFrame,
    y: np.ndarray,
    alt_col: str,
    n_splits: int = 5,
    n_jobs: int = 1,
    selector_cfg: Optional[dict] = None,
    poly_cfg: Optional[dict] = None,
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    cv_repeats: int = 1,
    seed: int = SEED,
    oof_repeats: int = 1,
    sample_weight: Optional[np.ndarray] = None,
    dense_mask: Optional[np.ndarray] = None,
):
    """Evaluate a model with ALT target imputation using nested CV.

    This is a more sophisticated CV evaluation that properly handles ALT
    target imputation to prevent leakage. For each outer fold:
    1. Generate OOF ALT predictions for training data (inner CV)
    2. Train ALT model on full training data for validation predictions
    3. Train final model on training data with OOF ALT
    4. Evaluate on validation data with full-fit ALT predictions

    This nested structure ensures the ALT model never sees validation data
    during training, giving realistic estimates of deployment performance.

    Args:
        spec: ModelSpec defining the model to evaluate.
        X_df: Feature DataFrame including alt_col.
        y: Target array.
        alt_col: Name of the ALT target column in X_df.
        n_splits: Number of outer CV folds. Defaults to 5.
        n_jobs: Parallel workers. Defaults to 1.
        selector_cfg: Feature selection config for target model. Defaults to None.
        poly_cfg: Polynomial interaction config. Defaults to None.
        splits: Pre-computed CV splits. Defaults to None.
        cv_repeats: Number of CV repeats (new seed per repeat). Defaults to 1.
        seed: Base seed for CV splits. Defaults to SEED.
        oof_repeats: Repeats for inner ALT OOF CV. Defaults to 1.

    Returns:
        Tuple of (mean_rmse, std_rmse, oof_preds, oof_folds) across folds.
        oof_preds: array of OOF predictions (NaN if not covered).
        oof_folds: array of fold assignments (-1 if not covered).

    Note:
        Falls back to simple CV if alt_col not present in X_df.
    """
    if alt_col not in X_df.columns:
        m, s = cross_val_rmse_for_model(
            spec,
            X_df.values,
            y,
            n_splits=n_splits,
            n_jobs=n_jobs,
            splits=splits,
            cv_repeats=cv_repeats,
            seed=seed,
            sample_weight=sample_weight,
            dense_mask=dense_mask,
        )
        return m, s, np.full(len(y), np.nan), np.full(len(y), -1, dtype=int)

    alt_feature_names = [c for c in X_df.columns if c != alt_col]
    X_no_alt_df = X_df[alt_feature_names].copy()
    alt_numeric = pd.to_numeric(X_df[alt_col], errors="coerce")

    if splits is None:
        splits = _build_repeated_splits(len(X_df), n_splits, cv_repeats, seed)
    if not splits:
        raise ValueError("n_splits must be >= 2 for cross-validation.")

    selector_cfg = selector_cfg or {"enabled": False}
    poly_cfg = poly_cfg or {"enabled": False}
    poly_enabled = bool(poly_cfg.get("enabled", False))
    poly_limit = int(poly_cfg.get("limit", 0) or 0)
    poly_include_squares = bool(poly_cfg.get("include_squares", False))

    def _eval_fold(tr, va):
        # 1. Generate OOF predictions for the TRAINING set of this fold
        X_tr_fold = X_df.iloc[tr]
        y_alt_tr_fold = alt_numeric.iloc[tr]
        w_tr = sample_weight[tr] if sample_weight is not None else None

        oof_preds_tr = _generate_oof_alt_predictions(
            X_tr_fold,
            y_alt_tr_fold,
            ALT_SELECTOR_CFG,
            n_splits=5,
            repeats=oof_repeats,
            seed=seed,
            sample_weight=w_tr,
        )

        # 2. Train Alt Model on FULL 'tr' to predict for 'va'
        alt_fit_full = _fit_alt_model_on_rows(X_no_alt_df, alt_numeric, ALT_SELECTOR_CFG, tr,
                                              sample_weight=sample_weight)
        alt_pred_all = alt_fit_full["pred_all"]

        # 3. Prepare Training Data for Final Model (using OOF)
        Xtr_df = X_df.iloc[tr].copy()
        Xtr_df[alt_col] = oof_preds_tr.values

        # 4. Prepare Validation Data for Final Model (using Full-Fit Preds)
        Xva_df = X_df.iloc[va].copy()
        Xva_df[alt_col] = alt_pred_all.loc[Xva_df.index].values

        # 5. Feature selection on THIS fold's training data
        sel_cols = _select_target_cols_for_train(Xtr_df, y[tr], selector_cfg)
        Xtr_sel = Xtr_df[sel_cols].copy()
        Xva_sel = Xva_df[sel_cols].copy()

        if poly_enabled:
            Xtr_sel, core = expand_poly_interactions(
                Xtr_sel,
                include_squares=poly_include_squares,
                limit=poly_limit,
                return_core=True,
            )
            Xva_sel = expand_poly_interactions(
                Xva_sel,
                include_squares=poly_include_squares,
                limit=poly_limit,
                preset_core=core,
            )

        model = spec.build(Xtr_sel.values, y[tr], w_tr)
        pred = model.predict(Xva_sel.values)
        if dense_mask is not None:
            dense_va = dense_mask[va]
            rmse_val = rmse(y[va][dense_va], pred[dense_va]) if dense_va.any() else float("nan")
        else:
            rmse_val = rmse(y[va], pred)
        return rmse_val, va, pred

    if n_jobs == 1:
        results = [_eval_fold(tr, va) for tr, va in splits]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_eval_fold)(tr, va) for tr, va in splits
        )
    rmses = np.array([r[0] for r in results], dtype=float)

    # Accumulate OOF predictions (average if repeated CV)
    oof_preds = np.full(len(y), np.nan)
    oof_folds = np.full(len(y), -1, dtype=int)
    oof_counts = np.zeros(len(y))
    for fold_k, (rmse_val, va, pred) in enumerate(results):
        fold_within = fold_k % n_splits
        for i, vi in enumerate(va):
            if oof_counts[vi] == 0:
                oof_preds[vi] = pred[i]
            else:
                oof_preds[vi] = (oof_preds[vi] * oof_counts[vi] + pred[i]) / (oof_counts[vi] + 1)
            oof_counts[vi] += 1
            oof_folds[vi] = fold_within

    return float(np.mean(rmses)), float(np.std(rmses, ddof=1)), oof_preds, oof_folds


def choose_best_model_with_alt(
    specs: List[ModelSpec],
    X_df: pd.DataFrame,
    y: np.ndarray,
    alt_col: str,
    n_splits: int = 5,
    model_n_jobs: int = 1,
    cv_n_jobs: int = 1,
    selector_cfg: Optional[dict] = None,
    poly_cfg: Optional[dict] = None,
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    cv_repeats: int = 1,
    seed: int = SEED,
    oof_repeats: int = 1,
    sample_weight: Optional[np.ndarray] = None,
    dense_mask: Optional[np.ndarray] = None,
):
    def _eval_spec(spec: ModelSpec):
        m, s, oof_preds, oof_folds = cross_val_rmse_with_alt(
            spec,
            X_df,
            y,
            alt_col,
            n_splits=n_splits,
            n_jobs=cv_n_jobs,
            selector_cfg=selector_cfg,
            poly_cfg=poly_cfg,
            splits=splits,
            cv_repeats=cv_repeats,
            seed=seed,
            oof_repeats=oof_repeats,
            sample_weight=sample_weight,
            dense_mask=dense_mask,
        )
        return {"model": spec.name, "rmse_mean": m, "rmse_std": s,
                "oof_preds": oof_preds, "oof_folds": oof_folds}

    if model_n_jobs == 1:
        rows = [_eval_spec(spec) for spec in specs]
    else:
        rows = Parallel(n_jobs=model_n_jobs, prefer="processes")(
            delayed(_eval_spec)(spec) for spec in specs
        )

    eval_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("oof_preds", "oof_folds")}
                            for r in rows]).sort_values("rmse_mean", ascending=True).reset_index(drop=True)
    best_name = eval_df.iloc[0]["model"]
    best_spec = next(sp for sp in specs if sp.name == best_name)
    best_row = next(r for r in rows if r["model"] == best_name)
    return best_spec, eval_df, best_row["oof_preds"], best_row["oof_folds"]

def choose_best_model(
    specs: List[ModelSpec],
    X: np.ndarray,
    y: np.ndarray,
    model_n_jobs: int = 1,
    cv_n_jobs: int = 1,
    cv_repeats: int = 1,
    seed: int = SEED,
):
    def _eval_spec(spec: ModelSpec):
        m, s = cross_val_rmse_for_model(
            spec,
            X,
            y,
            n_splits=5,
            n_jobs=cv_n_jobs,
            cv_repeats=cv_repeats,
            seed=seed,
        )
        return {"model": spec.name, "rmse_mean": m, "rmse_std": s}

    if model_n_jobs == 1:
        rows = [_eval_spec(spec) for spec in specs]
    else:
        rows = Parallel(n_jobs=model_n_jobs, prefer="processes")(
            delayed(_eval_spec)(spec) for spec in specs
        )

    eval_df = pd.DataFrame(rows).sort_values("rmse_mean", ascending=True).reset_index(drop=True)
    best_name = eval_df.iloc[0]["model"]
    best_spec = next(sp for sp in specs if sp.name == best_name)
    return best_spec, eval_df

def summarize_model(model, feature_names: List[str]) -> pd.DataFrame:
    # If it's a pipeline, inspect the last step too
    last = None
    if isinstance(model, Pipeline):
        last = list(model.named_steps.values())[-1]

    target = last or model  # where we'll look for attributes

    # 1) Linear-style models
    if hasattr(target, "coef_"):
        coefs = np.ravel(target.coef_)
        df = pd.DataFrame({"feature": feature_names, "importance": coefs})
        df["abs_importance"] = df["importance"].abs()
        return df.sort_values("abs_importance", ascending=False).drop(columns=["abs_importance"])

    # 2) Native tree/GBM-style importances
    if hasattr(target, "feature_importances_"):
        fi = np.ravel(target.feature_importances_)
        df = pd.DataFrame({"feature": feature_names, "importance": fi})
        return df.sort_values("importance", ascending=False)

    # 3) Generic fallback: permutation importance
    if hasattr(model, "_X_train_") and hasattr(model, "_y_train_"):
        r = permutation_importance(
            model, model._X_train_, model._y_train_,
            n_repeats=10, random_state=42, scoring="neg_mean_squared_error"
        )
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": r.importances_mean,
            "importance_std": r.importances_std
        })
        return df.sort_values("importance", ascending=False)

    # 4) If all else fails
    return pd.DataFrame({"feature": feature_names, "importance": np.nan})


def compute_variance_contributions(
    model,
    X: np.ndarray,
    feature_names: List[str],
    alt_info: Optional[dict] = None,
) -> pd.DataFrame:
    """Compute variance contribution of each feature to predictions.

    For linear models, computes (beta_j * std(X_j))^2 / var(predictions).
    This represents the fraction of prediction variance attributable to each feature.

    Args:
        model: Fitted model (Pipeline or estimator with coef_).
        X: Feature matrix used for predictions.
        feature_names: Names of features matching X columns.
        alt_info: Optional dict from alt_result with pca/pca_scaler/int_scaler
            for the post-PCA interaction architecture. When provided, PCA
            coefficients are projected back to original feature space via
            pca.components_.T, and interaction contributions are computed
            directly. feature_names should be the BASE feature names.

    Returns:
        DataFrame with columns: feature, variance_contribution, base_feature, grouped_contribution
    """
    # --- Post-PCA interaction architecture: project PCA back + interactions ---
    if alt_info is not None and not isinstance(model, Pipeline):
        pca_obj = alt_info.get("pca")
        int_names = alt_info.get("interaction_names", [])
        if pca_obj is not None and hasattr(model, "coef_"):
            coefs = np.ravel(model.coef_)
            n_pca = pca_obj.n_components_
            n_pca_input = pca_obj.components_.shape[1]
            # Dimensionality guard: feature_names must match PCA input dimensionality
            if n_pca_input != len(feature_names):
                return pd.DataFrame({
                    "feature": feature_names,
                    "variance_contribution": np.nan,
                    "base_feature": feature_names,
                    "grouped_contribution": np.nan,
                })
            coef_pca = coefs[:n_pca]
            coef_int = coefs[n_pca:]

            # PCA portion: project back to original features via squared loadings
            pc_contrib = coef_pca ** 2 * pca_obj.explained_variance_
            var_pca_per_base = (pca_obj.components_.T ** 2) @ pc_contrib  # (n_base,)

            # Interaction portion: coef^2 (interactions are standardized, std=1)
            var_int = np.zeros(len(int_names))
            for j in range(len(int_names)):
                if j < len(coef_int):
                    # Interactions were standardized — coef is on std=1 scale
                    var_int[j] = coef_int[j] ** 2

            # Combine and normalize
            all_names = list(feature_names) + list(int_names)
            all_contrib = np.concatenate([var_pca_per_base, var_int])
            total = all_contrib.sum()
            if total < 1e-12:
                total = 1.0
            all_contrib = all_contrib / total

            df = pd.DataFrame({
                "feature": all_names,
                "variance_contribution": all_contrib,
            })
            df["base_features"] = df["feature"].apply(lambda f: _feature_components(f))
            base_contrib = defaultdict(float)
            for _, row in df.iterrows():
                contrib_per_base = row["variance_contribution"] / len(row["base_features"])
                for base in row["base_features"]:
                    base_contrib[base] += contrib_per_base
            df["primary_base"] = df["base_features"].apply(lambda x: x[0] if x else "")
            df["grouped_contribution"] = df["primary_base"].map(base_contrib)
            return df.sort_values("variance_contribution", ascending=False).reset_index(drop=True)

    # Get coefficients
    if isinstance(model, Pipeline):
        last = list(model.named_steps.values())[-1]
    else:
        last = model

    if not hasattr(last, "coef_"):
        return pd.DataFrame({
            "feature": feature_names,
            "variance_contribution": np.nan,
            "base_feature": feature_names,
            "grouped_contribution": np.nan,
        })

    coefs = np.ravel(last.coef_)

    # Handle PCA pipelines: coefs are in PCA space (n_components), not feature space.
    # Project back to per-feature contributions using explained variance weighting.
    if isinstance(model, Pipeline) and "pca" in model.named_steps:
        pca_step = model.named_steps["pca"]
        n_pca_features = pca_step.components_.shape[1]
        if n_pca_features != len(feature_names):
            # PCA was trained on a different number of features than provided.
            # Fall back to NaN contributions.
            return pd.DataFrame({
                "feature": feature_names,
                "variance_contribution": np.nan,
                "base_feature": feature_names,
                "grouped_contribution": np.nan,
            })
        # Per-PC contribution: coef^2 * variance of that PC
        pc_contrib = coefs ** 2 * pca_step.explained_variance_
        # Distribute to original features via squared loadings
        var_contrib = (pca_step.components_.T ** 2) @ pc_contrib
        total = var_contrib.sum()
        if total < 1e-12:
            total = 1.0
        var_contrib = var_contrib / total
    elif len(coefs) != X.shape[1]:
        return pd.DataFrame({
            "feature": feature_names,
            "variance_contribution": np.nan,
            "base_feature": feature_names,
            "grouped_contribution": np.nan,
        })
    elif isinstance(model, Pipeline) and "scaler" in model.named_steps:
        # For Pipeline with StandardScaler, coefficients are on standardized scale (std=1)
        # So variance contribution is simply coef^2 / sum(coef^2)
        total_coef_sq = np.sum(coefs ** 2)
        if total_coef_sq < 1e-12:
            total_coef_sq = 1.0
        var_contrib = (coefs ** 2) / total_coef_sq
    else:
        # Raw coefficients - weight by feature variance
        preds = model.predict(X)
        pred_var = np.var(preds)
        if pred_var < 1e-12:
            pred_var = 1.0
        X_std = np.std(X, axis=0)
        var_contrib = (coefs * X_std) ** 2 / pred_var

    # Create DataFrame
    df = pd.DataFrame({
        "feature": feature_names,
        "variance_contribution": var_contrib,
    })

    # Extract base features using _feature_components
    df["base_features"] = df["feature"].apply(lambda f: _feature_components(f))

    # Group by base feature (a feature can contribute to multiple base features if it's an interaction)
    base_contrib = defaultdict(float)
    for _, row in df.iterrows():
        contrib_per_base = row["variance_contribution"] / len(row["base_features"])
        for base in row["base_features"]:
            base_contrib[base] += contrib_per_base

    # Add grouped contribution for each feature's primary base
    df["primary_base"] = df["base_features"].apply(lambda x: x[0] if x else "")
    df["grouped_contribution"] = df["primary_base"].map(base_contrib)

    return df.sort_values("variance_contribution", ascending=False).reset_index(drop=True)


def filter_features_by_importance(
    feature_names: List[str],
    variance_contributions: pd.DataFrame,
    threshold: float = 0.01,
) -> set:
    """Filter features keeping only those above importance threshold.

    Keeps features where the feature's own variance contribution >= threshold.

    Args:
        feature_names: List of all feature names.
        variance_contributions: DataFrame from compute_variance_contributions.
        threshold: Minimum variance contribution (default 1%).

    Returns:
        Set of feature names that meet the threshold.
    """
    if variance_contributions.empty or variance_contributions["variance_contribution"].isna().all():
        return set(feature_names)

    # Build lookup of feature -> variance_contribution
    feat_to_contrib = {}
    for _, row in variance_contributions.iterrows():
        feat = row["feature"]
        contrib = row.get("variance_contribution", 0.0)
        if pd.notna(contrib):
            feat_to_contrib[feat] = contrib

    # Keep features where own variance contribution >= threshold
    kept_features = set()
    for feat in feature_names:
        if feat_to_contrib.get(feat, 0.0) >= threshold:
            kept_features.add(feat)

    return kept_features


def compute_imputer_variance_contributions(
    imputer,
    X_df: pd.DataFrame,
    threshold: float = 0.01,
) -> Dict[str, set]:
    """Compute important predictors for each imputer model based on variance contribution.

    For each column model in the imputer, computes importance of predictors:
    - For linear models: uses coefficient-based variance contribution
    - For non-linear models: uses squared correlation (R²) as importance proxy

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
            # Compute R² for each predictor with the target column
            y = X_df[col].values
            valid_mask = ~np.isnan(y)

            if valid_mask.sum() < 10:
                # Not enough data points
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
                # Compute correlation
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


def mmss(delta_secs: float) -> str:
    m, s = divmod(delta_secs, 60)
    return f"{int(m)}m {s:05.2f}s"  # e.g., 0m 09.37s, 2m 03.04s

def _parse_alt_mode(s: str):
    s = str(s).strip().lower()
    if s == "auto":
        return "auto"
    if s == "none":
        return "none"
    try:
        return int(s)
    except Exception:
        return "auto"

def _normalize_cv_repeats(repeats: int) -> int:
    try:
        reps = int(repeats)
    except Exception:
        reps = 1
    return max(1, reps)


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
# GLOBAL CONFIGURATION
# ==============================================================================

ALT_SELECTOR_CFG = {}        # Configuration for ALT target feature selection
ALT_REGRESSOR_NAME = "bayes" # Regressor type for ALT target model
TARGET_SELECTOR_CFG = {}     # Configuration for main target feature selection


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Main entry point for the LMSYS prediction pipeline.

    This function orchestrates the complete prediction workflow:

    1. **Argument Parsing**: Configure imputation, feature selection, parallelism
    2. **Data Loading**: Read benchmark CSV, identify numeric columns
    3. **Low-Variance Filtering**: Drop constant/near-constant columns
    4. **Imputation**: Run SpecializedColumnImputer (cached if available)
    5. **Feature Selection**: Tree-based ranking with 1-SE rule
    6. **ALT Target Handling**: OOF stacking for alternative target imputation
    7. **Model Comparison**: CV evaluation of BayesianRidge
    8. **Prediction**: Generate final predictions with uncertainty intervals
    9. **Dependency Graph**: Build column dependency graph from imputer

    Outputs (written to timestamped output directory):
        - imputed_full.csv: Complete imputed benchmark matrix
        - predictions_*.csv: Per-model predictions with intervals
        - model_comparison.csv: CV results for all models
        - feature_ranking_*.csv: Feature importance rankings
        - run_config.json: Configuration used for this run
        - column_dependency_*.csv/json: Column dependency information
        - imputation_quality_*.csv: Quality metrics (if not cached)

    Command Line Args:
        See argparse configuration for full list. Key options:
        --csv_path: Input benchmark CSV
        --output_root: Base output directory
        --passes: Imputation iterations
        --alpha: Prediction interval confidence level
        --feature_selector: none/lgbm/xgb
        --top_k_features: auto/int/none

    Example:
        $ python lmsys_predictor5.py --csv_path data.csv --passes 12 --alpha 0.95

    Note:
        Results are cached based on input file hash and imputation parameters.
        Re-running with same parameters will reuse cached imputation.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, default="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv")
    ap.add_argument("--output_root", type=str, default="analysis_output")
    ap.add_argument("--passes", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=0.9361)
    ap.add_argument("--categorical_threshold", type=int, default=0,
                    help="Max distinct numeric values to auto-treat as categorical; 0 disables auto-detection.")
    ap.add_argument("--tolerance_percentile", type=float, default=91.1553,
                    help="Percentile of training uncertainties used as initial tolerance threshold (90-100).")
    ap.add_argument("--tolerance_relaxation_factor", type=float, default=1.2704,
                    help="Multiplicative factor to relax tolerance when no cells are imputed in a pass.")
    ap.add_argument("--tolerance_multiplier", type=float, default=5.8849,
                    help="Multiplier on initial tolerance to account for missing rows having higher uncertainty.")
    # v7.2: Per-column tolerance calibration
    ap.add_argument("--calibrate_tolerances", action="store_true",
                    help="Enable per-column tolerance calibration based on masked evaluation.")
    ap.add_argument("--calibration_target_rmse_ratio", type=float, default=0.6266,
                    help="Target RMSE/SD ratio for calibration. Lower = stricter.")
    ap.add_argument("--calibration_n_rounds", type=int, default=3,
                    help="Monte Carlo rounds for tolerance calibration.")
    ap.add_argument("--calibration_holdout_frac", type=float, default=0.2,
                    help="Fraction of known values to hold out for calibration.")
    ap.add_argument("--recalibrate_every_n_passes", type=int, default=5,
                    help="Recalibrate tolerances every N passes. 0 = only at start.")
    ap.add_argument("--no_feature_selector", dest="use_feature_selector", action="store_false",
                    help="Disable per-column feature selection before imputation.")
    ap.set_defaults(use_feature_selector=True)
    ap.add_argument("--selector_tau", type=float, default=0.9012,
                    help="Maximum allowed |corr| with already kept predictors during selection.")
    ap.add_argument("--selector_k_max", type=int, default=37,
                    help="Upper bound on selected predictors per column.")
    ap.add_argument("--gp_selector_k_max", type=int, default=28,
                    help="Max features for GP models (mRMR selection). Default 28.")
    ap.add_argument("--imputer_n_jobs", type=int, default=-1,
                    help="Parallel workers for the SpecializedColumnImputer (-1 uses all available cores).")
    ap.add_argument("--tier_quantiles", type=str, default="0.33,0.67",
                    help="Comma-separated quantiles that define easy/medium/hard imputation tiers.")
    ap.add_argument("--selector_n_jobs", type=int, default=-2,
                    help="Parallel workers for feature selection; 1 disables parallelism.")
    ap.add_argument("--cv_n_jobs", type=int, default=1,
                    help="Parallel workers for cross-validation folds.")
    ap.add_argument("--model_n_jobs", type=int, default=1,
                    help="Parallel workers across model specifications.")
    ap.add_argument("--cv_splits_path", type=str, default="",
                    help="Optional path to persist/reuse target-model CV splits (JSON; includes repeats if enabled).")
    ap.add_argument("--cv_repeats", type=int, default=1,
                    help="Global default for CV repeats (overridden by *_repeats flags).")
    ap.add_argument("--cv_repeats_outer", type=int, default=None,
                    help="Repeats for outer model CV + uncertainty calibration (overrides --cv_repeats).")
    ap.add_argument("--cv_repeats_inner", type=int, default=None,
                    help="Repeats for inner ALT OOF CV (overrides --cv_repeats).")
    ap.add_argument("--feature_cv_repeats", type=int, default=None,
                    help="Repeats for tree-based feature selection CV (overrides --cv_repeats).")
    ap.add_argument("--alt_cv_repeats", type=int, default=None,
                    help="Repeats for ALT imputation CV report (overrides --cv_repeats).")
    ap.add_argument("--cv_seed", type=int, default=SEED,
                    help="Base seed for CV splits (seed + repeat_index).")
    ap.add_argument("--feature_selector", choices=["none", "lgbm", "xgb"], default="lgbm")
    ap.add_argument("--top_k_features", default="auto", help="'auto' for 1-SE, integer for K, or 'none'")
    ap.add_argument("--selector_cv", type=int, default=5)
    ap.add_argument("--selector_k_grid", default="4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,all")
        # interaction options
    ap.add_argument("--poly_interactions", action="store_true",
                    help="Add degree-2 pairwise interactions before selection.")
    ap.add_argument("--poly_include_squares", action="store_true",
                    help="Also include squared terms (A^2).")
    ap.add_argument("--poly_limit", type=int, default=6,
                    help="If >0, only interact top-K variance columns; keep others as main effects.")
    ap.add_argument("--alt_feature_selector", choices=["none", "lgbm", "xgb"], default="lgbm",
                help="Tree-based selector for the ALT model (per-fold).")
    ap.add_argument("--alt_top_k_features", default="auto",
                    help="'auto' for 1-SE, integer for K, or 'none' (use all).")
    ap.add_argument("--alt_selector_cv", type=int, default=5,
                    help="CV folds for ALT selector.")
    ap.add_argument("--alt_selector_k_grid", default="4,5,6,7,10,15,20,all",
                    help="K grid for 1-SE rule when alt_top_k_features='auto'.")
    ap.add_argument("--max_workers", type=int, default=0,
                    help="Global cap on worker processes/threads (0 disables cap).")
    ap.add_argument("--alt_interaction_prescreen", type=int, default=50,
                    help="Number of candidate pairs to keep after Phase 1 residual correlation pre-screen.")
    ap.add_argument("--alt_interaction_max_pairs", type=int, default=20,
                    help="Maximum number of interaction pairs to select in greedy forward search.")
    ap.add_argument("--alt_interaction_min_improvement", type=float, default=0.05,
                    help="Minimum RMSE improvement to continue adding interaction pairs.")
    ap.add_argument("--margin", type=float, default=20.0,
                    help="Margin for 'top_by_margin_prob' column (default 20 points).")
    args = ap.parse_args()
    args.cv_repeats = max(1, int(args.cv_repeats))
    def _resolve_repeats(value: Optional[int], default: int) -> int:
        return max(1, int(value)) if value is not None else default
    args.cv_repeats_outer = _resolve_repeats(args.cv_repeats_outer, args.cv_repeats)
    args.cv_repeats_inner = _resolve_repeats(args.cv_repeats_inner, args.cv_repeats)
    args.feature_cv_repeats = _resolve_repeats(args.feature_cv_repeats, args.cv_repeats)
    args.alt_cv_repeats = _resolve_repeats(args.alt_cv_repeats, args.cv_repeats)
    args.cv_seed = int(args.cv_seed)
    _configure_parallelism(args)

    tier_quantiles: Optional[List[float]] = None
    parsed_tiers: List[float] = []
    for part in str(getattr(args, "tier_quantiles", "")).split(","):
        val = part.strip()
        if not val:
            continue
        try:
            num = float(val)
        except ValueError:
            print(f"WARNING: ignoring invalid tier_quantiles entry '{val}'", file=sys.stderr)
            continue
        if num < 0 or num > 1:
            print(f"WARNING: tier_quantiles entry {num} is outside [0, 1]; skipping", file=sys.stderr)
            continue
        parsed_tiers.append(num)
    if parsed_tiers:
        tier_quantiles = sorted(parsed_tiers)

    start = time.time()

    stamp = now_pst_timestamp()
    out_dir = os.path.join(args.output_root, f"output_{stamp}")
    ensure_dir(out_dir)

    global ALT_SELECTOR_CFG
    ALT_SELECTOR_CFG = {
        "enabled": args.alt_feature_selector != "none",
        "model_type": args.alt_feature_selector,
        "mode": _parse_alt_mode(args.alt_top_k_features),
        "k_grid": [int(x) if x.isdigit() else x for x in args.alt_selector_k_grid.split(",")],
        "cv": int(args.alt_selector_cv),
        "cv_repeats": int(args.feature_cv_repeats),
        "cv_seed": int(args.cv_seed),
    }

    global ALT_REGRESSOR_NAME
    ALT_REGRESSOR_NAME = "bayes"

    if not os.path.exists(args.csv_path):
        print(f"ERROR: CSV not found at {args.csv_path}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.csv_path)
    if ID_COL not in df.columns:
        raise ValueError(f"CSV must contain '{ID_COL}' column.")

    numeric_cols = [c for c in df.columns if c != ID_COL and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = get_feature_cols(df)
    categorical_numeric_cols = _find_numeric_categoricals(df, feature_cols, max_unique=10)
    if categorical_numeric_cols:
        print(f"Detected {len(categorical_numeric_cols)} low-cardinality integer-like column(s): {categorical_numeric_cols}")
        _coerce_discrete_columns_to_int(df, categorical_numeric_cols)

    def _is_low_variance(col: pd.Series, dominance_thresh: float = 0.7, min_std: float = 1e-8, min_minority: int = 2) -> bool:
        s = col.dropna()
        if len(s) <= 1:
            return True
        if s.std(ddof=0) < min_std:
            return True
        vc = s.value_counts(normalize=True)
        if vc.empty:
            return True
        # Drop if one category dominates or there are too few non-mode values
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

    # Diagnostics: catch degenerate columns (NaN-only or zero std) that can trigger divide-by-zero warnings
    arr = df[feature_cols].to_numpy(dtype=float, na_value=np.nan)
    stds = np.nanstd(arr, axis=0)
    nan_only_mask = np.all(np.isnan(arr), axis=0)
    zero_or_nan_std_mask = (~np.isfinite(stds)) | (stds <= 1e-12)
    degenerate_cols = [feature_cols[i] for i, bad in enumerate(nan_only_mask | zero_or_nan_std_mask) if bad]
    if degenerate_cols:
        print(f"Warning: degenerate columns before imputation (NaN-only or zero std): {degenerate_cols}")

    # Diagnostics: flag any columns that are still degenerate (zero std or too few non-null)
    stds = df[feature_cols].std(ddof=0, skipna=True)
    low_std_cols = stds[stds <= 1e-12].index.tolist()
    nn = df[feature_cols].count()
    few_nonnull_cols = nn[nn < 3].index.tolist()
    if low_std_cols or few_nonnull_cols:
        msg_parts = []
        if low_std_cols:
            msg_parts.append(f"near-zero std: {low_std_cols}")
        if few_nonnull_cols:
            msg_parts.append(f"few non-null (<3): {few_nonnull_cols}")
        print("Warning: potential degenerate columns before imputation -> " + "; ".join(msg_parts))

    missing_count_by_col = df[feature_cols].isna().sum().to_dict()

    # Compute completeness-based sample weights for sparse model handling
    row_missing_ratio = df[feature_cols].isna().mean(axis=1).values
    completeness = 1.0 - row_missing_ratio
    sample_weights_all = completeness ** COMPLETENESS_WEIGHT_POWER  # quadratic weighting
    dense_mask_all = row_missing_ratio <= DENSE_THRESHOLD
    n_dense = int(dense_mask_all.sum())
    n_sparse = int((~dense_mask_all).sum())
    print(f"Completeness weighting: {n_dense} dense (≤{DENSE_THRESHOLD:.0%} missing), "
          f"{n_sparse} sparse, power={COMPLETENESS_WEIGHT_POWER}")

    if TARGET not in numeric_cols:
        raise ValueError(f"CSV must contain numeric target column '{TARGET}'. Found numeric: {numeric_cols}")

    y_orig = df[TARGET].copy()
    y_missing_mask = y_orig.isna().values

    # ---- Track imputer metadata regardless of cache reuse ----
    imputer: Optional[SpecializedColumnImputer] = None
    imputer_predictors_map: Dict[str, List[str]] = {}
    imputer_important_predictors: Dict[str, set] = {}  # Filtered by 1% variance contribution

    # ---- NEW: cache key + cache dir ----
    cache_dir = os.path.join(args.output_root, "_cache")
    ensure_dir(cache_dir)
    csv_hash = _sha256_file(args.csv_path)
    tier_key = "none" if tier_quantiles is None else "_".join(f"{q:.3f}" for q in tier_quantiles)
    imp_key = (
        f"imputed_full_{csv_hash}_passes{args.passes}_alpha{args.alpha:.6f}_"
        f"sel{int(args.use_feature_selector)}_st{args.selector_tau:.3f}_"
        f"skmax{args.selector_k_max}_imnj{PARALLELISM_CFG['imputer_n_jobs']}_"
        f"tolp{args.tolerance_percentile:.1f}_tolr{args.tolerance_relaxation_factor:.2f}_"
        f"tolm{args.tolerance_multiplier:.2f}_tier{tier_key}_"
        f"catthr{args.categorical_threshold}_catovr{len(categorical_numeric_cols)}_skt2.0.csv"
    )
    cache_csv = os.path.join(cache_dir, imp_key)
    cache_meta = os.path.join(cache_dir, imp_key + ".meta.json")

    # ---- NEW: reuse from cache if available ----
    imputed_path = os.path.join(out_dir, "imputed_full.csv")
    if os.path.exists(cache_csv):
        imputed_df = pd.read_csv(cache_csv)
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
                # Load important predictors (filtered by variance contribution)
                raw_important = cache_payload.get("important_predictors", {})
                if isinstance(raw_important, dict):
                    imputer_important_predictors = {
                        str(col): set(deps) if isinstance(deps, list) else set()
                        for col, deps in raw_important.items()
                    }
            except Exception as exc:
                print(f"WARNING: failed to load imputer metadata cache ({exc}).", file=sys.stderr)
        # Restore SVD row factors and trajectory features from cache
        class _CachedImputer:
            pass
        svd_cache = cache_csv + ".svd_factors.csv"
        traj_cache = cache_csv + ".trajectory.csv"
        if os.path.exists(svd_cache) or os.path.exists(traj_cache):
            imputer = _CachedImputer()
            imputer.svd_row_factors_ = pd.read_csv(svd_cache, index_col=0) if os.path.exists(svd_cache) else None
            imputer.trajectory_features_ = pd.read_csv(traj_cache, index_col=0) if os.path.exists(traj_cache) else None
        already_done = True
        load_end = time.time()
        print(f"load data: {mmss(load_end - start)}")
    else:
        load_end = time.time()
        print(f"load data: {mmss(load_end - start)}")
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
            # v7.2: Per-column tolerance calibration
            calibrate_tolerances=args.calibrate_tolerances,
            calibration_target_rmse_ratio=args.calibration_target_rmse_ratio,
            calibration_n_rounds=args.calibration_n_rounds,
            calibration_holdout_frac=args.calibration_holdout_frac,
            recalibrate_every_n_passes=args.recalibrate_every_n_passes,
        )
        imputed_df.to_csv(cache_csv, index=False)
        imputer_predictors_map = {
            col: sorted(set(deps))
            for col, deps in getattr(imputer, "predictors_map_", {}).items()
        }
        # Compute important predictors (filtered by 1% variance contribution)
        imputer_important_predictors = compute_imputer_variance_contributions(
            imputer, imputed_df, threshold=0.01
        )
        # Save both predictors_map and important_predictors to cache
        with open(cache_meta, "w", encoding="utf-8") as fh:
            json.dump({
                "predictors_map": imputer_predictors_map,
                "important_predictors": {col: sorted(deps) for col, deps in imputer_important_predictors.items()},
            }, fh, indent=2)
        # Cache SVD row factors and trajectory features
        if hasattr(imputer, 'svd_row_factors_') and imputer.svd_row_factors_ is not None:
            imputer.svd_row_factors_.to_csv(cache_csv + ".svd_factors.csv", index=True)
        if hasattr(imputer, 'trajectory_features_') and imputer.trajectory_features_ is not None:
            imputer.trajectory_features_.to_csv(cache_csv + ".trajectory.csv", index=True)
        already_done = False

    # Always write the copy for this run
    imputed_df.to_csv(imputed_path, index=False)
    id_series = imputed_df[[ID_COL]] if ID_COL in imputed_df.columns else None
    impute_end = time.time()
    print(f"impute:    {mmss(impute_end - load_end)}")

    run_config = {
        "csv_path": args.csv_path,
        "passes": int(args.passes),
        "alpha": float(args.alpha),
        "use_feature_selector": bool(args.use_feature_selector),
        "selector_tau": float(args.selector_tau),
        "selector_k_max": int(args.selector_k_max),
        "selector_n_jobs": int(args.selector_n_jobs),
        "imputer_n_jobs": int(args.imputer_n_jobs),
        "categorical_threshold": int(args.categorical_threshold),
        "forced_categorical_cols": list(categorical_numeric_cols),
        "tolerance_percentile": float(args.tolerance_percentile),
        "tolerance_relaxation_factor": float(args.tolerance_relaxation_factor),
        "tolerance_multiplier": float(args.tolerance_multiplier),
        "tier_quantiles": tier_quantiles or [],
        "cv_n_jobs": int(args.cv_n_jobs),
        "model_n_jobs": int(args.model_n_jobs),
        "cv_repeats": int(args.cv_repeats),
        "cv_repeats_outer": int(args.cv_repeats_outer),
        "cv_repeats_inner": int(args.cv_repeats_inner),
        "feature_cv_repeats": int(args.feature_cv_repeats),
        "alt_cv_repeats": int(args.alt_cv_repeats),
        "cv_seed": int(args.cv_seed),
        "selector_cv": int(args.selector_cv),
        "alt_selector_cv": int(args.alt_selector_cv),
        "poly_interactions": bool(args.poly_interactions),
        "poly_include_squares": bool(args.poly_include_squares),
        "poly_limit": int(args.poly_limit),
        "max_workers": int(args.max_workers),
        "cv_splits_path": args.cv_splits_path,
        "feature_selector": args.feature_selector,
        "alt_feature_selector": args.alt_feature_selector,
        "alt_regressor": "bayes",
        "used_cache": bool(already_done),
        "cache_key": imp_key,
        "cache_path": cache_csv,
        "output_dir": out_dir,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)


    # Imputation quality assessment - always output, either fresh or from cache
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

        # Save imputation importance (which predictors matter for each column's imputation)
        try:
            imp_importance = imputer.get_imputation_importance()
            if not imp_importance.empty:
                imp_importance.to_csv(os.path.join(out_dir, "imputation_importance.csv"), index=False)
                print(f"  Imputation importance: {len(imp_importance)} predictor links saved")
        except Exception as e:
            print(f"  Warning: Could not extract imputation importance: {e}")
        # Also save to cache for future runs
        for qf in quality_files:
            shutil.copy(os.path.join(out_dir, qf), os.path.join(cache_dir, imp_key + "." + qf))
    else:
        # Copy cached quality files to output directory
        for qf in quality_files:
            cached_qf = os.path.join(cache_dir, imp_key + "." + qf)
            if os.path.exists(cached_qf):
                shutil.copy(cached_qf, os.path.join(out_dir, qf))

    # ---------------- Safety check: ensure no NaNs/Infs remain after imputation ----------------
    if not already_done:
        post_nan_report = imputed_df[feature_cols].isna().sum().rename("nan_count").to_frame()
        post_nan_report.to_csv(os.path.join(out_dir, "post_imputation_nan_report.csv"))

    safe_features = imputed_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    medians = safe_features.median(axis=0, skipna=True)
    safe_features = safe_features.fillna(medians)
    still_nan_cols = safe_features.columns[safe_features.isna().any()].tolist()
    if still_nan_cols:
        safe_features[still_nan_cols] = safe_features[still_nan_cols].fillna(0.0)
    if safe_features.isna().any().any():
        safe_features = safe_features.fillna(0.0)
    
    alt_result = None
    alt_feature_matrix: Optional[pd.DataFrame] = None
    if ALT_TARGET in df.columns:
        # --- Runtime greedy search for ALT interaction terms (with caching) ---
        global ALT_POST_PCA_INTERACTIONS
        alt_known_mask = pd.to_numeric(df[ALT_TARGET], errors="coerce").notna()
        X_for_search = safe_features.loc[alt_known_mask]
        y_for_search = pd.to_numeric(df[ALT_TARGET], errors="coerce").loc[alt_known_mask]

        # Cache key: hash of feature values + ALT values + seed + n_repeats
        alt_n_repeats = 5
        search_hash_data = (
            X_for_search.values.tobytes()
            + y_for_search.values.tobytes()
            + str(SEED).encode()
            + str(alt_n_repeats).encode()
        )
        search_hash = hashlib.sha256(search_hash_data).hexdigest()[:16]
        interaction_cache_path = os.path.join(cache_dir, f"alt_interactions_{search_hash}.json")

        if os.path.exists(interaction_cache_path):
            with open(interaction_cache_path, "r", encoding="utf-8") as fh:
                cached_int = json.load(fh)
            ALT_POST_PCA_INTERACTIONS = [tuple(p) for p in cached_int["pairs"]]
            print(f"ALT interactions: loaded {len(ALT_POST_PCA_INTERACTIONS)} cached pairs from cache")
        else:
            search_t0 = time.time()
            selected_pairs, search_log_df = _greedy_select_alt_interactions(
                X_for_search,
                y_for_search,
                n_pca=ALT_PCA_N_COMPONENTS,
                n_folds=5,
                n_repeats=alt_n_repeats,
                seed=SEED,
                prescreen_top_k=args.alt_interaction_prescreen,
                max_pairs=args.alt_interaction_max_pairs,
                min_improvement=args.alt_interaction_min_improvement,
                verbose=True,
                n_jobs=PARALLELISM_CFG.get("selector_n_jobs", -1),
            )
            search_elapsed = time.time() - search_t0
            ALT_POST_PCA_INTERACTIONS = selected_pairs

            # With consensus search, per-step RMSE is not available
            baseline_rmse = float("nan")
            selection_rmse = float("nan")

            # Save to cache
            cache_payload = {
                "pairs": [list(p) for p in selected_pairs],
                "n_pairs": len(selected_pairs),
                "selection_rmse": selection_rmse,
                "baseline_rmse": baseline_rmse,
                "n_repeats": alt_n_repeats,
                "n_candidate_cols": X_for_search.shape[1],
            }
            with open(interaction_cache_path, "w", encoding="utf-8") as fh:
                json.dump(cache_payload, fh, indent=2)

            # Save diagnostics to output dir
            diag_payload = dict(cache_payload)
            diag_payload["search_time_s"] = round(search_elapsed, 1)
            with open(os.path.join(out_dir, "alt_interaction_pairs_selected.json"), "w", encoding="utf-8") as fh:
                json.dump(diag_payload, fh, indent=2)
            search_log_df.to_csv(os.path.join(out_dir, "alt_interaction_search_log.csv"), index=False)

            print(f"ALT interactions: selected {len(selected_pairs)} pairs in {search_elapsed:.1f}s")

        alt_feature_matrix = safe_features.copy()

        # A2: Add SVD row factors as extra ALT features (additive, alongside PCA)
        global _MODULE_SVD_ROW_FACTORS
        if imputer is not None and hasattr(imputer, 'svd_row_factors_') and imputer.svd_row_factors_ is not None:
            svd_factors = imputer.svd_row_factors_
            _MODULE_SVD_ROW_FACTORS = svd_factors  # expose for regime model
            # Add raw, squared, and pairwise interaction SVD factors
            svd_col_names = list(svd_factors.columns)
            for col in svd_col_names:
                alt_feature_matrix[col] = svd_factors[col].values
                alt_feature_matrix[f"{col}_sq"] = svd_factors[col].values ** 2
            # Top-k pairwise interactions (top 4 factors → 6 pairs)
            n_interact = min(4, len(svd_col_names))
            import itertools as _itertools
            for i, j in _itertools.combinations(range(n_interact), 2):
                ci, cj = svd_col_names[i], svd_col_names[j]
                alt_feature_matrix[f"{ci}x{cj}"] = svd_factors[ci].values * svd_factors[cj].values

        # X2: Imputation trajectory signatures
        if imputer is not None and hasattr(imputer, 'trajectory_features_') and imputer.trajectory_features_ is not None:
            for col in imputer.trajectory_features_.columns:
                alt_feature_matrix[col] = imputer.trajectory_features_[col].values

        print('impute alt for all')
        alt_result = impute_alt_for_all(alt_feature_matrix, df[ALT_TARGET], ALT_SELECTOR_CFG)
        # Store base features (pre-polynomial) for variance contribution calculation later
        alt_result["base_features"] = list(safe_features.columns)
        alt_result["poly_core"] = None
        alt_result["poly_limit"] = int(args.poly_limit) if args.poly_limit else 0
        alt_result["poly_include_squares"] = bool(args.poly_include_squares)
        alt_filled = alt_result["filled"]
        safe_features = safe_features.assign(**{ALT_TARGET: alt_filled.values})

        if ALT_REGRESSOR_NAME == "bayes":
            alt_model = alt_result.get("fitted_model")
            coeff_df = None
            if alt_model is not None:
                coeff_df = _summarize_bayes_model(alt_model, alt_result.get("selected_features", []),
                                                    alt_result=alt_result)
            if coeff_df is not None:
                coeff_df.to_csv(os.path.join(out_dir, "alt_bayesian_ridge_feature_importance.csv"), index=False)

        alt_pred_df = pd.DataFrame({
            ID_COL: imputed_df[ID_COL].values if ID_COL in imputed_df.columns else np.arange(len(alt_filled)),
            "alt_actual": pd.to_numeric(df[ALT_TARGET], errors="coerce").values,
            "alt_predicted": alt_result["predicted"].values,
            "alt_feature_used": alt_filled.values,
        })
        alt_pred_df.to_csv(os.path.join(out_dir, "alt_imputed_values.csv"), index=False)
        diag = {
            "selected_features": alt_result["selected_features"],
            "fallback_value": alt_result["fallback_value"],
            "n_known": alt_result["n_known"],
            "model": ALT_REGRESSOR_NAME,
            "calibration": alt_result.get("calibration_summary"),
        }
        with open(os.path.join(out_dir, "alt_imputation_summary.json"), "w") as f:
            json.dump(diag, f, indent=2)

    target_base_features = safe_features.copy()

    to_save = safe_features if id_series is None else pd.concat([id_series, safe_features], axis=1)
    to_save.to_csv(os.path.join(out_dir, "feature_matrix_used.csv"), index=False)

    mode = args.top_k_features
    if mode.lower() == "none":
        mode = "none"
    elif mode.lower() == "auto":
        mode = "auto"
    else:
        mode = int(mode)

    raw_k_grid = [x.strip() for x in args.selector_k_grid.split(",") if x.strip()]
    parsed_k_grid: List[Union[int, str]] = []
    for item in raw_k_grid:
        if item.lower() == "all":
            parsed_k_grid.append("all")
        elif item.isdigit():
            parsed_k_grid.append(int(item))
        else:
            parsed_k_grid.append(item)

    global TARGET_SELECTOR_CFG
    TARGET_SELECTOR_CFG = {
        "enabled": args.feature_selector != "none",
        "model_type": args.feature_selector,
        "mode": mode,
        "k_grid": parsed_k_grid,
        "cv": int(args.selector_cv),
        "cv_repeats": int(args.feature_cv_repeats),
        "cv_seed": int(args.cv_seed),
    }

    poly_cfg = {
        "enabled": bool(args.poly_interactions),
        "include_squares": bool(args.poly_include_squares),
        "limit": int(args.poly_limit) if args.poly_limit else 0,
    }
    print('more matrix making')
    y_all = pd.to_numeric(df[TARGET], errors="coerce").to_numpy()
    obs_mask = ~np.isnan(y_all)
    # Build matrix that ALT selector sees (exclude the ALT column itself)
    if ALT_TARGET in df.columns:
        X_no_alt_all = alt_feature_matrix if alt_feature_matrix is not None else safe_features.drop(columns=[ALT_TARGET], errors="ignore")
        # (Do NOT add ALT column here; selector uses only non-ALT columns)
        # Base features (pre-poly, no ALT) for inner greedy search —
        # matches the search space used by the global greedy selector
        X_base_no_alt = safe_features.drop(columns=[ALT_TARGET], errors="ignore")
        alt_cv_metrics = _alt_cv_report(
            X_no_alt_all,
            df[ALT_TARGET],
            ALT_SELECTOR_CFG,
            out_dir,
            repeats=args.alt_cv_repeats,
            seed=args.cv_seed,
            X_base_for_search=X_base_no_alt,
        )
        if alt_cv_metrics:
            print(f"ALT nested-CV RMSE: {alt_cv_metrics['rmse']:.2f}  (95% CI: {alt_cv_metrics['ci_lo']:.2f} – {alt_cv_metrics['ci_hi']:.2f}, RMSE/SD: {alt_cv_metrics['rmse_over_sd']:.3f})")
            # Append cv_rmse to interaction diagnostics JSON (if it exists)
            diag_json_path = os.path.join(out_dir, "alt_interaction_pairs_selected.json")
            if os.path.exists(diag_json_path):
                with open(diag_json_path, "r", encoding="utf-8") as fh:
                    diag_data = json.load(fh)
                diag_data["cv_rmse"] = alt_cv_metrics["rmse"]
                diag_data["cv_rmse_over_sd"] = alt_cv_metrics["rmse_over_sd"]
                with open(diag_json_path, "w", encoding="utf-8") as fh:
                    json.dump(diag_data, fh, indent=2)

        # Optional: global ranking snapshot for inspection (not used in training)
        if ALT_SELECTOR_CFG.get("enabled", False):
            # rank using all known ALT rows (for a human-readable report)
            known = df[ALT_TARGET].notna()
            _, alt_ranking, _ = select_features_tree(
                X_no_alt_all.loc[known],
                pd.to_numeric(df.loc[known, ALT_TARGET], errors="coerce"),
                model_type=ALT_SELECTOR_CFG["model_type"],
                mode="none",  # full ranking
                k_grid=ALT_SELECTOR_CFG["k_grid"],
                cv=ALT_SELECTOR_CFG["cv"],
                cv_repeats=ALT_SELECTOR_CFG.get("cv_repeats", 1),
                random_state=ALT_SELECTOR_CFG.get("cv_seed", SEED),
            )
            alt_ranking.to_csv(os.path.join(out_dir, "alt_feature_ranking_gain.csv"), index=False)

    train_idx = np.where(~y_missing_mask)[0]
    pred_idx = np.where(y_missing_mask)[0]

    if len(train_idx) < 3:
        raise ValueError("Not enough rows with observed lmsys_Score to train models.")

    # Note: `X_train`, `y_train`, etc. are defined right before fit_and_predict_all

    specs = build_model_specs()

    preprocess_end = time.time()
    print(f"preprocessing:    {mmss(preprocess_end - impute_end)}")
    
    X_cv = target_base_features.loc[~y_missing_mask].copy()
    weights_cv = sample_weights_all[~y_missing_mask]
    dense_cv = dense_mask_all[~y_missing_mask]
    target_cv_splits = None
    if args.cv_splits_path:
        target_cv_splits = get_or_create_splits(
            len(X_cv),
            args.selector_cv,
            args.cv_splits_path,
            repeats=args.cv_repeats_outer,
            seed=args.cv_seed,
        )
    best_spec, eval_df, oof_preds, oof_folds = choose_best_model_with_alt(
        specs,
        X_cv,
        y_all[~y_missing_mask],
        ALT_TARGET,
        n_splits=args.selector_cv,
        model_n_jobs=args.model_n_jobs,
        cv_n_jobs=args.cv_n_jobs,
        selector_cfg=TARGET_SELECTOR_CFG,
        poly_cfg=poly_cfg,
        splits=target_cv_splits,
        cv_repeats=args.cv_repeats_outer,
        seed=args.cv_seed,
        oof_repeats=args.cv_repeats_inner,
        sample_weight=weights_cv,
        dense_mask=dense_cv,
    )
    model_summary = ", ".join(
        f"{r['model']}: {r['rmse_mean']:.2f} \u00b1 {r['rmse_std']:.2f}"
        for _, r in eval_df.iterrows()
    )
    print(f"Model comparison: {model_summary} -> {best_spec.name}")

    # Apply global target feature selection for final training/prediction
    ranking_df = None
    safe_features = target_base_features.copy()
    if TARGET_SELECTOR_CFG.get("enabled", False):
        selected_cols, ranking_df, diags = select_features_tree(
            safe_features,
            y_all,
            model_type=TARGET_SELECTOR_CFG["model_type"],
            mode=TARGET_SELECTOR_CFG["mode"],
            k_grid=TARGET_SELECTOR_CFG["k_grid"],
            cv=TARGET_SELECTOR_CFG["cv"],
            cv_repeats=TARGET_SELECTOR_CFG.get("cv_repeats", 1),
            random_state=TARGET_SELECTOR_CFG.get("cv_seed", SEED),
        )
        ranking_df.to_csv(os.path.join(out_dir, "feature_ranking_gain.csv"), index=False)
        pd.Series(selected_cols, name="selected_feature").to_csv(
            os.path.join(out_dir, "selected_features.csv"), index=False
        )
        with open(os.path.join(out_dir, "feature_selection_diag.json"), "w") as f:
            json.dump(diags, f, indent=2)
        n_dropped = diags.get("collinearity", {}).get("n_dropped", 0)
        print(f"Feature selection: {diags.get('chosen_K', len(selected_cols))}/{len(target_base_features.columns)} features kept"
              f" (dropped {n_dropped} collinear), method={TARGET_SELECTOR_CFG['model_type']}")
        safe_features = safe_features[selected_cols]
    else:
        selected_cols = list(safe_features.columns)

    if poly_cfg.get("enabled", False):
        safe_features = expand_poly_interactions(
            safe_features,
            include_squares=poly_cfg["include_squares"],
            limit=poly_cfg["limit"],
        )

    X_for_pred = safe_features.copy()
    used_feature_names = list(X_for_pred.columns)

    predictions_by_model = fit_and_predict_all_with_alt(
        specs,
        X_for_pred,
        y_all,
        train_idx,
        pred_idx,
        ALT_TARGET,
        cv_n_jobs=args.cv_n_jobs,
        model_n_jobs=args.model_n_jobs,
        selector_cfg=TARGET_SELECTOR_CFG,
        poly_cfg=poly_cfg,
        cv_base_features=target_base_features,
        cv_splits=target_cv_splits,
        cv_repeats=args.cv_repeats_outer,
        cv_seed=args.cv_seed,
        oof_repeats=args.cv_repeats_inner,
        sample_weight=sample_weights_all,
    )
    eval_df.to_csv(os.path.join(out_dir, "model_eval_rmse.csv"), index=False)
    train_end = time.time()
    print(f"train:     {mmss(train_end - preprocess_end)}")

    best = predictions_by_model[best_spec.name]
    mu = best["mu"]
    std = best["std"]
    lower = best["lower"]
    upper = best["upper"]
    fitted = best["fitted"]

    # Save OOF predictions from model selection CV (moved up — needed for conformal)
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

    # --- Normalized conformal prediction intervals ---
    conformal_start = time.time()

    # Build feature_gains aligned to feature_cols (pre-imputation columns)
    if ranking_df is not None:
        gain_map = dict(zip(ranking_df["feature"], ranking_df["mean_gain"]))
        feature_gains = np.array([gain_map.get(c, 0.0) for c in feature_cols], dtype=float)
    else:
        # Uniform gains when feature selection is disabled
        feature_gains = np.ones(len(feature_cols), dtype=float)

    # Pre-imputation missing mask (aligned to feature_cols)
    pre_imputation_missing = df[feature_cols].isna().values  # shape (n_all, n_raw_features)

    # Filter to valid OOF predictions
    oof_valid_mask = ~np.isnan(oof_preds)
    oof_preds_valid = oof_preds[oof_valid_mask]
    y_train_valid = y_all[~y_missing_mask][oof_valid_mask]
    # Map valid OOF positions back to global indices
    train_idx_all = np.where(~y_missing_mask)[0]
    train_idx_valid = train_idx_all[oof_valid_mask]

    conformal = compute_normalized_conformal_intervals(
        mu=mu,
        oof_preds=oof_preds_valid,
        y_train=y_train_valid,
        X_features=X_for_pred.values,
        train_idx=train_idx_valid,
        pre_imputation_missing=pre_imputation_missing,
        feature_gains=feature_gains,
        feature_names=list(feature_cols),
        target_coverage=0.95,
        seed=SEED,
    )

    # Overwrite intervals with heteroscedastic conformal intervals
    std = conformal["std"]
    lower = conformal["lower"]
    upper = conformal["upper"]

    conformal_end = time.time()
    print(f"conformal: {mmss(conformal_end - conformal_start)}")
    print(f"  q_hat={conformal['q_hat']:.3f}  sigma_floor={conformal['sigma_floor']:.3f}  "
          f"sigma_cv={conformal['sigma_cv']:.1%}  oof_coverage={conformal['oof_coverage']:.1%}")

    # Compute probabilities with new heteroscedastic std
    max_observed = float(np.nanmax(y_all[train_idx])) # Use y_train equivalent
    num_one_prob = prob_above_threshold(mu, std, threshold=max_observed)

    # Probability of exceeding threshold by margin (only for models without actual scores)
    top_by_margin_prob = prob_above_threshold(mu, std, threshold=max_observed + args.margin)
    # Set to NaN for models that have actual scores (training rows)
    top_by_margin_prob[train_idx] = np.nan

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
    # Sort by predicted_score descending (highest on top)
    pred_df = pred_df.sort_values("predicted_score", ascending=False)
    pred_df.to_csv(os.path.join(out_dir, "predictions_best_model.csv"), index=False)

    # Save conformal diagnostics (1 row)
    diag_row = {
        "q_hat": conformal["q_hat"],
        "sigma_floor": conformal["sigma_floor"],
        "sigma_cv": conformal["sigma_cv"],
        "oof_coverage": conformal["oof_coverage"],
    }
    for k, v in conformal["scale_model_coef"].items():
        diag_row[f"scale_model_coef_{k}"] = v
    pd.DataFrame([diag_row]).to_csv(os.path.join(out_dir, "conformal_diagnostics.csv"), index=False)

    # Save per-model uncertainty features
    uf = conformal["uncertainty_features"].copy()
    uf.insert(0, ID_COL, imputed_df[ID_COL].values)
    uf["is_train"] = False
    uf.loc[train_idx, "is_train"] = True
    uf.to_csv(os.path.join(out_dir, "conformal_uncertainty_features.csv"), index=False)

    rows = []
    for mname, bundle in predictions_by_model.items():
        for i, mdl in enumerate(imputed_df[ID_COL].values):
            rows.append({
                "model": mname,
                ID_COL: mdl,
                "mu": float(bundle["mu"][i]),
                "lower": float(bundle["lower"][i]),
                "upper": float(bundle["upper"][i]),
                "std": float(bundle["std"][i]),
            })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "predictions_all_models_long.csv"), index=False)

    # This call now works because used_feature_names has the same length as the model's coefficients
    summary = summarize_model(fitted, used_feature_names)
    summary.to_csv(os.path.join(out_dir, "best_model_feature_importance.csv"), index=False)

    # Compute variance contributions for filtering dependency graph
    X_train_arr = X_for_pred.values[train_idx]
    var_contrib_df = compute_variance_contributions(fitted, X_train_arr, used_feature_names)
    var_contrib_df.to_csv(os.path.join(out_dir, "best_model_variance_contributions.csv"), index=False)

    # Filter to features with >= 1% grouped variance contribution
    importance_threshold = 0.01
    important_features = filter_features_by_importance(
        used_feature_names, var_contrib_df, threshold=importance_threshold
    )
    print(f"Dependency graph: {len(important_features)}/{len(used_feature_names)} features above {importance_threshold:.0%} variance threshold")

    # ---------------- Column dependency reporting ----------------
    dependency_graph: Dict[str, set[str]] = defaultdict(set)

    def register(source: str, targets: List[str], filter_by_importance: bool = False) -> None:
        if not source:
            return
        dep_set = dependency_graph.setdefault(source, set())
        for tgt in targets:
            if tgt and tgt != source:
                # Optionally filter by importance
                if filter_by_importance and tgt not in important_features:
                    continue
                dep_set.add(tgt)

    root_node = "__target_model__"
    # Register only important features for the target model
    register(root_node, used_feature_names, filter_by_importance=True)

    # Track filtered features for summary
    filtered_final_features = sorted(dependency_graph.get(root_node, set()))

    used_feature_set = set(used_feature_names)
    base_feature_set = set(feature_cols)
    alt_selected_features = []
    if alt_result is not None:
        raw = alt_result.get("selected_features")
        if isinstance(raw, (list, tuple)):
            alt_selected_features = [str(r) for r in raw]

    # Register feature components (interactions -> base features)
    # Also register base components directly as deps of __target_model__ so they're 1 hop away
    for feat in list(dependency_graph.get(root_node, [])):
        dependency_graph.setdefault(feat, set())
        components = _feature_components(feat)
        if components != [feat]:
            register(feat, components)
            # Also add base components directly to root so both bases are 1 hop
            for comp in components:
                dependency_graph[root_node].add(comp)

    # ALT model variance contributions and filtering
    # With the post-PCA interaction architecture, the model is a plain BayesianRidge
    # whose features are [PC1..PCn | interaction1..interactionM].  We project PCA
    # contributions back to original base features and report interactions directly.
    alt_important_features = set(alt_selected_features)  # Default: all
    if ALT_TARGET in df.columns and alt_result is not None:
        dependency_graph.setdefault(ALT_TARGET, set())
        alt_model = alt_result.get("fitted_model")
        if alt_model is not None and alt_selected_features:
            try:
                alt_known_mask = df[ALT_TARGET].notna()
                # Reconstruct combined feature matrix using stored transforms
                alt_base_features = alt_result.get("base_features", alt_selected_features)
                # Get the raw base feature data (may need poly expansion)
                missing_base = [c for c in alt_selected_features if c not in imputed_df.columns]
                if missing_base and alt_base_features:
                    X_alt_base = imputed_df[alt_base_features].loc[alt_known_mask]
                    saved_core = alt_result.get("poly_core")
                    X_alt_raw = expand_poly_interactions(
                        X_alt_base,
                        include_squares=alt_result.get("poly_include_squares", True),
                        limit=alt_result.get("poly_limit", 0),
                        preset_core=saved_core,
                    )
                else:
                    valid_base = [c for c in alt_selected_features if c in imputed_df.columns]
                    X_alt_raw = imputed_df.loc[alt_known_mask, valid_base]

                # Use compute_variance_contributions with alt_info for PCA projection
                # X is not directly used when alt_info is provided (projection uses PCA internals)
                int_names = alt_result.get("interaction_names", [])
                all_report_names = list(X_alt_raw.columns) + list(int_names)
                alt_var_contrib = compute_variance_contributions(
                    alt_model, X_alt_raw.values, list(X_alt_raw.columns),
                    alt_info=alt_result,
                )
                alt_var_contrib.to_csv(os.path.join(out_dir, "alt_model_variance_contributions.csv"), index=False)
                alt_important_features = filter_features_by_importance(
                    all_report_names, alt_var_contrib, threshold=importance_threshold
                )
                print(f"ALT dependency graph: {len(alt_important_features)}/{len(all_report_names)} features above {importance_threshold:.0%} variance threshold")
            except Exception as e:
                print(f"Warning: Could not compute ALT variance contributions: {e}")
                import traceback; traceback.print_exc()
                # Keep default: all features are important

        # Register only important ALT features (base features + interactions)
        int_names = alt_result.get("interaction_names", [])
        all_alt_features = list(alt_selected_features) + list(int_names)
        important_alt_list = [f for f in all_alt_features if f in alt_important_features]
        register(ALT_TARGET, important_alt_list)
        for feat in important_alt_list:
            parts = _feature_components(feat)
            if parts != [feat]:
                register(feat, parts)

    def _col_has_missing(col: str) -> bool:
        # Only treat columns with observed missing values as imputation targets.
        # Unknown columns default to fully observed (no missing).
        return missing_count_by_col.get(col, 0) > 0

    imputation_targets = {
        col for col in imputer_predictors_map.keys()
        if _col_has_missing(col)
    }
    # Register imputer dependencies, filtered by 1% variance contribution
    imputer_total_deps = 0
    imputer_filtered_deps = 0
    for col, preds in imputer_predictors_map.items():
        if not _col_has_missing(col):
            continue
        dependency_graph.setdefault(col, set())
        # Use importance-filtered predictors if available, else all
        important_preds = imputer_important_predictors.get(col)
        if important_preds is not None:
            filtered_preds = [p for p in (preds or []) if p in important_preds]
            imputer_total_deps += len(preds or [])
            imputer_filtered_deps += len(filtered_preds)
            register(col, filtered_preds)
        else:
            imputer_total_deps += len(preds or [])
            imputer_filtered_deps += len(preds or [])
            register(col, preds or [])
    if imputer_total_deps > 0:
        print(f"Imputer dependency graph: {imputer_filtered_deps}/{imputer_total_deps} predictor links kept (>= 1% contribution)")

    # Ensure every dependency node exists in the graph
    for deps in list(dependency_graph.values()):
        for dep in deps:
            dependency_graph.setdefault(dep, set())

    for col in base_feature_set:
        dependency_graph.setdefault(col, set())

    def _is_missingness_flag_name(col: str) -> bool:
        return str(col).endswith("__was_missing")

    _transform_suffix = re.compile(r"(.+)_([A-Za-z0-9]+)~$")

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
    transforms_used_by_base: Dict[str, set[str]] = defaultdict(set)
    for tcol, base in transform_base_map.items():
        if tcol in reachable_nodes and base in base_feature_set and not _is_missingness_flag_name(base):
            transforms_used_by_base[base].add(tcol)

    transform_used_cols = sorted([base for base, used in transforms_used_by_base.items() if base not in reachable_depth and used])
    contributing_cols = sorted([c for c in base_feature_set if c in reachable_depth])
    dead_weight_cols = sorted([c for c in base_feature_set if c not in reachable_depth and c not in transforms_used_by_base])

    dep_graph_path = os.path.join(out_dir, "column_dependency_graph.json")
    dep_summary_path = os.path.join(out_dir, "column_dependency_summary.json")
    dep_table_path = os.path.join(out_dir, "column_dependency_summary.csv")
    degrees_path = os.path.join(out_dir, "column_degrees_of_separation.csv")

    with open(dep_graph_path, "w", encoding="utf-8") as fh:
        json.dump({node: sorted(deps) for node, deps in dependency_graph.items()}, fh, indent=2)

    dep_summary_payload = {
        "important_features": filtered_final_features,  # Features with >= 1% variance contribution
        "all_model_features": used_feature_names,  # All features used by model (before filtering)
        "alt_model_inputs": alt_selected_features,
        "contributing_columns": contributing_cols,
        "transform_used_columns": transform_used_cols,
        "dead_weight_columns": dead_weight_cols,
        "total_base_columns": len(feature_cols),
        "important_count": len(filtered_final_features),
        "contributing_count": len(contributing_cols),
        "transform_used_count": len(transform_used_cols),
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

    alt_selected_set = set(alt_selected_features)
    rows = []
    all_nodes = sorted(dependency_graph.keys())
    for col in all_nodes:
        deps_sorted = sorted(dependency_graph[col])
        rows.append({
            "column": col,
            "reachable_from_target": col in reachable_depth,
            "min_hops_from_target": reachable_depth.get(col, ""),
            "direct_dependencies": ";".join(deps_sorted),
            "is_final_feature": col in used_feature_set,
            "is_alt_target_column": col == ALT_TARGET,
            "is_alt_model_input": col in alt_selected_set,
            "is_imputation_target": col in imputation_targets,
            "is_base_feature": col in base_feature_set,
            "is_transform_used_base": col in transform_used_cols,
            "used_transform_columns": ";".join(sorted(transforms_used_by_base.get(col, []))),
        })
    pd.DataFrame(rows).to_csv(dep_table_path, index=False)
    print("column dependency outputs:")
    print(f"  graph:   {dep_graph_path}")
    print(f"  summary: {dep_summary_path}")
    print(f"  table:   {dep_table_path}")
    print(f"  degrees: {degrees_path}")

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
        "target": TARGET,
        "alt_target": ALT_TARGET,
        "best_model": best_spec.name,
        "oof_rmse": round(oof_rmse, 4),
        "oof_rmse_ci_lo": round(ci_lo, 4),
        "oof_rmse_ci_hi": round(ci_hi, 4),
        "cv_repeats_outer": int(args.cv_repeats_outer),
        "cv_repeats_inner": int(args.cv_repeats_inner),
        "notes": [
            "Safety-filled any residual NaNs/Infs in features post-imputation with column medians, then 0 if still NaN.",
            "num_one_prob compares to current max observed lmsys_Score among training rows.",
            f"top_by_margin_prob = P(score > max_observed + {args.margin}), only for models without actual scores.",
            "predictions_best_model.csv is sorted by predicted_score descending (highest on top).",
        ]
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(
            "LMSYS ELO Prediction Run (v6)\n"
            "=============================\n\n"
            f"Input CSV: {os.path.abspath(args.csv_path)}\n"
            f"Target: {TARGET}\n"
            f"Alt target present: {ALT_TARGET in df.columns}\n"
            f"Rows: {df.shape[0]}  | Numeric features (incl. targets): {len(numeric_cols)}\n"
            f"Margin for top_by_margin_prob: {args.margin}\n\n"
            "Outputs:\n"
            "  - imputed_full.csv\n"
            "  - post_imputation_nan_report.csv\n"
            "  - imputation_quality_per_column.csv\n"
            "  - model_eval_rmse.csv\n"
            "  - best_model_feature_importance.csv\n"
            "  - predictions_best_model.csv (sorted by predicted_score desc)\n"
            "      Columns: model_name, predicted_score, actual_score, lower_bound,\n"
            "               upper_bound, num_one_prob, top_by_margin_prob\n"
            "      - num_one_prob: P(score > max_observed)\n"
            "      - top_by_margin_prob: P(score > max_observed + margin), NaN for known models\n"
            "  - predictions_all_models_long.csv\n"
            "  - feature_matrix_used.csv\n"
        )
    print(f"OOF RMSE:  {oof_rmse:.2f}  (95% CI: {ci_lo:.2f} \u2013 {ci_hi:.2f})")
    print(f"Done. Results saved to: {out_dir}")
    end = time.time()
    print(f"total:     {mmss(end - start)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
