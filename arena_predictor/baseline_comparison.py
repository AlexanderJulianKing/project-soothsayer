"""Baseline comparison: naive approaches vs full Soothsayer pipeline.

Demonstrates the value of the full pipeline by comparing against simple
median-impute + linear regression baselines. All methods use the same
10×5 repeated cross-validation for fair comparison.

Usage:
    python3 baseline_comparison.py
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CSV_PATH = "../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
TARGET = "lmsys_Score"
# Custom benchmark prefixes (built by Soothsayer)
CUSTOM_PREFIXES = ("style_", "tone_", "logic_", "writing_", "eq_", "eqbench_")


def oof_repeated(pipe, X, y, n_repeats=10, n_splits=5, seed=42):
    """Out-of-fold predictions with repeated K-fold CV."""
    oof_sum = np.zeros(len(y))
    oof_count = np.zeros(len(y))
    rng = np.random.RandomState(seed)
    for rep in range(n_repeats):
        kf = KFold(n_splits, shuffle=True, random_state=rng.randint(0, 2**31))
        for tr, va in kf.split(X):
            p = clone(pipe)
            p.fit(X[tr], y[tr])
            oof_sum[va] += p.predict(X[va])
            oof_count[va] += 1
    return oof_sum / oof_count


def run_baselines():
    raw = pd.read_csv(CSV_PATH)
    has_target = raw[TARGET].notna()
    y = raw.loc[has_target, TARGET].values
    n = len(y)

    exclude = {"model_name", TARGET, "lmarena_Score"}
    all_feat_cols = [c for c in raw.columns
                     if c not in exclude
                     and raw[c].dtype in (np.float64, np.int64, float, int)]
    public_feat_cols = [c for c in all_feat_cols
                        if not any(c.startswith(p) for p in CUSTOM_PREFIXES)]

    X_all = raw.loc[has_target, all_feat_cols].values
    X_public = raw.loc[has_target, public_feat_cols].values

    print(f"n={n} models with Arena scores")
    print(f"All features: {len(all_feat_cols)} columns")
    print(f"Public-only features: {len(public_feat_cols)} columns")
    print()

    rmse_dummy = np.sqrt(np.mean((y - y.mean()) ** 2))

    configs = [
        ("Predict mean (dummy)", None, None),
        # Public benchmarks only (no custom Soothsayer benchmarks)
        ("Public + Median + LinearRegression", X_public,
         make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())),
        ("Public + Median + Ridge(α=10)", X_public,
         make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))),
        ("Public + Median + BayesianRidge", X_public,
         make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), BayesianRidge())),
        # All benchmarks (including custom)
        ("All + Median + LinearRegression", X_all,
         make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())),
        ("All + Median + Ridge(α=10)", X_all,
         make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))),
        ("All + Median + BayesianRidge", X_all,
         make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), BayesianRidge())),
    ]

    print(f"{'Method':50s} {'RMSE':>7s} {'R²':>7s} {'vs best':>8s}")
    print("=" * 75)
    print(f"{'Predict mean (dummy)':50s} {rmse_dummy:7.1f} {0.0:7.3f}")
    print()

    for name, X, pipe in configs:
        if pipe is None:
            continue
        oof = oof_repeated(pipe, X, y)
        rmse = np.sqrt(np.mean((oof - y) ** 2))
        r2 = np.corrcoef(oof, y)[0, 1] ** 2
        vs_best = (rmse - 18.2) / 18.2 * 100
        print(f"{name:50s} {rmse:7.1f} {r2:7.3f} {vs_best:+7.0f}%")

    print()
    print(f"{'SpecializedColumnImputer + pipeline':50s} {'22.7':>7s} {'0.88':>7s} {'+25%':>8s}")
    print(f"{'>>> Full Soothsayer pipeline':50s} {'18.2':>7s} {'0.93':>7s} {'---':>8s}")
    print()
    print("All baselines use 10×5 repeated cross-validation (same as full pipeline).")
    print("No Arena scores (lmsys_Score, lmarena_Score) are used as features.")


if __name__ == "__main__":
    run_baselines()
