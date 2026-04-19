"""Baseline comparison for the lmarena_Score target.

Re-runs the classic four-row comparison table (dummy, public-only+median,
all-bench+median, full Soothsayer pipeline) against the current champion
CSV (clean_combined_all_benches_with_sem_v4_d32.csv) using the same
KFold CV setup predict.py uses (SEED=42, 5 folds, shuffle=True), so the
numbers line up apples-to-apples with whatever predict.sh reports.

Updates vs. the older arena_predictor/baseline_comparison.py:
- Target is lmarena_Score (lmsys_Score was leakage; switched 2026-03-29).
- sem_ features are Soothsayer-derived, so they move from "public" to "custom".
- eqbench_ / eqmt_ stay "public" (external eqbench scraper, not in-house eq_).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
CSV_PATH = REPO / "benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_v4_d32.csv"
TARGET = "lmarena_Score"
LEAKAGE = {"lmsys_Score", "lmarena_Score"}
NON_FEATURE = {"model_name"}

# In-house Soothsayer features (NOT public benchmarks).
# eqbench_ / eqmt_ come from external scrapers — keep as public.
CUSTOM_PREFIXES = ("eq_", "writing_", "logic_", "style_", "tone_", "sem_")

SEED = 42
N_SPLITS = 5
N_REPEATS = 10


def oof_repeated(pipe, X, y, n_repeats=N_REPEATS, n_splits=N_SPLITS, seed=SEED):
    oof_sum = np.zeros(len(y))
    oof_count = np.zeros(len(y))
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for tr, va in kf.split(X):
            p = clone(pipe)
            p.fit(X[tr], y[tr])
            oof_sum[va] += p.predict(X[va])
            oof_count[va] += 1
    return oof_sum / oof_count


def metrics(pred, y):
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    r2 = float(np.corrcoef(pred, y)[0, 1] ** 2)
    rho = float(spearmanr(pred, y).correlation)
    return rmse, r2, rho


def is_numeric(s):
    return s.dtype.kind in "fiu"


def run():
    raw = pd.read_csv(CSV_PATH)
    has_t = raw[TARGET].notna()
    y = raw.loc[has_t, TARGET].values
    n = len(y)

    num_cols = [c for c in raw.columns if is_numeric(raw[c])]
    all_feat = [c for c in num_cols if c not in LEAKAGE and c not in NON_FEATURE]
    public_feat = [c for c in all_feat if not any(c.startswith(p) for p in CUSTOM_PREFIXES)]
    custom_feat = [c for c in all_feat if c not in public_feat]

    print(f"CSV: {CSV_PATH.name}")
    print(f"Rows with {TARGET}: {n} / {len(raw)}")
    print(f"All features:     {len(all_feat)}")
    print(f"  Public:         {len(public_feat)}")
    print(f"  Custom:         {len(custom_feat)}   {CUSTOM_PREFIXES}")
    print(f"CV: {N_REPEATS}× {N_SPLITS}-fold, SEED={SEED}")
    print()

    X_all = raw.loc[has_t, all_feat].values
    X_public = raw.loc[has_t, public_feat].values

    pipe = lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearRegression(),
    )

    results = []

    # Dummy
    pred = np.full_like(y, fill_value=y.mean(), dtype=float)
    rmse, r2, rho = metrics(pred, y)
    results.append(("Predict mean (dummy)", rmse, 0.0, float("nan")))

    pred = oof_repeated(pipe(), X_public, y)
    results.append(("Public benchmarks + median impute", *metrics(pred, y)))

    pred = oof_repeated(pipe(), X_all, y)
    results.append(("All benchmarks + median impute", *metrics(pred, y)))

    print(f"{'Method':40s} {'RMSE':>7s} {'R²':>7s} {'Spearman ρ':>11s}")
    print("-" * 70)
    for name, rmse, r2, rho in results:
        rho_s = f"{rho:11.3f}" if not np.isnan(rho) else f"{'—':>11s}"
        print(f"{name:40s} {rmse:7.2f} {r2:7.3f} {rho_s}")
    print(f"{'Full Soothsayer pipeline':40s}    see predict.sh (current: 14.24 @ 5-fold single)")


if __name__ == "__main__":
    run()
