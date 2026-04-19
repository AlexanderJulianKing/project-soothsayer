"""Chronological-holdout (honest walk-forward) for the four baseline table rows.

For each step i in [n_init..n): train on rows [0..i-1] (release-date ordered),
predict row i. This matches _walkforward_honest.py's protocol but swaps the
predictor for the three table baselines (dummy mean, public+median+Ridge,
all+median+Ridge). Full Soothsayer pipeline numbers come from
_walkforward_honest.py.

Prints a side-by-side table ready to drop into the explainer comparison.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _walkforward_honest import build_pooled_embeddings  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
CSV_PATH = REPO / "benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_v4_d32.csv"
DATES_PATH = REPO / "benchmark_combiner/benchmarks/openbench_release_dates.csv"
TARGET = "lmarena_Score"
LEAKAGE = {"lmsys_Score", "lmarena_Score"}
NON_FEATURE = {"model_name", "release_date", "all_slots_present"}
CUSTOM_PREFIXES = ("eq_", "writing_", "logic_", "style_", "tone_", "sem_")


def is_numeric(s):
    return s.dtype.kind in "fiu"


def metrics(pred, y):
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    r2 = float(np.corrcoef(pred, y)[0, 1] ** 2) if np.std(pred) > 0 else 0.0
    rho = spearmanr(pred, y).correlation if np.std(pred) > 0 else float("nan")
    return rmse, r2, float(rho) if rho == rho else float("nan")


def walk_forward(X, y, n_init, pipe_factory):
    preds = []
    for i in range(n_init, len(y)):
        p = pipe_factory()
        p.fit(X[:i], y[:i])
        preds.append(float(p.predict(X[i:i + 1])[0]))
    return np.array(preds)


def run():
    raw = pd.read_csv(CSV_PATH)
    dates = pd.read_csv(DATES_PATH).rename(columns={"Model": "model_name", "Release_Date": "release_date"})
    dates["release_date"] = pd.to_datetime(dates["release_date"], errors="coerce")

    pooled = build_pooled_embeddings()
    df = raw.merge(dates[["model_name", "release_date"]], on="model_name", how="left")
    df = df.merge(pooled[["model_name", "all_slots_present"]], on="model_name", how="left")
    mask = (df[TARGET].notna()
            & df["release_date"].notna()
            & df["all_slots_present"].fillna(False))
    df = df[mask].sort_values("release_date").reset_index(drop=True)
    n = len(df)
    n_init = int(n * 0.80)
    n_test = n - n_init
    print(f"n total: {n}, train (oldest 80%): {n_init}, test (newest 20%): {n_test}")

    num_cols = [c for c in df.columns if is_numeric(df[c])]
    all_feat = [c for c in num_cols if c not in LEAKAGE and c not in NON_FEATURE]
    public_feat = [c for c in all_feat if not any(c.startswith(p) for p in CUSTOM_PREFIXES)]
    print(f"all features: {len(all_feat)}, public: {len(public_feat)}\n")

    y = df[TARGET].values.astype(float)
    X_all = df[all_feat].values.astype(float)
    X_public = df[public_feat].values.astype(float)
    y_test = y[n_init:]

    pipe = lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearRegression(),
    )

    results = []

    # 1. Dummy — predict train mean
    preds_dummy = []
    for i in range(n_init, n):
        preds_dummy.append(y[:i].mean())
    results.append(("Predict mean (dummy)", *metrics(np.array(preds_dummy), y_test)))

    # 2. Public + median impute + linear regression
    preds_public = walk_forward(X_public, y, n_init, pipe)
    results.append(("Public benchmarks + median impute", *metrics(preds_public, y_test)))

    # 3. All + median impute + linear regression
    preds_all = walk_forward(X_all, y, n_init, pipe)
    results.append(("All benchmarks + median impute", *metrics(preds_all, y_test)))

    print(f"{'Method':40s} {'RMSE':>7s} {'R²':>7s} {'ρ':>7s}")
    print("-" * 65)
    for name, rmse, r2, rho in results:
        rho_s = f"{rho:7.3f}" if not np.isnan(rho) else f"{'—':>7s}"
        print(f"{name:40s} {rmse:7.2f} {r2:7.3f} {rho_s}")
    print(f"{'Full Soothsayer pipeline':40s} {'14.69':>7s} {'0.900':>7s} {'0.940':>7s}  (from _walkforward_honest.py)")


if __name__ == "__main__":
    run()
