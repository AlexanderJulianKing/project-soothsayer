"""Walk-forward baseline comparison for the lmarena_Score target.

The walk-forward analogue of baseline_comparison_lmarena.py. It runs the same
four-row ladder (dummy, public-only+median, all-bench+median, full pipeline) but
under the HONEST TEMPORAL protocol used by _walkforward_honest.py instead of
10×5-fold OOF:

  - Use the exact same pool/split as the full-pipeline walk-forward: models with
    a target + a release date + all 5 embedding slots, sorted by release date,
    oldest 80% as the initial train, each of the newest 20% predicted one at a
    time (train = all strictly-older models, re-fit every step).
  - Dummy = mean of the training targets at that step (a RISING mean, since newer
    models trend stronger — so it is NOT a constant predictor here).
  - Public / All = SimpleImputer(median) -> StandardScaler -> RidgeCV, fit on the
    training rows only at each step, then predict the held-out newer model.

Why RidgeCV, not OLS: with ~122 features and ~91 training rows the design is p>n,
where unregularized OLS is rank-deficient and wildly unstable in the temporal
setting (it lands at 52-73 RMSE and swings with tiny data changes). Ridge gives
the naive baseline a fair, stable shot at high dimension. R² is sklearn r2_score
(not Pearson²), matching the full-pipeline row.

Caveat: the "All benchmarks" row uses the precomputed sem_f* columns, whose PCA
basis was fit on all models, so it carries mild forward-leakage that slightly
flatters the baseline. The full pipeline (_walkforward_honest.py) avoids this by
re-fitting PCA every step; its 14.16 RMSE is reproduced there, not here.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _walkforward_honest import build_pooled_embeddings  # noqa: E402

REPO = SCRIPT_DIR.parent
CSV_PATH = REPO / "benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_v4_d32.csv"
DATES_PATH = REPO / "benchmark_combiner/benchmarks/openbench_release_dates.csv"
TARGET = "lmarena_Score"
LEAKAGE = {"lmsys_Score", "lmarena_Score"}
CUSTOM_PREFIXES = ("eq_", "writing_", "logic_", "style_", "tone_", "sem_")
RIDGE_ALPHAS = np.logspace(-1, 4, 30)
# Full-pipeline walk-forward result (from _walkforward_honest.py, shipped config).
FULL_PIPELINE = dict(rmse=14.16, r2=0.911, rho=0.954)


def build_pool():
    raw = pd.read_csv(CSV_PATH)
    dates = pd.read_csv(DATES_PATH).rename(
        columns={"Model": "model_name", "Release_Date": "release_date"})
    dates["release_date"] = pd.to_datetime(dates["release_date"], errors="coerce")
    pooled = build_pooled_embeddings()
    src = (raw.merge(dates[["model_name", "release_date"]], on="model_name", how="left")
              .merge(pooled, on="model_name", how="left"))
    mask = (src[TARGET].notna() & src["release_date"].notna()
            & src["all_slots_present"].fillna(False))
    return src[mask].sort_values("release_date").reset_index(drop=True)


def feature_cols(src):
    pcols = {f"p{i:04d}" for i in range(5 * 384)}
    drop = pcols | {"model_name", "release_date", "all_slots_present"} | LEAKAGE
    allf = [c for c in src.columns if c not in drop and src[c].dtype.kind in "fiu"]
    pub = [c for c in allf if not any(c.startswith(p) for p in CUSTOM_PREFIXES)]
    return pub, allf


def walk_forward(src, y, n_init, feats=None, dummy=False):
    n = len(src)
    preds = []
    for i in range(n_init, n):
        ytr = y[:i]
        if dummy:
            preds.append(ytr.mean())
            continue
        Xtr = src.iloc[:i][feats].values
        Xte = src.iloc[i:i + 1][feats].values
        imp = SimpleImputer(strategy="median").fit(Xtr)
        sc = StandardScaler().fit(imp.transform(Xtr))
        model = RidgeCV(alphas=RIDGE_ALPHAS).fit(sc.transform(imp.transform(Xtr)), ytr)
        preds.append(float(model.predict(sc.transform(imp.transform(Xte)))[0]))
    return np.array(preds)


def metrics(pred, y_te):
    rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
    r2 = float(r2_score(y_te, pred))
    rho = spearmanr(pred, y_te).correlation
    return rmse, r2, rho


def run():
    warnings.filterwarnings("ignore")
    src = build_pool()
    y = src[TARGET].values.astype(float)
    n = len(src)
    n_init = int(n * 0.80)
    y_te = y[n_init:]
    pub, allf = feature_cols(src)

    print(f"CSV: {CSV_PATH.name}")
    print(f"Pool: {n} models with target + release date + all 5 embedding slots")
    print(f"Split: oldest {n_init} train -> newest {n - n_init} predicted one at a time")
    print(f"Features: public={len(pub)}  all={len(allf)}")
    print()

    rows = [
        ("Predict mean (dummy)", walk_forward(src, y, n_init, dummy=True)),
        ("Public benchmarks + median impute", walk_forward(src, y, n_init, pub)),
        ("All benchmarks + median impute", walk_forward(src, y, n_init, allf)),
    ]
    print(f"{'Method (walk-forward, n_test=' + str(n - n_init) + ')':40s} "
          f"{'RMSE':>7s} {'R²':>7s} {'Spearman ρ':>11s}")
    print("-" * 70)
    for name, pred in rows:
        rmse, r2, rho = metrics(pred, y_te)
        print(f"{name:40s} {rmse:7.2f} {r2:7.3f} {rho:11.3f}")
    fp = FULL_PIPELINE
    print(f"{'Full Soothsayer pipeline':40s} "
          f"{fp['rmse']:7.2f} {fp['r2']:7.3f} {fp['rho']:11.3f}   (from _walkforward_honest.py)")


if __name__ == "__main__":
    run()
