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
import os
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] output dir: {OUT_DIR}", flush=True)

    CSV = Path(__file__).parent.parent / "benchmark_combiner" / "benchmarks" / "clean_combined_all_benches_with_sem_v4_d32.csv"
    DATES = Path(__file__).parent.parent / "benchmark_combiner" / "benchmarks" / "openbench_release_dates.csv"

    src = pd.read_csv(CSV)
    dates = pd.read_csv(DATES).rename(columns={"Model": "model_name", "Release_Date": "release_date"})
    dates["release_date"] = pd.to_datetime(dates["release_date"], errors="coerce")

    # build_pooled_embeddings() uses relative paths anchored at arena_predictor/
    _orig_dir = os.getcwd()
    os.chdir(Path(__file__).parent)
    try:
        pooled = build_pooled_embeddings()
    finally:
        os.chdir(_orig_dir)

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


if __name__ == "__main__":
    main()
