"""Walk-forward honest-eval + m-fit for predictor calibration.

Extends arena_predictor/_walkforward_honest.py by (a) running a nested LOO inside
each WF step to produce per-step OOF residuals and y_nb_std, (b) fitting per-step
gate + t_df + q_hat + s_floor on that prefix, (c) fitting a global scalar m across
steps, and (d) emitting diagnostics (PIT, coverage, Brier, log-loss) + wf_residuals.csv
for downstream consumption by predict.py via --walkforward_calibration_path.
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


if __name__ == "__main__":
    main()
