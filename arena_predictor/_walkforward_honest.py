"""Fully-honest walk-forward CV.

At each step i (release-date order, starting at oldest 80%):
  1. Re-fit ModelBankImputer on rows [0..i] with TRUE feature set.
     Imputer uses cross-benchmark patterns only, never touches lmarena_Score.
  2. Re-fit PCA-32 on pooled embeddings for rows [0..i-1] only, transform row i.
  3. Re-build full 119-feature production matrix for rows [0..i] + sem PCA.
  4. Fit PLS-3 on rows [0..i-1], transform both train + test.
  5. predict_adaptive_knn on model i.

Prints per-step progress + final metrics (RMSE, MAE, R^2, Pearson, Spearman).
"""
from __future__ import annotations
import itertools
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
from predict import predict_adaptive_knn, run_imputation, ID_COL, TARGET, ALT_TARGET


def build_pooled_embeddings() -> pd.DataFrame:
    emb = pd.read_parquet('../embeddings/cache/response_embeddings_bge_small.parquet')
    emb_cols = [f'e{i:03d}' for i in range(384)]

    def slot_of(row):
        b, pid = row['benchmark'], str(row['prompt_id'])
        if b == 'eq':
            if pid.endswith('_t1'):
                return 'eq_t1'
            if pid.endswith('_t3'):
                return 'eq_t3'
            return None
        return b

    emb['slot'] = emb.apply(slot_of, axis=1)
    emb = emb[emb['slot'].notna()]

    pooled = emb.groupby(['model', 'slot'])[emb_cols].mean().reset_index()
    norms = np.linalg.norm(pooled[emb_cols].values, axis=1, keepdims=True)
    pooled[emb_cols] = pooled[emb_cols].values / np.clip(norms, 1e-8, None)

    slots = ['eq_t1', 'eq_t3', 'logic', 'style', 'writing']
    rows = []
    for model, g in pooled.groupby('model'):
        slot2vec = {r['slot']: r[emb_cols].values for _, r in g.iterrows()}
        vec = []
        ok = True
        for s in slots:
            if s in slot2vec:
                vec.extend(slot2vec[s])
            else:
                vec.extend([np.nan] * 384)
                ok = False
        rows.append((model, ok, vec))
    cols = ['model_name', 'all_slots_present'] + [f'p{i:04d}' for i in range(5 * 384)]
    return pd.DataFrame([(r[0], r[1]) + tuple(r[2]) for r in rows], columns=cols)


def main():
    CSV = '../benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_v4_d32.csv'
    DATES = '../benchmark_combiner/benchmarks/openbench_release_dates.csv'

    src = pd.read_csv(CSV)
    dates = pd.read_csv(DATES).rename(columns={'Model': 'model_name', 'Release_Date': 'release_date'})
    dates['release_date'] = pd.to_datetime(dates['release_date'], errors='coerce')

    pooled = build_pooled_embeddings()

    # Merge date + embeddings, keep only models with all ingredients
    src = src.merge(dates[['model_name', 'release_date']], on='model_name', how='left')
    src = src.merge(pooled, on='model_name', how='left')
    mask = (src['lmarena_Score'].notna()
            & src['release_date'].notna()
            & src['all_slots_present'].fillna(False))
    src = src[mask].sort_values('release_date').reset_index(drop=True)
    n = len(src)
    print(f'[pool] {n} models with target + date + all embedding slots', flush=True)

    # Feature columns for imputer: drop sem_f* (we re-fit PCA per step), drop lmarena/lmsys,
    # drop metadata (release_date, pooled p####, all_slots_present, model_name).
    pooled_cols = [f'p{i:04d}' for i in range(5 * 384)]
    drop_cols = (set(pooled_cols)
                 | {'model_name', 'release_date', 'all_slots_present', TARGET, ALT_TARGET})
    sem_cols_csv = [c for c in src.columns if c.startswith('sem_')]
    drop_cols |= set(sem_cols_csv)

    feature_cols = [c for c in src.columns
                    if c not in drop_cols and pd.api.types.is_numeric_dtype(src[c])]
    print(f'[setup] imputer feature cols: {len(feature_cols)} (sem_f* dropped, rebuilt per step)', flush=True)

    y = src['lmarena_Score'].values.astype(float)
    P = src[pooled_cols].values.astype(float)

    # Production imputer settings (from predict.sh + argparse defaults)
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
        imputer_type='model_bank', confidence_threshold=0.4,
        coherence_lambda=8.0, coherence_shape='exp', coherence_gate='fixed',
        iterative_coherence=False,
        predictor_selection='loo_forward',
        use_svd_predictors=False, n_expansion_passes=1, max_confident_extras=1,
    )

    n_init = int(n * 0.80)
    print(f'[start] walk-forward from oldest {n_init} -> newest {n - n_init}', flush=True)

    preds, actuals, step_times = [], [], []
    t_start = time.time()

    for i in range(n_init, n):
        t0 = time.time()
        # ----- 1. Honest imputation on rows [0..i] -----
        sub = src.iloc[:i + 1].copy()
        imp_df, imputer = run_imputation(
            sub[[ID_COL] + feature_cols],
            **imp_kwargs,
        )
        # imputer adds svd_row_factors_ and trajectory_features_ (indexed 0..i)
        svd = imputer.svd_row_factors_.reset_index(drop=True)
        traj = imputer.trajectory_features_.reset_index(drop=True)

        # ----- 2. Build feature matrix (imputed + SVD + SVD_sq + SVD_x + traj) -----
        feat = imp_df.copy()
        for c in svd.columns:
            feat[c] = svd[c].values
            feat[f'{c}_sq'] = svd[c].values ** 2
        svd_cols = list(svd.columns)
        for a, b in itertools.combinations(range(min(4, len(svd_cols))), 2):
            feat[f'{svd_cols[a]}x{svd_cols[b]}'] = svd[svd_cols[a]].values * svd[svd_cols[b]].values
        for c in traj.columns:
            feat[c] = traj[c].values

        # drop_style_tone on static feature set
        static_cols = [c for c in feat.columns
                       if c != ID_COL
                       and not c.startswith('style_')
                       and not c.startswith('tone_')]
        X_static = feat[static_cols].values.astype(float)
        med = np.nanmedian(X_static, axis=0)
        inds = np.where(np.isnan(X_static))
        X_static[inds] = np.take(med, inds[1])
        X_static = np.nan_to_num(X_static)

        # ----- 3. PCA-32 on older-only pooled embeddings -----
        n_comp = min(32, i - 1, P.shape[1])
        pca = PCA(n_components=n_comp, random_state=42).fit(P[:i])
        sem_tr = pca.transform(P[:i])
        sem_te = pca.transform(P[i:i + 1])

        Xtr_raw = np.hstack([X_static[:i], sem_tr])
        Xte_raw = np.hstack([X_static[i:i + 1], sem_te])

        # ----- 4. Scale + PLS-3 + KNN -----
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr_raw)
        Xte = sc.transform(Xte_raw)
        pls = PLSRegression(n_components=min(3, Xtr.shape[1], i - 1)).fit(Xtr, y[:i])
        Xtr = np.hstack([Xtr, pls.transform(Xtr)])
        Xte = np.hstack([Xte, pls.transform(Xte)])

        p, _, _ = predict_adaptive_knn(Xtr, y[:i], Xte,
                                       max_k=min(80, i), min_k=min(20, i))
        preds.append(p)
        actuals.append(y[i])
        dt = time.time() - t0
        step_times.append(dt)
        eta = np.mean(step_times) * (n - i - 1)
        print(f'[step {i - n_init + 1}/{n - n_init}] model={src.iloc[i]["model_name"]!r} '
              f'actual={y[i]:.0f} pred={p:.1f} err={p - y[i]:+.1f} '
              f'({dt:.1f}s, eta {eta/60:.1f}m)', flush=True)

    a = np.array(actuals)
    p = np.array(preds)

    elapsed = time.time() - t_start
    print(f'\n=== DONE in {elapsed/60:.1f}m ===', flush=True)
    print(f'\n== honest walk-forward, oldest 80% -> newest 20%, n_test={len(a)} ==')
    print(f'  RMSE         : {np.sqrt(np.mean((a - p)**2)):.3f}')
    print(f'  MAE          : {np.mean(np.abs(a - p)):.3f}')
    print(f'  R^2          : {r2_score(a, p):.4f}')
    print(f'  Pearson r    : {np.corrcoef(a, p)[0, 1]:.4f}')
    print(f'  Spearman rho : {spearmanr(a, p)[0]:.4f}')

    # Save per-step results for later inspection
    out = pd.DataFrame({
        'model_name': src.iloc[range(len(y) - len(a), len(y))]['model_name'].values,
        'release_date': src.iloc[range(len(y) - len(a), len(y))]['release_date'].values,
        'actual': a, 'predicted': p, 'err': p - a,
    })
    out.to_csv('/tmp/walkforward_honest_80.csv', index=False)
    print(f'per-step results -> /tmp/walkforward_honest_80.csv')


if __name__ == '__main__':
    main()
