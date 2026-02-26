
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

def _spearman_on_pairs(y: np.ndarray, x: np.ndarray, mask: np.ndarray) -> float:
    yv, xv = y[mask], x[mask]
    if yv.size < 3:
        return float("nan")
    c = spearmanr(yv, xv).correlation
    return float(abs(c)) if c is not None and np.isfinite(c) else float("nan")

def _knn_mi_on_pairs(y: np.ndarray, x: np.ndarray, mask: np.ndarray, seed: int = 0) -> float:
    yv, xv = y[mask], x[mask]
    if yv.size < 10:
        return 0.0
    mi = mutual_info_regression(xv.reshape(-1, 1), yv, random_state=seed)
    return float(mi[0])

def _relevance_scores_df(X: pd.DataFrame, y_col: str, min_pairs: int, use_mi: bool, seed: int) -> List[str]:
    y = X[y_col].to_numpy(dtype=float)
    mask_y = ~np.isnan(y)
    scores = []
    for x_col in X.columns:
        if x_col == y_col:
            continue
        x = X[x_col].to_numpy(dtype=float)
        m = mask_y & (~np.isnan(x))
        if int(m.sum()) < int(min_pairs):
            continue
        s = _spearman_on_pairs(y, x, m)
        if not np.isfinite(s):
            s = 0.0
        if use_mi:
            s = max(s, _knn_mi_on_pairs(y, x, m, seed=seed))
        scores.append((float(s), x_col))
    scores.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scores]

def _decorrelate_df(X: pd.DataFrame, ranked_cols: List[str], tau: float, y_col: str, min_pairs: int) -> List[str]:
    keep: List[str] = []
    for c in ranked_cols:
        ok = True
        x = X[c].to_numpy(dtype=float)
        for k in keep:
            z = X[k].to_numpy(dtype=float)
            m = (~np.isnan(x)) & (~np.isnan(z))
            if int(m.sum()) < int(min_pairs):
                continue
            corr = np.corrcoef(x[m], z[m])[0, 1]
            if np.isfinite(corr) and abs(float(corr)) > float(tau):
                ok = False; break
        if ok:
            keep.append(c)
    return keep

def _cv_ridge_mse(X: np.ndarray, y: np.ndarray, cv_folds: int, seed: int) -> float:
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    mses = []
    for tr, te in kf.split(X):
        model = Ridge(alpha=1.0, fit_intercept=False, random_state=seed)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        mses.append(float(np.mean((y[te]-pred)**2)))
    return float(np.mean(mses))

def _low_pairs_fallback_df(X: pd.DataFrame, y_col: str) -> List[str]:
    y = X[y_col].to_numpy(dtype=float); mask_y = ~np.isnan(y)
    scores = []
    for c in X.columns:
        if c==y_col: continue
        x = X[c].to_numpy(dtype=float); m = mask_y & (~np.isnan(x))
        if int(m.sum())<3: continue
        s = _spearman_on_pairs(y,x,m)
        if np.isfinite(s): scores.append((float(s), c))
    scores.sort(key=lambda t: t[0], reverse=True)
    return [c for _,c in scores[:5]]

def select_predictors_for_target(
    X_df: pd.DataFrame,
    y_col: str,
    min_pairs: int = 200,
    tau: float = 0.90,
    k_seed: int = 12,
    k_max: int = 30,
    cv_folds: int = 5,
    delta_improve: float = 0.01,
    use_mi: bool = True,
    seed: int = 42
) -> Tuple[List[str], Tuple[np.ndarray, np.ndarray]]:
    ranked = _relevance_scores_df(X_df, y_col, min_pairs=min_pairs, use_mi=use_mi, seed=seed)
    if len(ranked) == 0:
        cols = _low_pairs_fallback_df(X_df, y_col)
        m = (~X_df[y_col].isna()).to_numpy()
        if cols:
            Xf = np.column_stack([X_df[c].to_numpy(dtype=float)[m] for c in cols])
            mu, sd = Xf.mean(0), Xf.std(0)+1e-8
        else:
            mu, sd = np.zeros(0), np.ones(0)
        return cols, (mu, sd)
    seed_set = ranked[:min(k_seed, len(ranked))]
    decor = _decorrelate_df(X_df, seed_set, tau=tau, y_col=y_col, min_pairs=min_pairs)
    candidates = [c for c in ranked if c not in decor]
    # residual rescue
    y = X_df[y_col].to_numpy(dtype=float)
    def complete_mask(cols): 
        m = ~np.isnan(y)
        for c in cols: m &= ~X_df[c].isna().to_numpy()
        return m
    cols = list(decor)
    while True:
        base_m = complete_mask(cols)
        if int(base_m.sum()) < max(int(min_pairs), int(cv_folds*5)): break
        Yb = y[base_m]
        Xb = np.column_stack([X_df[c].to_numpy(dtype=float)[base_m] for c in cols])
        mu = Xb.mean(0); sd = Xb.std(0)+1e-8
        base = _cv_ridge_mse((Xb-mu)/sd, Yb, cv_folds=cv_folds, seed=seed)
        best_gain, best_c = 0.0, None
        for c in list(candidates):
            xc = X_df[c].to_numpy(dtype=float)
            # tau-gate
            too = False
            for k in cols:
                zk = X_df[k].to_numpy(dtype=float)
                m2 = (~np.isnan(xc)) & (~np.isnan(zk))
                if int(m2.sum()) >= int(min_pairs):
                    corr = np.corrcoef(xc[m2], zk[m2])[0,1]
                    if np.isfinite(corr) and abs(float(corr))>float(tau): too=True; break
            if too: continue
            m2 = base_m & (~np.isnan(xc))
            if int(m2.sum())<int(min_pairs): continue
            Xcand = np.column_stack([X_df[k].to_numpy(dtype=float)[m2] for k in cols] + [xc[m2]])
            mu2 = Xcand.mean(0); sd2 = Xcand.std(0)+1e-8
            gain = base - _cv_ridge_mse((Xcand-mu2)/sd2, Yb[m2], cv_folds=cv_folds, seed=seed)
            if gain>best_gain: best_gain, best_c = float(gain), c
        if best_c is None or best_gain < float(delta_improve)*base: break
        cols.append(best_c); candidates.remove(best_c)
        if len(cols)>=int(k_max): break
    mfin = complete_mask(cols)
    if int(mfin.sum())>=int(min_pairs):
        Xf = np.column_stack([X_df[c].to_numpy(dtype=float)[mfin] for c in cols])
        mu_f, sd_f = Xf.mean(0), Xf.std(0)+1e-8
    else:
        mu_f, sd_f = np.zeros(len(cols)), np.ones(len(cols))
    return cols, (mu_f, sd_f)
