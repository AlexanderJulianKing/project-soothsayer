#!/usr/bin/env python3
"""
Joint Prediction: SCMF and BHLT approaches for Arena ELO prediction.

Two latent-factor approaches that jointly impute missing benchmark scores
and predict Arena ELO in a unified framework:

1. SCMF (Supervised Collective Matrix Factorization):
   ALS-based matrix factorization with a supervised target loss term.
   The latent factors Z are encouraged to both reconstruct the observed
   benchmark matrix X and predict the Arena ELO target y.

2. BHLT (Bayesian Hierarchical Latent-Trait):
   EM-based factor analysis with hierarchical priors on loadings (grouped
   by benchmark family) and full posterior uncertainty on latent factors.
   Produces calibrated prediction intervals.

Both approaches are transductive by default (use all rows' observed X for
learning factors), with an optional inductive mode that holds out validation
rows entirely during fitting.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import RepeatedKFold


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET = "lmsys_Score"
ALT_TARGET = "lmarena_Score"
ID_COL = "model_name"
EXCLUDE_COLS = {TARGET, ALT_TARGET, ID_COL, "Unified_Name"}

SEED = 42


# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------

def load_data(csv_path: str):
    """Load CSV and return feature matrix, target, mask, column names, model names.

    Returns:
        X_obs_df: DataFrame (n, p) with NaN for missing benchmark values.
        y: ndarray (n,) target values (NaN where unknown).
        y_mask: ndarray (n,) boolean, True where y is known (not NaN).
        feature_cols: list of feature column names.
        model_names: list of model name strings.
    """
    df = pd.read_csv(csv_path)

    # Identify feature columns: everything not in EXCLUDE_COLS
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    model_names = df[ID_COL].tolist()

    X_obs_df = df[feature_cols].copy()
    y = df[TARGET].values.astype(float)
    y_mask = ~np.isnan(y)

    return X_obs_df, y, y_mask, feature_cols, model_names


def build_cv_splits(
    n_labeled: int,
    n_splits: int = 10,
    repeats: int = 10,
    seed: int = SEED,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build repeated K-fold splits on labeled indices only.

    Returns list of (train_idx, val_idx) tuples where indices are into
    the labeled subset (0..n_labeled-1).
    """
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=repeats, random_state=seed)
    splits = []
    dummy = np.arange(n_labeled)
    for tr, va in rkf.split(dummy):
        splits.append((tr, va))
    return splits


def compute_oof_rmse(
    y_true: np.ndarray,
    oof_preds: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = SEED,
) -> Tuple[float, float, float]:
    """Compute RMSE with bootstrap 95% confidence interval.

    Args:
        y_true: ground truth values for labeled models.
        oof_preds: out-of-fold predictions (same length as y_true).
        n_bootstrap: number of bootstrap resamples.
        seed: random seed.

    Returns:
        (rmse, ci_lo, ci_hi): point estimate and 95% CI bounds.
    """
    residuals = y_true - oof_preds
    rmse = np.sqrt(np.mean(residuals ** 2))

    rng = np.random.RandomState(seed)
    n = len(residuals)
    boot_rmses = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_rmses[b] = np.sqrt(np.mean(residuals[idx] ** 2))

    ci_lo = float(np.percentile(boot_rmses, 2.5))
    ci_hi = float(np.percentile(boot_rmses, 97.5))

    return float(rmse), ci_lo, ci_hi


# ---------------------------------------------------------------------------
# SCMF: Supervised Collective Matrix Factorization
# ---------------------------------------------------------------------------

@dataclass
class SCMFConfig:
    """Configuration for SupervisedCMF."""
    rank: int = 6
    lambda_rec: float = 1.0
    lambda_target: float = 5.0
    lambda_reg: float = 0.01
    max_iter: int = 100
    tol: float = 1e-4
    n_restarts: int = 1


class SupervisedCMF:
    """ALS-based supervised collective matrix factorization.

    Jointly factorizes the benchmark matrix X ~ Z @ W.T and predicts
    target y ~ Z @ w_target + b_target, by minimizing:

        lambda_rec * ||X_obs - Z @ W.T||^2
      + lambda_target * ||y_obs - Z_obs @ w_target - b_target||^2
      + lambda_reg * (||Z||^2 + ||W||^2)
    """

    def __init__(self, config: Optional[SCMFConfig] = None):
        self.config = config or SCMFConfig()
        self.Z_ = None
        self.W_ = None
        self.w_target_ = None
        self.b_target_ = None
        self.col_mean_ = None
        self.col_std_ = None
        self.loss_history_ = []

    def fit(
        self,
        X_obs: np.ndarray,
        y: np.ndarray,
        y_mask: np.ndarray,
    ) -> "SupervisedCMF":
        """Fit the supervised CMF model.

        Args:
            X_obs: (n, p) matrix with NaN for missing entries.
            y: (n,) target values (NaN or arbitrary where y_mask is False).
            y_mask: (n,) boolean, True where y is known.

        Returns:
            self
        """
        cfg = self.config
        n, p = X_obs.shape
        obs_mask = ~np.isnan(X_obs)  # (n, p) boolean

        # 1. Standardize columns (observed values only)
        col_mean = np.nanmean(X_obs, axis=0)
        col_std = np.nanstd(X_obs, axis=0)
        col_std[col_std < 1e-8] = 1.0
        self.col_mean_ = col_mean
        self.col_std_ = col_std

        X_std = (X_obs - col_mean) / col_std
        X_std = np.where(obs_mask, X_std, 0.0)

        # 2. Median-fill for SVD init
        col_median_std = np.zeros(p)
        for j in range(p):
            vals = X_std[obs_mask[:, j], j]
            col_median_std[j] = np.median(vals) if len(vals) > 0 else 0.0
        X_filled = X_std.copy()
        for j in range(p):
            X_filled[~obs_mask[:, j], j] = col_median_std[j]

        # 3. SVD warm-start
        k = min(cfg.rank, min(n, p) - 1)
        U, S, Vt = svds(X_filled, k=k)
        # svds returns in ascending order of singular values; reverse
        idx = np.argsort(-S)
        U, S, Vt = U[:, idx], S[idx], Vt[idx, :]

        sqrt_S = np.sqrt(S)
        Z = U * sqrt_S[np.newaxis, :]         # (n, k)
        W = (Vt.T * sqrt_S[np.newaxis, :])    # (p, k)

        # 4. Init w_target via Ridge on labeled rows
        labeled_idx = np.where(y_mask)[0]
        y_labeled = y[labeled_idx]
        ridge = Ridge(alpha=cfg.lambda_reg, fit_intercept=True)
        ridge.fit(Z[labeled_idx], y_labeled)
        w_target = ridge.coef_.copy()
        b_target = float(ridge.intercept_)

        # 5. ALS loop
        self.loss_history_ = []
        prev_loss = np.inf

        for it in range(cfg.max_iter):
            # 5a. Update W column-by-column
            for j in range(p):
                rows_j = np.where(obs_mask[:, j])[0]
                if len(rows_j) == 0:
                    continue
                Z_j = Z[rows_j]  # (n_obs, k)
                x_j = X_std[rows_j, j]
                A = Z_j.T @ Z_j + cfg.lambda_reg * np.eye(k)
                b_vec = Z_j.T @ x_j
                W[j] = np.linalg.solve(A, b_vec)

            # 5b. Update Z row-by-row
            for i in range(n):
                cols_i = np.where(obs_mask[i])[0]
                W_obs = W[cols_i]  # (n_obs, k)
                x_obs = X_std[i, cols_i]

                Prec = cfg.lambda_rec * (W_obs.T @ W_obs) + cfg.lambda_reg * np.eye(k)
                info = cfg.lambda_rec * (W_obs.T @ x_obs)

                if y_mask[i]:
                    wt_outer = np.outer(w_target, w_target)
                    Prec += cfg.lambda_target * wt_outer
                    info += cfg.lambda_target * (y[i] - b_target) * w_target

                Z[i] = np.linalg.solve(Prec, info)

            # 5c. Update w_target, b_target via BayesianRidge
            br = BayesianRidge(fit_intercept=True, max_iter=300, tol=1e-6)
            br.fit(Z[labeled_idx], y_labeled)
            w_target = br.coef_.copy()
            b_target = float(br.intercept_)

            # 5d. Compute loss, check convergence
            loss = self._compute_loss(X_std, y, y_mask, obs_mask, Z, W, w_target, b_target)
            self.loss_history_.append(loss)

            delta = prev_loss - loss
            if (it + 1) % 10 == 0 or it == 0:
                print(f"  SCMF iter {it+1:3d}: loss={loss:.4f}, delta={delta:.6f}")

            if abs(delta) < cfg.tol and it > 0:
                print(f"  SCMF converged at iteration {it+1}")
                break
            prev_loss = loss

        self.Z_ = Z
        self.W_ = W
        self.w_target_ = w_target
        self.b_target_ = b_target
        return self

    def predict(self) -> np.ndarray:
        """Predict target for all rows using learned factors.

        Returns:
            (n,) array of predictions.
        """
        return self.Z_ @ self.w_target_ + self.b_target_

    def get_factors(self) -> np.ndarray:
        """Return latent factors Z.

        Returns:
            (n, rank) array.
        """
        return self.Z_

    def _compute_loss(
        self,
        X_std: np.ndarray,
        y: np.ndarray,
        y_mask: np.ndarray,
        obs_mask: np.ndarray,
        Z: np.ndarray,
        W: np.ndarray,
        w_target: np.ndarray,
        b_target: float,
    ) -> float:
        """Compute the full objective value."""
        cfg = self.config

        # Reconstruction loss (observed cells only)
        recon = Z @ W.T  # (n, p)
        resid = np.where(obs_mask, X_std - recon, 0.0)
        rec_loss = np.sum(resid ** 2)

        # Target loss (labeled rows only)
        labeled_idx = np.where(y_mask)[0]
        pred = Z[labeled_idx] @ w_target + b_target
        target_loss = np.sum((y[labeled_idx] - pred) ** 2)

        # Regularization
        reg_loss = np.sum(Z ** 2) + np.sum(W ** 2)

        return (
            cfg.lambda_rec * rec_loss
            + cfg.lambda_target * target_loss
            + cfg.lambda_reg * reg_loss
        )


# ---------------------------------------------------------------------------
# BHLT: Bayesian Hierarchical Latent-Trait
# ---------------------------------------------------------------------------

@dataclass
class BHLTConfig:
    """Configuration for BayesianHierarchicalLT."""
    n_factors: int = 6
    n_iter: int = 50
    family_prior_strength: float = 1.0
    tol: float = 1e-4
    clustering_method: str = "correlation"   # or 'prefix'
    clustering_threshold: float = 0.5


def _build_families_prefix(feature_cols: List[str]) -> List[List[int]]:
    """Group columns by prefix (text before first '_')."""
    family_map = {}
    for j, col in enumerate(feature_cols):
        prefix = col.split("_")[0] if "_" in col else col
        family_map.setdefault(prefix, []).append(j)
    return list(family_map.values())


def _build_families_correlation(
    X_obs: np.ndarray,
    threshold: float = 0.5,
) -> List[List[int]]:
    """Group columns by agglomerative clustering on pairwise correlations."""
    n, p = X_obs.shape
    if p <= 1:
        return [[j] for j in range(p)]

    # Compute pairwise correlations (using pairwise-complete observations)
    corr_mat = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            mask = ~np.isnan(X_obs[:, i]) & ~np.isnan(X_obs[:, j])
            if np.sum(mask) < 3:
                corr_mat[i, j] = corr_mat[j, i] = 0.0
            else:
                xi = X_obs[mask, i]
                xj = X_obs[mask, j]
                si = np.std(xi)
                sj = np.std(xj)
                if si < 1e-8 or sj < 1e-8:
                    corr_mat[i, j] = corr_mat[j, i] = 0.0
                else:
                    r = np.corrcoef(xi, xj)[0, 1]
                    corr_mat[i, j] = corr_mat[j, i] = r if np.isfinite(r) else 0.0

    # Convert correlation to distance
    dist_mat = 1.0 - np.abs(corr_mat)
    np.fill_diagonal(dist_mat, 0.0)

    # Ensure symmetry and non-negativity
    dist_mat = np.maximum(dist_mat, 0.0)
    dist_mat = (dist_mat + dist_mat.T) / 2.0

    # Agglomerative clustering
    condensed = squareform(dist_mat, checks=False)
    Z_link = linkage(condensed, method="average")
    labels = fcluster(Z_link, t=threshold, criterion="distance")

    families = {}
    for j, lab in enumerate(labels):
        families.setdefault(int(lab), []).append(j)
    return list(families.values())


class BayesianHierarchicalLT:
    """Bayesian hierarchical latent-trait model with EM inference.

    Factor analysis model:
        x_ij | z_i ~ N(mu_j + Lambda_j' z_i, psi_j)
        y_i  | z_i ~ N(w' z_i + b, sigma_y2)
        z_i ~ N(0, I)

    Hierarchical prior on loadings: Lambda_j ~ N(Lambda_family_mean, tau*I)
    where tau = 1/family_prior_strength.
    """

    def __init__(self, config: Optional[BHLTConfig] = None, feature_cols: Optional[List[str]] = None):
        self.config = config or BHLTConfig()
        self.feature_cols = feature_cols
        self.mu_post_ = None     # (n, k) posterior means
        self.sigma_post_ = None  # (n, k, k) posterior covariances
        self.w_ = None
        self.b_ = None
        self.sigma_y2_ = None
        self.Lambda_ = None
        self.mu_ = None
        self.psi_ = None
        self.families_ = None
        self.col_mean_ = None
        self.col_std_ = None
        self.loss_history_ = []

    def fit(
        self,
        X_obs: np.ndarray,
        y: np.ndarray,
        y_mask: np.ndarray,
    ) -> "BayesianHierarchicalLT":
        """Fit the BHLT model via EM.

        Args:
            X_obs: (n, p) matrix with NaN for missing entries.
            y: (n,) target values.
            y_mask: (n,) boolean, True where y is known.

        Returns:
            self
        """
        cfg = self.config
        n, p = X_obs.shape
        k = cfg.n_factors
        obs_mask = ~np.isnan(X_obs)
        labeled_idx = np.where(y_mask)[0]
        y_labeled = y[labeled_idx]

        # 1. Standardize (observed only)
        col_mean = np.nanmean(X_obs, axis=0)
        col_std = np.nanstd(X_obs, axis=0)
        col_std[col_std < 1e-8] = 1.0
        self.col_mean_ = col_mean
        self.col_std_ = col_std

        X_std = (X_obs - col_mean) / col_std
        X_std = np.where(obs_mask, X_std, 0.0)

        # 2. Build families
        if cfg.clustering_method == "prefix" and self.feature_cols is not None:
            self.families_ = _build_families_prefix(self.feature_cols)
        else:
            self.families_ = _build_families_correlation(X_obs, cfg.clustering_threshold)

        # 3. Initialize via SVD
        # Median-fill for SVD
        col_median_std = np.zeros(p)
        for j in range(p):
            vals = X_std[obs_mask[:, j], j]
            col_median_std[j] = np.median(vals) if len(vals) > 0 else 0.0
        X_filled = X_std.copy()
        for j in range(p):
            X_filled[~obs_mask[:, j], j] = col_median_std[j]

        k_svd = min(k, min(n, p) - 1)
        U, S, Vt = svds(X_filled, k=k_svd)
        idx = np.argsort(-S)
        U, S, Vt = U[:, idx], S[idx], Vt[idx, :]

        sqrt_S = np.sqrt(S)
        # If k_svd < k, pad with zeros
        Lambda = np.zeros((p, k))
        Lambda[:, :k_svd] = Vt[:k_svd, :].T * sqrt_S[:k_svd][np.newaxis, :]

        mu = np.zeros(p)  # column means are zero after standardization
        psi = np.ones(p) * 0.5  # init residual variance

        # Init target model
        Z_init = np.zeros((n, k))
        Z_init[:, :k_svd] = U[:, :k_svd] * sqrt_S[:k_svd][np.newaxis, :]
        br = BayesianRidge(fit_intercept=True, max_iter=300, tol=1e-6)
        br.fit(Z_init[labeled_idx], y_labeled)
        w = br.coef_.copy()
        b = float(br.intercept_)
        sigma_y2 = max(np.var(y_labeled - br.predict(Z_init[labeled_idx])), 1e-4)

        # Tau for family prior
        tau = 1.0 / max(cfg.family_prior_strength, 1e-6)

        # Storage for posteriors
        mu_post = np.zeros((n, k))
        sigma_post = np.zeros((n, k, k))

        self.loss_history_ = []
        prev_ll = -np.inf

        for it in range(cfg.n_iter):
            # ------ E-step ------
            I_k = np.eye(k)
            for i in range(n):
                obs_j = np.where(obs_mask[i])[0]
                Lambda_obs = Lambda[obs_j]    # (n_obs, k)
                psi_obs = psi[obs_j]          # (n_obs,)
                x_obs = X_std[i, obs_j]       # (n_obs,)
                mu_obs = mu[obs_j]            # (n_obs,)

                # Precision from observations: I + sum_j Lambda_j Lambda_j^T / psi_j
                # = I + Lambda_obs^T diag(1/psi_obs) Lambda_obs
                inv_psi = 1.0 / np.maximum(psi_obs, 1e-8)
                Prec = I_k + Lambda_obs.T @ (Lambda_obs * inv_psi[:, np.newaxis])

                # Info from observations
                info = Lambda_obs.T @ ((x_obs - mu_obs) * inv_psi)

                # Supervised term
                if y_mask[i]:
                    Prec += np.outer(w, w) / max(sigma_y2, 1e-8)
                    info += w * (y[i] - b) / max(sigma_y2, 1e-8)

                Sigma_i = np.linalg.solve(Prec, I_k)
                Sigma_i = (Sigma_i + Sigma_i.T) / 2.0  # symmetrize
                mu_i = Sigma_i @ info

                mu_post[i] = mu_i
                sigma_post[i] = Sigma_i

            # ------ M-step ------

            # Precompute E[z_i z_i^T] = Sigma_i + outer(mu_i, mu_i) for each i
            # We'll compute these on the fly to save memory

            # Update Lambda_j with family prior
            # Compute family means
            family_means = {}
            for fam_idx, fam_cols in enumerate(self.families_):
                fam_lambda = np.mean(Lambda[fam_cols], axis=0) if len(fam_cols) > 0 else np.zeros(k)
                family_means[fam_idx] = fam_lambda

            col_to_family = {}
            for fam_idx, fam_cols in enumerate(self.families_):
                for j in fam_cols:
                    col_to_family[j] = fam_idx

            for j in range(p):
                rows_j = np.where(obs_mask[:, j])[0]
                if len(rows_j) == 0:
                    continue

                # sum_i E[z_i z_i^T] for rows observing column j
                EzzT_sum = np.zeros((k, k))
                Ez_x_sum = np.zeros(k)
                for i in rows_j:
                    Ezz = sigma_post[i] + np.outer(mu_post[i], mu_post[i])
                    EzzT_sum += Ezz
                    Ez_x_sum += mu_post[i] * (X_std[i, j] - mu[j])

                # Family prior
                fam_idx = col_to_family.get(j, 0)
                fam_mean = family_means.get(fam_idx, np.zeros(k))

                A = EzzT_sum + (1.0 / tau) * I_k
                b_vec = Ez_x_sum + (1.0 / tau) * fam_mean
                Lambda[j] = np.linalg.solve(A, b_vec)

            # Update psi_j (residual variance per column)
            for j in range(p):
                rows_j = np.where(obs_mask[:, j])[0]
                if len(rows_j) < 2:
                    continue
                resid_sq_sum = 0.0
                for i in rows_j:
                    diff = X_std[i, j] - mu[j] - Lambda[j] @ mu_post[i]
                    Ezz = sigma_post[i] + np.outer(mu_post[i], mu_post[i])
                    resid_sq_sum += diff ** 2 + Lambda[j] @ sigma_post[i] @ Lambda[j]
                psi[j] = max(resid_sq_sum / len(rows_j), 1e-4)

            # Update w, b via BayesianRidge
            br = BayesianRidge(fit_intercept=True, max_iter=300, tol=1e-6)
            br.fit(mu_post[labeled_idx], y_labeled)
            w = br.coef_.copy()
            b = float(br.intercept_)

            # Update sigma_y2
            pred_lab = mu_post[labeled_idx] @ w + b
            resid_y = y_labeled - pred_lab
            # Include uncertainty from posterior: E[(y-w'z-b)^2] = (y-w'mu)^2 + w'Sigma w
            sigma_y2_sum = np.sum(resid_y ** 2)
            for idx_l, i in enumerate(labeled_idx):
                sigma_y2_sum += w @ sigma_post[i] @ w
            sigma_y2 = max(sigma_y2_sum / len(labeled_idx), 1e-4)

            # Compute approximate log-likelihood for convergence check
            ll = self._approx_log_likelihood(
                X_std, y, y_mask, obs_mask, mu_post, sigma_post,
                Lambda, mu, psi, w, b, sigma_y2,
            )
            self.loss_history_.append(ll)

            delta = ll - prev_ll
            if (it + 1) % 10 == 0 or it == 0:
                print(f"  BHLT iter {it+1:3d}: log-lik={ll:.4f}, delta={delta:.6f}")

            if abs(delta) < cfg.tol and it > 0:
                print(f"  BHLT converged at iteration {it+1}")
                break
            prev_ll = ll

        self.mu_post_ = mu_post
        self.sigma_post_ = sigma_post
        self.w_ = w
        self.b_ = b
        self.sigma_y2_ = sigma_y2
        self.Lambda_ = Lambda
        self.mu_ = mu
        self.psi_ = psi
        return self

    def predict(self, return_std: bool = False):
        """Predict target from posterior factor means.

        Args:
            return_std: if True, also return prediction standard deviations.

        Returns:
            pred: (n,) array of predictions.
            std: (n,) array of stds (only if return_std=True).
        """
        pred = self.mu_post_ @ self.w_ + self.b_
        if return_std:
            std = np.sqrt(
                np.array([self.w_ @ S @ self.w_ for S in self.sigma_post_])
                + self.sigma_y2_
            )
            return pred, std
        return pred

    def get_factors(self):
        """Return posterior factor means and standard deviations.

        Returns:
            means: (n, k) array.
            stds: (n, k) array of marginal standard deviations.
        """
        means = self.mu_post_
        stds = np.sqrt(np.diagonal(self.sigma_post_, axis1=1, axis2=2))
        return means, stds

    def _approx_log_likelihood(
        self,
        X_std, y, y_mask, obs_mask,
        mu_post, sigma_post,
        Lambda, mu, psi, w, b, sigma_y2,
    ) -> float:
        """Approximate expected complete-data log-likelihood (ELBO proxy)."""
        n, p = X_std.shape
        labeled_idx = np.where(y_mask)[0]

        ll = 0.0

        # Observation term
        for j in range(p):
            rows_j = np.where(obs_mask[:, j])[0]
            if len(rows_j) == 0:
                continue
            for i in rows_j:
                diff = X_std[i, j] - mu[j] - Lambda[j] @ mu_post[i]
                var_term = Lambda[j] @ sigma_post[i] @ Lambda[j]
                ll -= 0.5 * (np.log(max(psi[j], 1e-8)) + (diff ** 2 + var_term) / max(psi[j], 1e-8))

        # Target term
        for i in labeled_idx:
            diff = y[i] - w @ mu_post[i] - b
            var_term = w @ sigma_post[i] @ w
            ll -= 0.5 * (np.log(max(sigma_y2, 1e-8)) + (diff ** 2 + var_term) / max(sigma_y2, 1e-8))

        return ll


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(
    approach_class,
    config,
    X_obs: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    mode: str = "transductive",
    feature_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """Run cross-validation and return out-of-fold predictions.

    Args:
        approach_class: SupervisedCMF or BayesianHierarchicalLT class.
        config: configuration dataclass for the approach.
        X_obs: (n_total, p) full benchmark matrix with NaN.
        y: (n_total,) target array.
        y_mask: (n_total,) boolean mask.
        splits: list of (train_idx, val_idx) into labeled-only indices.
        mode: 'transductive' or 'inductive'.
        feature_cols: column names (needed for BHLT prefix clustering).

    Returns:
        oof_preds: (n_labeled,) array of OOF predictions.
    """
    labeled_idx = np.where(y_mask)[0]
    n_labeled = len(labeled_idx)
    oof_sum = np.zeros(n_labeled)
    oof_count = np.zeros(n_labeled)

    n_splits = len(splits)
    for fold_i, (tr_local, va_local) in enumerate(splits):
        if (fold_i + 1) % 10 == 0 or fold_i == 0:
            print(f"  Fold {fold_i+1}/{n_splits}...")

        # tr_local and va_local index into labeled_idx
        tr_global = labeled_idx[tr_local]
        va_global = labeled_idx[va_local]

        if mode == "transductive":
            # All rows see their X, but only training targets are used
            y_fold_mask = np.zeros_like(y_mask)
            y_fold_mask[tr_global] = True

            kwargs = {}
            if approach_class is BayesianHierarchicalLT:
                kwargs["feature_cols"] = feature_cols
                model = approach_class(config=config, **kwargs)
            else:
                model = approach_class(config=config)

            model.fit(X_obs, y, y_fold_mask)
            preds_all = model.predict() if not isinstance(model, BayesianHierarchicalLT) else model.predict(return_std=False)

            for local_i, global_i in zip(va_local, va_global):
                oof_sum[local_i] += preds_all[global_i]
                oof_count[local_i] += 1

        elif mode == "inductive":
            # Train only on train rows, then infer validation factors
            # For inductive: fit on train rows only, then use learned loadings
            # to infer factors for val rows
            y_fold_mask_tr = np.ones(len(tr_global), dtype=bool)

            # Build training data
            X_tr = X_obs[tr_global]
            y_tr = y[tr_global]

            kwargs = {}
            if approach_class is BayesianHierarchicalLT:
                kwargs["feature_cols"] = feature_cols
                model = approach_class(config=config, **kwargs)
            else:
                model = approach_class(config=config)

            model.fit(X_tr, y_tr, y_fold_mask_tr)

            # Infer validation factors from learned loadings
            preds_va = _inductive_predict(model, X_obs[va_global])

            for i, local_i in enumerate(va_local):
                oof_sum[local_i] += preds_va[i]
                oof_count[local_i] += 1

        else:
            raise ValueError(f"Unknown mode: {mode}")

    # Average over repeats
    oof_count[oof_count == 0] = 1
    oof_preds = oof_sum / oof_count
    return oof_preds


def _inductive_predict(model, X_new: np.ndarray) -> np.ndarray:
    """Predict for new rows using learned loadings/weights.

    For SCMF: solve for Z_new given fixed W, w_target, b_target.
    For BHLT: E-step with fixed Lambda, mu, psi, w, b, sigma_y2.
    """
    n_new = X_new.shape[0]

    if isinstance(model, SupervisedCMF):
        # Standardize
        X_std = (X_new - model.col_mean_) / model.col_std_
        obs_mask = ~np.isnan(X_new)
        X_std = np.where(obs_mask, X_std, 0.0)

        W = model.W_
        k = W.shape[1]
        cfg = model.config
        Z_new = np.zeros((n_new, k))

        for i in range(n_new):
            cols_i = np.where(obs_mask[i])[0]
            W_obs = W[cols_i]
            x_obs = X_std[i, cols_i]

            Prec = cfg.lambda_rec * (W_obs.T @ W_obs) + cfg.lambda_reg * np.eye(k)
            info = cfg.lambda_rec * (W_obs.T @ x_obs)
            Z_new[i] = np.linalg.solve(Prec, info)

        return Z_new @ model.w_target_ + model.b_target_

    elif isinstance(model, BayesianHierarchicalLT):
        # E-step with fixed parameters, no target supervision
        X_std = (X_new - model.col_mean_) / model.col_std_
        obs_mask = ~np.isnan(X_new)
        X_std = np.where(obs_mask, X_std, 0.0)

        Lambda = model.Lambda_
        mu_col = model.mu_
        psi = model.psi_
        k = Lambda.shape[1]
        I_k = np.eye(k)

        preds = np.zeros(n_new)
        for i in range(n_new):
            obs_j = np.where(obs_mask[i])[0]
            Lambda_obs = Lambda[obs_j]
            psi_obs = psi[obs_j]
            x_obs = X_std[i, obs_j]
            mu_obs = mu_col[obs_j]

            inv_psi = 1.0 / np.maximum(psi_obs, 1e-8)
            Prec = I_k + Lambda_obs.T @ (Lambda_obs * inv_psi[:, np.newaxis])
            info = Lambda_obs.T @ ((x_obs - mu_obs) * inv_psi)

            Sigma_i = np.linalg.solve(Prec, I_k)
            mu_i = Sigma_i @ info

            preds[i] = model.w_ @ mu_i + model.b_

        return preds

    else:
        raise TypeError(f"Unknown model type: {type(model)}")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> dict:
    """Run a single joint prediction experiment.

    Returns:
        metadata dict with results.
    """
    print(f"\n{'='*60}")
    print(f"Joint Prediction Experiment")
    print(f"  Approach: {args.approach}")
    print(f"  Mode:     {args.mode}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # Load data
    X_obs_df, y, y_mask, feature_cols, model_names = load_data(args.csv_path)
    X_obs = X_obs_df.values.astype(float)

    n_total = len(y)
    n_labeled = int(np.sum(y_mask))
    n_features = len(feature_cols)
    print(f"Data: {n_total} models, {n_labeled} labeled, {n_features} features")
    print(f"Missing rate: {np.mean(np.isnan(X_obs)):.1%}\n")

    # Build config and approach
    if args.approach == "scmf":
        config = SCMFConfig(
            rank=args.rank,
            lambda_rec=1.0,
            lambda_target=args.lambda_target,
            lambda_reg=args.lambda_reg,
            max_iter=args.max_iter,
            tol=1e-4,
        )
        approach_class = SupervisedCMF
    elif args.approach == "bhlt":
        config = BHLTConfig(
            n_factors=args.n_factors,
            n_iter=args.max_iter,
            family_prior_strength=args.family_prior,
            tol=1e-4,
            clustering_method=args.clustering,
            clustering_threshold=0.5,
        )
        approach_class = BayesianHierarchicalLT
    else:
        raise ValueError(f"Unknown approach: {args.approach}")

    # Build CV splits
    splits = build_cv_splits(n_labeled, n_splits=args.cv_splits, repeats=args.cv_repeats, seed=args.seed)
    print(f"CV: {args.cv_splits}-fold x {args.cv_repeats} repeats = {len(splits)} folds\n")

    # Cross-validate
    print("Running cross-validation...")
    oof_preds = cross_validate(
        approach_class, config, X_obs, y, y_mask, splits,
        mode=args.mode, feature_cols=feature_cols,
    )

    # Compute RMSE
    y_labeled = y[y_mask]
    rmse, ci_lo, ci_hi = compute_oof_rmse(y_labeled, oof_preds)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"OOF RMSE: {rmse:.2f}  [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*60}\n")

    # Fit final model on all data for predictions
    print("Fitting final model on all data...")
    if args.approach == "scmf":
        final_model = SupervisedCMF(config=config)
    else:
        final_model = BayesianHierarchicalLT(config=config, feature_cols=feature_cols)
    final_model.fit(X_obs, y, y_mask)
    final_preds = final_model.predict() if not isinstance(final_model, BayesianHierarchicalLT) else final_model.predict(return_std=False)

    # Save results
    os.makedirs(args.output_root, exist_ok=True)

    # Metadata
    config_dict = asdict(config)
    metadata = {
        "approach": args.approach,
        "mode": args.mode,
        "config": config_dict,
        "oof_rmse": rmse,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_labeled": n_labeled,
        "n_total": n_total,
        "n_features": n_features,
        "cv_splits": args.cv_splits,
        "cv_repeats": args.cv_repeats,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(args.output_root, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    # Predictions CSV
    labeled_idx = np.where(y_mask)[0]
    pred_df = pd.DataFrame({
        "model_name": model_names,
        "y_true": y,
        "y_pred": final_preds,
        "is_labeled": y_mask,
    })
    # Replace OOF predictions for labeled models
    pred_df.loc[labeled_idx, "y_pred_oof"] = oof_preds

    pred_path = os.path.join(args.output_root, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Joint prediction: SCMF and BHLT approaches for Arena ELO."
    )
    ap.add_argument("--approach", choices=["scmf", "bhlt"], required=True,
                     help="Which approach to use.")
    ap.add_argument("--mode", choices=["transductive", "inductive"], default="transductive",
                     help="Transductive (use all X) or inductive (hold out val rows).")
    ap.add_argument("--csv_path", type=str, required=True,
                     help="Path to clean_combined_all_benches.csv.")
    ap.add_argument("--output_root", type=str, required=True,
                     help="Output directory for results.")
    ap.add_argument("--rank", type=int, default=6,
                     help="Latent rank for SCMF (default: 6).")
    ap.add_argument("--n_factors", type=int, default=6,
                     help="Number of factors for BHLT (default: 6).")
    ap.add_argument("--lambda_target", type=float, default=5.0,
                     help="Target loss weight for SCMF (default: 5.0).")
    ap.add_argument("--lambda_reg", type=float, default=0.01,
                     help="Regularization weight for SCMF (default: 0.01).")
    ap.add_argument("--family_prior", type=float, default=1.0,
                     help="Family prior strength for BHLT (default: 1.0).")
    ap.add_argument("--clustering", choices=["correlation", "prefix"], default="correlation",
                     help="Clustering method for BHLT families (default: correlation).")
    ap.add_argument("--cv_splits", type=int, default=10,
                     help="Number of CV folds (default: 10).")
    ap.add_argument("--cv_repeats", type=int, default=10,
                     help="Number of CV repeats (default: 10).")
    ap.add_argument("--seed", type=int, default=42,
                     help="Random seed (default: 42).")
    ap.add_argument("--max_iter", type=int, default=100,
                     help="Maximum iterations for ALS/EM (default: 100).")

    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
