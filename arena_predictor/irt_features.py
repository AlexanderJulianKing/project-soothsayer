"""Continuous 2PL IRT model for latent ability extraction.

Fits a sigmoid-link model: P(correct_ij) = sigmoid(a_j * (theta_i - b_j))
where theta_i is model ability, a_j is item discrimination, b_j is item difficulty.

Operates on observed cells only (no imputation needed).
Outputs polynomial features of theta for downstream prediction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from typing import Optional, Tuple


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fit_irt_2pl(
    X: np.ndarray,
    k: int = 4,
    reg_lambda: float = 0.0001,
    lr: float = 0.01,
    max_iter: int = 2000,
    tol: float = 1e-7,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit continuous 2PL IRT via gradient descent on observed cells.

    Args:
        X: (n_models, n_items) matrix with NaN for missing cells.
           Values should be in [0, 1] (normalized benchmark scores).
        k: Number of latent dimensions.
        reg_lambda: L2 regularization on theta and discrimination.
        lr: Learning rate for Adam optimizer.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on loss change.
        seed: Random seed.

    Returns:
        theta: (n_models, k) latent ability matrix
        a: (n_items, k) discrimination parameters
        b: (n_items,) difficulty parameters
    """
    rng = np.random.RandomState(seed)
    n_models, n_items = X.shape

    # Find observed cells
    obs_mask = ~np.isnan(X)
    obs_rows, obs_cols = np.where(obs_mask)
    obs_vals = X[obs_mask]
    n_obs = len(obs_vals)

    # Initialize
    theta = rng.randn(n_models, k) * 0.1
    a = rng.randn(n_items, k) * 0.1 + 1.0  # start near 1
    b = np.zeros(n_items)

    # Adam optimizer state
    params = [theta, a, b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    prev_loss = np.inf
    for iteration in range(max_iter):
        # Forward: logit = sum_k(a[j,k] * theta[i,k]) - b[j]
        logits = np.sum(theta[obs_rows] * a[obs_cols], axis=1) - b[obs_cols]
        probs = _sigmoid(logits)

        # MSE loss + L2 reg
        residuals = probs - obs_vals
        loss = 0.5 * np.mean(residuals ** 2) + reg_lambda * (
            np.sum(theta ** 2) + np.sum(a ** 2)
        )

        if abs(prev_loss - loss) < tol and iteration > 100:
            break
        prev_loss = loss

        # Gradients
        d_logit = residuals * probs * (1 - probs) / n_obs

        # theta gradients
        g_theta = np.zeros_like(theta)
        np.add.at(g_theta, obs_rows, d_logit[:, None] * a[obs_cols])
        g_theta += 2 * reg_lambda * theta

        # a gradients
        g_a = np.zeros_like(a)
        np.add.at(g_a, obs_cols, d_logit[:, None] * theta[obs_rows])
        g_a += 2 * reg_lambda * a

        # b gradients
        g_b = np.zeros_like(b)
        np.add.at(g_b, obs_cols, -d_logit)

        grads = [g_theta, g_a, g_b]

        # Adam update
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g ** 2
            m_hat = m[i] / (1 - beta1 ** (iteration + 1))
            v_hat = v[i] / (1 - beta2 ** (iteration + 1))
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return theta, a, b


def compute_irt_features(
    df: pd.DataFrame,
    feature_cols: list,
    k: int = 4,
    poly_degree: int = 2,
    reg_lambda: float = 0.0001,
    seed: int = 42,
) -> pd.DataFrame:
    """Extract IRT latent features from a benchmark DataFrame.

    Args:
        df: DataFrame with model_name + benchmark columns.
        feature_cols: Benchmark columns to use.
        k: Number of latent dimensions.
        poly_degree: Polynomial expansion degree (2 = include cross-terms).
        reg_lambda: IRT regularization.
        seed: Random seed.

    Returns:
        DataFrame with model_name + IRT polynomial features.
    """
    # Normalize benchmarks to [0, 1]
    X_raw = df[feature_cols].values.astype(float)
    col_min = np.nanmin(X_raw, axis=0)
    col_max = np.nanmax(X_raw, axis=0)
    col_range = col_max - col_min
    col_range[col_range < 1e-12] = 1.0
    X_norm = (X_raw - col_min) / col_range

    # Fit IRT
    theta, a, b = fit_irt_2pl(X_norm, k=k, reg_lambda=reg_lambda, seed=seed)

    # Polynomial expansion
    if poly_degree >= 2:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        theta_poly = poly.fit_transform(theta)
        names = [f"irt_{n.replace(' ', '_')}" for n in poly.get_feature_names_out(
            [f"t{i}" for i in range(k)]
        )]
    else:
        theta_poly = theta
        names = [f"irt_t{i}" for i in range(k)]

    # Standardize
    scaler = StandardScaler()
    theta_scaled = scaler.fit_transform(theta_poly)

    result = pd.DataFrame(theta_scaled, columns=names, index=df.index)
    result.insert(0, "model_name", df["model_name"].values if "model_name" in df.columns else range(len(df)))

    return result
