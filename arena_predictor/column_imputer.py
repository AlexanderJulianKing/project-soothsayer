"""
Per-Column Specialized Imputer

A drop-in replacement for GatedIterativeImputer that uses different models
for different column types, with dependency-aware ordering and native uncertainty.

Key features:
- Auto-classifies columns (categorical, linear, nonlinear, bounded)
- Assigns appropriate model per column type
- Dependency-aware ordering (easy columns first, organized into tiers)
- Native model uncertainty (no learned gate)
- Iterative refinement: multiple rounds over all tiers with tolerance relaxation
"""

from enum import Enum


def get_benchmark_suite(col: str) -> str:
    """Extract benchmark suite prefix from a column name.

    Used to identify same-suite columns in dependency graph output.
    Same-suite links (e.g. livebench_X → livebench_Y) appear in the
    imputer's predictor lists but are neutralized by the availability
    filter when the entire suite is missing for a model.
    """
    for prefix in ('livebench_', 'aa_eval_', 'aa_pricing_', 'lechmazur_',
                    'eqbench_', 'style_', 'logic_', 'writing_', 'eq_',
                    'tone_', 'arc_', 'contextarena_', 'aaomniscience_',
                    'aagdpval_', 'aacritpt_', 'weirdml_', 'yupp_',
                    'simplebench_', 'openbench_'):
        if col.startswith(prefix):
            return prefix.rstrip('_')
    return '_other'
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr, entropy, skew as compute_skew

from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Matern, WhiteKernel, ConstantKernel

from sklearn.utils.extmath import randomized_svd
from joblib import Parallel, delayed


class ColumnType(Enum):
    """Column model family for specialized imputation."""
    CATEGORICAL = "categorical"
    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    BOUNDED = "bounded"  # Legacy: kept for backward compat, treated as GP + bounded tag
    EXTRAPOLATION_PRONE = "extrapolation_prone"
    GP_LINEAR_MATERN = "gp_linear_matern"  # Safe default


class ColumnTags:
    """Distribution tags that control wrappers/transforms (independent of model family).

    Tags are stored as a set of strings on the classification metadata.
    They control how the target variable is transformed before fitting
    and how predictions are post-processed, without changing which base model
    is used.
    """
    BOUNDED = "bounded"              # Values in [L, U] — apply logit link
    FLOOR_INFLATED = "floor_inflated"  # Bimodal: cluster at floor + spread above — use hurdle
    CEILING_INFLATED = "ceiling_inflated"  # Mass at ceiling — future use


class CorrelationWeightedImputer:
    """
    Fill missing predictor values based on row's percentile profile, weighted by correlations.

    Instead of filling missing values with the population median (which causes regression
    to the mean for high-performing models), this imputer:
    1. Computes the percentile of each observed value in the row
    2. Weights those percentiles by R^2 correlation with the missing column
    3. Fills the missing value at the weighted-average percentile

    This preserves the "performance tier" of a model across benchmarks.
    """

    def __init__(self, correlation_matrix: pd.DataFrame, min_weight: float = 0.01):
        """
        Args:
            correlation_matrix: Precomputed correlation matrix for all columns.
            min_weight: Minimum R^2 weight to include a column (default 0.01 = corr 0.1).
        """
        self.correlation_matrix = correlation_matrix
        self.min_weight = min_weight
        self.column_values_: Dict[str, np.ndarray] = {}  # col -> sorted observed values
        self.column_medians_: Dict[str, float] = {}  # fallback medians

    def fit(self, X: pd.DataFrame) -> "CorrelationWeightedImputer":
        """Store sorted values for each column for percentile lookups."""
        for col in X.columns:
            observed = X[col].dropna().values
            if len(observed) > 0:
                self.column_values_[col] = np.sort(observed)
                self.column_medians_[col] = np.median(observed)
            else:
                self.column_values_[col] = np.array([0.0])
                self.column_medians_[col] = 0.0
        return self

    def transform(self, X: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        Fill missing values in feature_cols using correlation-weighted percentiles.

        Args:
            X: DataFrame with potential missing values.
            feature_cols: List of feature columns to fill.

        Returns:
            np.ndarray with missing values filled.
        """
        result = X[feature_cols].copy()

        for idx in result.index:
            row = result.loc[idx]
            missing_mask = row.isna()

            if not missing_mask.any():
                continue

            missing_cols = row.index[missing_mask].tolist()
            observed_cols = row.index[~missing_mask].tolist()

            for missing_col in missing_cols:
                if missing_col not in self.column_values_:
                    continue

                # Compute weighted percentile from observed columns
                weights = []
                percentiles = []

                for obs_col in observed_cols:
                    # Check if correlation exists
                    if (missing_col not in self.correlation_matrix.index or
                        obs_col not in self.correlation_matrix.columns):
                        continue

                    corr = self.correlation_matrix.loc[missing_col, obs_col]
                    if pd.isna(corr):
                        continue

                    weight = corr ** 2  # R^2 weighting
                    if weight < self.min_weight:
                        continue

                    # Compute percentile of this row's value in obs_col
                    val = row[obs_col]
                    if obs_col not in self.column_values_:
                        continue
                    obs_vals = self.column_values_[obs_col]
                    if len(obs_vals) == 0:
                        continue

                    # Percentile: fraction of values <= val
                    pct = np.searchsorted(obs_vals, val, side='right') / len(obs_vals)

                    weights.append(weight)
                    percentiles.append(pct)

                # Compute weighted average percentile
                if weights:
                    weighted_pct = np.average(percentiles, weights=weights)
                else:
                    weighted_pct = 0.5  # fallback to median

                # Fill at weighted percentile
                target_vals = self.column_values_[missing_col]
                if len(target_vals) == 0:
                    fill_val = self.column_medians_.get(missing_col, 0.0)
                else:
                    fill_idx = int(weighted_pct * (len(target_vals) - 1))
                    fill_idx = max(0, min(fill_idx, len(target_vals) - 1))
                    fill_val = target_vals[fill_idx]

                result.loc[idx, missing_col] = fill_val

        return result.values


class BaseColumnModel(ABC):
    """Abstract base class for all column models."""

    def __init__(self, feature_names: List[str], alpha: float = 0.1, seed: int = 42):
        self.feature_names = feature_names
        self.alpha = alpha
        self.seed = seed

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model on observed data."""
        pass

    @abstractmethod
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty.

        Returns:
            predictions: Predicted values
            half_widths: Half-width of prediction intervals
        """
        pass


class BayesianRidgeModel(BaseColumnModel):
    """Bayesian Ridge regression for LINEAR columns."""

    def __init__(
        self,
        feature_names: List[str],
        alpha: float = 0.1,
        seed: int = 42,
        correlation_matrix: Optional[pd.DataFrame] = None
    ):
        super().__init__(feature_names, alpha, seed)
        self.model = BayesianRidge(compute_score=False, fit_intercept=True)
        self.scaler = StandardScaler()
        self.correlation_matrix = correlation_matrix
        # Use CorrelationWeightedImputer if correlation matrix provided, else fallback to median
        if correlation_matrix is not None:
            self.imputer = CorrelationWeightedImputer(correlation_matrix)
        else:
            self.imputer = SimpleImputer(strategy='median')
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_obs = y.dropna()
        if len(y_obs) == 0:
            self.fitted_ = False
            return

        # Fit the imputer on ALL data (for percentile lookups)
        if isinstance(self.imputer, CorrelationWeightedImputer):
            self.imputer.fit(X[self.feature_names])
            X_obs = X.loc[y_obs.index, self.feature_names]
            X_prep = self.imputer.transform(X_obs, self.feature_names)
        else:
            X_obs = X.loc[y_obs.index, self.feature_names]
            X_prep = self.imputer.fit_transform(X_obs)

        X_scaled = self.scaler.fit_transform(X_prep)
        self.model.fit(X_scaled, y_obs)
        self.fitted_ = True

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted_:
            return np.zeros(len(X)), np.ones(len(X)) * 1e6

        if isinstance(self.imputer, CorrelationWeightedImputer):
            X_prep = self.imputer.transform(X, self.feature_names)
        else:
            X_prep = self.imputer.transform(X[self.feature_names])

        X_scaled = self.scaler.transform(X_prep)
        mean, std = self.model.predict(X_scaled, return_std=True)
        z = norm.ppf(1 - self.alpha / 2)
        half_width = z * std
        return mean, half_width


class GPModel(BaseColumnModel):
    """Gaussian Process for NONLINEAR, EXTRAPOLATION_PRONE, BOUNDED, and default columns."""

    def __init__(
        self,
        feature_names: List[str],
        kernel_type: str = "linear_matern",
        alpha: float = 0.1,
        seed: int = 42,
        bounds: Optional[Tuple[float, float]] = None,
        correlation_matrix: Optional[pd.DataFrame] = None
    ):
        super().__init__(feature_names, alpha, seed)
        self.kernel_type = kernel_type
        self.bounds = bounds
        self.correlation_matrix = correlation_matrix
        self.fitted_ = False

        # Build kernel
        if kernel_type == "matern":
            kernel = (
                ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
                + WhiteKernel(noise_level=1.0)
            )
        elif kernel_type == "linear_matern":
            # Proven kernel from lmsys_predictor5.py
            kernel = (
                ConstantKernel(1.0) * DotProduct(sigma_0=1.0)
                + ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
                + WhiteKernel(noise_level=1.0)
            )
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=seed,
            alpha=1e-10
        )
        self.scaler = StandardScaler()
        # Use CorrelationWeightedImputer if correlation matrix provided, else fallback to median
        if correlation_matrix is not None:
            self.imputer = CorrelationWeightedImputer(correlation_matrix)
        else:
            self.imputer = SimpleImputer(strategy='median')

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_obs = y.dropna()
        if len(y_obs) < 3:  # GP needs at least 3 points
            self.fitted_ = False
            return

        # Fit the imputer on ALL data (for percentile lookups)
        if isinstance(self.imputer, CorrelationWeightedImputer):
            self.imputer.fit(X[self.feature_names])
            X_obs = X.loc[y_obs.index, self.feature_names]
            X_prep = self.imputer.transform(X_obs, self.feature_names)
        else:
            X_obs = X.loc[y_obs.index, self.feature_names]
            X_prep = self.imputer.fit_transform(X_obs)

        X_scaled = self.scaler.fit_transform(X_prep)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                self.model.fit(X_scaled, y_obs)
                self.fitted_ = True
            except Exception:
                self.fitted_ = False

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted_:
            return np.zeros(len(X)), np.ones(len(X)) * 1e6

        if isinstance(self.imputer, CorrelationWeightedImputer):
            X_prep = self.imputer.transform(X, self.feature_names)
        else:
            X_prep = self.imputer.transform(X[self.feature_names])

        X_scaled = self.scaler.transform(X_prep)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            mean, std = self.model.predict(X_scaled, return_std=True)

        # Note: We intentionally do NOT clip to bounds here.
        # Even for "bounded" columns (e.g., 0-100 scores), a model may legitimately
        # score outside the observed range if it's at the extreme end.

        z = norm.ppf(1 - self.alpha / 2)
        half_width = z * std
        return mean, half_width


class CategoricalModel(BaseColumnModel):
    """Classification with entropy-based uncertainty for CATEGORICAL columns."""

    def __init__(self, feature_names: List[str], n_classes: int, alpha: float = 0.1, seed: int = 42):
        super().__init__(feature_names, alpha, seed)
        self.n_classes = n_classes
        if n_classes <= 2:
            self.model = LogisticRegression(random_state=seed, max_iter=1000)
        else:
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=1
            )
        self.imputer = SimpleImputer(strategy='median')
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_obs = y.dropna()
        if len(y_obs) == 0:
            self.fitted_ = False
            return

        X_obs = X.loc[y_obs.index, self.feature_names]
        X_prep = self.imputer.fit_transform(X_obs)
        self.model.fit(X_prep, y_obs)
        self.fitted_ = True

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted_:
            return np.zeros(len(X)), np.ones(len(X)) * 1e6

        X_prep = self.imputer.transform(X[self.feature_names])
        probas = self.model.predict_proba(X_prep)
        predictions = self.model.classes_[np.argmax(probas, axis=1)]

        # Shannon entropy as uncertainty
        uncertainties = entropy(probas.T)  # Entropy per sample
        max_entropy = np.log(len(self.model.classes_))
        normalized_uncertainty = uncertainties / max_entropy if max_entropy > 0 else uncertainties

        return predictions, normalized_uncertainty


class BoundedLinkModel(BaseColumnModel):
    """Wraps a base regression model with logit/sigmoid target transform.

    For columns with values in [L, U], transforms y to logit space:
        z = logit((y - L + eps) / (U - L + 2*eps))
    Fits the base model on z, then inverse-transforms predictions back.
    This expands compressed near-ceiling/floor bands so the model can
    see meaningful differences.
    """

    def __init__(
        self,
        base_model: BaseColumnModel,
        bounds: Tuple[float, float],
        n_obs: int = 100,
    ):
        super().__init__(base_model.feature_names, base_model.alpha, base_model.seed)
        self.base_model = base_model
        self.lower = float(bounds[0])
        self.upper = float(bounds[1])
        self.eps = max(1e-3, 0.5 / (n_obs + 1))
        self.range_ = self.upper - self.lower + 2 * self.eps
        self.fitted_ = False

    def _to_logit(self, y: np.ndarray) -> np.ndarray:
        """Transform y from [L, U] to logit space."""
        p = (y - self.lower + self.eps) / self.range_
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def _from_logit(self, z: np.ndarray) -> np.ndarray:
        """Transform z from logit space back to [L, U]."""
        p = 1 / (1 + np.exp(-z))
        return p * self.range_ + self.lower - self.eps

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_obs = y.dropna()
        if len(y_obs) == 0:
            self.fitted_ = False
            return
        y_logit = pd.Series(self._to_logit(y_obs.to_numpy(dtype=float)), index=y_obs.index)
        # Check for degenerate transforms (all values map to same logit)
        if y_logit.std() < 1e-8:
            self.fitted_ = False
            return
        self.base_model.fit(X, y_logit)
        self.fitted_ = getattr(self.base_model, 'fitted_', True)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted_:
            return np.zeros(len(X)), np.ones(len(X)) * 1e6
        z_mean, z_hw = self.base_model.predict_with_uncertainty(X)
        # Transform mean back through sigmoid
        y_mean = self._from_logit(z_mean)
        # Approximate half-width: use derivative of sigmoid at z_mean
        # dy/dz = p*(1-p) * range_, where p = sigmoid(z)
        p = 1 / (1 + np.exp(-z_mean))
        dydz = p * (1 - p) * self.range_
        y_hw = z_hw * np.maximum(dydz, 1e-8)
        return y_mean, y_hw


class HurdleModel(BaseColumnModel):
    """Two-stage model for floor-inflated (capability wall) distributions.

    Stage 1 (gate): LogisticRegression predicting P(above threshold)
    Stage 2 (value): Base regression model trained only on above-threshold data

    Prediction: E[y] = p * mu_capable + (1-p) * floor_mean
    This soft combination avoids discontinuities in the iterative imputation loop.
    """

    def __init__(
        self,
        feature_names: List[str],
        threshold: float,
        floor_mean: float,
        alpha: float = 0.1,
        seed: int = 42,
        correlation_matrix: Optional[pd.DataFrame] = None,
    ):
        super().__init__(feature_names, alpha, seed)
        self.threshold = threshold
        self.floor_mean = floor_mean
        self.correlation_matrix = correlation_matrix

        # Gate: logistic classifier
        self.gate_model = LogisticRegression(
            random_state=seed, max_iter=1000, class_weight='balanced'
        )
        self.gate_scaler = StandardScaler()
        self.gate_imputer = SimpleImputer(strategy='median')

        # Value: BayesianRidge on above-threshold data
        if correlation_matrix is not None:
            self.value_imputer = CorrelationWeightedImputer(correlation_matrix)
        else:
            self.value_imputer = SimpleImputer(strategy='median')
        self.value_model = BayesianRidge(compute_score=False, fit_intercept=True)
        self.value_scaler = StandardScaler()

        self.gate_fitted_ = False
        self.value_fitted_ = False
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_obs = y.dropna()
        if len(y_obs) < 10:
            self.fitted_ = False
            return

        # Binary labels: above threshold
        is_capable = (y_obs > self.threshold).astype(int)
        n_capable = is_capable.sum()
        n_floor = len(is_capable) - n_capable

        if n_capable < 5 or n_floor < 3:
            self.fitted_ = False
            return

        # Fit gate
        X_obs_gate = X.loc[y_obs.index, self.feature_names]
        X_gate_prep = self.gate_imputer.fit_transform(X_obs_gate)
        X_gate_scaled = self.gate_scaler.fit_transform(X_gate_prep)
        try:
            self.gate_model.fit(X_gate_scaled, is_capable)
            self.gate_fitted_ = True
        except Exception:
            self.fitted_ = False
            return

        # Fit value model on above-threshold data only
        capable_mask = y_obs > self.threshold
        y_capable = y_obs[capable_mask]
        if len(y_capable) < 5:
            self.fitted_ = False
            return

        if isinstance(self.value_imputer, CorrelationWeightedImputer):
            self.value_imputer.fit(X[self.feature_names])
            X_cap = X.loc[y_capable.index, self.feature_names]
            X_cap_prep = self.value_imputer.transform(X_cap, self.feature_names)
        else:
            X_cap = X.loc[y_capable.index, self.feature_names]
            X_cap_prep = self.value_imputer.fit_transform(X_cap)

        X_cap_scaled = self.value_scaler.fit_transform(X_cap_prep)
        self.value_model.fit(X_cap_scaled, y_capable)
        self.value_fitted_ = True
        self.fitted_ = self.gate_fitted_ and self.value_fitted_

        # Update floor_mean from actual data
        floor_vals = y_obs[~capable_mask]
        if len(floor_vals) > 0:
            self.floor_mean = float(floor_vals.mean())

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted_:
            return np.zeros(len(X)), np.ones(len(X)) * 1e6

        # Gate prediction: P(capable)
        X_gate_prep = self.gate_imputer.transform(X[self.feature_names])
        X_gate_scaled = self.gate_scaler.transform(X_gate_prep)
        p_capable = self.gate_model.predict_proba(X_gate_scaled)
        # Handle case where gate only saw one class
        if self.gate_model.classes_.shape[0] == 2:
            p_cap = p_capable[:, 1]  # P(class=1) = P(capable)
        else:
            p_cap = np.ones(len(X)) * 0.5

        # Value prediction: E[y | capable]
        if isinstance(self.value_imputer, CorrelationWeightedImputer):
            X_val_prep = self.value_imputer.transform(X, self.feature_names)
        else:
            X_val_prep = self.value_imputer.transform(X[self.feature_names])
        X_val_scaled = self.value_scaler.transform(X_val_prep)
        mu_cap, std_cap = self.value_model.predict(X_val_scaled, return_std=True)

        # Combined prediction: E[y] = p * mu_capable + (1-p) * floor_mean
        y_pred = p_cap * mu_cap + (1 - p_cap) * self.floor_mean

        # Mixture variance: Var = p*sigma^2 + p*(1-p)*(mu_cap - floor_mean)^2
        var = p_cap * std_cap**2 + p_cap * (1 - p_cap) * (mu_cap - self.floor_mean)**2
        std_total = np.sqrt(np.maximum(var, 1e-8))

        z = norm.ppf(1 - self.alpha / 2)
        half_width = z * std_total
        return y_pred, half_width


class ColumnClassifier:
    """Classifies columns into appropriate model types."""

    @staticmethod
    def _is_integer_like(series: pd.Series, tol: float = 1e-6) -> bool:
        """Return True when non-null values are all (near) integers."""
        vals = series.dropna().to_numpy()
        if vals.size == 0:
            return False
        finite = np.isfinite(vals)
        vals = vals[finite]
        if vals.size == 0:
            return False
        return np.all(np.abs(vals - np.round(vals)) <= tol)

    @staticmethod
    def _detect_floor_inflation(observed: pd.Series, min_val: float, max_val: float) -> Optional[Dict[str, Any]]:
        """Detect bimodal floor-cluster distributions (capability wall pattern).

        Looks for a cluster of values near the floor with a gap before the
        'capable' models.  Uses an adaptive threshold: the valley between
        the floor cluster and the rest, found via a simple histogram method.

        Returns metadata dict with threshold info if floor-inflated, else None.
        """
        if len(observed) < 20:
            return None

        vals = observed.to_numpy(dtype=float)
        data_range = max_val - min_val
        if data_range < 1e-8:
            return None

        # Normalize to [0, 1]
        normed = (vals - min_val) / data_range

        # Check if there's a cluster in the bottom 25% of the range
        bottom_quarter = (normed <= 0.25).sum()
        bottom_frac = bottom_quarter / len(vals)

        # Need at least 15% in the floor cluster and at least 20 above
        n_above_quarter = (normed > 0.25).sum()
        if bottom_frac < 0.15 or n_above_quarter < 10:
            return None

        # Find the valley: scan percentiles 10-40 to find the best split point
        # using the gap between the cluster and the rest
        best_gap = 0.0
        best_threshold_normed = 0.25
        sorted_normed = np.sort(normed)

        for pct in range(10, 45, 5):
            idx = int(pct / 100 * len(sorted_normed))
            if idx <= 0 or idx >= len(sorted_normed) - 10:
                continue
            # Gap = distance between this value and the next
            gap = sorted_normed[idx] - sorted_normed[idx - 1]
            # Also check: the cluster below must be at least 15% of data
            frac_below = idx / len(sorted_normed)
            if frac_below >= 0.15 and gap > best_gap:
                best_gap = gap
                best_threshold_normed = sorted_normed[idx]

        # Require a meaningful gap (at least 5% of range)
        if best_gap < 0.05:
            return None

        threshold = min_val + best_threshold_normed * data_range
        n_floor = int((vals <= threshold).sum())
        n_capable = int((vals > threshold).sum())

        # Need enough in both clusters
        if n_floor < 5 or n_capable < 10:
            return None

        floor_mean = float(vals[vals <= threshold].mean())
        capable_mean = float(vals[vals > threshold].mean())

        return {
            'threshold': threshold,
            'n_floor': n_floor,
            'n_capable': n_capable,
            'floor_mean': floor_mean,
            'capable_mean': capable_mean,
            'gap': best_gap * data_range,
        }

    @staticmethod
    def classify(
        series: pd.Series,
        predictors_df: pd.DataFrame,
        categorical_threshold: int = 10,
        force_categorical_cols: Optional[set] = None,
    ) -> Tuple[ColumnType, Dict[str, Any]]:
        """
        Classify column into model family + distribution tags.

        Classification logic:
        1. Cardinality ≤ 10 → CATEGORICAL
        2. Compute correlations (Pearson + Spearman) with all predictors
        3. Determine model family from correlation evidence:
           - Spearman >> Pearson → NONLINEAR (GP)
           - High missingness + low correlation → EXTRAPOLATION_PRONE (GP)
           - Otherwise → LINEAR (BayesianRidge)
        4. Add distribution tags independently:
           - Values in [0,1] or [0,100] → tag: bounded
           - Bimodal floor cluster detected → tag: floor_inflated
        """
        metadata = {}
        metadata['tags'] = set()
        observed = series.dropna()

        if force_categorical_cols and series.name in force_categorical_cols:
            metadata['n_classes'] = observed.nunique()
            return ColumnType.CATEGORICAL, metadata

        if len(observed) == 0:
            return ColumnType.GP_LINEAR_MATERN, metadata

        n_unique = observed.nunique()
        is_bool = pd.api.types.is_bool_dtype(series)
        is_str_like = pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series)
        is_integer_like = pd.api.types.is_integer_dtype(series) or ColumnClassifier._is_integer_like(observed)

        # Check 1: Categorical (explicit string/bool or low-cardinality integer-like)
        if is_bool or is_str_like:
            metadata['n_classes'] = n_unique
            return ColumnType.CATEGORICAL, metadata

        cardinality_limit = max(2, categorical_threshold)
        if n_unique <= cardinality_limit and is_integer_like:
            metadata['n_classes'] = n_unique
            return ColumnType.CATEGORICAL, metadata

        # --- Compute distribution tags (independent of model family) ---
        min_val, max_val = observed.min(), observed.max()

        # Tag: bounded
        is_bounded = (min_val >= 0 and max_val <= 1) or (min_val >= 0 and max_val <= 100)
        if is_bounded:
            metadata['tags'].add(ColumnTags.BOUNDED)
            metadata['bounds'] = (float(min_val), float(max_val))

        # Tag: floor_inflated (capability wall detection)
        floor_info = ColumnClassifier._detect_floor_inflation(observed, float(min_val), float(max_val))
        if floor_info is not None:
            metadata['tags'].add(ColumnTags.FLOOR_INFLATED)
            metadata['floor_info'] = floor_info

        # --- Determine model family from correlation evidence ---
        correlations = []
        for col in predictors_df.columns:
            if col != series.name:
                common_idx = observed.index.intersection(predictors_df[col].dropna().index)
                if len(common_idx) >= 20:
                    try:
                        pearson_r, _ = pearsonr(observed.loc[common_idx], predictors_df.loc[common_idx, col])
                        spearman_r, _ = spearmanr(observed.loc[common_idx], predictors_df.loc[common_idx, col])
                        correlations.append((abs(pearson_r), abs(spearman_r)))
                    except Exception:
                        continue

        if correlations:
            max_pearson = max(c[0] for c in correlations)
            max_spearman = max(c[1] for c in correlations)
            metadata['max_pearson'] = max_pearson
            metadata['max_spearman'] = max_spearman

            if max_spearman > max_pearson + 0.1:
                return ColumnType.NONLINEAR, metadata

            # Extrapolation-prone
            missingness = series.isna().mean()
            if missingness > 0.3 and max_pearson < 0.6:
                metadata['missingness'] = missingness
                return ColumnType.EXTRAPOLATION_PRONE, metadata

            return ColumnType.LINEAR, metadata

        # No correlations computed — safe fallback to GP
        # If bounded with no correlates, still use GP (legacy behavior)
        if is_bounded:
            return ColumnType.BOUNDED, metadata

        return ColumnType.GP_LINEAR_MATERN, metadata


class DependencyAnalyzer:
    """Analyzes column dependencies for optimal imputation order."""

    @staticmethod
    def compute_imputation_order(
        df: pd.DataFrame,
        columns: List[str],
        tier_quantiles: List[float] = [0.33, 0.67]
    ) -> List[Tuple[str, float, int]]:
        """
        Compute imputation order based on difficulty.

        Returns:
            List of (column, difficulty, tier) sorted by difficulty (easy first)

        Difficulty = missingness_pct × (1 - max_correlation_with_observed)
        Tiers: 1 (easy), 2 (medium), 3 (hard)
        """
        difficulties = {}

        for col in columns:
            missingness = df[col].isna().mean()
            observed_mask = df[col].notna()

            # Find max correlation with other columns
            correlations = []
            for other_col in columns:
                if other_col != col:
                    common_mask = observed_mask & df[other_col].notna()
                    if common_mask.sum() >= 20:
                        try:
                            corr = abs(df.loc[common_mask, col].corr(df.loc[common_mask, other_col]))
                            if not np.isnan(corr):
                                correlations.append(corr)
                        except Exception:
                            continue

            max_corr = max(correlations) if correlations else 0.0
            difficulty = missingness * (1 - max_corr)
            difficulties[col] = difficulty

        # Sort by difficulty
        sorted_cols = sorted(difficulties.items(), key=lambda x: x[1])

        # Assign tiers using quantiles
        difficulty_values = [d for _, d in sorted_cols]
        if len(difficulty_values) > 0:
            quantile_thresholds = [np.percentile(difficulty_values, q * 100) for q in tier_quantiles]
        else:
            quantile_thresholds = []

        result = []
        for col, diff in sorted_cols:
            tier = 1
            for threshold in quantile_thresholds:
                if diff > threshold:
                    tier += 1
            result.append((col, diff, tier))

        return result


class ModelFactory:
    """Creates appropriate model for each column type, with optional tag-based wrappers."""

    @staticmethod
    def create_model(
        column_type: ColumnType,
        feature_names: List[str],
        alpha: float = 0.1,
        seed: int = 42,
        metadata: Optional[Dict] = None,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> BaseColumnModel:
        """Create model instance based on column type and distribution tags.

        Model family is determined by column_type.
        Distribution tags in metadata['tags'] add wrappers:
          - FLOOR_INFLATED → HurdleModel (overrides base model entirely)
          - BOUNDED → BoundedLinkModel wrapper (if base model is linear)
        """
        metadata = metadata or {}
        tags = metadata.get('tags', set())

        # --- Categorical: no wrappers ---
        if column_type == ColumnType.CATEGORICAL:
            return CategoricalModel(feature_names, metadata.get('n_classes', 2), alpha, seed)

        # --- Floor-inflated: HurdleModel takes priority ---
        if ColumnTags.FLOOR_INFLATED in tags:
            floor_info = metadata.get('floor_info', {})
            threshold = floor_info.get('threshold', 0.0)
            floor_mean = floor_info.get('floor_mean', 0.0)
            return HurdleModel(
                feature_names, threshold, floor_mean,
                alpha=alpha, seed=seed,
                correlation_matrix=correlation_matrix,
            )

        # --- Build base model from family ---
        if column_type == ColumnType.LINEAR:
            base = BayesianRidgeModel(feature_names, alpha, seed, correlation_matrix=correlation_matrix)
        elif column_type == ColumnType.NONLINEAR:
            base = GPModel(feature_names, kernel_type="matern", alpha=alpha, seed=seed, correlation_matrix=correlation_matrix)
        elif column_type == ColumnType.BOUNDED:
            # Legacy path: bounded with no correlation evidence → GP
            bounds = metadata.get('bounds', (0, 1))
            base = GPModel(feature_names, kernel_type="linear_matern", alpha=alpha, seed=seed, bounds=bounds, correlation_matrix=correlation_matrix)
        elif column_type == ColumnType.EXTRAPOLATION_PRONE:
            base = GPModel(feature_names, kernel_type="linear_matern", alpha=alpha, seed=seed, correlation_matrix=correlation_matrix)
        else:  # GP_LINEAR_MATERN (default)
            base = GPModel(feature_names, kernel_type="linear_matern", alpha=alpha, seed=seed, correlation_matrix=correlation_matrix)

        # --- Apply bounded-link wrapper for LINEAR models with bounded tag ---
        if ColumnTags.BOUNDED in tags and column_type == ColumnType.LINEAR:
            bounds = metadata.get('bounds', (0.0, 1.0))
            data_range = bounds[1] - bounds[0]
            # Only apply logit transform if the range is non-trivial
            # and not too compressed (logit on [0.7, 0.9] is fine,
            # but on [0.0, 0.001] is pointless)
            if data_range > 0.01:
                n_obs = metadata.get('n_obs', 100)
                base = BoundedLinkModel(base, bounds, n_obs=n_obs)

        return base


def _iterative_svd_impute(X: np.ndarray, rank: int = 5, max_iter: int = 30,
                          tol: float = 1e-4) -> np.ndarray:
    """Low-rank matrix completion via iterative SVD (SoftImpute-style).

    Fills missing values by iteratively computing a rank-k SVD approximation.
    Observed values are preserved; only missing entries are updated each iteration.

    Args:
        X: 2D array with np.nan for missing entries.
        rank: Target rank for SVD approximation.
        max_iter: Maximum iterations.
        tol: Convergence tolerance (relative change in Frobenius norm of fills).

    Returns:
        Completed matrix (same shape as X).
    """
    mask = np.isnan(X)
    # Initialize missing with column medians
    col_medians = np.nanmedian(X, axis=0)
    # Handle all-NaN columns
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    Z = X.copy()
    Z[mask] = np.take(col_medians, np.where(mask)[1])

    effective_rank = min(rank, min(Z.shape) - 1)
    if effective_rank < 1:
        return Z

    prev_fills = Z[mask].copy()

    for _ in range(max_iter):
        U, s, Vt = randomized_svd(Z, n_components=effective_rank, random_state=42)
        Z_approx = U @ np.diag(s) @ Vt
        # Only update missing entries
        Z[mask] = Z_approx[mask]

        # Check convergence
        new_fills = Z[mask]
        change = np.linalg.norm(new_fills - prev_fills)
        scale = np.linalg.norm(new_fills) + 1e-10
        if change / scale < tol:
            break
        prev_fills = new_fills.copy()

    return Z


class SpecializedColumnImputer:
    """
    Drop-in replacement for GatedIterativeImputer with per-column specialization.

    Uses different models for different column types:
    - CATEGORICAL: LogisticRegression or RandomForest
    - LINEAR: BayesianRidge
    - NONLINEAR: GP with Matérn kernel
    - BOUNDED: GP with Linear+Matérn (clipped to bounds)
    - EXTRAPOLATION_PRONE: GP with Linear+Matérn
    - Default: GP with Linear+Matérn

    Features:
    - Dependency-aware ordering (3 tiers)
    - Native model uncertainty (no learned gate)
    - 3 passes instead of 14-30
    """

    def __init__(
        self,
        passes: int = 14,
        alpha: float = 0.1,
        seed: int = 42,
        verbose: int = 1,
        n_jobs: int = -1,
        use_feature_selector: bool = True,
        selector_tau: float = 0.8,
        selector_k_max: int = 30,
        gp_selector_k_max: int = 10,  # Feature cap for GP models (mRMR selection)
        tier_quantiles: List[float] = None,
        categorical_threshold: int = 10,
        force_categorical_cols: Optional[List[str]] = None,
        tolerance_percentile: float = 95.0,
        tolerance_relaxation_factor: float = 1.3,
        tolerance_multiplier: float = 3.0,
        # v7.2: Per-column tolerance calibration
        calibrate_tolerances: bool = False,
        calibration_target_rmse_ratio: float = 0.5,
        calibration_n_rounds: int = 3,
        calibration_holdout_frac: float = 0.2,
        recalibrate_every_n_passes: int = 0,
        skew_threshold: float = 2.0,
        **kwargs
    ):
        self.passes = passes
        self.alpha = alpha
        self.seed = seed
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.use_feature_selector = use_feature_selector
        self.selector_tau = selector_tau
        self.selector_k_max = selector_k_max
        self.gp_selector_k_max = gp_selector_k_max
        self.tier_quantiles = tier_quantiles if tier_quantiles is not None else [0.33, 0.67]
        self.categorical_threshold = categorical_threshold
        self.force_categorical_cols = set(force_categorical_cols or [])
        self.tolerance_percentile = tolerance_percentile
        self.tolerance_relaxation_factor = tolerance_relaxation_factor
        self.tolerance_multiplier = tolerance_multiplier
        # v7.2: Per-column tolerance calibration
        self.calibrate_tolerances = calibrate_tolerances
        self.calibration_target_rmse_ratio = calibration_target_rmse_ratio
        self.calibration_n_rounds = calibration_n_rounds
        self.calibration_holdout_frac = calibration_holdout_frac
        self.recalibrate_every_n_passes = recalibrate_every_n_passes
        self.skew_threshold = skew_threshold

        # API-compatible attributes (populated by fit_transform)
        self.models_: Dict[str, BaseColumnModel] = {}
        self.tolerances_: Dict[str, float] = {}
        self.logs_: List[Dict[str, Any]] = []
        self.predictors_map_: Dict[str, List[str]] = {}

        # New attributes specific to this imputer
        self.column_types_: Dict[str, ColumnType] = {}
        self.column_metadata_: Dict[str, Dict[str, Any]] = {}  # Classification metadata (tags, bounds, etc.)
        self.imputation_order_: List[Tuple[str, float, int]] = []
        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.spearman_matrix_: Optional[pd.DataFrame] = None  # For GP mRMR selection
        self.log_transforms_: Dict[str, Dict[str, Any]] = {}  # col -> {shift, skew_before}

    def fit_transform(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main imputation pipeline.

        Args:
            X_df: DataFrame with missing values

        Returns:
            DataFrame with imputed values
        """
        # Stage 0: Identify columns with missing values
        cols_to_impute = [col for col in X_df.columns if X_df[col].isna().any()]

        if len(cols_to_impute) == 0:
            if self.verbose:
                print("No missing values to impute")
            return X_df.copy()

        if self.verbose:
            print(f"Columns to impute: {len(cols_to_impute)}")

        # Don't modify caller's data
        X_df = X_df.copy()

        # Stage 0.5: Variance-stabilizing log transforms for highly skewed columns
        # Positively skewed columns (long right tail) benefit from log1p to linearize
        # relationships with other features, improving BayesianRidge fit quality.
        self.log_transforms_ = {}
        if self.skew_threshold > 0:
            numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
            n_transformed = 0
            for col in numeric_cols:
                observed = X_df[col].dropna()
                if len(observed) < 20:
                    continue
                try:
                    obs_vals = observed.to_numpy(dtype=float)
                except (ValueError, TypeError):
                    continue
                col_skew = float(compute_skew(obs_vals))
                if col_skew > self.skew_threshold:
                    min_val = float(observed.min())
                    shift = -min_val + 1.0 if min_val < 0 else 0.0
                    # Only transform if it actually improves linearity with other
                    # columns. Compare max |Pearson| before and after log1p.
                    log_obs = np.log1p(obs_vals + shift)
                    best_r_orig, best_r_log = 0.0, 0.0
                    for other_col in numeric_cols:
                        if other_col == col:
                            continue
                        other = X_df[other_col].dropna()
                        common_idx = observed.index.intersection(other.index)
                        if len(common_idx) < 20:
                            continue
                        try:
                            r_orig = abs(float(pearsonr(
                                observed.loc[common_idx].to_numpy(dtype=float),
                                other.loc[common_idx].to_numpy(dtype=float)
                            )[0]))
                            r_log = abs(float(pearsonr(
                                np.log1p(observed.loc[common_idx].to_numpy(dtype=float) + shift),
                                other.loc[common_idx].to_numpy(dtype=float)
                            )[0]))
                        except (ValueError, TypeError):
                            continue
                        best_r_orig = max(best_r_orig, r_orig)
                        best_r_log = max(best_r_log, r_log)
                    # Require meaningful improvement (>0.02) in max Pearson
                    if best_r_log <= best_r_orig + 0.02:
                        if self.verbose >= 2:
                            print(f"  Skipping {col}: skew={col_skew:.2f} but log doesn't improve "
                                  f"Pearson ({best_r_orig:.3f} -> {best_r_log:.3f})")
                        continue
                    self.log_transforms_[col] = {
                        'shift': shift,
                        'skew_before': col_skew,
                    }
                    X_df[col] = np.log1p(X_df[col] + shift)
                    n_transformed += 1
                    if self.verbose >= 2:
                        new_skew = float(compute_skew(X_df[col].dropna(), nan_policy='omit'))
                        print(f"  Log-transformed {col}: skew {col_skew:.2f} -> {new_skew:.2f} "
                              f"(Pearson {best_r_orig:.3f} -> {best_r_log:.3f})")
            if self.verbose and n_transformed > 0:
                print(f"Stage 0.5: Log-transformed {n_transformed} positively-skewed columns (threshold={self.skew_threshold:.1f})")

        # Save original missing mask BEFORE any filling (needed for SVD warm-start)
        self.original_missing_: Dict[str, pd.Series] = {}
        for col in cols_to_impute:
            self.original_missing_[col] = X_df[col].isna().copy()

        # Stage 1: Column Classification
        if self.verbose:
            print("Stage 1: Classifying columns...")

        n_hurdle = 0
        n_bounded_linear = 0
        for col in cols_to_impute:
            col_type, metadata = ColumnClassifier.classify(
                X_df[col],
                X_df,
                self.categorical_threshold,
                self.force_categorical_cols
            )
            self.column_types_[col] = col_type
            self.column_metadata_[col] = metadata
            tags = metadata.get('tags', set())
            if ColumnTags.FLOOR_INFLATED in tags:
                n_hurdle += 1
            if ColumnTags.BOUNDED in tags and col_type == ColumnType.LINEAR:
                n_bounded_linear += 1
            if self.verbose >= 2:
                tag_str = f" [{', '.join(sorted(tags))}]" if tags else ""
                print(f"  {col}: {col_type.value}{tag_str}")
        if self.verbose and (n_hurdle > 0 or n_bounded_linear > 0):
            parts = []
            if n_hurdle:
                parts.append(f"{n_hurdle} hurdle (floor-inflated)")
            if n_bounded_linear:
                parts.append(f"{n_bounded_linear} bounded-link (linear+bounded)")
            print(f"  New model types: {', '.join(parts)}")

        # Stage 2: Dependency Analysis
        if self.verbose:
            print("Stage 2: Computing imputation order...")

        self.imputation_order_ = DependencyAnalyzer.compute_imputation_order(
            X_df,
            cols_to_impute,
            self.tier_quantiles
        )

        if self.verbose >= 2:
            for col, diff, tier in self.imputation_order_[:5]:
                print(f"  {col}: difficulty={diff:.4f}, tier={tier}")

        # Stage 2.5: Precompute correlation matrices
        if self.verbose:
            print("Computing correlation matrices (Pearson + Spearman)...")
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        self.correlation_matrix_ = X_df[numeric_cols].corr(method='pearson')
        self.spearman_matrix_ = X_df[numeric_cols].corr(method='spearman')

        # Stage 2.75: Compute SVD anchor for warm-start with rank selection via
        # masked-cell cross-validation. Tries a small grid of ranks and picks the
        # one that best reconstructs held-out observed cells.
        numeric_cols_for_svd = X_df.select_dtypes(include=[np.number]).columns.tolist()
        self.anchor_df_ = None
        if len(numeric_cols_for_svd) >= 3 and len(cols_to_impute) > 0:
            X_mat = X_df[numeric_cols_for_svd].to_numpy(dtype=float, na_value=np.nan)
            col_means = np.nanmean(X_mat, axis=0)
            col_stds = np.nanstd(X_mat, axis=0)
            col_stds[col_stds < 1e-10] = 1.0
            X_standardized = (X_mat - col_means) / col_stds

            rank_max = min(12, len(numeric_cols_for_svd) // 3, X_mat.shape[0] // 5)
            rank_candidates = [r for r in [4, 6, 8, 10, 12] if 2 <= r <= rank_max]
            if not rank_candidates:
                rank_candidates = [max(2, rank_max)]

            if len(rank_candidates) > 1:
                # Masked-cell CV: hold out 10% of observed cells, measure reconstruction
                observed_mask = ~np.isnan(X_standardized)
                obs_rows, obs_cols_idx = np.where(observed_mask)
                n_obs = len(obs_rows)
                rng = np.random.RandomState(42)
                holdout_idx = rng.choice(n_obs, size=max(1, n_obs // 10), replace=False)
                holdout_rows = obs_rows[holdout_idx]
                holdout_cols = obs_cols_idx[holdout_idx]
                holdout_vals = X_standardized[holdout_rows, holdout_cols]

                X_masked = X_standardized.copy()
                X_masked[holdout_rows, holdout_cols] = np.nan

                best_rank = rank_candidates[0]
                best_err = float('inf')
                for r in rank_candidates:
                    Z_trial = _iterative_svd_impute(X_masked, rank=r)
                    recon = Z_trial[holdout_rows, holdout_cols]
                    err = float(np.sqrt(np.mean((recon - holdout_vals) ** 2)))
                    if err < best_err:
                        best_err = err
                        best_rank = r
                svd_rank = best_rank
            else:
                svd_rank = rank_candidates[0]

            Z_std = _iterative_svd_impute(X_standardized, rank=svd_rank)
            Z = Z_std * col_stds + col_means

            self.anchor_df_ = pd.DataFrame(Z, index=X_df.index, columns=numeric_cols_for_svd)

            # Extract SVD row factors (U×S) for downstream use as ALT features
            U, s, Vt = randomized_svd(Z_std, n_components=svd_rank, random_state=42)
            self.svd_row_factors_ = pd.DataFrame(
                U * s[np.newaxis, :],
                index=X_df.index,
                columns=[f"_svd_f{i+1}" for i in range(svd_rank)],
            )

            if self.verbose:
                rank_info = f"rank={svd_rank}" if len(rank_candidates) <= 1 else f"rank={svd_rank} selected from {rank_candidates}"
                print(f"Stage 2.75: SVD anchor computed ({rank_info})")

        # Stage 3: Feature Selection & Model Fitting
        if self.verbose:
            print("Stage 3: Fitting models...")

        def fit_column_model(col, tier, difficulty):
            # Feature selection - use mRMR for GP models, regular for others
            col_type = self.column_types_[col]
            col_metadata = self.column_metadata_.get(col, {})
            tags = col_metadata.get('tags', set())

            # Determine if this is a GP-based model (for feature selection strategy)
            # Floor-inflated uses linear gate + value model, so use linear selection
            is_hurdle = ColumnTags.FLOOR_INFLATED in tags
            is_gp_model = (not is_hurdle) and col_type in (
                ColumnType.NONLINEAR,
                ColumnType.BOUNDED,
                ColumnType.EXTRAPOLATION_PRONE,
                ColumnType.GP_LINEAR_MATERN
            )

            if self.use_feature_selector:
                if is_gp_model:
                    # Use mRMR with Spearman for GP models (handles collinearity better)
                    predictors = self._select_predictors_mrmr(X_df, col, k=self.gp_selector_k_max)
                else:
                    # Use regular correlation-based selection for linear models
                    predictors = self._select_predictors(X_df, col)
            else:
                predictors = [c for c in X_df.columns if c != col]

            # Ensure at least one predictor
            if len(predictors) == 0:
                predictors = [c for c in X_df.columns if c != col]

            self.predictors_map_[col] = predictors

            # Build metadata for ModelFactory (use stored classification metadata)
            metadata = dict(col_metadata)  # Copy to avoid mutation
            if col_type == ColumnType.CATEGORICAL:
                metadata['n_classes'] = X_df[col].dropna().nunique()
            # Add n_obs for BoundedLinkModel eps calculation
            metadata['n_obs'] = int(X_df[col].notna().sum())

            model = ModelFactory.create_model(
                col_type, predictors, self.alpha, self.seed, metadata,
                correlation_matrix=self.correlation_matrix_
            )
            model.fit(X_df, X_df[col])

            # Compute initial tolerance (configurable percentile of training uncertainty)
            # Multiplier accounts for missing rows having higher uncertainty than observed rows
            X_obs = X_df.loc[X_df[col].notna()]
            if len(X_obs) > 0:
                _, uncertainties = model.predict_with_uncertainty(X_obs)
                tolerance = np.percentile(uncertainties, self.tolerance_percentile) * self.tolerance_multiplier
            else:
                tolerance = 1e6  # Very high tolerance for empty columns

            return col, model, tolerance

        if self.n_jobs == 1:
            results = [
                fit_column_model(col, tier, diff)
                for col, diff, tier in self.imputation_order_
            ]
        else:
            results = Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(fit_column_model)(col, tier, diff)
                for col, diff, tier in self.imputation_order_
            )

        for col, model, tolerance in results:
            self.models_[col] = model
            self.tolerances_[col] = tolerance  # Initial tolerance from model fitting

        # v7.2: Run per-column calibration if enabled
        cols_to_impute = [col for col, _, _ in self.imputation_order_]
        if self.calibrate_tolerances:
            if self.verbose:
                print("Running per-column tolerance calibration...")
            self._run_calibration(X_df, cols_to_impute)

        # Stage 4: Iterative Imputation (multiple rounds over all tiers)
        if self.verbose:
            print(f"Stage 4: Iterative imputation ({self.passes} rounds)...")

        # Initialize current_df: fill missing with SVD anchor (warm-start) or keep NaN
        current_df = X_df.copy()
        if self.anchor_df_ is not None:
            n_svd_filled = 0
            for col in cols_to_impute:
                if col in self.anchor_df_.columns:
                    missing_mask = self.original_missing_.get(col, current_df[col].isna())
                    if missing_mask.any():
                        # Ensure column is float-compatible (handles Int64 nullable)
                        if hasattr(current_df[col].dtype, 'numpy_dtype'):
                            current_df[col] = current_df[col].astype(float)
                        current_df.loc[missing_mask, col] = self.anchor_df_.loc[missing_mask, col].values
                        n_svd_filled += int(missing_mask.sum())
            if self.verbose:
                print(f"  SVD warm-start: {n_svd_filled} cells initialized")
        tiers = sorted(set(tier for _, _, tier in self.imputation_order_))
        global_pass = 0

        for round_idx in range(1, self.passes + 1):
            round_writes = 0

            if self.verbose:
                print(f"  Round {round_idx}/{self.passes}")

            for tier in tiers:
                tier_cols = [col for col, _, t in self.imputation_order_ if t == tier]

                if not tier_cols:
                    continue

                # Check if any columns in this tier have originally-missing values
                has_missing = any(
                    self.original_missing_.get(col, current_df[col].isna()).any()
                    for col in tier_cols
                )
                if not has_missing:
                    continue

                global_pass += 1

                if self.verbose >= 2:
                    print(f"    Tier {tier}: {len(tier_cols)} columns")

                snapshot_df = current_df.copy()  # Jacobi-style

                def impute_column(col, pass_num=global_pass):
                    writes = []
                    # Use original missing mask if available (SVD pre-filled cells
                    # still need refinement by per-column models)
                    missing_mask = self.original_missing_.get(col, snapshot_df[col].isna())
                    if not missing_mask.any():
                        return col, writes

                    X_missing = snapshot_df.loc[missing_mask]
                    preds, uncertainties = self.models_[col].predict_with_uncertainty(X_missing)

                    # If tolerance_multiplier is very large, skip gating entirely
                    skip_gating = self.tolerance_multiplier >= 1e6
                    tolerance = self.tolerances_[col]

                    # Collect all predictions with uncertainties
                    all_preds = list(zip(X_missing.index, preds, uncertainties))

                    if skip_gating:
                        # No gating - write everything
                        for idx, pred, unc in all_preds:
                            writes.append({
                                'col': col,
                                'row': idx,
                                'y_pred': pred,
                                'h_blend': unc,
                                'pass_num': pass_num
                            })
                    else:
                        # Greedy best-first strategy
                        passed = [(idx, pred, unc) for idx, pred, unc in all_preds if unc <= tolerance]
                        rejected = [(idx, pred, unc) for idx, pred, unc in all_preds if unc > tolerance]

                        if passed:
                            # Write all that passed tolerance
                            for idx, pred, unc in passed:
                                writes.append({
                                    'col': col,
                                    'row': idx,
                                    'y_pred': pred,
                                    'h_blend': unc,
                                    'pass_num': pass_num
                                })
                        elif rejected:
                            # Nothing passed - take the single best (lowest uncertainty)
                            best_idx, best_pred, best_unc = min(rejected, key=lambda x: x[2])
                            writes.append({
                                'col': col,
                                'row': best_idx,
                                'y_pred': best_pred,
                                'h_blend': best_unc,
                                'pass_num': pass_num
                            })

                    return col, writes

                if self.n_jobs == 1:
                    results = [impute_column(col) for col in tier_cols]
                else:
                    results = Parallel(n_jobs=self.n_jobs, backend='threading')(
                        delayed(impute_column)(col) for col in tier_cols
                    )

                # Apply writes
                tier_writes = 0
                for col, writes in results:
                    for write in writes:
                        current_df.loc[write['row'], col] = write['y_pred']
                        self.logs_.append(write)
                    tier_writes += len(writes)

                round_writes += tier_writes

                if self.verbose >= 2:
                    print(f"      Wrote {tier_writes} cells")

                # Relax tolerances if no writes for this tier (configurable increase)
                if tier_writes == 0:
                    for col in tier_cols:
                        self.tolerances_[col] *= self.tolerance_relaxation_factor
                    if self.verbose >= 2:
                        pct_increase = int((self.tolerance_relaxation_factor - 1) * 100)
                        print(f"      No writes, relaxing tolerances by {pct_increase}%")

            if self.verbose:
                print(f"    Round {round_idx} total: {round_writes} cells written")

            # v7.2: Periodic recalibration
            if (self.calibrate_tolerances and
                self.recalibrate_every_n_passes > 0 and
                round_idx % self.recalibrate_every_n_passes == 0 and
                round_writes > 0):  # Only recalibrate if we made progress
                if self.verbose >= 1:
                    print(f"    Recalibrating tolerances at round {round_idx}...")
                self._run_calibration(current_df, cols_to_impute)

            # Early exit if no progress in this round
            if round_writes == 0:
                if self.verbose:
                    print(f"    No progress in round {round_idx}, stopping early")
                break

        # Stage 5: Fallback (fill remaining with median)
        for col in cols_to_impute:
            if current_df[col].isna().any():
                median = current_df[col].median()
                n_remaining = current_df[col].isna().sum()
                current_df[col] = current_df[col].fillna(median)
                if self.verbose >= 2:
                    print(f"  Filled {n_remaining} remaining cells in {col} with median={median:.4f}")

        # Stage 6: Inverse log transforms (back to original scale)
        if self.log_transforms_:
            for col, info in self.log_transforms_.items():
                if col in current_df.columns:
                    current_df[col] = np.expm1(current_df[col]) - info['shift']
            if self.verbose:
                print(f"Stage 6: Inverse-transformed {len(self.log_transforms_)} columns back to original scale")

        if self.verbose:
            print(f"Imputation complete. Total writes: {len(self.logs_)}")

        # Stage 7: Imputation trajectory — per-row divergence from SVD anchor
        self.trajectory_features_ = None
        if self.anchor_df_ is not None and self.original_missing_:
            traj_mean_delta = pd.Series(0.0, index=current_df.index, dtype=float)
            traj_max_delta = pd.Series(0.0, index=current_df.index, dtype=float)
            for col, missing_mask in self.original_missing_.items():
                if col in self.anchor_df_.columns and missing_mask.any():
                    delta = (current_df.loc[missing_mask, col].values -
                             self.anchor_df_.loc[missing_mask, col].values)
                    abs_delta = np.abs(delta)
                    for i, idx in enumerate(missing_mask[missing_mask].index):
                        traj_mean_delta.loc[idx] += abs_delta[i]
                        if abs_delta[i] > traj_max_delta.loc[idx]:
                            traj_max_delta.loc[idx] = abs_delta[i]
            # Normalize by number of imputed columns per row
            n_imputed = pd.Series(0, index=current_df.index, dtype=float)
            for col, mask in self.original_missing_.items():
                n_imputed.loc[mask[mask].index] += 1
            n_imputed = n_imputed.clip(lower=1)
            traj_mean_delta = traj_mean_delta / n_imputed
            self.trajectory_features_ = pd.DataFrame({
                '_traj_mean_delta': traj_mean_delta,
                '_traj_max_delta': traj_max_delta,
                '_traj_n_imputed': n_imputed,
            }, index=current_df.index)

        return current_df

    def get_imputation_importance(self) -> pd.DataFrame:
        """Extract per-column feature importances from fitted imputation models.

        For each imputed column, reports which predictor columns contribute
        most to its imputation. Uses model-specific importance extraction:
        - BayesianRidge / BoundedLinkModel: |standardized coefficients|
        - HurdleModel: combined gate + value coefficients
        - GP: not extractable (returns equal weights)
        - Categorical: not extractable (returns equal weights)

        Returns:
            DataFrame with columns: [target_col, predictor_col, importance,
            model_type, rank]. Importances are normalized to sum to 1.0
            per target column.
        """
        records = []
        for col, model in self.models_.items():
            predictors = self.predictors_map_.get(col, [])
            if not predictors:
                continue

            importances = self._extract_model_importance(model, predictors)
            model_type = type(model).__name__

            # Normalize to sum to 1
            total = sum(importances.values())
            if total < 1e-12:
                # Equal weights fallback
                n = len(predictors)
                importances = {p: 1.0 / n for p in predictors}
                total = 1.0

            sorted_preds = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            for rank, (pred, imp) in enumerate(sorted_preds, 1):
                records.append({
                    'target_col': col,
                    'predictor_col': pred,
                    'importance': imp / total,
                    'model_type': model_type,
                    'rank': rank,
                })

        return pd.DataFrame(records)

    @staticmethod
    def _extract_model_importance(model: BaseColumnModel, predictors: List[str]) -> Dict[str, float]:
        """Extract feature importance from a fitted model."""
        n = len(predictors)
        equal = {p: 1.0 / n for p in predictors}

        if isinstance(model, BoundedLinkModel):
            # Unwrap to get base model coefficients
            return SpecializedColumnImputer._extract_model_importance(model.base_model, predictors)

        if isinstance(model, HurdleModel):
            imp = {}
            # Gate coefficients
            if model.gate_fitted_ and hasattr(model.gate_model, 'coef_'):
                gate_coefs = np.abs(model.gate_model.coef_.ravel())
                if len(gate_coefs) == n:
                    for i, p in enumerate(predictors):
                        imp[p] = imp.get(p, 0) + float(gate_coefs[i])
            # Value coefficients
            if model.value_fitted_ and hasattr(model.value_model, 'coef_'):
                val_coefs = np.abs(model.value_model.coef_.ravel())
                if len(val_coefs) == n:
                    for i, p in enumerate(predictors):
                        imp[p] = imp.get(p, 0) + float(val_coefs[i])
            return imp if imp else equal

        if isinstance(model, BayesianRidgeModel):
            if model.fitted_ and hasattr(model.model, 'coef_'):
                coefs = np.abs(model.model.coef_.ravel())
                if len(coefs) == n:
                    return {p: float(coefs[i]) for i, p in enumerate(predictors)}
            return equal

        # GP and Categorical: no extractable importances
        return equal

    def _select_predictors(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """Correlation-based feature selection with co-missingness awareness.

        Predictors that are always missing when the target is missing are
        useless for imputation — they provide no information for the rows
        that actually need filling.  We discount correlation by the fraction
        of target-missing rows where the predictor is available.
        """
        target_missing = df[target_col].isna()
        n_missing = target_missing.sum()

        correlations = []
        for col in df.columns:
            if col != target_col:
                common_mask = df[target_col].notna() & df[col].notna()
                if common_mask.sum() >= 20:
                    try:
                        corr = abs(df.loc[common_mask, target_col].corr(df.loc[common_mask, col]))
                        if not np.isnan(corr):
                            # Compute availability: fraction of target-missing rows
                            # where this predictor is NOT missing
                            if n_missing > 0:
                                pred_available = (~df[col].isna() & target_missing).sum()
                                availability = pred_available / n_missing
                            else:
                                availability = 1.0
                            # Effective score: correlation * availability
                            # A predictor with r=0.9 but 0% availability is useless
                            # A predictor with r=0.7 but 100% availability is much better
                            eff_score = corr * availability
                            correlations.append((col, eff_score, corr, availability))
                    except Exception:
                        continue

        # Sort by effective score, keep top k_max
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected = [col for col, _, _, _ in correlations[:self.selector_k_max]]

        return selected

    def _select_predictors_mrmr(
        self,
        df: pd.DataFrame,
        target_col: str,
        k: Optional[int] = None,
        min_relevance: float = 0.1
    ) -> List[str]:
        """
        mRMR (minimum Redundancy Maximum Relevance) feature selection using Spearman correlation.

        Selects features that are:
        1. Highly correlated with target (relevance)
        2. Not highly correlated with already-selected features (low redundancy)

        Uses Spearman (rank) correlation to capture nonlinear monotonic relationships,
        which is appropriate for GP models on nonlinear data.

        Args:
            df: DataFrame with features and target.
            target_col: Name of the target column.
            k: Maximum number of features to select. Defaults to self.gp_selector_k_max.
            min_relevance: Minimum |correlation| with target to consider a feature.

        Returns:
            List of selected feature column names.
        """
        if k is None:
            k = self.gp_selector_k_max
        if self.spearman_matrix_ is None:
            # Fallback to regular selection if no Spearman matrix
            return self._select_predictors(df, target_col)

        # Get candidate features (exclude target)
        candidates = [c for c in df.columns if c != target_col and c in self.spearman_matrix_.columns]

        if not candidates:
            return []

        # Precompute availability for co-missingness awareness
        target_missing = df[target_col].isna()
        n_missing = target_missing.sum()
        availability = {}
        for col in candidates:
            if n_missing > 0:
                pred_available = (~df[col].isna() & target_missing).sum()
                availability[col] = pred_available / n_missing
            else:
                availability[col] = 1.0

        # Compute relevance: |spearman(feature, target)| * availability
        relevance = {}
        for col in candidates:
            if target_col in self.spearman_matrix_.index and col in self.spearman_matrix_.columns:
                corr = self.spearman_matrix_.loc[target_col, col]
                if not pd.isna(corr) and abs(corr) >= min_relevance:
                    # Discount by availability: a perfectly correlated predictor
                    # that's always missing when the target is missing scores 0
                    relevance[col] = abs(corr) * availability.get(col, 1.0)

        if not relevance:
            # No features pass minimum relevance threshold
            return self._select_predictors(df, target_col)

        selected = []
        remaining = set(relevance.keys())

        while len(selected) < k and remaining:
            best_score = -np.inf
            best_col = None

            for col in remaining:
                rel = relevance[col]

                # Compute redundancy: average |spearman(col, selected_features)|
                if selected:
                    redundancies = []
                    for sel_col in selected:
                        if col in self.spearman_matrix_.index and sel_col in self.spearman_matrix_.columns:
                            r = self.spearman_matrix_.loc[col, sel_col]
                            if not pd.isna(r):
                                redundancies.append(abs(r))
                    redundancy = np.mean(redundancies) if redundancies else 0.0
                else:
                    redundancy = 0.0

                # mRMR score: relevance - redundancy
                score = rel - redundancy

                if score > best_score:
                    best_score = score
                    best_col = col

            if best_col is not None:
                selected.append(best_col)
                remaining.remove(best_col)
            else:
                break

        return selected

    def _compute_knn_ratio(
        self,
        X_train: np.ndarray,
        x_query: np.ndarray,
        avail_idx: np.ndarray,
        k: int = 5
    ) -> float:
        """
        Compute kNN distance ratio for extrapolation detection.

        Returns ratio of (distance to k-th neighbor) / (median pairwise distance in training).
        Higher values indicate the query point is far from training data.
        """
        if len(avail_idx) == 0 or X_train.shape[0] < k:
            return 0.0

        # Use only available features
        X_sub = X_train[:, avail_idx]
        q_sub = x_query[avail_idx]

        # Handle NaNs in query
        valid_mask = ~np.isnan(q_sub)
        if valid_mask.sum() < 2:
            return 0.0

        X_sub = X_sub[:, valid_mask]
        q_sub = q_sub[valid_mask]

        # Compute distances to query
        dists = np.linalg.norm(X_sub - q_sub, axis=1)
        dists = dists[~np.isnan(dists)]

        if len(dists) < k:
            return 0.0

        # k-th nearest neighbor distance
        knn_dist = np.partition(dists, k - 1)[k - 1]

        # Median pairwise distance in training (sample for efficiency)
        n = min(50, X_sub.shape[0])
        sample_idx = np.random.choice(X_sub.shape[0], size=n, replace=False)
        X_sample = X_sub[sample_idx]
        pairwise = np.linalg.norm(X_sample[:, None] - X_sample[None, :], axis=2)
        np.fill_diagonal(pairwise, np.nan)
        med_dist = np.nanmedian(pairwise)

        if med_dist < 1e-8:
            return 0.0

        return float(knn_dist / med_dist)

    # =========================================================================
    # v7.2: Per-column tolerance calibration
    # =========================================================================

    def _compute_global_tolerance(self, df: pd.DataFrame, col: str) -> float:
        """
        Compute tolerance using global tolerance_percentile (original 7.1 behavior).

        Args:
            df: DataFrame with data.
            col: Target column name.

        Returns:
            Tolerance threshold for this column.
        """
        if col not in self.models_:
            return 1e6

        model = self.models_[col]
        X_obs = df.loc[df[col].notna()]

        if len(X_obs) == 0:
            return 1e6

        _, uncertainties = model.predict_with_uncertainty(X_obs)
        tolerance = np.percentile(uncertainties, self.tolerance_percentile) * self.tolerance_multiplier
        return tolerance

    def _calibrate_column_tolerance(self, df: pd.DataFrame, col: str) -> Optional[float]:
        """
        Find uncertainty threshold for column where predictions are reliable.

        Uses masked evaluation: hold out known values, predict them, find threshold
        where RMSE is acceptable (< column_std * calibration_target_rmse_ratio).

        Args:
            df: DataFrame with current data state.
            col: Column to calibrate.

        Returns:
            Calibrated tolerance threshold, or None if calibration fails.
        """
        if col not in self.models_:
            return None

        model = self.models_[col]
        known_mask = df[col].notna()
        n_known = known_mask.sum()

        if n_known < 10:
            # Not enough data to calibrate
            return None

        errors_and_uncertainties = []
        rng = np.random.RandomState(self.seed)

        for round_idx in range(self.calibration_n_rounds):
            # Hold out a fraction of known values
            known_indices = df.index[known_mask].tolist()
            n_holdout = max(1, int(len(known_indices) * self.calibration_holdout_frac))
            rng.shuffle(known_indices)
            holdout_indices = known_indices[:n_holdout]

            if len(holdout_indices) == 0:
                continue

            # Get the predictors for this column
            predictors = self.predictors_map_.get(col, [])
            if not predictors:
                continue

            # Build feature matrix for holdout samples
            X_holdout = df.loc[holdout_indices, predictors]
            y_true = df.loc[holdout_indices, col].values

            # Handle missing values in holdout predictors
            if X_holdout.isna().any().any():
                # Convert to float to avoid Int64 dtype issues, then fill with median
                X_holdout = X_holdout.astype(float).fillna(X_holdout.median())

            try:
                preds, uncertainties = model.predict_with_uncertainty(X_holdout)
                errors = np.abs(preds - y_true)

                for err, unc in zip(errors, uncertainties):
                    if np.isfinite(err) and np.isfinite(unc):
                        errors_and_uncertainties.append((float(err), float(unc)))
            except Exception:
                continue

        if len(errors_and_uncertainties) < 5:
            # Not enough samples to calibrate reliably
            return None

        # Sort by uncertainty (ascending)
        errors_and_uncertainties.sort(key=lambda x: x[1])

        # Compute target RMSE
        col_std = df[col].std()
        if col_std < 1e-8:
            return None
        target_rmse = col_std * self.calibration_target_rmse_ratio

        # Find the largest uncertainty threshold where cumulative RMSE <= target
        best_threshold = None
        for i in range(1, len(errors_and_uncertainties) + 1):
            subset_errors = [e for e, u in errors_and_uncertainties[:i]]
            rmse = np.sqrt(np.mean(np.array(subset_errors) ** 2))

            if rmse <= target_rmse:
                best_threshold = errors_and_uncertainties[i - 1][1]
            else:
                # Once RMSE exceeds target, stop
                break

        return best_threshold

    def _run_calibration(self, df: pd.DataFrame, columns: List[str]) -> None:
        """
        Run calibration for all columns, falling back to global tolerance on failure.

        Args:
            df: Current DataFrame state.
            columns: Columns to calibrate.
        """
        calibrated_count = 0
        fallback_count = 0

        for col in columns:
            calibrated = self._calibrate_column_tolerance(df, col)
            if calibrated is not None:
                self.tolerances_[col] = calibrated
                calibrated_count += 1
            else:
                # Fallback to global tolerance_percentile
                self.tolerances_[col] = self._compute_global_tolerance(df, col)
                fallback_count += 1

        if self.verbose >= 1:
            print(f"  Tolerance calibration: {calibrated_count} calibrated, {fallback_count} fallback to global")

    def evaluate_quality(
        self,
        X_df: pd.DataFrame,
        n_rounds: int = 5,
        frac: float = 0.2,
        random_state: Optional[int] = None,
        knn_bins: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, np.inf),
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Monte Carlo mask-eval on observed cells to estimate imputation error/coverage.

        For each column with a fitted model, samples a fraction of observed (non-missing)
        cells, masks them, predicts using the model, and compares to ground truth.

        Parameters
        ----------
        X_df : pd.DataFrame
            The fully imputed DataFrame (after fit_transform).
        n_rounds : int
            Number of Monte Carlo rounds per column.
        frac : float
            Fraction of observed cells to probe per round.
        random_state : int, optional
            Random seed for reproducibility.
        knn_bins : tuple of floats
            Bin edges for grouping by kNN ratio (extrapolation severity).

        Returns
        -------
        per_cell : pd.DataFrame
            One row per probed cell: [col, row, y_true, y_pred, abs_err, within,
            h_blend, log_knn_ratio].
        per_col : pd.DataFrame
            Aggregated metrics per column: n, mae, rmse, coverage, med_h, sd_y, rmse_over_sd.
        by_bin : pd.DataFrame
            Metrics binned by kNN ratio across all columns.
        """
        if not self.models_:
            raise RuntimeError("Call fit_transform before evaluate_quality.")

        rng = np.random.default_rng(self.seed if random_state is None else random_state)
        records: List[Dict[str, Any]] = []

        for col, model in self.models_.items():
            predictors = self.predictors_map_.get(col, [])
            if not predictors:
                continue

            # Predictor matrix
            X_pred = X_df[predictors]
            obs_mask = ~X_df[col].isna().values
            idx_obs = np.where(obs_mask)[0]

            if idx_obs.size < 5:
                continue

            m = max(1, int(frac * idx_obs.size))

            # Precompute training rows for kNN
            X_train_vals = X_pred.loc[obs_mask].to_numpy(dtype=float, na_value=np.nan)

            for _ in range(n_rounds):
                probe = rng.choice(idx_obs, size=m, replace=False)
                for ridx in probe:
                    row_pred_vals = X_pred.iloc[ridx].to_numpy(dtype=float, na_value=np.nan)
                    avail_idx = np.where(~np.isnan(row_pred_vals))[0]

                    # Extrapolation metric
                    knn_ratio = self._compute_knn_ratio(
                        X_train_vals, row_pred_vals, avail_idx, k=5
                    )
                    log_knn_ratio = float(np.log1p(max(knn_ratio, 1e-8)))

                    # Predict with uncertainty
                    X_row = X_df.loc[[X_df.index[ridx]], predictors]
                    try:
                        preds, uncertainties = model.predict_with_uncertainty(X_row)
                        y_pred = float(preds[0])
                        h_blend = float(uncertainties[0])
                    except Exception:
                        # Fallback for models that fail
                        continue

                    y_true = float(X_df.iloc[ridx][col])
                    abs_err = abs(y_true - y_pred)
                    within = float(abs_err <= h_blend) if h_blend > 0 else 0.0

                    records.append({
                        "col": col,
                        "row": int(ridx),
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "abs_err": float(abs_err),
                        "within": within,
                        "h_blend": h_blend,
                        "log_knn_ratio": log_knn_ratio,
                    })

        per_cell = pd.DataFrame.from_records(records)
        if per_cell.empty:
            if self.verbose:
                print("Warning: evaluate_quality produced no records")
            return per_cell, pd.DataFrame(), pd.DataFrame()

        # Per-column aggregation
        def _rmse(x: pd.Series) -> float:
            return float(np.sqrt(np.mean(np.square(x))))

        per_col = (
            per_cell.groupby("col")
            .agg(
                n=("row", "count"),
                mae=("abs_err", "mean"),
                rmse=("abs_err", _rmse),
                coverage=("within", "mean"),
                med_h=("h_blend", "median"),
                med_log_knn=("log_knn_ratio", "median"),
            )
            .reset_index()
        )

        # Add normalization: RMSE / std(y) per column
        sd_map = {c: float(np.nanstd(X_df[c].to_numpy(dtype=float, na_value=np.nan))) for c in per_col["col"]}
        per_col["sd_y"] = per_col["col"].map(sd_map)
        per_col["rmse_over_sd"] = per_col["rmse"] / per_col["sd_y"].astype(float).clip(lower=1e-8)

        # Bin by kNN ratio (extrapolation severity)
        bins = np.array(knn_bins, dtype=float)
        labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins) - 1)]
        cut = pd.cut(
            np.expm1(per_cell["log_knn_ratio"]),
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        )
        tmp_eval = per_cell.assign(knn_bin=cut)

        # Aggregate by bin
        by_bin = (
            tmp_eval
            .groupby("knn_bin", observed=True)
            .agg(
                n=("row", "count"),
                mae=("abs_err", "mean"),
                rmse=("abs_err", _rmse),
                coverage=("within", "mean"),
            )
            .reset_index()
        )

        # Bin-wise normalization
        sd_by_bin = tmp_eval.groupby("knn_bin", observed=True)["y_true"].std(ddof=0)
        by_bin["sd_y"] = by_bin["knn_bin"].map(sd_by_bin).astype(float)
        by_bin["rmse_over_sd"] = by_bin["rmse"] / by_bin["sd_y"].astype(float).clip(lower=1e-8)

        if self.verbose:
            print(f"evaluate_quality: {len(per_cell)} cells probed across {len(per_col)} columns")

        return per_cell, per_col, by_bin


    def evaluate_quality_oof(
        self,
        X_df: pd.DataFrame,
        n_splits: int = 5,
        n_rounds: int = 1,
        frac: float = 0.2,
        random_state: Optional[int] = None,
        knn_bins: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, np.inf),
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Out-of-fold mask-eval on observed cells to estimate imputation error/coverage.

        Fits a fresh per-column model on each training fold and evaluates
        on held-out rows. Uses the already-imputed feature matrix for predictors,
        so some leakage remains (other columns may encode information about the target).
        """
        if not self.models_:
            raise RuntimeError("Call fit_transform before evaluate_quality.")
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2 for OOF evaluation.")

        # Apply the same log transforms used during fit_transform
        orig_X_df = X_df  # Keep reference for original-scale sd computation
        if self.log_transforms_:
            X_df = X_df.copy()
            for col, info in self.log_transforms_.items():
                if col in X_df.columns:
                    X_df[col] = np.log1p(X_df[col] + info['shift'])

        rng = np.random.default_rng(self.seed if random_state is None else random_state)
        records: List[Dict[str, Any]] = []

        corr_matrix = self.correlation_matrix_
        if corr_matrix is None:
            num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
            corr_matrix = X_df[num_cols].corr(method="pearson")

        for col in self.models_.keys():
            predictors = self.predictors_map_.get(col, [])
            if not predictors:
                continue

            obs_mask = ~X_df[col].isna().values
            idx_obs = np.where(obs_mask)[0]
            if idx_obs.size < max(5, n_splits * 2):
                continue

            col_type = self.column_types_.get(col)
            # Use stored metadata from classification (includes tags, bounds, floor_info)
            metadata = dict(self.column_metadata_.get(col, {}))
            if col_type is None:
                col_type, metadata = ColumnClassifier.classify(
                    X_df[col],
                    X_df,
                    self.categorical_threshold,
                    self.force_categorical_cols,
                )
            else:
                if col_type == ColumnType.CATEGORICAL:
                    metadata["n_classes"] = int(X_df[col].dropna().nunique())
                elif col_type == ColumnType.BOUNDED and 'bounds' not in metadata:
                    obs = X_df[col].dropna()
                    if len(obs):
                        metadata["bounds"] = (float(obs.min()), float(obs.max()))
            metadata['n_obs'] = int(X_df[col].notna().sum())

            kf = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.seed if random_state is None else random_state,
            )
            X_pred = X_df[predictors]

            for tr_pos, va_pos in kf.split(idx_obs):
                tr_idx = idx_obs[tr_pos]
                va_idx = idx_obs[va_pos]
                if tr_idx.size < 3 or va_idx.size == 0:
                    continue

                model = ModelFactory.create_model(
                    col_type,
                    predictors,
                    self.alpha,
                    self.seed,
                    metadata,
                    correlation_matrix=corr_matrix,
                )
                X_train = X_df.iloc[tr_idx]
                y_train = X_df.iloc[tr_idx][col]
                model.fit(X_train, y_train)

                X_train_vals = X_pred.iloc[tr_idx].to_numpy(dtype=float, na_value=np.nan)

                m = max(1, int(frac * len(va_idx)))
                m = min(m, len(va_idx))
                rounds = max(1, int(n_rounds))
                for _ in range(rounds):
                    probe = rng.choice(va_idx, size=m, replace=False)
                    for ridx in probe:
                        row_pred_vals = X_pred.iloc[ridx].to_numpy(dtype=float, na_value=np.nan)
                        avail_idx = np.where(~np.isnan(row_pred_vals))[0]

                        knn_ratio = self._compute_knn_ratio(
                            X_train_vals, row_pred_vals, avail_idx, k=5
                        )
                        log_knn_ratio = float(np.log1p(max(knn_ratio, 1e-8)))

                        X_row = X_df.loc[[X_df.index[ridx]], predictors]
                        try:
                            preds, uncertainties = model.predict_with_uncertainty(X_row)
                            y_pred = float(preds[0])
                            h_blend = float(uncertainties[0])
                        except Exception:
                            continue

                        y_true = float(X_df.iloc[ridx][col])
                        abs_err = abs(y_true - y_pred)
                        within = float(abs_err <= h_blend) if h_blend > 0 else 0.0

                        records.append({
                            "col": col,
                            "row": int(ridx),
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "abs_err": float(abs_err),
                            "within": within,
                            "h_blend": h_blend,
                            "log_knn_ratio": log_knn_ratio,
                        })

        # Inverse-transform y_true/y_pred for log-transformed columns so metrics
        # are in original scale (RMSE in original units, RMSE/SD uses original SD).
        if self.log_transforms_ and records:
            for rec in records:
                col = rec['col']
                if col in self.log_transforms_:
                    info = self.log_transforms_[col]
                    rec['y_true'] = float(np.expm1(rec['y_true'])) - info['shift']
                    rec['y_pred'] = float(np.expm1(rec['y_pred'])) - info['shift']
                    rec['abs_err'] = abs(rec['y_true'] - rec['y_pred'])
                    # Coverage: keep log-space value (multiplicative interval is more
                    # appropriate for heavy-tailed variables)

        per_cell = pd.DataFrame.from_records(records)
        if per_cell.empty:
            if self.verbose:
                print("Warning: evaluate_quality_oof produced no records")
            return per_cell, pd.DataFrame(), pd.DataFrame()

        # Per-column aggregation
        def _rmse(x: pd.Series) -> float:
            return float(np.sqrt(np.mean(np.square(x))))

        per_col = (
            per_cell.groupby("col")
            .agg(
                n=("row", "count"),
                mae=("abs_err", "mean"),
                rmse=("abs_err", _rmse),
                coverage=("within", "mean"),
                med_h=("h_blend", "median"),
                med_log_knn=("log_knn_ratio", "median"),
            )
            .reset_index()
        )

        # Add normalization: RMSE / std(y) per column — use original-scale data
        sd_map = {c: float(np.nanstd(orig_X_df[c].to_numpy(dtype=float, na_value=np.nan))) for c in per_col["col"]}
        per_col["sd_y"] = per_col["col"].map(sd_map)
        per_col["rmse_over_sd"] = per_col["rmse"] / per_col["sd_y"].astype(float).clip(lower=1e-8)

        # Bin by kNN ratio (extrapolation severity)
        bins = np.array(knn_bins, dtype=float)
        labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins) - 1)]
        cut = pd.cut(
            np.expm1(per_cell["log_knn_ratio"]),
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        )
        tmp_eval = per_cell.assign(knn_bin=cut)

        # Aggregate by bin
        by_bin = (
            tmp_eval
            .groupby("knn_bin", observed=True)
            .agg(
                n=("row", "count"),
                mae=("abs_err", "mean"),
                rmse=("abs_err", _rmse),
                coverage=("within", "mean"),
            )
            .reset_index()
        )

        # Bin-wise normalization
        sd_by_bin = tmp_eval.groupby("knn_bin", observed=True)["y_true"].std(ddof=0)
        by_bin["sd_y"] = by_bin["knn_bin"].map(sd_by_bin).astype(float)
        by_bin["rmse_over_sd"] = by_bin["rmse"] / by_bin["sd_y"].astype(float).clip(lower=1e-8)

        if self.verbose:
            print(f"evaluate_quality_oof: {len(per_cell)} cells probed across {len(per_col)} columns")

        return per_cell, per_col, by_bin


# =============================================================================
# ModelBankImputer — Per-Cell Predictor Selection with Uncertainty Tracking
# =============================================================================


@dataclass
class FittedCellModel:
    """Cached fitted model for a (column, predictor_subset) combination."""
    predictor_names: List[str]
    sigma2_loo: float
    n_train: int
    model_kind: str  # 'ridge', 'bounded_ridge', 'hurdle', 'categorical'
    _model: Any = field(default=None, repr=False)
    _scaler: Any = field(default=None, repr=False)
    # Original-space coefficients for σ² propagation in pass 2
    _coefficients: Optional[np.ndarray] = field(default=None, repr=False)
    # Bounded model extras
    _lower: float = 0.0
    _upper: float = 1.0
    _logit_eps: float = 1e-3
    _logit_range: float = 1.0
    # Hurdle model extras
    _gate_model: Any = field(default=None, repr=False)
    _gate_scaler: Any = field(default=None, repr=False)
    _threshold: float = 0.0
    _floor_mean: float = 0.0


class ModelBankImputer:
    """Per-cell predictor selection imputer with uncertainty tracking.

    Drop-in replacement for SpecializedColumnImputer.  Each missing cell uses
    the best model compatible with what that specific row actually has observed,
    tracks uncertainty per cell, and only uses imputed values as predictors
    when they are confident enough.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        seed: int = 42,
        verbose: int = 1,
        n_jobs: int = -1,
        selector_k_max: int = 30,
        categorical_threshold: int = 10,
        force_categorical_cols: Optional[List[str]] = None,
        skew_threshold: float = 2.0,
        confidence_threshold: float = 0.4,
        min_support: int = 10,
        coherence_lambda: float = 1.0,
        coherence_shape: str = "linear",
        coherence_gate: str = "fixed",
        predictor_selection: str = "corr",
        iterative_coherence: bool = False,
        use_svd_predictors: bool = False,
        n_expansion_passes: int = 1,
        max_confident_extras: int = 1,
        **kwargs,
    ):
        self.alpha = alpha
        self.seed = seed
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.selector_k_max = selector_k_max
        self.categorical_threshold = categorical_threshold
        self.force_categorical_cols = set(force_categorical_cols or [])
        self.skew_threshold = skew_threshold
        self.confidence_threshold = confidence_threshold
        self.min_support = min_support
        self.coherence_lambda = coherence_lambda
        self.coherence_shape = coherence_shape
        self.coherence_gate = coherence_gate
        self.predictor_selection = predictor_selection
        self.iterative_coherence = iterative_coherence
        self.use_svd_predictors = use_svd_predictors
        self.n_expansion_passes = n_expansion_passes
        self.max_confident_extras = max_confident_extras

        # ---- populated by fit_transform ----
        self.model_bank_: Dict[Tuple[str, frozenset], Optional[FittedCellModel]] = {}
        self.sigma2_matrix_: Optional[pd.DataFrame] = None
        self.cell_predictors_: Dict[Tuple, List[str]] = {}
        self.logs_: List[Dict[str, Any]] = []
        # API-compatible attributes (representative per-column)
        self.models_: Dict[str, BaseColumnModel] = {}
        self.predictors_map_: Dict[str, List[str]] = {}
        self.predictor_freqs_: Dict[str, Dict[str, int]] = {}
        self.svd_row_factors_: Optional[pd.DataFrame] = None
        self.trajectory_features_: Optional[pd.DataFrame] = None
        self.column_types_: Dict[str, ColumnType] = {}
        self.column_metadata_: Dict[str, Dict[str, Any]] = {}
        self.log_transforms_: Dict[str, Dict[str, Any]] = {}
        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.spearman_matrix_: Optional[pd.DataFrame] = None
        self.original_missing_: Dict[str, pd.Series] = {}
        self._candidate_rankings: Dict[str, List[Tuple[str, float, int]]] = {}
        self._loo_forward_selections: Dict[str, List[str]] = {}
        self._col_sd: Dict[str, float] = {}
        self._svd_rank: Optional[int] = None
        self._svd_anchor: Optional[np.ndarray] = None
        self._svd_anchor_std: Optional[np.ndarray] = None
        self._svd_anchor_cols: Optional[List[str]] = None
        self._svd_anchor_means: Optional[np.ndarray] = None
        self._svd_anchor_stds: Optional[np.ndarray] = None
        self._coherence_gate_model = None

    # --------------------------------------------------------------------- #
    #  fit_transform — main entry point                                      #
    # --------------------------------------------------------------------- #
    def fit_transform(self, X_df: pd.DataFrame) -> pd.DataFrame:
        cols_to_impute = [c for c in X_df.columns if X_df[c].isna().any()]
        if not cols_to_impute:
            if self.verbose:
                print("No missing values to impute")
            return X_df.copy()

        X_df = X_df.copy()

        if self.verbose:
            print(f"ModelBankImputer: {len(cols_to_impute)} columns to impute")

        # ---- Phase 0: preprocessing ----
        self._apply_log_transforms(X_df)
        for col in cols_to_impute:
            self.original_missing_[col] = X_df[col].isna().copy()
        self._classify_columns(X_df, cols_to_impute)
        self._compute_correlations(X_df)
        self._compute_svd_factors(X_df, cols_to_impute)

        # Inject SVD row factors as candidate predictors (fully observed, no NaNs)
        self._svd_predictor_cols = []
        if self.svd_row_factors_ is not None and self.use_svd_predictors:
            for fc in self.svd_row_factors_.columns:
                X_df[fc] = self.svd_row_factors_[fc].values
                self._svd_predictor_cols.append(fc)
            if self.verbose:
                print(f"  Added {len(self._svd_predictor_cols)} SVD factors as predictors")

        # Store column SDs for normalization
        for col in cols_to_impute:
            obs = X_df[col].dropna()
            self._col_sd[col] = float(obs.std()) if len(obs) > 1 else 1.0

        # ---- Phase 1: per-column candidate rankings ----
        self._build_candidate_rankings(X_df, cols_to_impute)
        if self.predictor_selection == "loo_forward":
            self._build_loo_forward_selections(X_df, cols_to_impute)

        # ---- Phase 2: pass 1 — observed-only imputation ----
        self.sigma2_matrix_ = pd.DataFrame(
            np.inf, index=X_df.index, columns=X_df.columns, dtype=float
        )
        # Mark observed cells as 0 uncertainty
        for col in X_df.columns:
            observed_mask = X_df[col].notna()
            self.sigma2_matrix_.loc[observed_mask, col] = 0.0

        self._pass1_observed_only(X_df, cols_to_impute)

        # ---- Phase 3: expansion passes — uncertainty-gated ----
        if self.iterative_coherence:
            # Iterative: pass1 → coherence → pass2 → coherence
            self._fill_median_fallback(X_df, cols_to_impute)
            self._coherence_projection(X_df, cols_to_impute)
            for pass_num in range(self.n_expansion_passes):
                self._pass2_expansion(X_df, cols_to_impute)
            self._coherence_projection(X_df, cols_to_impute)
        else:
            # Standard: pass2 → fallback → coherence
            for pass_num in range(self.n_expansion_passes):
                self._pass2_expansion(X_df, cols_to_impute)
            self._fill_median_fallback(X_df, cols_to_impute)
            self._coherence_projection(X_df, cols_to_impute)

        # Trajectory features (computed in log space, before inverse transforms,
        # so delta-to-anchor and σ² normalization use consistent units)
        self._compute_trajectory_features(X_df, cols_to_impute)

        # ---- Phase 5: post-processing ----
        # Inverse log transforms
        if self.log_transforms_:
            for col, info in self.log_transforms_.items():
                if col in X_df.columns:
                    X_df[col] = np.expm1(X_df[col]) - info['shift']
            if self.verbose:
                print(f"Inverse-transformed {len(self.log_transforms_)} columns")

        # Representative models for API compatibility
        self._build_representative_models(X_df, cols_to_impute)

        # Remove SVD predictor columns from output (predict.py adds them separately)
        if self._svd_predictor_cols:
            X_df = X_df.drop(columns=self._svd_predictor_cols, errors='ignore')

        if self.verbose:
            print(f"ModelBankImputer complete. {len(self.logs_)} cell writes, "
                  f"{len(self.model_bank_)} cached models")

        return X_df

    # --------------------------------------------------------------------- #
    #  Phase 0 helpers                                                       #
    # --------------------------------------------------------------------- #
    def _apply_log_transforms(self, X_df: pd.DataFrame) -> None:
        """Variance-stabilizing log transforms for highly skewed columns."""
        if self.skew_threshold <= 0:
            return
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        n_transformed = 0
        for col in numeric_cols:
            observed = X_df[col].dropna()
            if len(observed) < 20:
                continue
            try:
                obs_vals = observed.to_numpy(dtype=float)
            except (ValueError, TypeError):
                continue
            col_skew = float(compute_skew(obs_vals))
            if col_skew <= self.skew_threshold:
                continue
            min_val = float(observed.min())
            shift = -min_val + 1.0 if min_val < 0 else 0.0
            log_obs = np.log1p(obs_vals + shift)
            best_r_orig, best_r_log = 0.0, 0.0
            for other_col in numeric_cols:
                if other_col == col:
                    continue
                other = X_df[other_col].dropna()
                common_idx = observed.index.intersection(other.index)
                if len(common_idx) < 20:
                    continue
                try:
                    r_orig = abs(float(pearsonr(
                        observed.loc[common_idx].to_numpy(dtype=float),
                        other.loc[common_idx].to_numpy(dtype=float),
                    )[0]))
                    r_log = abs(float(pearsonr(
                        np.log1p(observed.loc[common_idx].to_numpy(dtype=float) + shift),
                        other.loc[common_idx].to_numpy(dtype=float),
                    )[0]))
                except (ValueError, TypeError):
                    continue
                best_r_orig = max(best_r_orig, r_orig)
                best_r_log = max(best_r_log, r_log)
            if best_r_log <= best_r_orig + 0.02:
                continue
            self.log_transforms_[col] = {'shift': shift, 'skew_before': col_skew}
            X_df[col] = np.log1p(X_df[col] + shift)
            n_transformed += 1
        if self.verbose and n_transformed > 0:
            print(f"  Log-transformed {n_transformed} skewed columns")

    def _classify_columns(self, X_df: pd.DataFrame, cols: List[str]) -> None:
        for col in cols:
            col_type, metadata = ColumnClassifier.classify(
                X_df[col], X_df, self.categorical_threshold,
                self.force_categorical_cols,
            )
            self.column_types_[col] = col_type
            self.column_metadata_[col] = metadata
            if self.verbose >= 2:
                tags = metadata.get('tags', set())
                tag_str = f" [{', '.join(sorted(tags))}]" if tags else ""
                print(f"  {col}: {col_type.value}{tag_str}")

    def _fill_median_fallback(self, X_df: pd.DataFrame, cols_to_impute: List[str]) -> None:
        """Fill any remaining NaN cells with column median."""
        for col in cols_to_impute:
            still_missing = X_df[col].isna()
            if still_missing.any():
                median = X_df[col].median()
                if pd.isna(median):
                    median = 0.0
                n_fill = int(still_missing.sum())
                X_df.loc[still_missing, col] = median
                col_var = self._col_sd.get(col, 1.0) ** 2
                self.sigma2_matrix_.loc[still_missing, col] = col_var
                if self.verbose >= 2:
                    print(f"  Fallback: {n_fill} cells in {col} filled with median")

    def _compute_correlations(self, X_df: pd.DataFrame) -> None:
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        self.correlation_matrix_ = X_df[numeric_cols].corr(method='pearson')
        self.spearman_matrix_ = X_df[numeric_cols].corr(method='spearman')
        if self.verbose:
            print("  Correlation matrices computed")

    # --------------------------------------------------------------------- #
    #  Phase 4b: low-rank coherence projection                               #
    # --------------------------------------------------------------------- #
    def _learn_coherence_gate(
        self,
        X_df: pd.DataFrame,
        cols_to_impute: List[str],
        X_lowrank: np.ndarray,
        col_idx: Dict[str, int],
        col_sds: np.ndarray,
        sigma2_svd_col: Dict[str, float],
    ):
        """Train a Ridge model to predict optimal coherence weight per cell.

        Uses observed cells in columns that have missing data as training data.
        For each observed cell, we know the true value, the SVD reconstruction,
        and we can estimate the model-bank LOO error from sigma2_loo.  The
        optimal weight minimises (w*svd + (1-w)*mb - true)^2.

        Features: tau, row_completeness, col_completeness, svd_quality, |gap|/sd.
        """
        from sklearn.linear_model import Ridge

        feat_rows = []
        target_rows = []

        # Precompute row completeness
        n_cols = len(X_df.columns)
        row_obs_frac = X_df.notna().sum(axis=1).values / max(n_cols, 1)

        for c in cols_to_impute:
            if c not in col_idx:
                continue
            j = col_idx[c]
            miss_ser = self.original_missing_.get(c)
            if miss_ser is None:
                continue
            obs_mask = ~miss_ser.values
            n_obs = obs_mask.sum()
            if n_obs < 5:
                continue

            sd = col_sds[j]
            sd2 = sd ** 2
            s2_svd = sigma2_svd_col.get(c, sd2)
            col_comp = n_obs / len(X_df)

            # For observed cells: true value and SVD reconstruction are known.
            # Model-bank sigma2_loo gives the expected squared error.
            # Optimal w per column: Bayesian = s2_model / (s2_model + s2_svd)
            # We refine this with per-cell features.
            true_vals = X_df[c].values[obs_mask]
            svd_vals = X_lowrank[obs_mask, j]
            svd_err2 = (true_vals - svd_vals) ** 2

            # Average model-bank sigma2 for this column (from fitted models)
            col_sigma2s = [
                m.sigma2_loo for key, m in self.model_bank_.items()
                if m is not None and key[0] == c
            ]
            mb_sigma2 = float(np.median(col_sigma2s)) if col_sigma2s else sd2

            # Per-cell optimal w: minimize (w*svd + (1-w)*true - true)^2
            # = (w*(svd-true))^2 → w*=0 trivially for observed cells.
            # Instead, use cross-validated target: w that balances mb vs svd error.
            # w_opt = mb_sigma2 / (mb_sigma2 + svd_err2) per cell
            w_opt = mb_sigma2 / (mb_sigma2 + svd_err2 + 1e-10)
            w_opt = np.clip(w_opt, 0.0, 1.0)

            # Features for each observed cell
            tau_obs = mb_sigma2 / max(sd2, 1e-10)
            for i_local, i_global in enumerate(np.where(obs_mask)[0]):
                feat_rows.append([
                    tau_obs,
                    row_obs_frac[i_global],
                    col_comp,
                    s2_svd / max(sd2, 1e-10),
                    abs(true_vals[i_local] - svd_vals[i_local]) / max(sd, 1e-10),
                ])
                target_rows.append(w_opt[i_local])

        if len(feat_rows) < 20:
            self._coherence_gate_model = None
            return

        X_gate = np.array(feat_rows)
        y_gate = np.array(target_rows)
        gate_model = Ridge(alpha=1.0)
        gate_model.fit(X_gate, y_gate)
        self._coherence_gate_model = gate_model

        if self.verbose:
            print(f"  Learned coherence gate: {len(feat_rows)} training cells, "
                  f"coefs={np.round(gate_model.coef_, 3)}")

    def _coherence_projection(self, X_df: pd.DataFrame, cols_to_impute: List[str]) -> None:
        """Blend imputed values toward a low-rank SVD reconstruction.

        Re-computes SVD on the completed matrix (post-imputation) for the
        coherence target.  The observed-only SVD anchor is stored separately
        for trajectory features.

        High-uncertainty cells are pulled more toward the SVD estimate (which
        preserves cross-column structure), while confident cells keep their
        per-cell model-bank predictions.  Observed cells are never touched.
        """
        if self._svd_rank is None or self.coherence_lambda <= 0:
            return

        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = {c for c, t in self.column_types_.items()
                    if t == ColumnType.CATEGORICAL}
        proj_cols = [c for c in numeric_cols if c not in cat_cols]
        if len(proj_cols) < 3:
            return

        col_idx = {c: i for i, c in enumerate(proj_cols)}

        # Standardize the completed matrix
        X_mat = X_df[proj_cols].to_numpy(dtype=float)
        col_means = X_mat.mean(axis=0)
        col_sds = np.array([self._col_sd.get(c, 1.0) for c in proj_cols])
        col_sds = np.where(col_sds < 1e-10, 1.0, col_sds)
        X_std = (X_mat - col_means) / col_sds

        # Rank-k SVD reconstruction
        k = self._svd_rank
        U, s, Vt = randomized_svd(X_std, n_components=k, random_state=self.seed)
        X_lowrank_std = U @ np.diag(s) @ Vt
        X_lowrank = X_lowrank_std * col_sds + col_means

        # Per-column SVD residual variance (on observed cells)
        sigma2_svd_col = {}
        for c in proj_cols:
            j = col_idx[c]
            if c in self.original_missing_:
                obs_mask = ~self.original_missing_[c].values
            else:
                obs_mask = np.ones(X_mat.shape[0], dtype=bool)
            if obs_mask.sum() > 0:
                resid = X_std[obs_mask, j] - X_lowrank_std[obs_mask, j]
                sigma2_svd_col[c] = float(np.mean(resid ** 2)) * (col_sds[j] ** 2)
            else:
                sigma2_svd_col[c] = col_sds[j] ** 2

        # Train learned gate if requested
        gate = self.coherence_gate
        if gate == "learned":
            self._learn_coherence_gate(
                X_df, cols_to_impute, X_lowrank, col_idx, col_sds, sigma2_svd_col
            )

        # Precompute row completeness for adaptive gates
        n_total_cols = len(X_df.columns)
        row_obs_frac = X_df.notna().sum(axis=1).values / max(n_total_cols, 1)

        # Blend imputed cells toward low-rank estimate, weighted by uncertainty
        lam_base = self.coherence_lambda
        n_adjusted = 0
        for c in cols_to_impute:
            if c not in col_idx or c in cat_cols:
                continue
            j = col_idx[c]
            miss_mask = self.original_missing_.get(c, pd.Series(False, index=X_df.index))
            miss_idx = miss_mask[miss_mask].index
            if len(miss_idx) == 0:
                continue

            sd2 = col_sds[j] ** 2
            s2_svd = sigma2_svd_col.get(c, sd2)

            sigma2_vals = self.sigma2_matrix_.loc[miss_idx, c].values
            finite = np.isfinite(sigma2_vals) & (sigma2_vals > 0)
            if not finite.any():
                continue

            active_idx = miss_idx[finite]
            s2 = sigma2_vals[finite]
            tau = s2 / max(sd2, 1e-10)

            row_positions = np.array([X_df.index.get_loc(idx) for idx in active_idx])
            old_vals = X_df.loc[active_idx, c].values
            svd_vals = X_lowrank[row_positions, j]

            if gate == "learned" and self._coherence_gate_model is not None:
                # Learned gate: predict w from features
                n_obs_col = (~self.original_missing_.get(c, pd.Series(False, index=X_df.index))).sum()
                col_comp = n_obs_col / len(X_df)
                gate_feats = np.column_stack([
                    tau,
                    row_obs_frac[row_positions],
                    np.full(len(tau), col_comp),
                    np.full(len(tau), s2_svd / max(sd2, 1e-10)),
                    np.abs(old_vals - svd_vals) / max(col_sds[j], 1e-10),
                ])
                w = np.clip(self._coherence_gate_model.predict(gate_feats), 0.0, 1.0)
            elif gate == "row_adaptive":
                # Scale lambda by row completeness² — sparse rows get more SVD pull
                row_comp = row_obs_frac[row_positions]
                lam_adj = lam_base * row_comp ** 2
                w = tau / (tau + lam_adj)
            else:
                # Fixed gate: use shape-based formula
                lam = lam_base
                shape = self.coherence_shape
                if shape == "squared":
                    w = tau ** 2 / (tau ** 2 + lam)
                elif shape == "power3":
                    w = tau ** 3 / (tau ** 3 + lam)
                elif shape == "exp":
                    w = 1.0 - np.exp(-tau / lam)
                elif shape == "step":
                    w = np.where(tau > lam, 1.0, 0.0)
                else:  # "linear" (default James-Stein)
                    w = tau / (tau + lam)

            X_df.loc[active_idx, c] = (1.0 - w) * old_vals + w * svd_vals
            self.sigma2_matrix_.loc[active_idx, c] = (
                (1.0 - w) ** 2 * s2 + w ** 2 * s2_svd
            )
            n_adjusted += len(active_idx)

        if self.verbose:
            print(f"  Coherence projection: {n_adjusted} cells adjusted "
                  f"(rank={k}, lambda={lam_base}, gate={gate})")

    def _compute_svd_factors(self, X_df: pd.DataFrame, cols_to_impute: List[str]) -> None:
        """Extract SVD row factors for downstream use — NO warm-start filling."""
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 3 or not cols_to_impute:
            return

        X_mat = X_df[numeric_cols].to_numpy(dtype=float, na_value=np.nan)
        col_means = np.nanmean(X_mat, axis=0)
        col_stds = np.nanstd(X_mat, axis=0)
        col_stds[col_stds < 1e-10] = 1.0
        X_std = (X_mat - col_means) / col_stds

        rank_max = min(12, len(numeric_cols) // 3, X_mat.shape[0] // 5)
        rank_candidates = [r for r in [4, 6, 8, 10, 12] if 2 <= r <= rank_max]
        if not rank_candidates:
            rank_candidates = [max(2, rank_max)]

        if len(rank_candidates) > 1:
            observed_mask = ~np.isnan(X_std)
            obs_rows, obs_cols_idx = np.where(observed_mask)
            n_obs = len(obs_rows)
            rng = np.random.RandomState(self.seed)
            holdout_idx = rng.choice(n_obs, size=max(1, n_obs // 10), replace=False)
            holdout_rows = obs_rows[holdout_idx]
            holdout_cols = obs_cols_idx[holdout_idx]
            holdout_vals = X_std[holdout_rows, holdout_cols]
            X_masked = X_std.copy()
            X_masked[holdout_rows, holdout_cols] = np.nan
            best_rank, best_err = rank_candidates[0], float('inf')
            for r in rank_candidates:
                Z_trial = _iterative_svd_impute(X_masked, rank=r)
                recon = Z_trial[holdout_rows, holdout_cols]
                err = float(np.sqrt(np.mean((recon - holdout_vals) ** 2)))
                if err < best_err:
                    best_err = err
                    best_rank = r
            svd_rank = best_rank
        else:
            svd_rank = rank_candidates[0]

        Z_std = _iterative_svd_impute(X_std, rank=svd_rank)
        U, s, Vt = randomized_svd(Z_std, n_components=svd_rank, random_state=self.seed)
        self.svd_row_factors_ = pd.DataFrame(
            U * s[np.newaxis, :],
            index=X_df.index,
            columns=[f"_svd_f{i+1}" for i in range(svd_rank)],
        )
        self._svd_rank = svd_rank

        # Store observed-only SVD anchor for coherence projection
        # This is the low-rank reconstruction computed BEFORE any imputation,
        # using only observed data (NaNs filled via iterative SVD internally).
        X_anchor_std = U @ np.diag(s) @ Vt   # rank-k reconstruction in standardized space
        self._svd_anchor_cols = numeric_cols
        self._svd_anchor_means = col_means
        self._svd_anchor_stds = col_stds
        self._svd_anchor_std = X_anchor_std          # standardized
        self._svd_anchor = X_anchor_std * col_stds + col_means  # original units

        if self.verbose:
            print(f"  SVD factors: rank={svd_rank}")

    # --------------------------------------------------------------------- #
    #  Phase 1: per-column candidate rankings                                #
    # --------------------------------------------------------------------- #
    def _build_candidate_rankings(self, X_df: pd.DataFrame, cols_to_impute: List[str]) -> None:
        """For each target column, rank candidate predictors by |corr| * sqrt(n_common)."""
        for col in cols_to_impute:
            target_obs = X_df[col].notna()
            candidates = []
            for other in X_df.columns:
                if other == col:
                    continue
                common = target_obs & X_df[other].notna()
                n_common = int(common.sum())
                if n_common < self.min_support:
                    continue
                # Use precomputed Pearson correlation
                if (self.correlation_matrix_ is not None and
                        col in self.correlation_matrix_.index and
                        other in self.correlation_matrix_.columns):
                    corr = self.correlation_matrix_.loc[col, other]
                else:
                    try:
                        corr, _ = pearsonr(
                            X_df.loc[common, col].to_numpy(dtype=float),
                            X_df.loc[common, other].to_numpy(dtype=float),
                        )
                    except Exception:
                        continue
                if pd.isna(corr):
                    continue
                score = abs(corr) * np.sqrt(n_common)
                candidates.append((other, score, n_common))
            candidates.sort(key=lambda x: x[1], reverse=True)
            self._candidate_rankings[col] = candidates[:self.selector_k_max]
        if self.verbose:
            print(f"  Candidate rankings built for {len(cols_to_impute)} columns")

    # --------------------------------------------------------------------- #
    #  Phase 1b: per-column LOO forward selection                            #
    # --------------------------------------------------------------------- #
    def _build_loo_forward_selections(
        self, X_df: pd.DataFrame, cols_to_impute: List[str]
    ) -> None:
        """For each target column, run greedy forward selection driven by LOO MSE.

        Uses the candidate pool from `_candidate_rankings`. At each step, evaluates
        adding each remaining candidate, computes BayesianRidge LOO MSE on rows
        where target + selected + candidate are all observed, and accepts the one
        with lowest LOO MSE. Stops when no candidate strictly improves LOO MSE
        (with a small margin) or when k_max selected.
        """
        k_cap = 8  # global cap on selected count per column
        rel_tol = 1e-3  # require >0.1% MSE improvement to keep adding
        for col in cols_to_impute:
            cands = [n for n, _, _ in self._candidate_rankings.get(col, [])]
            if not cands:
                self._loo_forward_selections[col] = []
                continue

            target = X_df[col].to_numpy(dtype=float)
            target_obs = ~np.isnan(target)

            selected: List[str] = []
            best_mse = np.inf
            remaining = list(cands)

            while remaining and len(selected) < k_cap:
                best_cand = None
                best_cand_mse = np.inf
                for cand in remaining:
                    pred_list = selected + [cand]
                    mask = target_obs.copy()
                    for p in pred_list:
                        mask &= X_df[p].notna().to_numpy()
                    n = int(mask.sum())
                    needed = max(self.min_support, 5 * (len(pred_list) + 1))
                    if n < needed:
                        continue
                    X_tr = X_df.loc[mask, pred_list].to_numpy(dtype=float)
                    y_tr = target[mask]
                    try:
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X_tr)
                        model = BayesianRidge(compute_score=False, fit_intercept=True)
                        model.fit(X_scaled, y_tr)
                        mse, ill = self._hat_matrix_loo(model, X_scaled, y_tr)
                    except Exception:
                        continue
                    if ill or not np.isfinite(mse):
                        continue

                    if mse < best_cand_mse:
                        best_cand_mse = mse
                        best_cand = cand
                if best_cand is None:
                    break
                if len(selected) > 0 and best_cand_mse >= best_mse * (1.0 - rel_tol):
                    break
                selected.append(best_cand)
                remaining.remove(best_cand)
                best_mse = best_cand_mse
            self._loo_forward_selections[col] = selected
        if self.verbose:
            sizes = [len(v) for v in self._loo_forward_selections.values()]
            avg = float(np.mean(sizes)) if sizes else 0.0
            print(f"  LOO forward selections built for {len(cols_to_impute)} cols "
                  f"(avg k={avg:.1f})")

    # --------------------------------------------------------------------- #
    #  Phase 2: pass 1 — observed-only imputation                            #
    # --------------------------------------------------------------------- #
    def _select_cell_predictors(
        self,
        available_cols: set,
        col: str,
        extra_cols: Optional[set] = None,
    ) -> List[str]:
        """Greedy predictor selection for a single cell from available columns."""
        candidates = self._candidate_rankings.get(col, [])
        pool = set(available_cols)
        if extra_cols:
            pool = pool | extra_cols
        pool.discard(col)

        # LOO forward selection branch: use precomputed ordering, no redundancy filter.
        if self.predictor_selection == "loo_forward":
            ordered = self._loo_forward_selections.get(col, [])
            cand_supports = {n: nc for n, _, nc in candidates}
            sel = [p for p in ordered if p in pool]
            if not sel:
                return []
            approx_support = cand_supports.get(sel[0], 0)
            if approx_support < 15:
                k_max = 1
            elif approx_support < 30:
                k_max = 2
            elif approx_support < 50:
                k_max = 3
            elif approx_support < 80:
                k_max = 5
            else:
                k_max = 8
            out: List[str] = []
            for name in sel:
                if len(out) >= k_max:
                    break
                needed_support = max(self.min_support, 5 * (len(out) + 2))
                nc = cand_supports.get(name, 0)
                if nc < needed_support:
                    continue
                out.append(name)
            return out

        filtered = [(n, s, nc) for n, s, nc in candidates if n in pool]
        if not filtered:
            return []

        # Adaptive k: conservative to avoid insufficient joint support.
        # With missingness p_miss per column, joint support for k predictors
        # + target ≈ n_common * (1-p_miss)^(k-1).  We need this above min_support.
        approx_support = filtered[0][2] if filtered else 0
        if approx_support < 15:
            k_max = 1
        elif approx_support < 30:
            k_max = 2
        elif approx_support < 50:
            k_max = 3
        elif approx_support < 80:
            k_max = 5
        else:
            k_max = 8

        selected: List[str] = []
        for name, score, n_common in filtered:
            if len(selected) >= k_max:
                break
            needed_support = max(self.min_support, 5 * (len(selected) + 2))
            if n_common < needed_support:
                continue
            # Redundancy check against already-selected
            redundant = False
            if self.correlation_matrix_ is not None:
                for sel in selected:
                    if (sel in self.correlation_matrix_.index and
                            name in self.correlation_matrix_.columns):
                        r = self.correlation_matrix_.loc[sel, name]
                        if not pd.isna(r) and abs(r) > 0.85:
                            redundant = True
                            break
            if not redundant:
                selected.append(name)
        return selected

    def _get_best_single_predictor(self, col: str, available: set) -> Optional[str]:
        """Return the highest-ranked single predictor that is available."""
        for name, score, n_common in self._candidate_rankings.get(col, []):
            if name in available and n_common >= self.min_support:
                return name
        return None

    # ---- hat-matrix LOO σ² ----
    @staticmethod
    def _hat_matrix_loo(model, X_scaled: np.ndarray, y: np.ndarray):
        """Compute LOO σ² analytically via hat matrix for BayesianRidge.

        Returns (sigma2, ill_conditioned).  ill_conditioned=True means
        the predictor set should be rejected.
        """
        y_hat = model.predict(X_scaled)
        n, p = X_scaled.shape
        # BayesianRidge stores regularisation precision as alpha_
        reg = getattr(model, 'alpha_', 1e-3)
        XtX = X_scaled.T @ X_scaled
        try:
            XtX_reg_inv = np.linalg.solve(
                XtX + reg * np.eye(p), np.eye(p)
            )
        except np.linalg.LinAlgError:
            resid = y - y_hat
            return float(np.mean(resid ** 2)), False

        H = X_scaled @ XtX_reg_inv @ X_scaled.T
        h_ii = np.diag(H)
        if np.max(h_ii) > 0.95:
            return float('inf'), True  # ill-conditioned
        h_ii = np.clip(h_ii, 0, 0.99)
        e_loo = (y - y_hat) / (1.0 - h_ii)
        return float(np.mean(e_loo ** 2)), False

    # ---- model fitting ----
    def _fit_ridge(self, pred_list, X_train, y_train):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        model = BayesianRidge(compute_score=False, fit_intercept=True)
        model.fit(X_scaled, y_train)
        sigma2, ill = self._hat_matrix_loo(model, X_scaled, y_train)
        if ill:
            return None
        coefficients = model.coef_ / scaler.scale_
        return FittedCellModel(
            predictor_names=pred_list, sigma2_loo=sigma2,
            n_train=len(y_train), model_kind='ridge',
            _model=model, _scaler=scaler, _coefficients=coefficients,
        )

    def _fit_bounded_ridge(self, pred_list, X_train, y_train, bounds):
        lower, upper = float(bounds[0]), float(bounds[1])
        n_obs = len(y_train)
        eps = max(1e-3, 0.5 / (n_obs + 1))
        range_ = upper - lower + 2 * eps
        p = (y_train - lower + eps) / range_
        p = np.clip(p, 1e-6, 1 - 1e-6)
        y_logit = np.log(p / (1 - p))
        if np.std(y_logit) < 1e-8:
            return None
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        model = BayesianRidge(compute_score=False, fit_intercept=True)
        model.fit(X_scaled, y_logit)

        # Compute LOO residuals analytically via hat matrix (in logit space),
        # then back-transform to ORIGINAL scale before reporting sigma2.
        # The earlier code reported sigma2 in logit space, which was orders
        # of magnitude larger than original-scale variance for bounded targets
        # — that broke EB parent shrinkage and coherence projection (both
        # consume sigma2 assuming original-scale units).
        y_hat_logit = model.predict(X_scaled)
        n, n_pred = X_scaled.shape
        reg = getattr(model, 'alpha_', 1e-3)
        XtX = X_scaled.T @ X_scaled
        try:
            XtX_reg_inv = np.linalg.solve(XtX + reg * np.eye(n_pred), np.eye(n_pred))
        except np.linalg.LinAlgError:
            resid_logit = y_logit - y_hat_logit
            y_hat_logit_loo = y_hat_logit
            h_ii = None
        else:
            H = X_scaled @ XtX_reg_inv @ X_scaled.T
            h_ii = np.diag(H)
            if np.max(h_ii) > 0.95:
                return None  # ill-conditioned
            h_ii = np.clip(h_ii, 0, 0.99)
            e_loo_logit = (y_logit - y_hat_logit) / (1.0 - h_ii)
            y_hat_logit_loo = y_logit - e_loo_logit

        # Back-transform LOO predictions to original scale
        p_loo = 1.0 / (1.0 + np.exp(-y_hat_logit_loo))
        y_hat_loo = p_loo * range_ + lower - eps
        sigma2 = float(np.mean((y_train - y_hat_loo) ** 2))

        coefficients = model.coef_ / scaler.scale_
        return FittedCellModel(
            predictor_names=pred_list, sigma2_loo=sigma2,
            n_train=len(y_train), model_kind='bounded_ridge',
            _model=model, _scaler=scaler, _coefficients=coefficients,
            _lower=lower, _upper=upper, _logit_eps=eps, _logit_range=range_,
        )

    def _fit_hurdle(self, pred_list, X_train, y_train, floor_info):
        threshold = floor_info.get('threshold', 0.0)
        floor_mean = floor_info.get('floor_mean', 0.0)
        is_capable = (y_train > threshold).astype(int)
        n_cap, n_floor = int(is_capable.sum()), len(is_capable) - int(is_capable.sum())
        if n_cap < 5 or n_floor < 3:
            return self._fit_ridge(pred_list, X_train, y_train)

        gate_scaler = StandardScaler()
        X_gate = gate_scaler.fit_transform(X_train)
        gate = LogisticRegression(random_state=self.seed, max_iter=1000, class_weight='balanced')
        try:
            gate.fit(X_gate, is_capable)
        except Exception:
            return self._fit_ridge(pred_list, X_train, y_train)

        cap_mask = y_train > threshold
        X_cap = X_train[cap_mask]
        y_cap = y_train[cap_mask]
        if len(y_cap) < 5:
            return self._fit_ridge(pred_list, X_train, y_train)

        val_scaler = StandardScaler()
        X_val = val_scaler.fit_transform(X_cap)
        val_model = BayesianRidge(compute_score=False, fit_intercept=True)
        val_model.fit(X_val, y_cap)

        # Update floor_mean from data
        floor_vals = y_train[~cap_mask]
        if len(floor_vals) > 0:
            floor_mean = float(floor_vals.mean())

        # σ² via residual with DoF correction
        p_proba = gate.predict_proba(X_gate)
        p_cap = p_proba[:, 1] if gate.classes_.shape[0] == 2 else np.full(len(y_train), 0.5)
        mu_cap = val_model.predict(val_scaler.transform(X_train))
        y_hat = p_cap * mu_cap + (1.0 - p_cap) * floor_mean
        resid = y_train - y_hat
        dof = max(1, len(y_train) - len(pred_list) - 2)
        sigma2 = float(np.sum(resid ** 2) / dof)

        return FittedCellModel(
            predictor_names=pred_list, sigma2_loo=sigma2,
            n_train=len(y_train), model_kind='hurdle',
            _model=val_model, _scaler=val_scaler, _coefficients=None,
            _gate_model=gate, _gate_scaler=gate_scaler,
            _threshold=threshold, _floor_mean=floor_mean,
        )

    def _fit_categorical(self, pred_list, X_train, y_train):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        n_classes = len(np.unique(y_train))
        if n_classes <= 2:
            model = LogisticRegression(random_state=self.seed, max_iter=1000)
        else:
            model = RandomForestClassifier(
                n_estimators=100, max_depth=6, min_samples_leaf=2,
                random_state=self.seed, n_jobs=1,
            )
        try:
            model.fit(X_scaled, y_train)
        except Exception:
            return None
        # σ² = classification entropy (proxy)
        probas = model.predict_proba(X_scaled)
        ent = entropy(probas.T)
        sigma2 = float(np.mean(ent)) if len(ent) else 1.0
        return FittedCellModel(
            predictor_names=pred_list, sigma2_loo=sigma2,
            n_train=len(y_train), model_kind='categorical',
            _model=model, _scaler=scaler,
        )

    def _fit_or_lookup(
        self, col: str, pred_key: frozenset, X_df: pd.DataFrame,
    ) -> Optional[FittedCellModel]:
        cache_key = (col, pred_key)
        if cache_key in self.model_bank_:
            return self.model_bank_[cache_key]

        pred_list = sorted(pred_key)
        # Training data: rows where target AND all predictors are observed
        mask = X_df[col].notna()
        for p in pred_list:
            mask = mask & X_df[p].notna()
        train_idx = mask[mask].index
        n_train = len(train_idx)
        min_needed = max(self.min_support, 5 * (len(pred_list) + 1))
        if n_train < min_needed:
            self.model_bank_[cache_key] = None
            return None

        X_train = X_df.loc[train_idx, pred_list].to_numpy(dtype=float)
        y_train = X_df.loc[train_idx, col].to_numpy(dtype=float)

        col_type = self.column_types_.get(col, ColumnType.LINEAR)
        tags = self.column_metadata_.get(col, {}).get('tags', set())

        if col_type == ColumnType.CATEGORICAL:
            fitted = self._fit_categorical(pred_list, X_train, y_train)
        elif ColumnTags.FLOOR_INFLATED in tags:
            floor_info = self.column_metadata_.get(col, {}).get('floor_info', {})
            fitted = self._fit_hurdle(pred_list, X_train, y_train, floor_info)
        elif ColumnTags.BOUNDED in tags:
            bounds = self.column_metadata_.get(col, {}).get('bounds', (0.0, 100.0))
            fitted = self._fit_bounded_ridge(pred_list, X_train, y_train, bounds)
        else:
            fitted = self._fit_ridge(pred_list, X_train, y_train)

        self.model_bank_[cache_key] = fitted
        return fitted

    # ---- prediction ----
    def _predict_cell(self, fitted: FittedCellModel, X_row: np.ndarray) -> float:
        X_2d = X_row.reshape(1, -1)
        if fitted.model_kind == 'hurdle':
            X_gate = fitted._gate_scaler.transform(X_2d)
            proba = fitted._gate_model.predict_proba(X_gate)
            p_cap = float(proba[0, 1]) if fitted._gate_model.classes_.shape[0] == 2 else 0.5
            X_val = fitted._scaler.transform(X_2d)
            mu_cap = float(fitted._model.predict(X_val)[0])
            return p_cap * mu_cap + (1.0 - p_cap) * fitted._floor_mean

        X_scaled = fitted._scaler.transform(X_2d)

        if fitted.model_kind == 'categorical':
            return float(fitted._model.predict(X_scaled)[0])

        y_pred = float(fitted._model.predict(X_scaled)[0])

        if fitted.model_kind == 'bounded_ridge':
            p = 1.0 / (1.0 + np.exp(-y_pred))
            y_pred = p * fitted._logit_range + fitted._lower - fitted._logit_eps

        return y_pred

    # ---- pass 1 ----
    def _pass1_observed_only(self, X_df: pd.DataFrame, cols_to_impute: List[str]) -> None:
        if self.verbose:
            print("Pass 1: observed-only imputation")

        # Precompute per-row observed column sets (fast numpy check)
        obs_matrix = X_df.notna().values
        col_names = list(X_df.columns)
        col_to_pos = {c: i for i, c in enumerate(col_names)}
        idx_list = list(X_df.index)
        idx_to_pos = {idx: i for i, idx in enumerate(idx_list)}

        total_writes = 0
        for col in cols_to_impute:
            missing_mask = self.original_missing_[col]
            missing_indices = missing_mask[missing_mask].index.tolist()
            if not missing_indices:
                continue

            col_pos = col_to_pos[col]

            # Group rows by predictor subset
            groups: Dict[frozenset, List] = {}

            for row_idx in missing_indices:
                row_pos = idx_to_pos[row_idx]
                observed = {col_names[j] for j in range(len(col_names))
                            if obs_matrix[row_pos, j] and j != col_pos}

                predictors = self._select_cell_predictors(observed, col)
                if not predictors:
                    best = self._get_best_single_predictor(col, observed)
                    if best:
                        predictors = [best]
                    else:
                        continue

                key = frozenset(predictors)
                groups.setdefault(key, []).append(row_idx)

            # Fit and predict per group
            for pred_key, row_indices in groups.items():
                fitted = self._fit_or_lookup(col, pred_key, X_df)

                # Single-proxy challenger (shared across group)
                single_fitted = None
                if len(pred_key) > 1 or fitted is None:
                    all_available = set()
                    for ri in row_indices:
                        rp = idx_to_pos[ri]
                        all_available |= {col_names[j] for j in range(len(col_names))
                                          if obs_matrix[rp, j] and j != col_pos}
                    best_single = self._get_best_single_predictor(col, all_available)
                    if best_single:
                        single_key = frozenset([best_single])
                        single_fitted = self._fit_or_lookup(col, single_key, X_df)

                # If multi-predictor failed, use single proxy as primary
                if fitted is None:
                    if single_fitted is None:
                        continue
                    fitted = single_fitted

                for row_idx in row_indices:
                    use_fitted = fitted

                    # Check single-proxy challenger (only when multi succeeded)
                    if (single_fitted is not None and single_fitted is not fitted and
                            single_fitted.sigma2_loo < fitted.sigma2_loo):
                        single_name = single_fitted.predictor_names[0]
                        sp = col_to_pos.get(single_name)
                        rp = idx_to_pos[row_idx]
                        if sp is not None and obs_matrix[rp, sp]:
                            use_fitted = single_fitted

                    rp = idx_to_pos[row_idx]
                    if not all(
                        col_to_pos.get(p) is not None
                        and obs_matrix[rp, col_to_pos[p]]
                        for p in use_fitted.predictor_names
                    ):
                        # Predictor unobserved for this row (can happen when
                        # `fitted = single_fitted` fallback was chosen group-wide
                        # but the single predictor isn't observed for every row).
                        continue
                    X_row = np.array([float(X_df.loc[row_idx, p])
                                      for p in use_fitted.predictor_names])
                    y_pred = self._predict_cell(use_fitted, X_row)

                    X_df.loc[row_idx, col] = y_pred
                    self.sigma2_matrix_.loc[row_idx, col] = use_fitted.sigma2_loo
                    self.cell_predictors_[(row_idx, col)] = list(use_fitted.predictor_names)

                    freq = self.predictor_freqs_.setdefault(col, {})
                    for p in use_fitted.predictor_names:
                        freq[p] = freq.get(p, 0) + 1

                    self.logs_.append({
                        'col': col, 'row': row_idx,
                        'y_pred': float(y_pred),
                        'h_blend': float(np.sqrt(max(0, use_fitted.sigma2_loo))),
                        'pass_num': 1,
                    })
                    total_writes += 1

        if self.verbose:
            print(f"  Pass 1: {total_writes} cells written, "
                  f"{len(self.model_bank_)} models cached")

    # ---- pass 2: uncertainty-gated expansion ----
    def _pass2_expansion(self, X_df: pd.DataFrame, cols_to_impute: List[str]) -> None:
        """One expansion pass using confident imputations as additional predictors."""
        if self.verbose:
            print("Pass 2: uncertainty-gated expansion")

        # Snapshot pass-1 state (Jacobi-style: reads from snapshot, writes to X_df)
        snapshot = X_df.copy()
        sigma2_snap = self.sigma2_matrix_.copy()

        obs_matrix_orig = pd.DataFrame(False, index=X_df.index, columns=X_df.columns)
        for col in X_df.columns:
            if col in self.original_missing_:
                obs_matrix_orig[col] = ~self.original_missing_[col]
            else:
                obs_matrix_orig[col] = True

        # Collect all imputed cells, sorted worst-first
        imputed_cells = []
        for col in cols_to_impute:
            missing_mask = self.original_missing_[col]
            for row_idx in missing_mask[missing_mask].index:
                s2 = self.sigma2_matrix_.loc[row_idx, col]
                if np.isfinite(s2) and s2 > 0:
                    imputed_cells.append((row_idx, col, s2))
        imputed_cells.sort(key=lambda x: x[2], reverse=True)

        # Precompute confident imputed columns per row
        confident_per_row: Dict[Any, set] = {}
        for row_idx in X_df.index:
            confident = set()
            for col in cols_to_impute:
                if not obs_matrix_orig.loc[row_idx, col]:
                    s2 = sigma2_snap.loc[row_idx, col]
                    sd_col = self._col_sd.get(col, 1.0)
                    if sd_col > 0 and np.isfinite(s2):
                        if np.sqrt(s2) / sd_col < self.confidence_threshold:
                            confident.add(col)
            confident_per_row[row_idx] = confident

        n_improved = 0
        max_extras = self.max_confident_extras
        for row_idx, col, old_sigma2 in imputed_cells:
            observed = set(c for c in X_df.columns
                           if c != col and obs_matrix_orig.loc[row_idx, c])
            confident = confident_per_row.get(row_idx, set()) - {col} - observed
            if not confident:
                continue

            # Pick top-N confident imputed columns (highest relevance to target)
            scored_extras = []
            for cc in confident:
                for name, score, _ in self._candidate_rankings.get(col, []):
                    if name == cc:
                        scored_extras.append((cc, score))
                        break
            if not scored_extras:
                continue
            scored_extras.sort(key=lambda x: x[1], reverse=True)
            extra_cols = {cc for cc, _ in scored_extras[:max_extras]}

            new_predictors = self._select_cell_predictors(
                observed, col, extra_cols=extra_cols
            )
            if not new_predictors:
                continue

            old_predictors = self.cell_predictors_.get((row_idx, col), [])
            if set(new_predictors) == set(old_predictors):
                continue

            new_key = frozenset(new_predictors)
            new_fitted = self._fit_or_lookup(col, new_key, snapshot)
            if new_fitted is None:
                continue

            # Compute σ²_total with input uncertainty propagation (diagonal approx)
            sigma2_model = new_fitted.sigma2_loo
            sigma2_input = 0.0
            if new_fitted._coefficients is not None:
                for ec in extra_cols:
                    if ec in new_fitted.predictor_names:
                        try:
                            coef_idx = new_fitted.predictor_names.index(ec)
                            beta = new_fitted._coefficients[coef_idx]
                            s2_in = sigma2_snap.loc[row_idx, ec]
                            if np.isfinite(s2_in):
                                sigma2_input += beta ** 2 * s2_in
                        except (ValueError, IndexError):
                            pass
            sigma2_total = sigma2_model + sigma2_input

            if sigma2_total >= old_sigma2 * 0.95:
                continue

            # Predict from snapshot (Jacobi-style)
            X_row = np.array([float(snapshot.loc[row_idx, p])
                              for p in new_fitted.predictor_names])
            y_pred = self._predict_cell(new_fitted, X_row)

            X_df.loc[row_idx, col] = y_pred
            self.sigma2_matrix_.loc[row_idx, col] = sigma2_total
            self.cell_predictors_[(row_idx, col)] = list(new_fitted.predictor_names)

            freq = self.predictor_freqs_.setdefault(col, {})
            for p in new_fitted.predictor_names:
                freq[p] = freq.get(p, 0) + 1

            self.logs_.append({
                'col': col, 'row': row_idx,
                'y_pred': float(y_pred),
                'h_blend': float(np.sqrt(max(0, sigma2_total))),
                'pass_num': 2,
            })
            n_improved += 1

        if self.verbose:
            print(f"  Pass 2: {n_improved} cells improved, "
                  f"{len(self.model_bank_)} total cached models")

    # --------------------------------------------------------------------- #
    #  Phase 5: trajectory features and representative models                #
    # --------------------------------------------------------------------- #
    def _compute_trajectory_features(self, X_df: pd.DataFrame, cols_to_impute: List[str]) -> None:
        """Compute trajectory features: σ²-based + delta-to-SVD-anchor per row."""
        traj_sigma_mean = pd.Series(0.0, index=X_df.index, dtype=float)
        traj_sigma_max = pd.Series(0.0, index=X_df.index, dtype=float)
        traj_anchor_mean = pd.Series(0.0, index=X_df.index, dtype=float)
        traj_anchor_max = pd.Series(0.0, index=X_df.index, dtype=float)
        n_imputed = pd.Series(0.0, index=X_df.index, dtype=float)

        # Pre-build anchor lookup if available
        anchor_col_idx = {}
        if self._svd_anchor is not None:
            anchor_col_idx = {c: i for i, c in enumerate(self._svd_anchor_cols)}

        for col in cols_to_impute:
            missing_mask = self.original_missing_.get(col, pd.Series(False, index=X_df.index))
            if not missing_mask.any():
                continue
            sd = self._col_sd.get(col, 1.0)
            if sd < 1e-10:
                sd = 1.0

            miss_idx = missing_mask[missing_mask].index
            n_imputed.loc[miss_idx] += 1

            # σ²-based trajectory
            s2_vals = self.sigma2_matrix_.loc[miss_idx, col].values
            sigma_norm = np.where(
                np.isfinite(s2_vals), np.sqrt(np.maximum(0, s2_vals)) / sd, 1.0
            )
            traj_sigma_mean.loc[miss_idx] += sigma_norm
            traj_sigma_max.loc[miss_idx] = np.maximum(
                traj_sigma_max.loc[miss_idx].values, sigma_norm
            )

            # Delta-to-anchor trajectory (if anchor available)
            if col in anchor_col_idx:
                j = anchor_col_idx[col]
                row_positions = np.array([X_df.index.get_loc(idx) for idx in miss_idx])
                imputed_vals = X_df.loc[miss_idx, col].values
                anchor_vals = self._svd_anchor[row_positions, j]
                delta_norm = np.abs(imputed_vals - anchor_vals) / sd
                traj_anchor_mean.loc[miss_idx] += delta_norm
                traj_anchor_max.loc[miss_idx] = np.maximum(
                    traj_anchor_max.loc[miss_idx].values, delta_norm
                )

        n_imputed_safe = n_imputed.clip(lower=1)
        traj_sigma_mean = traj_sigma_mean / n_imputed_safe
        traj_anchor_mean = traj_anchor_mean / n_imputed_safe

        self.trajectory_features_ = pd.DataFrame({
            '_traj_mean_delta': traj_sigma_mean,
            '_traj_max_delta': traj_sigma_max,
            '_traj_n_imputed': n_imputed,
        }, index=X_df.index)

    def _build_representative_models(self, X_df: pd.DataFrame, cols_to_impute: List[str]) -> None:
        """Build one representative BaseColumnModel per column for API compat."""
        for col in cols_to_impute:
            freq = self.predictor_freqs_.get(col, {})
            if not freq:
                continue
            sorted_preds = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            k = min(8, len(sorted_preds))
            top_preds = [p for p, _ in sorted_preds[:k]]
            self.predictors_map_[col] = top_preds

            col_type = self.column_types_.get(col, ColumnType.LINEAR)
            metadata = dict(self.column_metadata_.get(col, {}))
            if col_type == ColumnType.CATEGORICAL:
                metadata['n_classes'] = int(X_df[col].dropna().nunique())
            metadata['n_obs'] = int(X_df[col].notna().sum())
            try:
                model = ModelFactory.create_model(
                    col_type, top_preds, self.alpha, self.seed, metadata,
                    correlation_matrix=self.correlation_matrix_,
                )
                model.fit(X_df, X_df[col])
                self.models_[col] = model
            except Exception:
                pass

    # --------------------------------------------------------------------- #
    #  API-compatible methods                                                #
    # --------------------------------------------------------------------- #
    def get_imputation_importance(self) -> pd.DataFrame:
        """Extract per-column feature importances (delegates to representative models)."""
        records = []
        for col, model in self.models_.items():
            predictors = self.predictors_map_.get(col, [])
            if not predictors:
                continue
            importances = SpecializedColumnImputer._extract_model_importance(model, predictors)
            model_type = type(model).__name__
            total = sum(importances.values())
            if total < 1e-12:
                n = len(predictors)
                importances = {p: 1.0 / n for p in predictors}
                total = 1.0
            sorted_preds = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            for rank, (pred, imp) in enumerate(sorted_preds, 1):
                records.append({
                    'target_col': col,
                    'predictor_col': pred,
                    'importance': imp / total,
                    'model_type': model_type,
                    'rank': rank,
                })
        return pd.DataFrame(records)

    def evaluate_quality_oof(
        self,
        X_df: pd.DataFrame,
        n_splits: int = 5,
        n_rounds: int = 1,
        frac: float = 0.2,
        random_state: Optional[int] = None,
        knn_bins: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, np.inf),
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """OOF mask-eval that emulates per-cell predictor selection.

        For each fold, held-out rows get per-cell predictor selection and
        model fitting on the training fold only.
        """
        if not self.models_:
            raise RuntimeError("Call fit_transform before evaluate_quality_oof.")

        orig_X_df = X_df
        if self.log_transforms_:
            X_df = X_df.copy()
            for col, info in self.log_transforms_.items():
                if col in X_df.columns:
                    X_df[col] = np.log1p(X_df[col] + info['shift'])

        rng = np.random.default_rng(self.seed if random_state is None else random_state)
        records: List[Dict[str, Any]] = []

        for col in self.models_.keys():
            candidates = self._candidate_rankings.get(col, [])
            if not candidates:
                continue

            obs_mask = ~X_df[col].isna().values
            idx_obs = np.where(obs_mask)[0]
            if idx_obs.size < max(5, n_splits * 2):
                continue

            col_type = self.column_types_.get(col)
            metadata = dict(self.column_metadata_.get(col, {}))
            if col_type == ColumnType.CATEGORICAL:
                metadata['n_classes'] = int(X_df[col].dropna().nunique())
            metadata['n_obs'] = int(X_df[col].notna().sum())

            kf = KFold(
                n_splits=n_splits, shuffle=True,
                random_state=self.seed if random_state is None else random_state,
            )

            for tr_pos, va_pos in kf.split(idx_obs):
                tr_idx = idx_obs[tr_pos]
                va_idx = idx_obs[va_pos]
                if tr_idx.size < 3 or va_idx.size == 0:
                    continue

                m = max(1, int(frac * len(va_idx)))
                m = min(m, len(va_idx))
                probe = rng.choice(va_idx, size=m, replace=False)

                fold_bank: Dict[frozenset, Optional[FittedCellModel]] = {}

                for ridx in probe:
                    row = X_df.iloc[ridx]
                    observed = set(c for c in X_df.columns
                                   if c != col and pd.notna(row[c]))
                    predictors = self._select_cell_predictors(observed, col)
                    if not predictors:
                        best = self._get_best_single_predictor(col, observed)
                        if best:
                            predictors = [best]
                        else:
                            continue
                    pred_key = frozenset(predictors)

                    if pred_key not in fold_bank:
                        pred_list = sorted(pred_key)
                        mask = X_df[col].notna()
                        for p_name in pred_list:
                            mask = mask & X_df[p_name].notna()
                        train_rows = set(X_df.index[tr_idx])
                        valid = mask & X_df.index.isin(train_rows)
                        tidx = valid[valid].index
                        min_needed = max(self.min_support, 5 * (len(pred_list) + 1))
                        if len(tidx) < min_needed:
                            fold_bank[pred_key] = None
                            continue
                        X_tr = X_df.loc[tidx, pred_list].to_numpy(dtype=float)
                        y_tr = X_df.loc[tidx, col].to_numpy(dtype=float)
                        tags = metadata.get('tags', set())
                        if col_type == ColumnType.CATEGORICAL:
                            fitted = self._fit_categorical(pred_list, X_tr, y_tr)
                        elif ColumnTags.FLOOR_INFLATED in tags:
                            fi = metadata.get('floor_info', {})
                            fitted = self._fit_hurdle(pred_list, X_tr, y_tr, fi)
                        elif ColumnTags.BOUNDED in tags:
                            bounds = metadata.get('bounds', (0.0, 100.0))
                            fitted = self._fit_bounded_ridge(pred_list, X_tr, y_tr, bounds)
                        else:
                            fitted = self._fit_ridge(pred_list, X_tr, y_tr)
                        fold_bank[pred_key] = fitted

                    fitted = fold_bank[pred_key]
                    if fitted is None:
                        continue

                    X_row = np.array([float(X_df.iloc[ridx][p_name])
                                      for p_name in fitted.predictor_names])
                    try:
                        y_pred = self._predict_cell(fitted, X_row)
                    except Exception:
                        continue
                    h_blend = float(np.sqrt(max(0, fitted.sigma2_loo)))

                    y_true = float(X_df.iloc[ridx][col])
                    abs_err = abs(y_true - y_pred)
                    within = float(abs_err <= h_blend) if h_blend > 0 else 0.0

                    records.append({
                        "col": col, "row": int(ridx),
                        "y_true": y_true, "y_pred": y_pred,
                        "abs_err": float(abs_err), "within": within,
                        "h_blend": h_blend, "log_knn_ratio": 0.0,
                    })

        # Inverse-transform for log-transformed columns
        if self.log_transforms_ and records:
            for rec in records:
                if rec['col'] in self.log_transforms_:
                    info = self.log_transforms_[rec['col']]
                    rec['y_true'] = float(np.expm1(rec['y_true'])) - info['shift']
                    rec['y_pred'] = float(np.expm1(rec['y_pred'])) - info['shift']
                    rec['abs_err'] = abs(rec['y_true'] - rec['y_pred'])

        per_cell = pd.DataFrame.from_records(records)
        if per_cell.empty:
            if self.verbose:
                print("Warning: evaluate_quality_oof produced no records")
            return per_cell, pd.DataFrame(), pd.DataFrame()

        def _rmse(x: pd.Series) -> float:
            return float(np.sqrt(np.mean(np.square(x))))

        per_col = (
            per_cell.groupby("col")
            .agg(
                n=("row", "count"),
                mae=("abs_err", "mean"),
                rmse=("abs_err", _rmse),
                coverage=("within", "mean"),
                med_h=("h_blend", "median"),
                med_log_knn=("log_knn_ratio", "median"),
            )
            .reset_index()
        )
        sd_map = {c: float(np.nanstd(orig_X_df[c].to_numpy(dtype=float, na_value=np.nan)))
                   for c in per_col["col"]}
        per_col["sd_y"] = per_col["col"].map(sd_map)
        per_col["rmse_over_sd"] = per_col["rmse"] / per_col["sd_y"].astype(float).clip(lower=1e-8)

        bins = np.array(knn_bins, dtype=float)
        labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins) - 1)]
        cut = pd.cut(
            np.expm1(per_cell["log_knn_ratio"]),
            bins=bins, labels=labels, include_lowest=True, right=False,
        )
        tmp_eval = per_cell.assign(knn_bin=cut)
        by_bin = (
            tmp_eval.groupby("knn_bin", observed=True)
            .agg(n=("row", "count"), mae=("abs_err", "mean"),
                 rmse=("abs_err", _rmse), coverage=("within", "mean"))
            .reset_index()
        )
        sd_by_bin = tmp_eval.groupby("knn_bin", observed=True)["y_true"].std(ddof=0)
        by_bin["sd_y"] = by_bin["knn_bin"].map(sd_by_bin).astype(float)
        by_bin["rmse_over_sd"] = by_bin["rmse"] / by_bin["sd_y"].astype(float).clip(lower=1e-8)

        if self.verbose:
            print(f"evaluate_quality_oof: {len(per_cell)} cells across {len(per_col)} columns")

        return per_cell, per_col, by_bin
