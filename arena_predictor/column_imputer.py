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
from abc import ABC, abstractmethod
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

        current_df = X_df.copy()
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

                # Check if any columns in this tier still have missing values
                has_missing = any(current_df[col].isna().any() for col in tier_cols)
                if not has_missing:
                    continue

                global_pass += 1

                if self.verbose >= 2:
                    print(f"    Tier {tier}: {len(tier_cols)} columns")

                snapshot_df = current_df.copy()  # Jacobi-style

                def impute_column(col, pass_num=global_pass):
                    writes = []
                    missing_mask = snapshot_df[col].isna()
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

        return current_df

    def _select_predictors(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """Simple correlation-based feature selection."""
        correlations = []
        for col in df.columns:
            if col != target_col:
                common_mask = df[target_col].notna() & df[col].notna()
                if common_mask.sum() >= 20:
                    try:
                        corr = abs(df.loc[common_mask, target_col].corr(df.loc[common_mask, col]))
                        if not np.isnan(corr):
                            correlations.append((col, corr))
                    except Exception:
                        continue

        # Sort by correlation, keep top k_max
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected = [col for col, _ in correlations[:self.selector_k_max]]

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

        # Compute relevance: |spearman(feature, target)|
        relevance = {}
        for col in candidates:
            if target_col in self.spearman_matrix_.index and col in self.spearman_matrix_.columns:
                corr = self.spearman_matrix_.loc[target_col, col]
                if not pd.isna(corr) and abs(corr) >= min_relevance:
                    relevance[col] = abs(corr)

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
