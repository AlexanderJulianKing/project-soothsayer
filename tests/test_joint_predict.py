"""Tests for joint_predict.py (SCMF and BHLT approaches)."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# Add arena_predictor to path so we can import joint_predict
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "arena_predictor"))

from joint_predict import (
    SupervisedCMF,
    SCMFConfig,
    BayesianHierarchicalLT,
    BHLTConfig,
    load_data,
    build_cv_splits,
    compute_oof_rmse,
    cross_validate,
    _build_families_prefix,
    _build_families_correlation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_data(
    n: int = 50,
    p: int = 20,
    rank: int = 3,
    missing_frac: float = 0.3,
    n_labeled: int = 35,
    seed: int = 42,
    noise: float = 0.1,
):
    """Generate synthetic low-rank data with missing values and a target.

    Returns:
        X_obs: (n, p) array with NaN for missing.
        y: (n,) target array (NaN where unlabeled).
        y_mask: (n,) boolean.
        X_full: (n, p) complete ground truth matrix (no missing).
        w_true: (rank,) true target weights.
    """
    rng = np.random.RandomState(seed)

    # Generate low-rank matrix
    Z_true = rng.randn(n, rank)
    W_true = rng.randn(p, rank)
    X_full = Z_true @ W_true.T + noise * rng.randn(n, p)

    # Generate target
    w_true = rng.randn(rank)
    b_true = rng.randn() * 10
    y_full = Z_true @ w_true + b_true + noise * rng.randn(n)

    # Create missing mask
    X_obs = X_full.copy()
    mask_missing = rng.rand(n, p) < missing_frac
    # Ensure at least 2 observed per row and per column
    for i in range(n):
        if np.sum(~mask_missing[i]) < 2:
            cols = rng.choice(p, 2, replace=False)
            mask_missing[i, cols] = False
    for j in range(p):
        if np.sum(~mask_missing[:, j]) < 2:
            rows = rng.choice(n, 2, replace=False)
            mask_missing[rows, j] = False
    X_obs[mask_missing] = np.nan

    # Create target mask
    y = y_full.copy()
    y_mask = np.zeros(n, dtype=bool)
    y_mask[:n_labeled] = True
    y[~y_mask] = np.nan

    return X_obs, y, y_mask, X_full, w_true


def _make_csv(tmp_path, n=50, p=20, rank=3, missing_frac=0.3, n_labeled=35, seed=42):
    """Write synthetic data as a CSV matching the expected format."""
    X_obs, y, y_mask, _, _ = _make_test_data(n, p, rank, missing_frac, n_labeled, seed)

    feature_cols = [f"bench_{i}" for i in range(p)]
    data = {}
    data["model_name"] = [f"model_{i}" for i in range(n)]
    for j, col in enumerate(feature_cols):
        data[col] = X_obs[:, j]
    data["lmsys_Score"] = y
    data["lmarena_Score"] = np.where(y_mask, y + np.random.randn(n) * 5, np.nan)

    df = pd.DataFrame(data)
    csv_path = os.path.join(str(tmp_path), "test_data.csv")
    df.to_csv(csv_path, index=False)
    return csv_path, feature_cols


# ---------------------------------------------------------------------------
# SCMF tests
# ---------------------------------------------------------------------------

class TestSCMF:
    """Tests for SupervisedCMF."""

    def test_low_rank_recovery(self):
        """SCMF should recover a low-rank matrix reasonably well."""
        X_obs, y, y_mask, X_full, _ = _make_test_data(
            n=60, p=15, rank=3, missing_frac=0.2, n_labeled=45, noise=0.05,
        )
        config = SCMFConfig(rank=3, lambda_rec=1.0, lambda_target=1.0, lambda_reg=0.01, max_iter=50)
        model = SupervisedCMF(config=config)
        model.fit(X_obs, y, y_mask)

        # Reconstruct
        X_recon = model.Z_ @ model.W_.T
        # Unstandardize
        X_recon_orig = X_recon * model.col_std_ + model.col_mean_

        # Check recovery on held-out cells
        missing_mask = np.isnan(X_obs)
        if np.sum(missing_mask) > 0:
            mae = np.mean(np.abs(X_recon_orig[missing_mask] - X_full[missing_mask]))
            # With low noise and decent rank match, MAE should be reasonable
            # (not perfect due to standardization and finite iterations)
            assert mae < np.std(X_full) * 2.0, f"MAE too high: {mae:.3f}"

    def test_loss_decreases(self):
        """Loss should generally decrease across iterations."""
        X_obs, y, y_mask, _, _ = _make_test_data(n=40, p=10, rank=3, missing_frac=0.2)
        config = SCMFConfig(rank=3, max_iter=30, tol=0.0)
        model = SupervisedCMF(config=config)
        model.fit(X_obs, y, y_mask)

        losses = model.loss_history_
        assert len(losses) > 1
        # Check that most consecutive pairs show decrease (allow small numerical bumps)
        n_decrease = sum(1 for i in range(1, len(losses)) if losses[i] <= losses[i-1] + 1e-2)
        assert n_decrease >= len(losses) * 0.7, (
            f"Loss should mostly decrease: {n_decrease}/{len(losses)-1} decreases"
        )

    def test_semi_supervised_benefit(self):
        """Using all X rows (transductive) should be at least as good as labeled-only."""
        X_obs, y, y_mask, _, _ = _make_test_data(n=60, p=15, rank=3, missing_frac=0.2, n_labeled=40)
        config = SCMFConfig(rank=3, lambda_target=5.0, max_iter=40)

        # Transductive: fit on all X, all labeled y
        model_trans = SupervisedCMF(config=config)
        model_trans.fit(X_obs, y, y_mask)
        pred_trans = model_trans.predict()
        labeled_idx = np.where(y_mask)[0]
        rmse_trans = np.sqrt(np.mean((y[labeled_idx] - pred_trans[labeled_idx]) ** 2))

        # Labeled-only: fit on labeled rows only
        model_lab = SupervisedCMF(config=config)
        X_labeled = X_obs[labeled_idx]
        y_labeled = y[labeled_idx]
        y_mask_labeled = np.ones(len(labeled_idx), dtype=bool)
        model_lab.fit(X_labeled, y_labeled, y_mask_labeled)
        pred_lab = model_lab.predict()
        rmse_lab = np.sqrt(np.mean((y_labeled - pred_lab) ** 2))

        # Transductive should do at least roughly as well (allow some tolerance)
        assert rmse_trans <= rmse_lab * 1.5, (
            f"Transductive ({rmse_trans:.2f}) should not be much worse than labeled-only ({rmse_lab:.2f})"
        )

    def test_predict_shape(self):
        """predict() returns correct shape."""
        X_obs, y, y_mask, _, _ = _make_test_data(n=30, p=10, rank=2)
        config = SCMFConfig(rank=2, max_iter=10)
        model = SupervisedCMF(config=config)
        model.fit(X_obs, y, y_mask)
        preds = model.predict()
        assert preds.shape == (30,)
        assert not np.any(np.isnan(preds))

    def test_get_factors_shape(self):
        """get_factors() returns (n, rank) array."""
        X_obs, y, y_mask, _, _ = _make_test_data(n=30, p=10, rank=2)
        config = SCMFConfig(rank=2, max_iter=10)
        model = SupervisedCMF(config=config)
        model.fit(X_obs, y, y_mask)
        Z = model.get_factors()
        assert Z.shape == (30, 2)

    def test_variable_missingness(self):
        """SCMF handles rows with different amounts of missingness."""
        rng = np.random.RandomState(123)
        n, p = 40, 12
        X_obs, y, y_mask, _, _ = _make_test_data(n=n, p=p, rank=2, missing_frac=0.0, n_labeled=30)

        # Make first 10 rows very sparse, rest very dense
        for i in range(10):
            cols_to_mask = rng.choice(p, p - 3, replace=False)
            X_obs[i, cols_to_mask] = np.nan
        for i in range(10, n):
            cols_to_mask = rng.choice(p, 2, replace=False)
            X_obs[i, cols_to_mask] = np.nan

        config = SCMFConfig(rank=2, max_iter=20)
        model = SupervisedCMF(config=config)
        model.fit(X_obs, y, y_mask)
        preds = model.predict()
        assert preds.shape == (n,)
        assert not np.any(np.isnan(preds))


# ---------------------------------------------------------------------------
# BHLT tests
# ---------------------------------------------------------------------------

class TestBHLT:
    """Tests for BayesianHierarchicalLT."""

    def test_uncertainty_vs_missingness(self):
        """Models with more missing data should have higher posterior uncertainty."""
        rng = np.random.RandomState(77)
        n, p = 40, 15
        X_obs, y, y_mask, _, _ = _make_test_data(n=n, p=p, rank=3, missing_frac=0.0, n_labeled=30)

        # Make first 5 rows very sparse
        for i in range(5):
            cols_to_mask = rng.choice(p, p - 3, replace=False)
            X_obs[i, cols_to_mask] = np.nan
        # Make rows 5-10 dense
        for i in range(5, 10):
            cols_to_mask = rng.choice(p, 1, replace=False)
            X_obs[i, cols_to_mask] = np.nan
        # Rest: moderate
        for i in range(10, n):
            cols_to_mask = rng.choice(p, 4, replace=False)
            X_obs[i, cols_to_mask] = np.nan

        config = BHLTConfig(n_factors=3, n_iter=20)
        model = BayesianHierarchicalLT(config=config)
        model.fit(X_obs, y, y_mask)

        means, stds = model.get_factors()
        # Average uncertainty for sparse rows should be higher
        avg_std_sparse = np.mean(stds[:5])
        avg_std_dense = np.mean(stds[5:10])
        assert avg_std_sparse > avg_std_dense, (
            f"Sparse rows should have higher uncertainty: {avg_std_sparse:.4f} vs {avg_std_dense:.4f}"
        )

    def test_family_prior_regularizes(self):
        """Stronger family prior should make loadings more similar within families."""
        X_obs, y, y_mask, _, _ = _make_test_data(n=50, p=12, rank=3, missing_frac=0.2, n_labeled=35)
        feature_cols = [f"fam{j//4}_{j}" for j in range(12)]  # 3 families of 4

        # Weak prior
        config_weak = BHLTConfig(n_factors=3, n_iter=20, family_prior_strength=0.01, clustering_method="prefix")
        model_weak = BayesianHierarchicalLT(config=config_weak, feature_cols=feature_cols)
        model_weak.fit(X_obs, y, y_mask)

        # Strong prior
        config_strong = BHLTConfig(n_factors=3, n_iter=20, family_prior_strength=100.0, clustering_method="prefix")
        model_strong = BayesianHierarchicalLT(config=config_strong, feature_cols=feature_cols)
        model_strong.fit(X_obs, y, y_mask)

        # Within-family variance of loadings should be smaller with strong prior
        def within_family_var(Lambda, families):
            total = 0.0
            count = 0
            for fam in families:
                if len(fam) > 1:
                    fam_loadings = Lambda[fam]
                    total += np.sum(np.var(fam_loadings, axis=0))
                    count += 1
            return total / max(count, 1)

        var_weak = within_family_var(model_weak.Lambda_, model_weak.families_)
        var_strong = within_family_var(model_strong.Lambda_, model_strong.families_)
        assert var_strong <= var_weak * 1.1, (
            f"Strong prior should reduce within-family variance: {var_strong:.4f} vs {var_weak:.4f}"
        )

    def test_predict_with_std(self):
        """predict(return_std=True) returns (pred, std) with correct shapes."""
        X_obs, y, y_mask, _, _ = _make_test_data(n=30, p=10, rank=2, n_labeled=20)
        config = BHLTConfig(n_factors=2, n_iter=10)
        model = BayesianHierarchicalLT(config=config)
        model.fit(X_obs, y, y_mask)

        pred, std = model.predict(return_std=True)
        assert pred.shape == (30,)
        assert std.shape == (30,)
        assert np.all(std > 0), "Standard deviations should be positive"
        assert not np.any(np.isnan(pred))
        assert not np.any(np.isnan(std))

    def test_get_factors_shapes(self):
        """get_factors() returns (means, stds) with correct shapes."""
        X_obs, y, y_mask, _, _ = _make_test_data(n=30, p=10, rank=2, n_labeled=20)
        config = BHLTConfig(n_factors=2, n_iter=10)
        model = BayesianHierarchicalLT(config=config)
        model.fit(X_obs, y, y_mask)

        means, stds = model.get_factors()
        assert means.shape == (30, 2)
        assert stds.shape == (30, 2)
        assert np.all(stds > 0)

    def test_correlation_clustering(self):
        """Correlation-based clustering produces sensible groups."""
        rng = np.random.RandomState(99)
        n, p = 50, 12
        # Create data with 3 clear groups of correlated columns
        Z = rng.randn(n, 3)
        X = np.zeros((n, p))
        for j in range(4):
            X[:, j] = Z[:, 0] + rng.randn(n) * 0.1
        for j in range(4, 8):
            X[:, j] = Z[:, 1] + rng.randn(n) * 0.1
        for j in range(8, 12):
            X[:, j] = Z[:, 2] + rng.randn(n) * 0.1

        families = _build_families_correlation(X, threshold=0.5)

        # Should produce roughly 3 groups
        assert len(families) >= 2, f"Expected at least 2 families, got {len(families)}"
        assert len(families) <= 6, f"Expected at most 6 families, got {len(families)}"

        # Check that correlated columns tend to be in same family
        col_to_fam = {}
        for fam_idx, fam in enumerate(families):
            for j in fam:
                col_to_fam[j] = fam_idx

        # Columns 0-3 should mostly be in the same family
        fam_0_3 = [col_to_fam[j] for j in range(4)]
        assert len(set(fam_0_3)) <= 2, "First 4 correlated columns should cluster together"


# ---------------------------------------------------------------------------
# Shared infrastructure tests
# ---------------------------------------------------------------------------

class TestSharedInfra:
    """Tests for load_data, build_cv_splits, compute_oof_rmse."""

    def test_load_data_shapes(self, tmp_path):
        """load_data returns correct shapes and types."""
        csv_path, feature_cols = _make_csv(tmp_path, n=50, p=20, n_labeled=35)
        X_obs_df, y, y_mask, feat_cols, model_names = load_data(csv_path)

        assert isinstance(X_obs_df, pd.DataFrame)
        assert X_obs_df.shape == (50, 20)
        assert y.shape == (50,)
        assert y_mask.shape == (50,)
        assert y_mask.dtype == bool
        assert np.sum(y_mask) == 35
        assert len(feat_cols) == 20
        assert len(model_names) == 50

    def test_load_data_excludes_targets(self, tmp_path):
        """load_data excludes target and alt target from features."""
        csv_path, _ = _make_csv(tmp_path)
        _, _, _, feat_cols, _ = load_data(csv_path)

        assert "lmsys_Score" not in feat_cols
        assert "lmarena_Score" not in feat_cols
        assert "model_name" not in feat_cols

    def test_build_cv_splits(self):
        """build_cv_splits returns correct number of splits."""
        splits = build_cv_splits(n_labeled=100, n_splits=5, repeats=3)
        assert len(splits) == 15  # 5 * 3

        # Each split should partition the labeled indices
        for tr, va in splits:
            assert len(tr) + len(va) == 100
            assert len(set(tr) & set(va)) == 0

    def test_build_cv_splits_coverage(self):
        """Each labeled index appears in validation exactly repeats times."""
        n_labeled = 50
        n_splits = 5
        repeats = 2
        splits = build_cv_splits(n_labeled, n_splits=n_splits, repeats=repeats)

        va_counts = np.zeros(n_labeled)
        for _, va in splits:
            for idx in va:
                va_counts[idx] += 1

        # Each index appears in validation exactly `repeats` times
        assert np.all(va_counts == repeats)

    def test_compute_oof_rmse(self):
        """compute_oof_rmse returns (float, float, float) with valid CI."""
        rng = np.random.RandomState(42)
        y_true = rng.randn(100) * 10 + 50
        oof_preds = y_true + rng.randn(100) * 3

        rmse, ci_lo, ci_hi = compute_oof_rmse(y_true, oof_preds)
        assert isinstance(rmse, float)
        assert isinstance(ci_lo, float)
        assert isinstance(ci_hi, float)
        assert ci_lo <= rmse <= ci_hi
        assert ci_lo > 0
        assert rmse > 0


# ---------------------------------------------------------------------------
# Cross-validation test
# ---------------------------------------------------------------------------

class TestCrossValidate:
    """Tests for cross_validate function."""

    def test_oof_predictions_valid(self):
        """OOF predictions should be valid (no NaN) for all labeled rows."""
        X_obs, y, y_mask, _, _ = _make_test_data(n=40, p=10, rank=2, n_labeled=30)
        config = SCMFConfig(rank=2, max_iter=15, lambda_target=3.0)
        splits = build_cv_splits(30, n_splits=3, repeats=1)

        oof_preds = cross_validate(
            SupervisedCMF, config, X_obs, y, y_mask, splits, mode="transductive",
        )

        assert oof_preds.shape == (30,)
        assert not np.any(np.isnan(oof_preds)), "OOF predictions should not contain NaN"

    def test_oof_predictions_reasonable(self):
        """OOF predictions should correlate with true values."""
        X_obs, y, y_mask, _, _ = _make_test_data(
            n=50, p=15, rank=3, missing_frac=0.15, n_labeled=40, noise=0.05,
        )
        config = SCMFConfig(rank=3, max_iter=30, lambda_target=5.0)
        splits = build_cv_splits(40, n_splits=5, repeats=1)

        oof_preds = cross_validate(
            SupervisedCMF, config, X_obs, y, y_mask, splits, mode="transductive",
        )

        y_labeled = y[y_mask]
        corr = np.corrcoef(y_labeled, oof_preds)[0, 1]
        assert corr > 0.3, f"OOF predictions should correlate with truth: r={corr:.3f}"


# ---------------------------------------------------------------------------
# Family building tests
# ---------------------------------------------------------------------------

class TestFamilyBuilding:
    """Tests for family/clustering helpers."""

    def test_prefix_families(self):
        """Prefix-based families group columns by prefix before first underscore."""
        cols = ["livebench_a", "livebench_b", "style_x", "style_y", "eq_z"]
        families = _build_families_prefix(cols)
        assert len(families) == 3

        # Flatten and check all indices present
        all_indices = sorted([j for fam in families for j in fam])
        assert all_indices == [0, 1, 2, 3, 4]

    def test_correlation_families_no_crash_sparse(self):
        """Correlation clustering handles sparse data without crashing."""
        rng = np.random.RandomState(55)
        X = rng.randn(20, 8)
        # Make it very sparse
        X[rng.rand(20, 8) < 0.5] = np.nan

        families = _build_families_correlation(X, threshold=0.5)
        all_indices = sorted([j for fam in families for j in fam])
        assert all_indices == list(range(8))
