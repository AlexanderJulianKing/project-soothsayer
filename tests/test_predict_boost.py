"""Regression tests for top-tier boost handling in predict.py."""

import os
import sys

import numpy as np
import pandas as pd

# Add arena_predictor to path so we can import predict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "arena_predictor"))

from predict import (  # type: ignore
    ALT_TARGET,
    _apply_top_tier_boost,
    _build_repeated_splits,
    _precompute_alt_fold_data,
)


def test_apply_top_tier_boost_accepts_explicit_settings():
    """Explicit boost args should work independently of module globals."""
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([1300.0, 1410.0, 1450.0, 1460.0, 1500.0, 1200.0])
    w = np.ones(len(y))

    X_aug, y_aug, w_aug = _apply_top_tier_boost(
        X,
        y,
        w,
        boost=3,
        threshold=1450.0,
    )

    # Rows at indices 2, 3, 4 should each appear 3x total.
    assert X_aug.shape[0] == 12
    assert y_aug.shape[0] == 12
    assert w_aug.shape[0] == 12


def test_precompute_alt_fold_data_parallel_respects_boost():
    """Sequential and process-parallel fold prep should apply the same boost."""
    n = 12
    X_df = pd.DataFrame({
        "feat_a": np.linspace(0.0, 1.0, n),
        "feat_b": np.linspace(1.0, 2.0, n),
        ALT_TARGET: np.linspace(1350.0, 1515.0, n),
    })
    y = np.linspace(1400.0, 1510.0, n)
    splits = _build_repeated_splits(n, n_splits=3, repeats=1, seed=42)

    fold_data_seq = _precompute_alt_fold_data(
        X_df,
        y,
        ALT_TARGET,
        splits,
        n_splits=3,
        oof_repeats=1,
        seed=42,
        sample_weight=None,
        selector_cfg={"enabled": False},
        poly_cfg={"enabled": False},
        n_jobs=1,
        top_tier_boost=3,
        top_tier_threshold=1450.0,
    )
    fold_data_par = _precompute_alt_fold_data(
        X_df,
        y,
        ALT_TARGET,
        splits,
        n_splits=3,
        oof_repeats=1,
        seed=42,
        sample_weight=None,
        selector_cfg={"enabled": False},
        poly_cfg={"enabled": False},
        n_jobs=2,
        top_tier_boost=3,
        top_tier_threshold=1450.0,
    )

    seq_sizes = [fd["Xtr"].shape[0] for fd in fold_data_seq]
    par_sizes = [fd["Xtr"].shape[0] for fd in fold_data_par]
    base_sizes = [len(tr) for tr, _ in splits]

    assert seq_sizes == par_sizes
    assert any(boosted > base for boosted, base in zip(seq_sizes, base_sizes))
