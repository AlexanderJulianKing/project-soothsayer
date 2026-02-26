
# Collinearity-Aware Per-Target Feature Selector

This update adds a per-target predictor selector to `column_imputer.py` that improves stability and accuracy when many weak or collinear features exist.

## Algorithm: Relevance → De-correlate → Residual rescue

For each target column `y`:

1. **Relevance (fast):** Rank candidates by `max(|Spearman|, kNN-MI)` computed on *complete pairs* `(y, x)` only. Keep the top `k_seed`.
2. **De-correlate (τ-cap):** Greedily keep features so that `|corr(x, s)| ≤ τ` for all kept `s`, measured on intersecting observed rows.
3. **Residual rescue (wrapper):** Fit a quick Ridge on complete cases, score remaining candidates against residuals, and add the best if it improves CV-MSE by ≥ `delta_improve` and passes the τ-cap. Repeat up to `k_max`.

Outputs an ordered list `S_y` plus per-target **means/stds** from the complete cases used during selection (used to standardize the linear branch).

## Integration

- Selection runs **once before the first pass** on the original observed data and is cached per target.
- Both the **Bayesian Ridge** linear branch and the **nonlinear** branch (CatBoost or HGBR) are trained and predicted **only on the selected features**.
- Linear branch uses `fit_intercept=False` and standardizes selected raw features with the **complete-case means/stds**; missing values are imputed with the per-feature mean, and missingness indicators for the selected features are appended.
- Gate diagnostics and any kNN-based measures use only the selected features for each target.
- Determinism: `random_state` is propagated to CV splits, MI, and all models.

## New parameters

```python
use_feature_selector: bool = True
selector_min_pairs: int = 200
selector_tau: float = 0.90
selector_k_seed: int = 12
selector_k_max: int = 30
selector_delta_improve: float = 0.01
selector_use_mi: bool = True
reselect_every_n_passes: Optional[int] = None

linear_model: Literal["bayesian_ridge","elasticnet","ard"] = "bayesian_ridge"
linear_standardize: bool = True
random_state: Optional[int] = 42
```

> Opinionated defaults: keep **Bayesian Ridge**, compute selection once, and apply the same selected features to both branches.

## Quick start

```python
from column_imputer import SpecializedColumnImputer
imp = SpecializedColumnImputer(use_feature_selector=True, selector_tau=0.9, random_state=42)
X_completed = imp.fit_transform(X_with_nans)
print(imp._selected_predictors["target_col"])
```

## Tests & Benchmark

- See `tests/` for selector unit tests (collinearity cap, residual rescue, and low-pairs fallback).
- Run `python benchmarks/benchmark_selector.py` for a small synthetic benchmark reporting RMSE/SD vs baseline.
