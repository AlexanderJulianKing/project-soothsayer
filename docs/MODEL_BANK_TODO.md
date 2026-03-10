# ModelBankImputer: Status & Findings

## Current Status: DEFAULT IMPUTER

ModelBankImputer is the production default. It outperforms SpecializedColumnImputer
on current data (15.2 vs 16.9 OOF RMSE). The previously-reported 14.32 specialized
baseline was unreproducible (likely a lucky ALT pair draw).

Best config: `--imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp`

## Completed Experiments (23 total)

| Experiment | OOF RMSE | Result |
|---|---|---|
| No coherence | 16.39 | Baseline model-bank |
| + Coherence λ=1.0 (linear) | 15.64 | +0.75 improvement |
| + Coherence (exp shape) | 15.57 | Best coherence shape |
| + Completeness² sample weights | ~15.02-15.64 | Small help, high variance |
| Coherence λ=0.5 | 15.17 | Worse |
| Per-column coherence (λ/completeness) | 16.31 | Harmful |
| Observed-only SVD anchor | 17.31 | Much worse |
| Delta-to-anchor trajectory | 16.00 | Worse |
| Both σ² + anchor trajectory | 16.98 | Worse |
| More expansion passes (2×1) | 15.73 | Neutral |
| More confident extras (1×2) | 15.70 | Neutral, ALT degraded |
| 2 passes × 2 extras | 15.66 | Neutral, ALT degraded |
| Coherence squared | 16.26 | Worse |
| Coherence power3 | 15.97 | Worse |
| Coherence step | 16.26 | Worse |
| SVD row factors as predictors | 16.11 | Worse |
| σ²-weighted downstream | 16.18-16.74 | Worse — too few rows |
| Iterative coherence | 15.81 | Neutral |
| Prediction-level stacking | LOO 15.06 | Marginal improvement |
| Row-adaptive coherence gate | 15.90 | Neutral |
| Learned coherence gate | 17.09 | Worse — overfitting |
| Multiple imputation K=5 | 20.19 | Much worse — σ² too large |
| Specialized baseline (fresh, current data) | 16.92 | Model-bank wins |

## Key Insights

- **Post-imputation coherence is the only lever that works** — +0.75 RMSE. Everything else is noise.
- **Imputer optimization is exhausted.** 23 experiments, none beat the simple exp coherence at λ=1.0.
- **ALT pair search variance dominates** — ±2 RMSE from different interaction pairs dwarfs imputer tuning. This is the biggest remaining opportunity.
- **MI fails fundamentally** — LOO MSE ≠ posterior variance. Adding noise calibrated to expected error just corrupts the data.
- **Correlation inflation in specialized is real** — but model-bank's honest imputation now wins anyway on raw RMSE.

## Next Steps: Downstream Pipeline

The imputer is done. Remaining gains are in the downstream pipeline:

### 1. Stabilize ALT interaction pair search
**Impact: High** — ALT pair variance (±2 RMSE) is the single biggest source of instability.
- More consensus repeats (10-20 instead of 5)
- Stability-weighted pair selection (penalize pairs that only appear in some repeats)
- Ensemble of top-N pair sets

### 2. Feature selection improvements
- Current tree-based ranking + 1-SE rule works but is noisy with 75 features and 140 rows
- Consider stability selection (repeated subsampling)
- Cross-validate the number of ALT interaction terms (currently greedy forward search)

### 3. Model diversity
- Current: ARDRegression, BayesianRidge, HuberRegressor, Blend50, BlendWeighted
- Consider: ElasticNet grid, Gaussian Process, ensemble methods

## Infrastructure Built

- `--freeze_alt_pairs <path>`: Frozen ALT interaction pairs for stable evaluation
- Pickle-based imputation cache: exact float round-trip
- `--coherence_shape {linear,squared,power3,exp,step}`: Parameterized shrinkage weight
- `--coherence_gate {fixed,row_adaptive,learned}`: Per-cell coherence gate variants
- `--n_imputations K`: Multiple imputation (experimental, not useful)
- `--iterative_coherence`, `--use_svd_predictors`, `--sigma2_weights`: Experiment flags
- `stack_imputers.py`: Prediction-level stacking harness
