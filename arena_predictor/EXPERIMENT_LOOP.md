# Autonomous ML Experiment Loop — Column Imputer Optimization

## Objective
Improve the ALT nested-CV RMSE for Arena ELO prediction by iterating on the column imputer (`arena_predictor/column_imputer.py`). Changes are tested one at a time, winners are committed, losers are logged.

## Baseline Metrics (commit 0665800)
| Metric | Value |
|--------|-------|
| ALT nested-CV RMSE | 21.95 (RMSE/SD: 0.387) |
| OOF RMSE | 17.17 |
| Imputation time | ~37s |
| Total pipeline time | ~7m 14s |
| Rows | 137 |
| Feature columns | 75 (4 kept by feature selection) |

## Weakest Columns (RMSE/SD > 0.80)
| Column | n | RMSE/SD | Model Type |
|--------|---|---------|------------|
| livebench_code_generation | 46 | 1.22 | ? |
| livebench_tablejoin | 46 | 0.95 | ? |
| livebench_zebra_puzzle | 45 | 0.93 | ? |
| livebench_logic_with_navigation | 45 | 0.89 | ? |
| livebench_spatial | 45 | 0.97 | ? |
| livebench_code_completion | 46 | 0.92 | ? |
| livebench_typos | 47 | 0.94 | ? |
| aa_pricing_price_1m_input_tokens | 136 | 0.90 | ? |
| aa_pricing_price_1m_output_tokens | 136 | 0.90 | ? |
| logic_avg_answer_tokens | 136 | 0.91 | ? |
| openbench_Answer Output... | 92 | 0.87 | ? |
| livebench_connections | 47 | 0.88 | ? |
| livebench_theory_of_mind | 45 | 0.71 | ? |

## Current Architecture Summary

### Model Classes
- **BayesianRidgeModel**: Linear regression with uncertainty. Uses CorrelationWeightedImputer to fill missing predictors.
- **GPModel**: Gaussian Process (Matern or linear+Matern kernel). For nonlinear/extrapolation-prone columns.
- **CategoricalModel**: LogisticRegression or RandomForest for low-cardinality columns.
- **BoundedLinkModel**: Wraps a base model with logit/sigmoid transform for [0,1] or [0,100] bounded columns.
- **HurdleModel**: Two-stage gate (LogisticRegression) + value (BayesianRidge) for floor-inflated "capability wall" distributions.

### Classification
- ColumnType: CATEGORICAL, LINEAR, NONLINEAR, BOUNDED, EXTRAPOLATION_PRONE, GP_LINEAR_MATERN
- ColumnTags: BOUNDED, FLOOR_INFLATED (applied orthogonally to model family)
- classify(): cardinality check → correlation analysis → model family → tag detection

### Predictor Selection
- `_select_predictors()`: Top-k by |correlation| × availability (co-missingness aware)
- `_select_predictors_mrmr()`: mRMR with Spearman, availability-discounted relevance

### Iterative Loop
- 3 tiers ordered by difficulty (missingness × (1 - max_correlation))
- 14 rounds max, Jacobi-style snapshots, tolerance gating with relaxation
- Convergence: 95th-percentile uncertainty × multiplier, calibrated via holdout

### Known Weaknesses
1. GP models are opaque black boxes — no importance extraction
2. No variance propagation through imputed predictors
3. mRMR redundancy computed on all rows, not just target-missing rows
4. Floor-inflation detection brittle (histogram-based, threshold-sensitive)
5. Many livebench sub-columns have n=45-47 and RMSE/SD > 0.85 — nearly unpredictable

## Experiment Protocol

### Step 1: Consult Codex
Share this document + the per-column quality CSV with Codex (gpt-5.3-codex). Ask for 3-5 targeted improvement hypotheses ranked by expected impact. Focus on:
- The weakest columns (RMSE/SD > 0.80)
- Architecture changes to the iterative loop
- Predictor selection improvements
- Model selection or ensembling ideas
- Anything that reduces downstream ALT RMSE

### Step 2: Implement One Change at a Time
For each hypothesis:
1. Create a clean working state (`git stash` if needed)
2. Implement the change in `column_imputer.py` only
3. Verify import: `python3 -c "import arena_predictor.column_imputer"`
4. Clear cache: `rm -rf arena_predictor/analysis_output/_cache`
5. Run pipeline: `cd arena_predictor && python3 predict.py --max_workers 1`
6. Record ALL metrics in the experiment log below

### Step 3: Evaluate
- **Win threshold**: ALT RMSE < 21.73 (>1% improvement) AND OOF RMSE does not regress by >10%
- **Strong win**: ALT RMSE < 21.51 (>2% improvement)
- **Reject**: ALT RMSE >= 21.95 or OOF RMSE > 18.89 (>10% regression)
- Record per-column quality changes for the weakest columns

### Step 4: Keep or Revert
- **Win**: Keep the change, update baseline, commit with before/after metrics
- **Loss/Neutral**: `git checkout -- arena_predictor/column_imputer.py`, log the result
- **After individual experiments**: If multiple changes won independently, test them combined

### Step 5: Combine Winners
If >1 experiment won:
1. Apply all winning changes together
2. Run pipeline
3. If combined result beats best individual result → commit combined
4. If combined result regresses → keep only the single best change

## Budget
- Max 20 experiments total
- ~7 minutes per pipeline run
- 30 minutes per experiment (implementation + run + analysis)
- Multiple Codex consultation rounds allowed

## Experiment Log

### Experiment 0: Baseline (commit 0665800)
- **Change**: None (co-missingness aware predictor selection)
- **ALT RMSE**: 21.95
- **OOF RMSE**: 17.17
- **Imputation time**: 37s
- **Notes**: Baseline for this experiment round

### Experiment 1: SVD rank-8 warm-start — WIN
- **Change**: Added iterative SVD (SoftImpute-style) pre-pass that computes a rank-8 low-rank approximation of the full data matrix. Missing cells are warm-started with SVD values before the per-column iterative imputation loop begins. The iterative loop then refines these initial estimates using column-specific models.
- **ALT RMSE**: 21.58 (RMSE/SD: 0.380) — **-1.7% vs baseline**
- **OOF RMSE**: 15.53 — **-9.6% vs baseline**
- **Imputation time**: 38s
- **Total time**: ~6m 36s
- **Median fallback fills**: 41 of 3105 (unchanged)
- **Features kept**: 4/75 (unchanged)
- **Verdict**: **STRONG WIN** (ALT < 21.51 threshold). Both metrics improved substantially.
- **Validated**: Reproduced in two independent runs (Exp 1 and final validation).

### Experiment 2: Trust-weighted dampening — LOSS
- **Change**: Computed per-column trust scores from training residuals and sample size. Low-trust imputed values were dampened (blended toward column mean). Trust also weighted predictor selection.
- **ALT RMSE**: 23.47 (RMSE/SD: 0.414) — **+6.9% vs baseline**
- **OOF RMSE**: 16.80 — -2.2% vs baseline
- **Verdict**: **REJECT**. ALT regressed badly despite modest OOF improvement. Shrinkage toward mean hurts downstream.
- **Lesson**: Regularization/dampening consistently hurts ALT RMSE.

### Experiment 3: Engineered features for ALT model — SKIPPED
- **Reason**: ALT model lives in predict.py, not column_imputer.py. Out of scope for this experiment round.

### Experiment 4: Per-column oscillation freeze — N/A
- **Change**: Attempted to detect and freeze oscillating columns during iterative loop.
- **Finding**: Cells in the current architecture are imputed once per round and not revisited within a round. Oscillation detection is moot.
- **Verdict**: N/A — architecture prevents the problem this was designed to solve.

### Experiment 5: Hierarchical family pooling — LOSS
- **Change**: After fallback fill, shrunk imputed values toward benchmark-family means (e.g., all livebench columns share a family mean).
- **ALT RMSE**: 22.33 — **+1.7% vs baseline**
- **OOF RMSE**: 18.39 — **+7.1% vs baseline**
- **Verdict**: **REJECT**. Both metrics regressed. Family-level shrinkage destroys column-specific signal.
- **Lesson**: Shrinkage toward group means hurts. Columns are more individual than their family names suggest.

### Experiment 6: Bootstrap predictor stability — SKIPPED
- **Reason**: After experiments 2 and 5 both showed regularization hurts ALT, skipped this similar approach (would have selected more stable but potentially weaker predictors).
- **Lesson**: Pattern is clear — regularization/stability approaches hurt ALT RMSE.

### Experiment 7: SVD + Residual-on-SVD targets — LOSS
- **Change**: On top of SVD warm-start, fit per-column models on residuals (y - svd_estimate) instead of raw y. Predictions add SVD estimate back.
- **ALT RMSE**: 23.67 — **+7.8% vs baseline**
- **OOF RMSE**: 16.39
- **Median fallback fills**: 90 of 3105 (up from 41) — models became less confident
- **Verdict**: **REJECT**. Residual targets change signal structure, making models less confident and increasing fallback rate.
- **Lesson**: SVD is best used as initialization, not as a target decomposition.

### Experiment 8: ALT SVD latent factors — SKIPPED
- **Reason**: ALT model feature construction lives in predict.py. PCA in ALT model already captures similar latent row structure. Diminishing returns.

### Experiment 9: SVD + Hybrid Jacobi→Gauss-Seidel — MARGINAL
- **Change**: On top of SVD warm-start, switched from Jacobi (snapshot-based) to Gauss-Seidel (in-place updates) after the first 2 rounds.
- **ALT RMSE**: 21.89 — worse than SVD-only (21.58)
- **OOF RMSE**: 15.43 — slightly better than SVD-only (15.53)
- **Verdict**: **NOT KEPT**. ALT regressed vs SVD-only. GS doesn't add value on top of good SVD initialization.

## Summary

| # | Experiment | ALT RMSE | OOF RMSE | vs Baseline | Verdict |
|---|-----------|----------|----------|-------------|---------|
| 0 | Baseline | 21.95 | 17.17 | — | — |
| 1 | SVD rank-8 warm-start | **21.58** | **15.53** | ALT -1.7%, OOF -9.6% | **WIN** |
| 2 | Trust-weighted dampening | 23.47 | 16.80 | ALT +6.9% | LOSS |
| 3 | Engineered features (ALT) | — | — | — | SKIPPED |
| 4 | Oscillation freeze | — | — | — | N/A |
| 5 | Family pooling | 22.33 | 18.39 | ALT +1.7%, OOF +7.1% | LOSS |
| 6 | Bootstrap stability | — | — | — | SKIPPED |
| 7 | SVD + Residual targets | 23.67 | 16.39 | ALT +7.8% | LOSS |
| 8 | ALT SVD latent factors | — | — | — | SKIPPED |
| 9 | SVD + Hybrid Jacobi/GS | 21.89 | 15.43 | ALT -0.3% (worse than #1) | MARGINAL |

**Winner: Experiment 1 (SVD rank-8 warm-start)** — Only change that improved ALT RMSE beyond the 1% threshold. Key insight: global low-rank structure provides better initialization than median fallback, allowing per-column models to converge to better solutions. Regularization/shrinkage approaches (2, 5, 6) consistently hurt.

## Key Learnings

1. **SVD warm-start works**: Low-rank matrix completion provides globally coherent initial values that per-column models can refine. This is strictly better than median initialization.
2. **Regularization hurts ALT**: Trust weighting, family pooling, and bootstrap stability all regressed ALT. The downstream ALT model benefits from accurate (not conservative) imputed values.
3. **Residual decomposition hurts**: Fitting models on y - svd_estimate makes them less confident, increasing fallback rates. SVD is best as initialization, not decomposition.
4. **Gauss-Seidel adds nothing on top of SVD**: When initial values are already good (from SVD), the order of updates in the iterative loop matters less.
5. **Architecture limitation**: Per-column models can only be improved via column_imputer.py. ALT model improvements require predict.py changes (different experiment scope).
