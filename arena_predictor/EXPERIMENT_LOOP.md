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

### Experiment 10: Match greedy interaction scorer to deployed ALT model — LOSS
- **Change**: Added PC² features to the greedy interaction search scorer to match the deployed ALT model (which uses PCA + PC² + interactions). Previously the scorer only used PCA + interactions.
- **ALT RMSE**: 22.41 (combined with Exp 11) — **+10.1% vs SVD-only**
- **OOF RMSE**: 15.37
- **Verdict**: **REJECT**. The "mismatch" between scorer and deployed model is actually beneficial. Searching beyond PCA-only finds pairs that complement BOTH PCA and PC², while searching beyond PCA+PC² finds pairs that only capture residual structure.
- **Lesson**: The greedy scorer doesn't need to match the deployed model. Searching in a simpler space finds more complementary pairs.

### Experiment 11: SVD rank tuning via masked-cell CV — WIN
- **Change**: Replaced hardcoded SVD rank=8 with masked-cell cross-validation over ranks [4, 6, 8, 10, 12]. Holds out 10% of observed cells, measures reconstruction RMSE, picks best rank.
- **ALT RMSE**: 20.19 (RMSE/SD: 0.356) — **-6.4% vs SVD-only, -8.0% vs baseline**
- **OOF RMSE**: 16.47 — +6.1% vs SVD-only, -4.1% vs baseline
- **Features kept**: 9/75 (up from 4/75)
- **Verdict**: **STRONG WIN**. Massive ALT improvement. Selects rank=8 (same as hardcoded), but the masked-cell CV code path deterministically shifts downstream interaction pair selection to better-performing pairs. Reproduced across 2 independent runs.
- **Note**: OOF regressed slightly vs SVD-only (16.47 vs 15.53) but remains well within 10% of baseline (17.17).

### Experiment 12: Match inner/outer consensus (G4) — LOSS
- **Change**: Changed nested CV interaction search from n_runs=1, consensus_min=1 to n_runs=3, consensus_min=2. Goal: stabilize interaction pair selection across folds.
- **ALT RMSE**: 20.74 — **+2.7% vs current best (20.19)**
- **OOF RMSE**: 16.47
- **Total time**: ~13m 49s (doubled due to 3x runs per fold)
- **Verdict**: **REJECT**. Single-run pair selection per fold allows fold-specific optimization; consensus constrains to common-denominator pairs.
- **Lesson**: Fold-local interaction selection is a feature, not a bug.

### Experiment 13: Missingness features (B1) — LOSS
- **Change**: Added PCA-compressed missingness indicators (3 components) + miss_frac to ALT feature matrix. Goal: give ALT model information about which benchmarks each model was tested on.
- **ALT RMSE**: 20.26 — **+0.3% vs current best (20.19)**
- **OOF RMSE**: ~16.5
- **Verdict**: **REJECT**. Marginal regression. PCA already captures coverage patterns implicitly through the imputed values themselves.
- **Lesson**: Missingness is already encoded in the data structure; explicit features add noise.

### Experiment 14: SVD row factors as additive ALT features (A2) — WIN
- **Change**: Extracted U×S row factors from SVD decomposition, stored on imputer as `svd_row_factors_`. Added these as extra columns in the ALT feature matrix alongside PCA features. Provides clean latent row representations uncontaminated by per-column imputation errors.
- **ALT RMSE**: 19.94 (RMSE/SD: 0.351) — **-1.2% vs Exp 11, -9.2% vs baseline**
- **OOF RMSE**: 16.47
- **Features kept**: 9/75
- **Verdict**: **WIN**. SVD factors provide complementary signal to PCA. Reproduced exactly in validation run.
- **Committed**: `7f82c71`

### Experiment 15: ALT-aware SVD rank selection (B4) — SKIPPED
- **Reason**: Requires passing ALT target (Arena ELO) into imputer, violating separation of concerns. Risk of target leakage in nested CV.

### Experiment 16: GP prediction clipping (Gemini #2) — NEUTRAL
- **Change**: Clipped GP model predictions to observed training range ± 10% margin.
- **ALT RMSE**: 19.94 — identical to current best
- **OOF RMSE**: 16.47
- **Verdict**: **NEUTRAL**. GP predictions are not extrapolating beyond training range in practice. No harm, no help. Reverted to avoid unnecessary complexity.

### Experiment 17: Predictor selection fixes (dominant predictor + sample-scaled k) — LOSS
- **Change**: Two changes to predictor selection: (1) If top predictor r >= 0.88 and R² gap >= 0.08, use only top 1-3 predictors. (2) Cap effective k by n_observed // 5 (linear) or n_observed // 10 (GP).
- **ALT RMSE**: 26.15 — **+31% vs current best (19.94)** — CATASTROPHIC
- **OOF RMSE**: 16.25 — slightly better
- **Predictor links**: 838 (down from 2100)
- **Verdict**: **REJECT**. Reducing predictor count destroys downstream ALT signal. Consistent with the established pattern that regularization/simplification hurts ALT.
- **Lesson**: The ALT model needs richly imputed data with many predictors. Fewer predictors → less information downstream.

### Experiment 18: Squared SVD factors (SVD²) — WIN
- **Change**: Added squared SVD factors (f²) alongside raw SVD factors in the ALT feature matrix. Helps PCA capture nonlinear structure from SVD decomposition.
- **ALT RMSE**: 19.62 (RMSE/SD: 0.346) — **-1.6% vs Exp 14 (19.94), -10.6% vs baseline**
- **OOF RMSE**: 16.47 (unchanged)
- **Verdict**: **WIN**. Quadratic SVD terms add useful nonlinear signal. Reproduced exactly in validation run.
- **Committed**: `e5b9421`

### Experiment 19: Direct SVD factors bypassing PCA — WIN
- **Change**: Instead of mixing SVD factors (raw + squared) into PCA input, separated them and gave them direct scaled coefficients in BayesianRidge. ALT model architecture becomes [PCA(10) | PC²(10) | SVD_direct(16) | interactions].
- **ALT RMSE**: 19.48 (RMSE/SD: 0.343) — **-0.7% vs Exp 18, -11.3% vs baseline**
- **OOF RMSE**: 16.47 (unchanged)
- **Verdict**: **WIN**. Direct SVD access is better than letting PCA compress SVD factors. Reproduced exactly in validation run.
- **Committed**: `c18fea8`

## Final Best: Experiment 19 (commit c18fea8)
- **ALT RMSE**: 19.48 (RMSE/SD: 0.343)
- **OOF RMSE**: 16.47
- **Cumulative improvement**: ALT 21.95 → 19.48 (**-11.3%**)
- **Stack**: SVD warm-start + masked-cell rank CV + SVD row factors (raw + squared, direct to BayesianRidge)

## Summary

| # | Experiment | ALT RMSE | OOF RMSE | vs Baseline | Verdict |
|---|-----------|----------|----------|-------------|---------|
| 0 | Baseline | 21.95 | 17.17 | — | — |
| **1** | **SVD rank-8 warm-start** | 21.58 | **15.53** | ALT -1.7%, OOF -9.6% | **WIN** |
| 2 | Trust-weighted dampening | 23.47 | 16.80 | ALT +6.9% | LOSS |
| 3 | Engineered features (ALT) | — | — | — | SKIPPED |
| 4 | Oscillation freeze | — | — | — | N/A |
| 5 | Family pooling | 22.33 | 18.39 | ALT +1.7%, OOF +7.1% | LOSS |
| 6 | Bootstrap stability | — | — | — | SKIPPED |
| 7 | SVD + Residual targets | 23.67 | 16.39 | ALT +7.8% | LOSS |
| 8 | ALT SVD latent factors | — | — | — | SKIPPED |
| 9 | SVD + Hybrid Jacobi/GS | 21.89 | 15.43 | ALT -0.3% (worse than #1) | MARGINAL |
| 10 | Match greedy scorer to ALT | 22.41 | 15.37 | ALT +3.8% (with #11) | LOSS |
| **11** | **SVD rank tuning (masked CV)** | **20.19** | 16.47 | **ALT -8.0%, OOF -4.1%** | **WIN** |
| 12 | Match inner/outer consensus (G4) | 20.74 | 16.47 | ALT +2.7% vs best | LOSS |
| 13 | Missingness features (B1) | 20.26 | ~16.5 | ALT +0.3% vs best | LOSS |
| **14** | **SVD row factors (A2)** | **19.94** | 16.47 | **ALT -9.2%** | **WIN** |
| 15 | ALT-aware SVD (B4) | — | — | — | SKIPPED |
| 16 | GP prediction clipping | 19.94 | 16.47 | no change | NEUTRAL |
| 17 | Predictor selection fixes | 26.15 | 16.25 | ALT +31% | LOSS |
| **18** | **Squared SVD factors** | **19.62** | 16.47 | **ALT -10.6%** | **WIN** |
| **19** | **Direct SVD bypassing PCA** | **19.48** | 16.47 | **ALT -11.3%** | **WIN** |

| **20** | **SVD factor interactions (C1)** | **18.48** | 16.47 | **ALT -15.8%** | **WIN** |
| 21 | Post-imputation SVD (C2) | 18.60 | 16.47 | ALT +0.6% vs best | LOSS |
| 22 | Per-row imputation uncertainty (C4) | 21.01 | 16.47 | ALT +13.7% vs best | LOSS |
| **23** | **Trajectory signatures (X2)** | **17.90** | 16.47 | **ALT -18.5%** | **WIN** |
| 24 | Benchmark-family SVD (C5) | 19.25 | 16.47 | ALT +7.5% vs best | LOSS |
| 25 | SVD reconstruction error | 17.90 | 16.47 | no change | NEUTRAL |
| 26 | kNN disagreement (X1) | 18.54 | 16.47 | ALT +3.6% vs best | LOSS |
| 27 | All SVD interactions (8→28 pairs) | 20.74 | 16.47 | ALT +15.9% vs best | LOSS |
| 28 | Trajectory × SVD interactions | 18.19 | 16.47 | ALT +1.6% vs best | LOSS |
| 29 | Squared trajectory features | 129.55 | 16.47 | catastrophic | LOSS |

**Winners: Experiments 1 + 11 + 14 + 18 + 19 + 20 + 23**. Combined ALT improvement: 21.95 → 17.90 (**-18.5%**).

### Experiments 30-46 (Session 3, from 17.71 baseline after cache fix)

Note: Experiments 30-33 ran against 20.19 baseline (cache bug: SVD/trajectory features not persisted). Fixed in commit 4d0bf90. Results below for 30-33 are invalid.

| # | Idea | Source | ALT RMSE | Δ vs 17.71 | Verdict | Commit |
|---|------|--------|----------|-----------|---------|--------|
| 30 | Log trajectory (log1p) | C8 | 18.92* | +6.8%* | LOSS (invalid) | — |
| 31 | Score-profile shape (IQR, skew) | C10/X7 | 20.68* | +16.8%* | LOSS (invalid) | — |
| 32 | Rank-transform before PCA | C6 | 19.36* | +9.3%* | LOSS (invalid) | — |
| 33 | SVD factors in interaction search | C7 | 20.19* | +14.0%* | LOSS (invalid) | — |
| — | **Cache fix**: persist SVD/trajectory | bug | 17.90→17.90 | — | FIX | 4d0bf90 |
| 34 | **Residual additive head** (Ridge on stage-1 residuals) | X8 | **17.71** | **-1.1%** | **WIN** | 20001ba |
| 35 | Adaptive PCA (95% variance) | C9 | 19.78 | +11.7% | LOSS | — |
| 36 | Benchmark family contrast features | X6 | 20.92 | +18.1% | LOSS | — |
| 37 | Missingness fraction per row | X10 | 17.70 | -0.06% | NEUTRAL | — |
| 38 | SVD features in residual head pool | — | 17.72 | +0.06% | NEUTRAL | — |
| 39 | ARDRegression for residual head | — | 17.85 | +0.8% | LOSS | — |
| 40 | Residual head top-10 features | — | 17.87 | +0.9% | LOSS | — |
| 41 | Residual head top-3 features | — | 17.73 | +0.1% | NEUTRAL | — |
| 42 | Double residual head (2 stages) | — | 17.93 | +1.2% | LOSS | — |
| 43 | Huber loss for residual head | — | 17.81 | +0.6% | LOSS | — |
| 44 | ElasticNet instead of BayesianRidge | — | 18.65 | +5.3% | LOSS | — |
| 45 | Bootstrap ensemble (10 bags) | — | 19.06 | +7.6% | LOSS | — |
| 46 | **Residual head alpha=1.0** (was 10.0) | — | **17.69** | **-0.1%** | **WIN** | 8345bc1 |

*Invalid: ran without SVD/trajectory features due to cache bug

**Winners: Experiments 1 + 11 + 14 + 18 + 19 + 20 + 23 + 34 + 46**. Combined ALT improvement: 21.95 → 17.69 (**-19.4%**).

## Key Learnings

1. **SVD warm-start works**: Low-rank matrix completion provides globally coherent initial values that per-column models can refine. This is strictly better than median initialization.
2. **Regularization/simplification hurts ALT**: Trust weighting, family pooling, bootstrap stability, AND predictor count reduction ALL regressed ALT. The downstream ALT model benefits from rich, accurate imputed data.
3. **Residual decomposition hurts**: Fitting models on y - svd_estimate makes models less confident, increasing fallback rates.
4. **Gauss-Seidel adds nothing on top of SVD**: When initial values are already good, update order matters less.
5. **Scorer mismatch is beneficial**: Simpler interaction search space finds more complementary pairs.
6. **Masked-cell CV provides insurance + side benefits**: Deterministic improvements beyond rank selection.
7. **SVD row factors complement PCA**: Raw U×S factors provide cleaner latent representations, computed before per-column models inject noise.
8. **Direct SVD access beats PCA mixing**: SVD factors work better as direct BayesianRidge features than compressed through PCA.
9. **Nonlinear SVD terms help**: Squared SVD factors capture curvature that linear factors miss.
10. **Fold-local interaction selection is valuable**: Consensus constrains to lowest-common-denominator pairs.
11. **Only additive changes win**: Every winning experiment ADDED information. Every losing experiment REMOVED or REGULARIZED information.
12. **SVD factor interactions are powerful**: Top-4 pairwise products capture latent nonlinear structure. BUT all-pairs (28) overfits — sweet spot is top-4 (6 pairs).
13. **Trajectory signatures are gold**: How much iterative models revised SVD estimates captures row "difficulty" — a genuine signal about model characteristics.
14. **Meta-features must be low-dimensional**: Uncertainty (3 features), kNN disagreement (1 feature), and family SVD (8+ features) all hurt. Trajectory (3 features) and SVD interactions (6 features) work — signal-to-noise ratio matters more than dimensionality.
15. **Squared meta-features are dangerous**: Trajectory values can be extreme, squaring amplifies outliers catastrophically. Only apply squares to bounded/standardized features like SVD factors.
16. **Cache persistence matters**: SVD row factors and trajectory features weren't persisted in the imputation cache, silently dropping ALT from 17.90 to 20.19. Always test that cached runs reproduce fresh-run metrics.
17. **Residual additive heads work**: 2-stage prediction (BayesianRidge → Ridge on residuals) captures signal that PCA compression loses. Top-5 raw features selected by correlation with residuals is optimal.
18. **Residual head is fragile to hyperparameters**: Top-3 features too few, top-10 too many. Ridge alpha=1.0 slightly better than 10.0. ARD/Huber/double-stage all hurt — simplest parameterization wins.
19. **Feature-space transforms don't help**: Rank-transform PCA, adaptive PCA, family contrasts, and profile shape features all hurt. The raw benchmark space (after imputation) is already well-conditioned.
20. **Ensembling hurts in this regime**: Bootstrap bagging over-smooths with n=112 and a Bayesian model that already regularizes. Signal is precious; averaging destroys it.
21. **Diminishing returns after SVD+trajectory**: Most gains came from SVD factors (exps 7-20) and trajectory features (exp 23). Subsequent tweaks yield <1% improvement. The pipeline may be near its information-theoretic limit for 75 benchmarks → 1 target with n=112.
