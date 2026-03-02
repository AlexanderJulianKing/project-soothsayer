# Findings: Alt-Target Prediction & Model Coverage

## Summary

Predicting LMArena (length-adjusted) scores from 74 benchmark features across 100 models. Best approach: PCA(10) → BayesianRidge, achieving RMSE/SD ≈ 0.345 (R² ≈ 0.88). This is near-ceiling for n=100 with benchmark-only features.

---

## 1. Model Selection

### PCA(10) → BayesianRidge wins decisively

Tested 22+ approaches with identical 5-fold CV splits:

| Approach | RMSE/SD | Notes |
|----------|---------|-------|
| PCA(10) → BayesianRidge | **0.345** | Best overall |
| CatBoost (tuned) | 0.343 | Overfits in family-aware CV |
| PCA(10) → BR + pooling (50 draws) | 0.343 | Marginal gain from multiple imputation |
| BayesianRidge (all features) | 0.361 | No dimensionality reduction |
| RidgeCV | 0.366 | Built-in regularization insufficient |
| BR + LGBM feature selection | 0.400 | **Feature selection hurts** |
| LightGBM direct | 0.417 | Tree models overfit at n=100 |
| Random Forest | 0.456 | Worst performer |

### Why PCA works

- **Denoises imputation artifacts**: 31% of cells are imputed. PCA averages noise across components, preventing the model from fitting imputation errors.
- **Eliminates multicollinearity**: 74 features with heavy correlation (many benchmarks measure similar abilities). PCA decorrelates them.
- **Captures latent factors**: The first 10 components capture "general capability", "reasoning vs chat", "code vs language" etc. — the actual latent structure arena scores depend on.
- **No interactions needed**: Degree-2 and degree-3 polynomial terms on PCA components show zero signal surviving Bonferroni correction (800 tests). The relationship is genuinely linear in latent space.

### Why LGBM feature selection hurts

Hard top-k feature selection at n=100 is unstable: different CV folds select different features, introducing variance. It also biases toward sparse/noisy features (eqbench) that have high importance in individual trees but poor generalization. PCA keeps all features and lets regularization handle the rest.

### Family-aware CV (most honest evaluation)

GroupKFold holding out entire model families (Claude, DeepSeek, GPT, etc.) simulates the real prediction task:

| Approach | Family CV RMSE/SD |
|----------|-------------------|
| PCA(10) → BR | **0.407** |
| CatBoost tuned | 0.419 |
| RidgeCV | 0.425 |
| LightGBM | 0.447 |

PCA → BR generalizes best to unseen model families.

---

## 2. Residual Analysis & Ceiling

### r(residual, y) ≈ RMSE/SD is algebraic

The persistent correlation between residuals and actual scores (~0.37) across all approaches is **not recoverable signal** — it's a mathematical consequence of R² < 1. For any model with RMSE/SD = σ, the correlation r(residual, y) ≈ σ. This was confirmed independently.

### Heteroscedasticity

The model is more accurate for high-scoring models and systematically overpredicts weak models:
- Top residuals: Nova 2 Lite (-60), Phi-4 (-55), Nova Micro (-40)
- These are models where arena performance diverges from benchmark performance (arena captures conversational qualities benchmarks don't fully measure for small models)

This is an **omitted variable problem**, not a modeling problem. The conversational quality features (eqbench r=0.88, writerbench r=0.81) ARE present and observed for these models — the issue is fundamental limits of benchmark-based prediction for arena scores.

### Nothing fixes it

Tested: target transforms (logit, rank), post-hoc calibration (linear, isotonic, spline), heteroscedastic weighted regression, two-regime gated models, asymmetric loss weighting, quantile regression. None improved RMSE/SD below 0.343.

---

## 3. Imputation Findings

### Multiple imputation pooling: marginal benefit

Approximated multiple imputation by adding N(0, column_RMSE) noise to imputed cells across 50 draws, averaging PCA(10)→BR predictions:

- Baseline (single imputation): 0.345
- Pooled (50 draws): 0.343
- Improvement: ~0.6%

PCA already denoises imputation artifacts, limiting the benefit. Between-imputation prediction std correlates with missingness (r=0.43) but not with prediction error (r=0.01).

### eqbench features: noisy but not critical

- eqbench columns are 28-36% populated with imputation RMSE/SD ≈ 0.47
- eq is 99% populated and serves as a dense proxy for the same signal
- Removing eqbench entirely barely hurts — signal flows through correlated features
- Running more EQ-Bench evaluations would have negligible impact vs. adding new training rows

### Imputation quality is not the bottleneck

The constraint is **n=100 training rows**, not imputation fidelity. SpecializedColumnImputerV2 or more benchmark coverage on existing models would not meaningfully move the needle.

---

## 4. Model Coverage Analysis

### Current state (as of 2026-02-09)

- 125 models in combined CSV, 100 with lmarena_Score
- 303 models on LMArena total
- 344 models on OpenRouter
- ~98 of 100 training models are on OpenRouter

### Custom benchmark coverage

| Benchmark | Models missing (of 100 training) |
|-----------|----------------------------------|
| logic | 0 |
| style | 0 |
| writing | 0 |
| tone | 0 (some partial: 2/3 judges) |
| eq | 1 (Claude 3.5 Haiku) |
| weirdml | **44** |

WeirdML is the largest coverage gap across existing training models.

### New models added to pipeline

10 models newly added (mappings + openbench entries). All need custom benchmark data to pass the 50.8% sparsity threshold:

| Model | LMArena | Status |
|-------|---------|--------|
| LongCat Flash Chat | 1399 | Already mapped, 78.9% missing |
| Kimi K2.5 (Non-Reasoning) | 1438 | Newly added |
| MiniMax M1 | 1367 | Newly added |
| GLM-4.7 Flash | 1362 | Already mapped, 60.5% missing |
| Intellect-3 | 1356 | Newly added |
| Llama 3.1 Nemotron Ultra 253B | 1347 | Newly added |
| OLMo 3.1 32B Instruct | 1330 | Newly added |
| Gemma 3n 4B | 1319 | Already mapped, 70.4% missing |
| Mercury | 1310 | Newly added |
| Olmo 3 32B (Reasoning) | 1306 | Already mapped, 72.4% missing |

### Additional candidates (not yet added)

14 more models on OpenRouter + LMArena that could be added. Particularly valuable for extending the score range below 1212 (current minimum):

- grok-3-mini-beta (1357), qwen-plus-0125 (1346), nemotron-70b (1299)
- llama-3-70b (1276), gemma-2-9b (1266), claude-3-haiku (1262)
- gpt-3.5-turbo (1225), llama-3-8b (1224), llama-3.2-3b (1167), llama-3.2-1b (1112)

Adding these would improve calibration for weaker models, which is where prediction error is currently highest.

---

## 5. What Would Actually Move the Needle

In order of expected impact:

1. **More training rows (n)** — Going from 100 → 120+ models is the single biggest lever. Each new model with arena scores AND benchmark data directly improves generalization. The 10 newly-added models represent the easiest wins here.

2. **Extending the score range** — Adding models scoring 1100-1250 would improve calibration at the bottom, where heteroscedasticity is worst. The 14 additional candidates above are ideal for this.

3. **WeirdML coverage** — 44 of 100 training models lack WeirdML data. Filling this is the biggest gap for existing models.

4. **Waiting** — New models appear on LMArena continuously. Each one that also has benchmarks is a free training row.

Things that would NOT meaningfully help:
- Better imputation (SpecializedColumnImputerV2)
- More EQ-Bench runs
- ~~Polynomial interactions or nonlinear models~~ (see Section 6 — ALT-centric interactions do help)
- Post-hoc calibration schemes
- More benchmark sources measuring the same abilities

---

## 6. Target Model Optimization (Feb–Mar 2026)

### Summary

Starting from the original PCA(10)→BayesianRidge pipeline, a multi-session experiment campaign optimized the target model (the model that predicts Arena ELO from imputed benchmark features). Cumulative improvement:

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| BlendWeighted CV RMSE | 15.70 | **14.78** | **-0.92 (-5.9%)** |
| OOF RMSE | 15.94 | **14.92** | **-1.02 (-6.4%)** |
| ALT nested-CV RMSE | 17.33 | 17.33 | 0 (unchanged) |
| Runtime | ~25min | ~9min | -64% |

### What won

**1. Imputation trajectory features (biggest single win: -0.69 OOF)**

The column imputer tracks three per-row statistics across its iterative passes:
- `_traj_mean_delta`: Mean absolute update across imputation passes
- `_traj_max_delta`: Max absolute update across passes
- `_traj_n_imputed`: Number of originally-missing cells per row

These were already used by the ALT model (Arena score imputation). Adding them to the *target* model (Arena ELO prediction) produced the largest single improvement in the entire campaign. The feature selector typically picks one trajectory feature alongside ~9 benchmark features.

**2. ALT-centric polynomial interactions (-0.15 CV)**

Instead of generic variance-based polynomial feature expansion (which picks the 6 highest-variance columns and creates all pairwise products), ALT-centric expansion:
1. Computes residuals after a simple ALT-only linear fit
2. Creates ALT×feature interaction terms for each non-ALT feature
3. Ranks them by |correlation| with the residual
4. Keeps top-k (k=5 optimal)

This is more principled because it specifically targets interactions that explain variance *beyond* the dominant ALT feature, rather than wasting interaction terms on high-variance features that may not interact meaningfully with the target.

**3. HuberRegressor in the model blend (-0.30 CV in original pre-trajectory setting)**

Adding HuberRegressor (Huber loss, epsilon=1.35) as a second model alongside ARDRegression creates a diverse 2-model blend. The key insight is *diversity of errors* — ARD uses Bayesian shrinkage (pulls toward zero), Huber uses robust loss (downweights outliers). They make different mistakes on different models, and their equal-weight average cancels out much of the error.

**4. Fast uncertainty fallback (-15min runtime, neutral accuracy)**

For models without native uncertainty estimation (Huber, BR_YeoJohnson), replaced an expensive full CV for prediction intervals with a simple training residual RMSE as flat uncertainty. This cut runtime by ~60% with no accuracy change.

### What lost (and why)

| Experiment | CV RMSE | Why it failed |
|-----------|---------|---------------|
| Orthogonalized ALT interactions | 15.51 | Removed useful signal; ARD handles multicollinearity via regularization |
| KernelRidge (RBF) | 220.02 | Catastrophic overfit at n=90; RBF kernel has far too many effective parameters |
| Residual correction head (Ridge on blend OOF residuals) | 16.10 | Overfits with ~72 training points per fold; residual signal is mostly noise |
| PLS (Partial Least Squares) | 20.46 | Catastrophic; PLS constructs latent factors to maximize covariance with target, which overfits wildly at n=90 |
| ElasticNet in blend | 15.40 | Marginal individual gain, but diluted the blend (too similar to ARD) |
| BlendRidge meta-learner (stacking) | 15.75 | Overfits; learning blend weights from OOF predictions with n=90 adds noise |
| SVD latent factors as features | 15.90 | Signal already captured by imputed features + ALT |
| ALT prediction uncertainty as feature | 17.03→19.19 | Leaked into inner ALT model; after fix, feature wasn't even selected |
| BayesianRidge / Ridge / TheilSen as 3rd blend member | 15.00–15.01 | All too similar to ARD; 3rd model consistently dilutes the 2-model blend |
| Row completeness as feature | 15.26 | Collinear with `_traj_n_imputed`; redundant information hurt via multicollinearity |
| BR_YeoJohnson (Yeo-Johnson target transform) | 15.36 | Small win as 4th model, but trimmed for runtime; 2-model blend nearly as good |

### Full experiment log

Session 1 (from ~15.70/15.94 baseline):
| # | Experiment | CV | OOF | Verdict |
|---|-----------|-----|------|---------|
| 1 | Blend50 (equal-weight avg of all models) | 15.61 | 15.84 | WIN (became default) |
| 2 | DeltaOnALT (predict y-ALT residual) | 16.80 | — | LOSS |
| 3 | Exclude ALT from poly interactions | 15.70 | 15.93 | NEUTRAL |
| 4 | Residual calibrator (spline on OOF residuals) | 15.59 | 15.95 | NEUTRAL |
| 5 | Feature selection on y-ALT residuals | 15.65 | 15.87 | MARGINAL |

Session 2 (from ~15.70/15.94 baseline):
| # | Experiment | CV | OOF | Verdict |
|---|-----------|-----|------|---------|
| 6 | SVD factors as target features | 15.90 | 16.15 | LOSS |
| 7 | **ALT-centric poly interactions** | **15.55** | **15.79** | **WIN (k=4)** |
| 8 | PLS | 20.46 | 20.62 | CATASTROPHIC |
| 9 | ALT uncertainty as feature | — | — | LOSS (leakage) |

Session 3 (from 15.55/15.79 after ALT-centric poly):
| # | Experiment | CV | OOF | Verdict |
|---|-----------|-----|------|---------|
| 10 | BR_YeoJohnson (target transform) | 15.36 | 15.62 | WIN (later removed for runtime) |
| 11 | **HuberRegressor in blend** | **15.45** | **15.68** | **WIN** |
| 12 | BlendRidge meta-learner | 15.75 | — | LOSS (overfits) |
| 13 | ElasticNet in blend | 15.40 | 15.66 | MARGINAL (dilutes) |
| — | Trim to 2 models (ARD+Huber) | 15.40 | 15.63 | KEPT (runtime: 25min→10min) |

Session 4 (from 15.40/15.63 after ARD+Huber trim):
| # | Experiment | CV | OOF | Verdict |
|---|-----------|-----|------|---------|
| 14 | Orthogonalized ALT interactions | 15.51 | 15.74 | LOSS |
| 15 | Residual correction head | 16.10 | — | LOSS |
| 16 | KernelRidge (RBF) | 220.02 | — | CATASTROPHIC |
| 17 | **Trajectory features in target model** | **14.81** | **14.94** | **BIG WIN** |
| 18 | Row completeness feature | 15.26 | 15.47 | LOSS (collinear) |
| 19 | ALT-centric k sweep (3,4,5,6) | 14.78 | 14.92 | k=5 marginal win |
| 20 | BayesianRidge back in 3-model blend | 15.00 | 15.11 | LOSS |
| 21 | TheilSen in blend | 14.74 | 14.91 | MARGINAL (+3.5min) |
| 22 | Ridge in blend | 14.99 | 15.11 | LOSS |
| 23 | Huber epsilon tuning (1.0, 1.5) | — | — | 1.35 optimal |

### Current best configuration

```bash
python3 predict.py \
    --csv_path ../benchmark_combiner/benchmarks/clean_combined_all_benches.csv \
    --poly_interactions --poly_include_squares \
    --alt_centric_poly --alt_centric_k 5 \
    --cv_repeats_outer 5 --cv_repeats_inner 3 \
    --feature_cv_repeats 1 --alt_cv_repeats 1
```

Models: ARDRegression + HuberRegressor (epsilon=1.35), equal-weight blend.
Features: ~10 selected from 78 (75 benchmarks + 3 trajectory) by LGBM-based selector.
Poly: ALT² + top-5 ALT×feature interactions ranked by residual correlation.

---

## 7. What the Experiments Reveal About the Prediction Task

### 7.1 This is a radically small-n, high-dimensional problem

~90 training rows, 75+ features, ~40% missing. This single fact explains most experiment outcomes:

- **Nonlinear models catastrophically overfit.** KernelRidge (RBF): 220. PLS: 20.5. LightGBM direct: 0.417 RMSE/SD (Section 1). Random Forest: 0.456 RMSE/SD. These models have far too many effective parameters for n=90.
- **Stacking and meta-learning overfit.** BlendRidge (learning blend weights from OOF predictions): worse than equal-weight average. With n=90, there isn't enough data to reliably estimate even 2-3 blend weights.
- **More than 2 blend members always hurts.** Adding a 3rd model (BayesianRidge, Ridge, TheilSen, ElasticNet) to the ARD+Huber blend consistently degrades performance. There isn't enough data to benefit from the extra averaging — the 3rd model's noise outweighs its diversity.
- **Aggressive feature selection is essential.** The selector keeps 10 out of 78 features. Attempts to use more features (k=15, k=20, "all") degrade performance. The data can't support more than ~10 predictors.
- **Regularization must be strong.** ARD (automatic relevance determination, which zeros out irrelevant features) outperforms Ridge, BayesianRidge, and ElasticNet. The sparsest regularizer wins.

### 7.2 The prediction is dominated by one feature

Feature importance from the final model:

| Feature | Importance |
|---------|-----------|
| lmarena_Score (ALT) | 46.2 |
| style_predicted_delta | -21.5 |
| lmarena_Score² | 10.3 |
| livebench_zebra_puzzle | -0.11 |
| All others combined | <0.01 |

The task is essentially: "given a model's existing Arena score (or an imputation of it) plus a small correction from style/personality benchmarks, predict the Arena ELO." The top 3 features (all involving the ALT target) account for ~78% of model importance. Every other benchmark contributes marginal corrections.

This has a key implication: **prediction quality is gated by ALT imputation quality.** For models that already have an Arena score, prediction is near-trivial (use the score). For models that don't, everything depends on how well the imputer can estimate their Arena score from benchmarks. The 21% improvement in ALT RMSE (21.95→17.33) from the imputer optimization campaign (Section EXPERIMENT_LOOP.md) directly flowed through to target model improvement.

### 7.3 Missingness is the most informative meta-feature

The trajectory features (`_traj_mean_delta`, `_traj_max_delta`, `_traj_n_imputed`) were the single largest improvement across all target model experiments: -0.69 OOF RMSE, a bigger gain than any model architecture change, polynomial expansion, or hyperparameter tune.

**Why missingness is so predictive:**

1. **Missingness proxies for model "establishment status."** Models that appear on many benchmarks are typically from major labs (OpenAI, Anthropic, Google) — the same labs whose models get the most Arena votes. More votes → more stable ELO → more predictable. Models with sparse benchmark coverage are often smaller labs, fine-tunes, or very new releases with volatile ELO. The trajectory feature `_traj_n_imputed` (count of missing cells) directly encodes this.

2. **Imputation noise compounds, and the model can't otherwise detect it.** When a row has 40% missing features, those values are filled by BayesianRidge predictions trained on other incomplete rows. Each imputed value carries prediction error. The target model trains on a mix of real and synthetic values but has no way to distinguish them. The trajectory features (`_traj_mean_delta`, `_traj_max_delta`) effectively tell the model: "trust this row's features less" or "this row's feature vector is partly synthetic." Without that signal, the model treats a cleanly-measured GPT-4o row identically to a heavily-imputed niche model row.

3. **Missing benchmarks are not missing at random.** A model missing LiveBench scores probably launched before LiveBench existed. A model missing ARC-AGI-2 might not support the required tool-use interface. These missingness patterns encode model *type* — reasoning-focused vs. chat-focused, large vs. small, old vs. new — which is genuinely predictive of Arena ELO. Trajectory features are a compressed representation of that structure.

4. **It's the one signal that survives imputation.** Every actual benchmark score gets smoothed by the imputer — outlier scores are pulled toward the mean, distinctive patterns are homogenized. But missingness patterns are binary facts from the raw data that pass through untouched. In a pipeline that aggressively regularizes and works with imputed features, the few bits of ground truth that survive carry disproportionate weight.

**Contrast with explicit missingness features:** Earlier experiments (Experiment 13 in the imputer loop, and the row completeness experiment in the target model session) tried adding explicit missingness indicators (PCA-compressed missing flags, raw missing fraction). These consistently failed. The trajectory features succeed where raw missingness fails because they encode *imputation dynamics* — not just "which cells were missing" but "how hard was it to fill them." A model missing 40% of benchmarks that are all highly correlated (easy to impute) is fundamentally different from one missing 40% of weakly correlated benchmarks (hard to impute). The trajectory captures this distinction; raw missingness counts don't.

### 7.4 Diversity of errors matters more than individual accuracy

The blend insight is the second most important finding. ARDRegression (16.86 individual RMSE) and HuberRegressor (16.41 individual RMSE) are both mediocre individually — worse than the old BayesianRidge baseline. But their equal-weight average (14.78) dramatically outperforms either.

The mechanism: ARD uses Bayesian shrinkage (pulls coefficients toward zero, implicitly does feature selection). Huber uses robust loss (squared error for small residuals, linear penalty for large ones). On models where the prediction is fundamentally hard — models whose Arena ELO is poorly explained by benchmarks — the two models err in *different directions*. Averaging cancels correlated noise while preserving shared signal.

This explains why adding similar models (BayesianRidge, Ridge, ElasticNet) consistently hurts: they use the same loss function (MSE) with similar regularization, so they make the same mistakes. The blend only gains from members with genuinely different inductive biases.

### 7.5 The signal is nearly linear and low-dimensional

Across 50+ experiments spanning both the imputer and target model:

- **Linear models dominate.** PCA→BayesianRidge beats CatBoost, LightGBM, Random Forest, and GaussianProcess in both standard and family-aware CV.
- **The only useful nonlinearity is low-order polynomial.** ALT² and 5 ALT×feature interaction terms are the only nonlinear features selected. Generic squared terms, cubic terms, and kernel methods all overfit.
- **10 features suffice.** Out of 78 available features, the selector consistently keeps ~10. Adding more degrades performance.
- **PCA works because the signal is latent and linear.** The first 10 PCA components capture "general capability," "reasoning vs. chat," "code vs. language" — the actual latent factors Arena scores depend on. Polynomial interactions on PCA components show zero significant signal (Section 1).

This makes physical sense: Arena ELO measures overall user preference, which is approximately a weighted sum of individual capabilities (reasoning + creativity + instruction following + ...). Each benchmark measures a noisy linear projection of the same underlying qualities. The relationship between benchmarks and ELO is therefore approximately linear in the latent capability space.

### 7.6 The ceiling is real

The 95% CI on our best OOF RMSE of 14.92 is [12.39, 17.27]. With Arena ELO ranging ~900–1400 (SD≈56), that's an RMSE/SD of ~0.27. The residual error likely comes from:

1. **Arena ELO measurement noise.** ELO scores shift as new users vote. A model's "true" ELO is itself uncertain, setting a floor on prediction error.
2. **Omitted variables.** Arena captures conversational qualities (humor, helpfulness, tone) that no benchmark fully measures. The heteroscedasticity analysis (Section 2) confirms this: the model systematically overpredicts weak models where arena-specific qualities diverge most from benchmark performance.
3. **Imputation noise for the ~40% of models missing key features.** Even with trajectory features informing the model about imputation quality, heavily-imputed rows remain fundamentally noisier.

The diminishing returns observed across the experiment campaign support this interpretation. The first few wins (SVD warm-start, trajectory features, ALT-centric interactions) each improved OOF by 0.3–0.7 points. Later experiments consistently yield <0.1 improvement or degrade performance. The pipeline is approaching the information-theoretic limit of what 75 benchmarks can predict about Arena ELO with n≈90 training rows.

### 7.7 Implications for the pipeline

1. **More training rows remain the single biggest lever.** Every model with Arena scores AND benchmark data is a free training row. Going from 90→120+ would meaningfully reduce the overfitting constraints that currently dominate.
2. **Imputation quality feeds directly into prediction quality.** Because the ALT target (Arena score itself) is the dominant feature, improving the ALT imputer directly improves the target model. The 21% ALT RMSE improvement from the imputer optimization (EXPERIMENT_LOOP.md) is likely the largest single contributor to the overall pipeline improvement.
3. **Feature engineering should focus on meta-features, not more benchmarks.** The trajectory features were worth more than any new benchmark column. Additional benchmark sources measuring the same abilities would be redundant; features that encode *data quality* and *model characteristics* have higher marginal value.
4. **The 2-model blend (ARD+Huber) is near-optimal for n≈90.** Adding models only helps if they bring genuinely different inductive biases. At this sample size, there isn't enough data to support more than 2 diverse models in the blend.
5. **The pipeline rewards restraint.** Almost every attempt at more expressiveness — more models, more features, nonlinear kernels, stacking, learned blend weights — made things worse. The best improvements came from giving the model information it didn't have (trajectory features, ALT-centric interactions) rather than making the model more complex.
