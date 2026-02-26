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
- Polynomial interactions or nonlinear models
- Post-hoc calibration schemes
- More benchmark sources measuring the same abilities
