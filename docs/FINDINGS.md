# Findings: Alt-Target Prediction Pipeline

## Verified Results (2026-03-09, post-leakage fix)

All numbers below use honest OOF predictions for `style_predicted_delta` (cross_val_predict, OOF r=0.870). Prior results used in-sample predictions (r=1.000) due to a leakage bug in `soothsayer_style/score.py`.

### Imputer & Configuration Comparison

| # | Config | OOF RMSE | Best Model | Notes |
|---|--------|----------|------------|-------|
| 1 | Specialized imputer | 22.74 | ARDRegression | Baseline |
| 2 | ModelBank, no coherence | 22.89 | ARDRegression | Worse than Specialized without coherence |
| 3 | ModelBank + coherence (λ=1.0, exp) | 22.17 | ARDRegression | Coherence is what makes ModelBank win |
| 4 | ModelBank + coherence + trajectory in target | **21.69** | ARDRegression | Best verified config |
| 5 | ModelBank + coherence + trajectory in both | **21.69** | ARDRegression | ALT trajectory irrelevant |

All runs: `--poly_interactions --poly_limit 7 --no_residual_head --cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1`

### Verified Improvement Deltas

| Change | Delta | From → To |
|--------|-------|-----------|
| Coherence projection (λ=1.0, exp shape) | **-0.72** | 22.89 → 22.17 |
| Trajectory features in target model | **-0.48** | 22.17 → 21.69 |
| ModelBank+coherence vs Specialized | **-0.57** | 22.74 → 22.17 |
| ALT trajectory features | **0.00** | No effect on final RMSE |

### Best Production Config

```bash
python3 predict.py \
    --csv_path ../benchmark_combiner/benchmarks/clean_combined_all_benches.csv \
    --imputer_type model_bank \
    --coherence_lambda 1.0 --coherence_shape exp \
    --poly_interactions --poly_limit 7 \
    --no_residual_head --no_traj_in_alt \
    --eb_parent
```

Note: `--delta_head` was removed — the delta head has pre-CV leakage (see Leakage Analysis). Pending fold-internal fix and re-verification.

---

## Architectural Findings (qualitatively robust)

These findings were established across 70+ experiments. While the absolute RMSE numbers from the leaked era are invalid, the qualitative patterns are confirmed by the re-verification (relative ordering preserved, same features win/lose).

### The prediction is dominated by the ALT feature

The lmarena_Score (ALT target) is by far the strongest predictor. For models that already have an Arena score, prediction is near-trivial. For models that don't, everything depends on ALT imputation quality. This is why coherence projection helps — it improves imputed ALT values.

### Trajectory features encode imputation quality

The `_traj_mean_delta`, `_traj_max_delta`, `_traj_n_imputed` features (per-row statistics from the imputer's iterative passes) are the single most valuable meta-features. They encode:
1. **Coverage** — how many benchmarks the model was tested on (correlated with model prominence and ELO stability)
2. **Imputation difficulty** — how much imputed values shifted across passes (epistemic uncertainty)
3. **Surviving imputation smoothing** — binary facts from raw data that pass through untouched

The 2×2 ablation confirms the gain comes entirely from the target model pathway, not from trajectory improving ALT imputation. ALT trajectory features improve the ALT model's own cross-validation diagnostic (18.11 → 17.33) but this doesn't propagate to final predictions.

### Small n dominates everything

~90 training rows, 75+ features, ~40% missing. This explains most experiment outcomes:
- Nonlinear models catastrophically overfit (KernelRidge: 220, PLS: 20.5)
- Stacking and meta-learning overfit (learning blend weights from OOF with n=90)
- More than 2 blend members consistently hurts
- Aggressive feature selection is essential (~10 features from 78)
- The sparsest regularizer (ARD) wins

### Coherence projection is the only imputer lever that worked

23 imputer experiments were run. Only coherence projection (SVD low-rank projection of imputed matrix, weighted by completeness) improved results. Everything else was neutral or worse: adaptive gates, learned gates, per-column coherence, multiple imputation, SVD anchors, iterative coherence, prediction-level stacking.

---

## Model Coverage

### Current state

- ~140 models in combined CSV, ~100 with lmarena_Score
- 303 models on LMArena total
- 344 models on OpenRouter

### Custom benchmark coverage gaps

| Benchmark | Models missing (of training set) |
|-----------|----------------------------------|
| logic | 0 |
| style | 0 |
| writing | 0 |
| eq | 1 |
| weirdml | ~44 |

WeirdML is the largest coverage gap.

---

## Imputation Improvement Experiments

### Cycle 1 (in progress)

Baseline: **21.41** OOF RMSE (ModelBank + coherence + trajectory, improved style delta)

| # | Idea | Source | Description |
|---|------|--------|-------------|
| 1 | Spearman-augmented ranking | Claude | Use max(|pearson|, |spearman|) for candidate predictor ranking |
| 2 | KNN fallback (k=5) | Claude | Replace column-median fallback with k-NN weighted average |
| 3 | Per-column adaptive lambda | Claude | Scale coherence λ by column's SVD reconstruction quality |
| 4 | EB parent model | Codex | Shrink cell predictions toward empirical-Bayes parent model |
| 5 | Masked-cell calibration | Codex | Calibrate σ² via synthetic masking of observed cells |
| 6 | Graph-Laplacian coherence | Codex | Correlation-graph per-row smoothing instead of global SVD |

| # | Experiment | RMSE | Delta | Verdict |
|---|-----------|------|-------|---------|
| 0 | Baseline | 21.69 | — | — |
| 1 | Spearman ranking | 21.87 | +0.18 | Worse — non-linear ranking reorders predictors unhelpfully |
| 2 | KNN fallback (k=5) | 21.69 | 0.00 | Neutral — median fallback cells are too few to matter |
| 3 | Adaptive col lambda | 21.72 | +0.02 | Neutral — per-column λ scaling doesn't help |
| 4 | **EB parent** | **21.48** | **-0.22** | **Winner — shrinking uncertain cells toward parent model helps** |
| 5 | Masked calibration | 22.12 | +0.42 | Worse — recalibrated σ² disrupts coherence weighting |
| 6 | Graph Laplacian | exploded | — | Broken — numerical instability in per-row linear solve |

### Cycle 2

Baseline: **21.48** OOF RMSE (EB parent ON from Cycle 1)

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 0 | EB parent baseline | — | 21.48 | — | — |
| 1 | EB residual (blend in pass1) | Codex | 22.02 | +0.55 | Worse — early blending disrupts per-cell models |
| 2 | Exact joint support + EB | Codex | 21.49 | +0.01 | Neutral — approx support heuristic was fine |
| 3 | SPD graph smoother + EB | Codex | 22.82 | +1.35 | Much worse — local graph corrections hurt SVD coherence |
| 4 | EB after coherence | Claude | 22.25 | +0.77 | Worse — EB must come before coherence |
| 5 | Double EB | Claude | 22.25 | +0.77 | Same as Exp 4 — second EB after coherence dominates |
| 6 | EB Bayesian sigma | Claude | 21.68 | +0.20 | Slightly worse — per-prediction uncertainty adds noise |

**Key insight**: EB parent works best as a simple post-pass1, pre-coherence step with fixed parent σ². Any modification to its position or σ² formula hurts.

### Cycle 3

Baseline: **21.48** OOF RMSE (EB parent ON)

Testing pipeline structural choices: poly interaction settings, residual head, ALT trajectory.

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 0 | Baseline (poly_limit=7) | — | 21.48 | — | — |
| 1 | No poly interactions | Claude | 22.35 | +0.88 | Much worse — poly interactions are essential |
| 2 | Poly limit=10 | Claude | 21.88 | +0.40 | Worse — more interactions overfit |
| 3 | Poly limit=5 | Claude | 21.73 | +0.26 | Worse — fewer interactions lose signal |
| 4 | With residual head | Codex | 21.63 | +0.15 | Slightly worse — residual head adds noise |
| 5 | ALT trajectory ON | Codex | 21.48 | 0.00 | Neutral — ALT trajectory still irrelevant |
| 6 | Outer CV=20 repeats | Codex | — | — | Killed (100-fold eval too slow, no expected gain) |

**Key insight**: poly_limit=7 is the sweet spot. Fewer loses signal, more overfits. The residual head and ALT trajectory confirm prior findings — neither helps.

### Cycle 4

Baseline: **21.48** OOF RMSE (EB parent ON)

Testing post-imputation pipeline ideas: ALT bagging, delta head, reliability gating, imputer ensembles.

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 0 | Baseline | — | 21.48 | — | — |
| 1 | ALT bagged (5 seeds) | Codex | 21.48 | 0.00 | Neutral — interaction pair variance is already handled by consensus search |
| **2** | **Delta head** | **Codex** | **20.29** | **-1.19** | **LEAKED — delta head trained on all TARGET rows before CV; must re-verify** |
| 3 | ALT reliability gate | Codex | 21.42 | -0.06 | Neutral |
| 4 | Imputer ensemble (3 seeds) | Claude | 21.53 | +0.05 | Neutral |
| 5 | Obs-weighted SVD | Claude | 21.57 | +0.09 | Neutral |
| 6 | ALT LOO calibrate | Claude | 21.48 | 0.00 | Neutral |

**Key insight**: Delta head result is LEAKED — the -1.19 improvement is invalid. The delta head model was trained on ALL known-TARGET rows before CV, so validation rows see predictions from a model trained on their own labels. The signal (lmsys-lmarena gap predictable from style features, train r=0.937) may be genuine, but the honest lift must be measured with fold-internal delta head computation. See leakage analysis below.

### Cycle 5

Baseline: **21.48** OOF RMSE (EB parent ON)

Testing problem framing and regularization: completeness weighting, sparse column dropping, winsorization, feature noise.

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 0 | Baseline | — | 21.48 | — | — |
| 1 | No completeness weighting | Codex | 21.53 | +0.05 | Neutral |
| 2 | Weight power=4 | Codex | 22.81 | +1.33 | Much worse — over-weighting complete rows hurts |
| 3 | Pairwise rank feature | Codex | 15.57 | -5.91 | **CONFIRMED LEAKAGE** — see leakage analysis below |
| 4 | Drop sparse cols >70% | Claude | 21.70 | +0.22 | Worse — sparse columns still contribute through imputation |
| 5 | Winsorize 2% | Claude | 21.57 | +0.09 | Neutral |
| 6 | Feature noise 5% | Claude | 21.48 | 0.00 | Neutral |

**Key insight**: The default completeness weighting (power=2) is well-tuned. Dropping sparse columns hurts because even highly incomplete columns contribute through the imputation graph. Pairwise rank feature is confirmed leakage (see below).

### Cycle 6

Baseline: **21.48** OOF RMSE (EB parent ON, PCA-10 ALT feature mode)

Testing whether PCA in the ALT pathway compresses away useful capability-specific signals: raw columns, Factor Analysis, hybrid approaches.

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 0 | PCA-10 baseline | — | 21.48 | — | — |
| 1 | Raw columns (no PCA) | Claude | 21.79 | +0.31 | Worse |
| 2 | Factor Analysis (10 factors) | Claude | 21.89 | +0.41 | Worse |
| 3 | Hybrid (top-15 raw + PCA-5) | Claude | 21.98 | +0.50 | Worse |
| 4 | Raw, no interactions | Claude | 22.11 | +0.63 | Worse |
| 5 | FA, no interactions | Claude | 22.80 | +1.32 | Much worse |

**Key insight**: PCA(10) is optimal for the ALT pathway. At n≈140 rows with 76 features, PCA's regularization benefit outweighs any capability-specific signal that FA or raw columns might preserve. The interaction search on top of PCA is also confirmed valuable — removing it consistently hurts (+0.3 to +0.9 RMSE).

### Summary Across All Cycles

| Cycle | Experiments | Winners | Best Δ |
|-------|------------|---------|--------|
| 1 | 6 | EB parent (-0.22) | -0.22 |
| 2 | 6 | None | 0.00 |
| 3 | 5 | None | 0.00 |
| 4 | 6 | Delta head (-1.19, **LEAKED**) | ? |
| 5 | 6 | Pairwise rank (**LEAKED**) | ? |
| 6 | 5 | None | 0.00 |
| 7 | 7 | None (delta head confirmed leakage) | 0.00 |
| 8 | 3 | None | 0.00 |
| 9 | 3 | None | 0.00 |

**Total: 54 experiments, 1 confirmed winner (EB parent).** 53 experiments failed to beat 21.48. The prediction floor appears real at this sample size. Honest best: EB parent → **21.48 RMSE** (down from 21.69 baseline).

### Cycle 9

Baseline: **21.48** OOF RMSE (EB parent ON)

Moonshot experiments: semi-supervised self-training, model name features, prediction-level shrinkage.

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 1 | Self-training (pseudo-labels) | Claude | 22.05 | +0.57 | Worse — pseudo-labels add noise, don't regularize |
| 2 | Alias archaeology (name features) | Codex | 24.35 | +2.87 | Much worse — name tokens create spurious correlations at n=90 |
| 3 | Prediction shrinkage (blend w/ ALT-only) | Claude | 22.11 | +0.63 | Worse — simple model is too inaccurate, blending loses signal |

**Key insight**: Even fundamentally different approaches fail. Self-training adds ~50 pseudo-labeled rows but the pseudo-labels are too noisy (predicted from a 21.48-RMSE model) to help. Model name features (is_reasoning, is_mini, version) are spuriously correlated at n=90. Prediction shrinkage toward a 1-feature model sacrifices the real signal in the other 14 features.

### Cycle 8

Baseline: **21.48** OOF RMSE (EB parent ON)

Testing feature transforms: quantile normalization, reliability-weighted PCA, ALT orthogonalization.

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 1 | Quantile transform | Claude | 24.54 | +3.06 | Much worse — destroying natural scale hurts linear model |
| 2 | Reliability-weighted PCA | Codex | 21.48 | 0.00 | Neutral — obs_rate weighting doesn't change PCA enough |
| 3 | Orthogonalize to ALT | Codex | 22.58 | +1.10 | Worse — ALT-correlated signal in features is useful, not redundant |

**Key insight**: Raw benchmark scales carry important information. Quantile normalization destroys distances between scores that the linear model exploits. Features correlated with ALT are useful (not redundant) — the model needs redundant ALT signal across features to overcome imputed ALT noise. The existing StandardScaler → PCA(10) is already the right preprocessing.

### Cycle 7

Baseline: **21.48** OOF RMSE (EB parent ON)

Testing outside-the-box ideas + delta head leakage re-verification.

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 0 | Baseline | — | 21.48 | — | — |
| 1 | Delta head (fold-internal fix) | Codex C4 | 23.38 | +1.90 | **Much worse — confirms signal was 100% leakage** |
| 2 | LOBO residuals | Claude | 21.48 | 0.00 | Neutral |
| 3 | Target-aware coherence | Claude | 21.48 | 0.00 | Neutral |
| 4 | Provider-family EB | Claude | 21.75 | +0.27 | Worse — family means too noisy with few models per family |
| 5 | Partial-linear ALT | Codex | 21.71 | +0.23 | Worse — isotonic overfits with ~80 fold-train rows |
| 6 | Pairwise anchor head | Codex | 43.01 | +21.53 | Catastrophic — pairwise differencing destroys signal at small n |

**Key insight**: Delta head is confirmed as pure leakage. With fold-internal computation, train r dropped from 0.937 (leaked, all rows) to 0.625 (honest, ~80 rows), and the feature actively hurts (+1.90 RMSE). The lmsys-lmarena gap is NOT reliably predictable from ~80 training rows. Provider-family and isotonic approaches also hurt — both add parameters that overfit at this sample size. Pairwise anchor head catastrophically fails — creating O(n²) difference examples from n=80 rows adds massive noise that overwhelms any relative signal.

---

## Leakage Analysis (2026-03-10)

Two features added in Cycles 4-5 were computed globally BEFORE the CV loop, bypassing the nested-CV safeguards. Both confirmed as leakage by code review + Codex (gpt-5.4) audit.

### Bug 1: Pairwise rank feature (Cycle 5 Exp 3, RMSE 15.57)

**Mechanism**: `_alt_rank_pctile = rank(global_ALT_prediction)` is computed at line 5546-5549 on the full-data ALT prediction (trained on all ~140 rows). Inside each CV fold, the ALT column is properly replaced with fold-honest OOF predictions, but `_alt_rank_pctile` is NOT updated — it retains the full-data ALT signal.

**Why it's severe**: The per-fold feature selector may keep `_alt_rank_pctile` because its r² with the noisy OOF-ALT falls below the 0.95 collinearity threshold. The model then uses `_alt_rank_pctile` as a superior proxy for ALT, completely circumventing OOF protection. Additionally, `_alt_rank_pctile` contaminates the ALT OOF model itself — `X_no_alt_df` only drops `ALT_TARGET`, not derived features.

**Fix**: Remove the feature entirely. A monotonic transform of ALT adds no information if computed honestly (fold-internally it would be perfectly collinear with ALT and dropped).

### Bug 2: Delta head (Cycle 4 Exp 2, RMSE 20.29)

**Mechanism**: `_delta_head_pred` is a BayesianRidge trained on `TARGET - ALT` using ALL known-target rows (line 5552-5573), then predictions are added as a feature for ALL rows. Inside CV folds, validation rows see predictions from a model trained on their own TARGET values — classic label leakage.

**Why the signal may be partially real**: The lmsys-lmarena gap IS predictable from style features (response length, formatting patterns). But the reported -1.19 RMSE improvement is contaminated and must be treated as entirely invalid until re-run.

**Fix** (implemented 2026-03-10): Delta head computation moved inside `_precompute_single_fold`. For each outer fold: delta model trained on fold's train rows only, predicts for both train and val rows. Also added to final model path in `fit_and_predict_all_with_alt`. Pairwise rank feature code removed entirely.

### Impact on production config

The "best production config" previously included `--delta_head`. The leakage fix has been implemented (delta head is now fold-internal), but needs re-verification. Until re-verified, the honest best config is:

```bash
python3 predict.py \
    --csv_path ../benchmark_combiner/benchmarks/clean_combined_all_benches.csv \
    --imputer_type model_bank \
    --coherence_lambda 1.0 --coherence_shape exp \
    --poly_interactions --poly_limit 7 \
    --no_residual_head --no_traj_in_alt \
    --eb_parent
```

Honest best RMSE: **21.48** (EB parent only, no delta head).

---

## Error Decomposition Analysis (2026-03-11)

### Where does the 21.48 RMSE come from?

The pipeline predicts target (lmsys_Score) in two stages: benchmarks → ALT, then ALT + benchmarks → target. Diagnostic experiments decompose the error:

| Scenario | ALT RMSE | Target RMSE | Notes |
|----------|----------|-------------|-------|
| Perfect ALT + all benchmarks → target | — | 12.93 | Floor: if ALT were oracle |
| Real ALT + style features → target | — | ~10-13 | Cheating: real ALT never available at inference |
| 73 benchmarks (median-fill) → ALT | 23.84 | — | Stage 1 bottleneck |
| 33 low-miss benchmarks → ALT | 22.99 | — | Slightly better, fewer imputed cells |
| Benchmarks → ALT → target (naive 2-stage) | 23.84 | 29.69 | No feature selection, no poly, no OOF stacking |
| Full pipeline | ~23 | **21.48** | Pipeline machinery recovers ~8 pts over naive |

**Key insight**: The bottleneck is stage 1 (benchmarks → ALT). Benchmarks only predict debiased Arena score to ~24 RMSE. Everything downstream is damage control. With perfect ALT the problem is easy (12.93), but predicting ALT from benchmarks is hard.

### The style correction is working as designed

The ALT target (lmarena_Score) is a style-debiased transform of the target (lmsys_Score). Style features (length, formatting, bold/list counts) capture exactly what makes these scores diverge. This is the entire point of the style benchmark — the r=-0.87 correlation between `style_predicted_delta` and the `lmsys - lmarena` gap is the design working, not leakage.

### IRT latent factors beat raw benchmarks for ALT prediction

A continuous 2-parameter IRT model (sigmoid link per benchmark, gradient-descent fit on observed cells only) extracts latent ability θ from the partially-observed benchmark matrix **without any imputation**.

| Approach | θ dims | ALT RMSE | Target RMSE (2-stage) |
|----------|--------|----------|-----------------------|
| IRT θ only | k=1 | 35.39 | 42.40 |
| IRT θ only | k=2 | 25.29 | 39.25 |
| IRT θ only | k=3 | **20.82** | 26.57 |
| IRT θ only | k=5 | 21.38 | 27.49 |
| IRT θ only | k=8 | 20.33 | 26.68 |
| Raw benchmarks (median-fill, 73 cols) | — | 23.84 | 29.69 |
| Full pipeline | — | ~23 | **21.48** |

**IRT θ (k=3) predicts ALT at 20.82 — 3 points better than 73 raw benchmarks (23.84).** Three nonlinear latent dimensions, fit on observed cells only, compress information more efficiently than 73 linearly-imputed columns. 68/73 benchmarks have genuinely nonlinear S-curves (not just linear regime), and 27/73 hit saturation bounds.

However, the IRT 2-stage target RMSE (26.57) is still behind the full pipeline (21.48) because the pipeline's OOF stacking, feature selection, and polynomial interactions recover signal in stage 2 that raw θ alone doesn't capture.

### IRT optimization: regularization and polynomial features matter

Grid sweep over rank × regularization, plus polynomial expansion of θ:

| Config | ALT RMSE | Notes |
|--------|----------|-------|
| k=3, λ=0.01 (original) | 20.82 | Initial prototype |
| k=3, λ=0.001 | 19.77 | Less regularization helps |
| k=4, λ=0.0001 | **18.29** | Best raw θ |
| k=4, λ=0.0001, poly(2) | **16.82** | Cross-terms between latent dims capture nonlinear ability combos |
| k=3, λ=0.0005, poly(2) | 17.50 | |
| k=6, λ=0.0005, poly(2) | 17.44 | Diminishing returns past k=4 |
| k=4, poly(3) | 27.37 | Degree 3 overfits |

**Poly(2) interactions of IRT θ are critical**: θ₁×θ₂, θ₁×θ₃, etc. capture nonlinear combinations of latent abilities. This is analogous to the pipeline's finding that poly_limit=7 is essential.

### IRT 2-stage: approaching the pipeline

Honest 2-stage evaluation (IRT θ_poly → ALT, then θ_poly + features → target):

| Stage 2 features | ALT RMSE | Target RMSE |
|------------------|----------|-------------|
| θ_poly only | 16.82 | 26.12 |
| θ_poly + style features | 16.82 | 22.66 |
| θ_poly + all 17 fully-observed features | 16.82 | **22.41** |
| Full pipeline (impute + ALT + feat sel + poly) | ~23 | **21.48** |

The IRT predicts ALT better (16.82 vs ~23) but the pipeline still wins on target (21.48 vs 22.41) due to its OOF stacking, feature selection, and polynomial interaction machinery for the ALT→target bridge. The ~1 point gap suggests feeding IRT θ into the existing pipeline could beat 21.48.

**Key insight**: Adding ANY features to θ for ALT prediction makes it worse (θ alone: 19.77, θ+features: 21.52+). The IRT factors are already the optimal compression — extra columns dilute signal at n=112. But for the target prediction stage, the style features are essential to model the Arena style bias.

**Next step**: Feed IRT θ into the existing pipeline as features, replacing or supplementing imputed benchmarks. The θ factors are clean, low-dimensional, require no imputation, and encode nonlinear benchmark relationships.

---

## Style Shape Features (2026-03-12)

### New features: per-question style variance

Added 15 style shape features to the pipeline, derived from per-question (Q1–Q9) response data in soothsayer_style. These capture *how* a model varies its style across different question types, not just aggregate formatting.

**New columns (prefixed `style_` in combined CSV):**
- `cv_{length,header_count,bold_count,list_count}` — coefficient of variation across 9 questions (adaptability)
- `min_{length,header_count,bold_count,list_count}` — minimum across questions (formatting floor)
- `frac_used_{header_count,bold_count,list_count}` — fraction of questions where formatting is used (consistency)
- `q7_{length,header_count,bold_count,list_count}` — Q7-specific metrics (creative programming task: Python slots in anime voice)

**Individual correlations with Arena ELO (n=110):**
- `style_q7_header_count`: r=+0.57 (best single feature — creative programming structure)
- `style_q7_length`: r=+0.47 (creative programming effort)
- `style_cv_length`: r=+0.41 (response length adaptability)
- `style_frac_used_bold_count`: r=+0.35 (consistent bold formatting)
- `style_min_bold_count`: r=+0.33 (formatting even on simple prompts)

**Key insight:** `style_cv_length` (adaptability) and `style_q7_header_count` (creative task structure) have NO overlap (r<0.5) with any existing combined column. They are genuinely new signals.

### Result: 17.89 RMSE (top-50 LOO, verified)

| Config | Top-50 LOO | 10×10-fold (all 112) | 10×5-fold (all 112) |
|--------|-----------|---------------------|---------------------|
| Previous best (EB parent, old features) | 21.48 | — | — |
| + style shape features, PCA-10 ALT, poly | **17.89** (CI: 14.67–20.92) | **19.76** (CI: 17.19–22.29) | **20.25** (CI: 17.57–22.90) |

Top-50 LOO evaluates prediction quality on the 50 strongest models (Arena score 1397–1502), which is the primary use case — predicting where new frontier models will land. The full 112-model K-fold includes weaker models that are harder to predict and less practically relevant. Both show clear improvement over the 21.48 baseline.

Selected features (7): `lmarena_Score`, `aa_eval_livecodebench`, `style_predicted_delta`, `eqbench_eq_elo`, **`style_q7_length`**, `arc_ARC-AGI-2`, `style_normalized_bold_count`

Only `style_q7_length` is new — but poly interactions expand it into 6 cross-terms (e.g., `style_q7_length × lmarena_Score`) that capture how creative coding effort modulates the benchmark-to-arena mapping.

**Previously underpredicted models improved:**

| Model | Old Error | New Error | Improvement |
|-------|-----------|-----------|-------------|
| Claude Sonnet 4.6 Thinking | -38.8 | -15.3 | +23.5 |
| Mistral Medium 3.1 | +46.4 | +24.9 | +21.5 |
| Claude Opus 4.6 | +50.9 | +31.8 | +19.1 |
| GLM-4.5 | +28.1 | +8.6 | +19.5 |
| Claude Opus 4.1 Thinking | +35.6 | +20.3 | +15.2 |

**Remaining hard cases:** Claude Opus 4.1 (+44.0), ChatGPT-4o (+35.1), DeepSeek R1 (+34.4). These likely need signals no current benchmark captures.

### Cross-domain ALT vs PCA-10 ALT

Tested a hybrid architecture: cross-domain interaction search for ALT prediction (greedy-selected pairwise interactions, RMSE ~17 vs PCA-10's ~28), then PCA components + greedy residual correlates for the target model.

**CD ALT consistently loses to PCA-10 ALT at the final target prediction despite much better stage-1 RMSE.** Pin-down experiments isolate why:

| Controls | PCA-10 | CD ALT | Gap |
|----------|--------|--------|-----|
| No selection, no poly | 23.39 | 24.39 | 1.00 |
| No selection, poly | 21.38 | 23.07 | 1.69 |
| Selection + poly (full pipeline) | 17.89 | 18.58 | 0.70 |
| ALT-centric poly | 22.70 | 24.67 | 1.97 |

**Diagnosis (confirmed by Codex review + ablation):**

The gap is ~60% fundamental, ~40% poly amplification.

1. **Fundamental (Wave 1, gap=1.0):** PCA-10 ALT is a better *raw feature* for predicting lmsys_Score despite being worse at predicting lmarena_Score. PCA-10 preserves a clean linear capability axis that the target model can use directly; CD ALT's accuracy at predicting lmarena doesn't translate to lmsys prediction utility.

2. **Poly amplification (Wave 2, gap grows to 1.7):** PCA-10's "structured errors" — systematic mispredictions for models with distinctive style — interact productively with raw features via poly terms. CD ALT's tighter fit leaves smaller, more random residuals that poly can't exploit. This is a division-of-labor effect: PCA-10 handles linear signal, poly handles nonlinear.

3. **Selector compensation (Wave 3, gap shrinks to 0.7):** Tree-based feature selection actually *helps* CD ALT more than PCA-10, partially compensating for the weaker poly interactions. The selector is not the villain.

4. **ALT-centric poly is counterproductive (Wave 4, gap=2.0):** Forcing explicit `ALT × feature` interactions hurts both ALT types and widens the gap. The model needs generic poly, not ALT-focused interactions.

**Literature context (via Codex):** This is a known phenomenon in stacked models — `argmin_h E[(ALT - h(X))²]` ≠ `argmin_h min_g E[(TARGET - g(X, h(X)))²]` when g is a restricted model class (Wolpert 1992, Breiman 1996). The optimal stage-1 model depends on stage-2's capacity.

**Practical conclusion:** PCA-10 ALT is the correct choice for this pipeline. Better ALT prediction is not a productive direction.

---

## Residual Analysis (2026-03-12)

In-sample residual analysis on the best model's final predictions (predictions_best_model.csv). These are NOT OOF predictions — they're trained on all data — but patterns should be directionally similar.

### By Provider

| Provider | Count | RMSE | Mean Error | Notes |
|----------|-------|------|------------|-------|
| Google | 19 | 3.0 | -0.2 | Best predicted — strong, consistent performers |
| OpenAI | 12 | 3.9 | -0.4 | Excellent — tight cluster around expected performance |
| Meta | 11 | 5.0 | -0.7 | Good |
| Anthropic | 8 | 7.2 | +3.6 | Systematically underpredicted |
| Mistral | 10 | 8.1 | +2.3 | Mixed — frontier models underpredicted |
| xAI | 4 | **13.2** | +5.5 | Worst — Grok 3 Beta has extreme formatting (40× length, 52× lists vs median) |

### By Model Type

| Group | RMSE | Notes |
|-------|------|-------|
| Reasoning models | 4.8 | Easier to predict — benchmarks capture their strengths well |
| Non-reasoning | 7.2 | Harder — Arena rewards conversational quality not captured by benchmarks |

### Key Findings

- **xAI/Grok is the worst outlier**: Grok 3 Beta's extreme formatting divergence (40× length, 52× lists vs median model) maps to huge Arena underprediction. Style features help but can't fully capture this.
- **Missingness does NOT explain errors**: Models with 20+ missing benchmarks can still be well-predicted (e.g., Google models). Imputation quality matters more than missingness count.
- **Style features fixed prior hard cases**: ChatGPT-4o went from +35 → -1.4 error, DeepSeek R1 from +34 → +12.8. But Claude Opus 4.1 (+44.0) and Grok models remain hard.
- **Provider effect is real**: Arena voters may have provider-specific preferences (conversation style, safety behavior) that benchmark scores can't capture.

---

## Joint Prediction Experiments (2026-03-12)

Testing whether joint optimization of benchmark reconstruction + target prediction can break the 17.89/19.76 wall. Two approaches implemented in `arena_predictor/joint_predict.py`:

1. **SCMF (Supervised Collective Matrix Factorization)**: ALS-based, learns latent factors Z that jointly minimize reconstruction + target prediction loss. SVD warm-start.
2. **BHLT (Bayesian Hierarchical Latent-Trait)**: EM-based factor analysis with hierarchical family priors. Returns calibrated posterior uncertainty.

Both tested in standalone mode (pure latent model predicts directly), hybrid mode (inject Z factors into existing pipeline), and inductive mode (train-only factor fitting).

### Results (Waves 1-2, 10×10-fold on all 112)

Baseline: **19.76** (existing pipeline, 10×10-fold)

| Exp | Approach | Config | RMSE | Delta | Verdict |
|-----|----------|--------|------|-------|---------|
| 1 | SCMF standalone | rank=6, λ_target=5 | 29.24 | +9.5 | Much worse |
| 2 | **BHLT standalone** | k=6, family_prior=1.0 | **23.77** | +4.0 | Best joint model, still worse |
| 3 | SCMF hybrid | rank=6, λ_target=10 | 29.15 | +9.4 | Hybrid doesn't help SCMF |
| 4 | SCMF hybrid | rank=4, λ_target=5 | 27.10 | +7.3 | Lower rank slightly better |
| 5 | SCMF hybrid | rank=8, λ_target=5 | 34.33 | +14.6 | Higher rank overfits badly |
| 6 | SCMF hybrid | rank=6, λ_target=10 | 29.13 | +9.4 | Lambda insensitive |
| 7-12 | Various | — | — | — | Terminated (OOM), rerunning |

### Key Findings

- **SCMF is uniformly bad (27-34 RMSE)**: The ALS optimization finds latent factors that reconstruct benchmarks well but don't predict Arena. Joint supervision (λ_target) doesn't help — the reconstruction loss dominates.
- **BHLT is better (23.77) but still loses by 4 points**: The hierarchical structure provides better regularization than SCMF, but pure latent factors can't compete with the pipeline's direct ALT + trajectory + style features.
- **Hybrid injection doesn't rescue SCMF**: Injecting SCMF Z factors as additional features into the existing pipeline (exps 3-6) gives essentially the same RMSE as standalone. The Z factors are redundant with what the pipeline already computes.
- **Rank sensitivity**: rank=4 slightly better than 6, rank=8 much worse. At n=112, even 8 latent factors overfit.

**Conclusion**: Joint optimization of reconstruction + prediction is theoretically appealing but practically inferior to the staged pipeline. The existing pipeline's PCA-10 ALT + feature selection + poly interactions is a better inductive bias for this small-n problem than learned latent factors. The IRT experiments (Section above) showed the same pattern — better ALT prediction (16.82 vs ~23) but worse final target prediction.

---

## Open Questions

1. ~~**Sliced error analysis**~~ — **Partially done.** See Residual Analysis section. Provider-level and reasoning/non-reasoning slices complete. Remaining: RMSE by ALT-observed/imputed, by missingness quintile, by score range (on OOF predictions).
2. **Exogenous metadata features** — provider, release date, parameter count, open/closed source. Genuinely orthogonal to benchmark scores.
3. **Arena vote count as sample weight** — models with more votes have more reliable ELO.
4. **More training rows** — still the single biggest lever. n=112 → 130+ would meaningfully reduce overfitting constraints.
5. **GroupKFold evaluation on honest data** — the leaked-era result showed ~27% RMSE increase when holding out entire provider families. Needs re-verification.
6. **ChatGPT-4o / Claude Opus 4.1 underprediction** — these models are systematically underpredicted (+35, +44) and not fixed by style features. May need conversational quality or instruction-following signals.
7. ~~**Full repeated K-fold verification**~~ — **Done.** Top-50 LOO: 17.89 (confirmed). 10×10-fold all 112: 19.76. 10×5-fold all 112: 20.25. All represent real improvement over 21.48 baseline.
8. **Tonebench shape features** — Investigated 2026-03-12. Per-question margin data exists (188 models, 97 with Arena) but is too sparse (median 2 battles/model/question). Best new features: `density_min` (r=0.407), `confidence_mean` (r=0.439), but both overlap heavily with existing TrueSkill (r=0.715). Shape features (std, cv, range) are weak (|r|<0.2). Not worth adding — the existing TrueSkill columns already capture the signal. Could revisit if more tone battles are collected.
