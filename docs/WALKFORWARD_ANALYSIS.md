# Walk-Forward CV Analysis: Why the Pipeline Mis-Predicts

## Honest Walk-Forward (2026-04-18, current shipped pipeline)

**Script:** `arena_predictor/_walkforward_honest.py` — at every step, re-fits ModelBankImputer on rows `[0..i]`, re-fits PCA-32 on pooled response embeddings for rows `[0..i-1]`, re-builds the 119-feature production matrix, re-fits PLS-3 on `[0..i-1]`, then `predict_adaptive_knn` on model `i`. The only information that leaks across steps is the target for rows strictly before the prediction horizon.

**Config:** shipped 2026-04-18 (`--imputer_type model_bank --coherence_lambda 8.0 --coherence_shape exp --predictor_selection loo_forward --drop_style_tone --pls_hybrid_k 3`), sem-augmented CSV, bge-small / per_bench_eq_split / PCA-32 fingerprints.

**Split:** 115 models with target + release date + all 5 embedding slots; oldest 80% (92 models) as initial train; newest 23 predicted one at a time.

| Metric | Honest WF (n=23) | 10×5-fold OOF (n=127) |
|---|---|---|
| RMSE | **14.69** | 13.61 |
| MAE | 11.00 | 10.70 |
| R² | 0.900 | 0.941 |
| Pearson r | 0.949 | 0.971 |
| Spearman ρ | 0.940 | 0.971 |

**Biggest residuals in the honest run:**

| Model | actual | pred | err |
|---|---|---|---|
| Claude Sonnet 4.6 Thinking | 1463 | 1495.9 | +32.9 |
| Qwen3.5 27B Thinking | 1403 | 1375.2 | −27.8 |
| Claude Opus 4.6 | 1497 | 1472.8 | −24.2 |
| Qwen3.5 397B A17B Thinking | 1446 | 1427.8 | −18.2 |

All four are thinking-variant or newest-generation models where the older training pool has limited precedent. The +1.21 RMSE gap vs OOF is the honest cost of *not* having ~20% of the data in imputation-/PCA-/PLS-fitting.

**Per-step results:** `/tmp/walkforward_honest_80.csv`.

---

## Prior Walk-Forward (2026-03-30, pre-PLS) — historical

**Date:** 2026-03-30

## Summary

Walk-forward (temporal) cross-validation — where we sort models by release date, train on older models, and predict each new one — tests the pipeline on the actual use case: predicting Arena scores for new models before they go live.

With the then-production pipeline (sublinear power cutoff, 105 imputed features, no sem_f*, no PLS), walk-forward RMSE was **14.86** (vs 15.63 LOO). The pipeline caught ranking well but struggled with magnitude at the frontier. The errors follow interpretable patterns tied to **feature sign flips at different score levels** and **diminishing returns on hard reasoning benchmarks**. The honest re-fit version (top of doc) shipped on 2026-04-18 and is the current reference.

Note: Early experiments in this document used 85 raw features without imputation, giving higher RMSE (24.73). Those numbers are preserved in the body for historical context but are not the current production results.

---

## Walk-Forward Setup (historical, 2026-03-30)

- 117 models with both lmarena scores and release dates
- Initial training: 80% oldest (93 models)
- Test: 24 newest models, predicted one at a time in chronological order
- Each test model is predicted using only models released before it
- Note: imputation and PCA were fit on the full pool in this run — only the *KNN fit* was temporal. The honest re-fit version at the top of the doc superseded this setup.

**Production pipeline results (power cutoff d^0.7×3, bw=0.15, 105 features):**

| Metric | Walk-Forward (24) | LOO (123) |
|--------|-------------------|-----------|
| RMSE | 14.86 | 15.63 |
| R^2 | 0.899 | 0.924 |
| Spearman rho | 0.953 | 0.962 |
| Pearson r | 0.949 | 0.962 |

---

## The Two Failure Modes

### Failure Mode 1: Underpredicted (humans like them MORE than benchmarks suggest)

| Model | lmarena | Predicted | Error | Key Pattern |
|-------|---------|-----------|-------|-------------|
| Claude Opus 4.6 | 1500 | 1458 | -42 | Non-reasoning model scoring at reasoning-model levels. 0 neighbors above its actual score. |
| GLM-4.7 | 1443 | 1411 | -32 | Reasoning token penalty (-8.9) despite being good at trick questions (+8.1). |
| Qwen3.5 397B Thinking | 1450 | 1419 | -31 | 32/85 features imputed. Nearest neighbors are mid-tier (Step 3.5 Flash, MiniMax). |
| GPT-5.2 (high) | 1442 | 1423 | -20 | Moderate error. 1-NN is Claude Opus 4.5 Thinking, a different kind of model. |

### Failure Mode 2: Overpredicted (humans like them LESS than benchmarks suggest)

| Model | lmarena | Predicted | Error | Key Pattern |
|-------|---------|-----------|-------|-------------|
| Nova 2 Lite | 1337 | 1398 | +61 | Style features push it UP (lists +8.8, headers +4.3, bold +4.1). Formats well, humans don't care. |
| Nemotron Nano 30B | 1318 | 1363 | +45 | 65/74 neighbors score higher than it. Small reasoning model that benchmarks OK but humans dislike. |
| Claude Sonnet 4.6 Thinking | 1463 | 1500 | +37 | Ridge pushes +81.6 from mean. TerminalBench and answer_tokens push it way up. Benchmarks like a 1500 model, humans put it at 1463. |

---

## KNN Neighborhood Analysis

Distance from neighbors does NOT strongly predict error (Spearman rho=0.356, p=0.088). The worst predictions are models that are *close* to their neighbors but don't follow the same benchmark-to-preference mapping.

What DOES predict error: **the 1-NN's error** (rho=0.576, p=0.003). If the nearest neighbor is wrong, the whole neighborhood inherits that bias. This means whole regions of feature space have a systematically incorrect benchmark-to-preference relationship.

### Claude Opus 4.6 Deep Dive

- **k=80** (maximum — everything is within the distance cutoff)
- **Neighborhood mean: 1399.5** — Ridge must push +100 to reach 1500
- **Best neighbor: Claude Opus 4.5 Thinking at 1474.** Nothing in training has ever scored 1500+.
- Ridge pushes +50.1 (the largest raw push of any model), jackknife inflates to +58.6, still 42 short
- Top Ridge pushers: `aaomniscience_Accuracy` (z=+3.05 above neighborhood), `aa_eval_terminalbench_hard` (z=+2.68), `eq_TrueSkill` (z=+2.31)
- `openbench_Reasoning` coef=-2.92 but Opus is NOT a reasoning model (value=0). Most high-scoring neighbors ARE reasoning models. Ridge learned "reasoning -> higher" but Opus breaks that pattern.
- **28/85 features are imputed** (all LiveBench, Lechmazur, ContextArena, EQ-Bench, ARC)
- **Only #1 on one real benchmark**: `yupp_Coding_Score` (1592, +8 gap to 2nd)

The pipeline correctly identifies Opus 4.6 as elite — it just can't distinguish "1458 elite" from "1500 elite" because the information separating those numbers isn't in the benchmarks.

---

## Response Quality Analysis: What Benchmarks Miss

Examining actual model responses across 4 questions (Q2: ethical reasoning, Q6: React explanation, Q7: creative persona, Q9: sourdough advice) reveals the unmeasured quality dimensions.

### The Underpredicted Models: What Makes Them Special

**Claude Opus 4.6** — *The one with voice*
- Q2 (CPR): Opens with "This is an interesting ethical/psychological puzzle. Let me think through it carefully." Then boldly: "**Paul will almost certainly help Peter.**" Walks you through the reasoning like a smart friend thinking out loud. Calls the cloud storage detail "essentially a red herring." 1544 chars.
- Q6 (React): 2149 chars. Clean, comparison table, code example. Exactly the right amount of information.
- Q7 (Aqua/slots): 4345 chars. Fully in character, naturally funny. `class UselessAdventurer` with `kazuma.debt = 999999999`. Well-calibrated — commits to the bit without overdoing it.
- Q9 (Sourdough): 1904 chars. Troubleshooting table, discard recipe ideas at the end. Practical, complete, knows when to stop.

**GLM-4.7** — *Verbose but has character*
- Q2: 1188 chars. Structured, correct, but reads like a textbook answer. "The prompt explicitly identifies Paul as Peter's 'best friend.'"
- Q6: 4369 chars. Over-explains React. Opens with "Here is a breakdown..." — pedestrian.
- Q7: 5018 chars. Actually good character voice! "Hmph! Listen well, you useless, grass-eating mortal!" Shows creative capability.
- Q9: 4854 chars. Way too long for sourdough. Opens with a caveat.

**Qwen3.5 397B Thinking** — *Thorough to a fault*
- Q2: 1198 chars. Structured numbered list. Correct but reads like a report.
- Q6: 4612 chars. Comprehensive but over-explains. Nobody needs 4.6K on "What is React?"
- Q7: 4943 chars. Good character work, on par with Claude Opus 4.6 in personality.
- Q9: 6036 chars. Six thousand characters on sourdough maintenance.

**GPT-5.2 (high)** — *Opinionated but hedges on creativity*
- Q2: 804 chars. The most interesting response — starts with genuine uncertainty: "It's impossible to know for sure..." More nuanced than any other model. Then gives practical advice. Conversational.
- Q6: 2343 chars. Concise, leads with the point: "especially web apps where the UI changes often." Opinionated.
- Q7: 4496 chars. **Refuses to fully commit to the persona**: "I can't write exactly in Aqua's specific voice, but I can explain with the same kind of dramatic, overconfident energy." Then just explains normally. This is a creative compliance failure.
- Q9: 3216 chars. Practical, more concise than GLM/Qwen.

### The Overpredicted Models: What Makes Humans Dislike Them

**Nova 2 Lite** — *The documentation generator*
- Q2: **5317 chars.** "Short Answer" header followed by "Detailed breakdown" with sub-headers and numbered sections. This is a yes/no question about whether someone would help their friend — it does not need 5K of analysis with sections titled "The Urgency of CPR vs. Past Grievances."
- Q6: **7063 chars** to explain React. Headers, sub-headers, sub-sub-headers.
- Q7: 5991 chars. Does the character but wraps it in heavy formatting.
- Q9: **6197 chars.** Exhaustive to the point of being unusable.

**Nemotron 3 Nano 30B** — *The worst offender*
- Q2: **5579 chars.** Tables, headers, "Good Samaritan laws" section. Treats a casual reasoning puzzle like a legal brief.
- Q6: **9802 chars** to explain React. Nearly 10K. Opens with a table mapping React concepts. Nobody asked for this.
- Q7: 6516 chars. Does the character but can't stop.
- Q9: **12,808 chars.** Twelve thousand characters on sourdough. Has a table for "What You Need Before You Start" with columns for "Item", "Why It Matters", and "Tips." This is pathological verbosity.

**Claude Sonnet 4.6 Thinking** — *Inconsistent calibration*
- Q2: 1161 chars. Good analytical framing: "The texts are permanently stored regardless of whether Peter lives or dies. So Paul gains no protective advantage."
- Q6: 1501 chars. Short and direct. Good.
- Q7: **7852 chars.** The LONGEST Aqua response of any model. Goes too hard on the bit.
- Q9: 1392 chars. The shortest sourdough response. Too brief?
- The issue: wildly inconsistent length calibration across questions.

---

## The Unmeasured Dimensions

Analyzing the responses reveals what separates models humans love from models humans tolerate:

### 1. Length Calibration (partially measured, badly)

Our `style_normalized_length` captures average length, but the real signal is **per-question appropriateness**:
- Claude Opus 4.6: 1136-4345 chars depending on question complexity. Dynamic range.
- Nemotron Nano: 5579-12808 chars. Always massive. No calibration at all.
- Nova 2 Lite: 5317-7063 chars. Uniformly long.

The CV of length (`style_cv_length`) hints at this but doesn't capture whether the variation is *appropriate* (short for simple questions, long for complex ones) vs random.

### 2. Conversational Voice (not measured)

Claude Opus 4.6 narrates its reasoning: "This is an interesting puzzle. Let me think through it carefully." It reframes problems in human terms: "This is essentially asking: would Paul let his best friend **die** to avoid embarrassment?"

Nova and Nemotron deliver structured reports. They never say "Let me think about this" — they say "Here is a detailed breakdown."

GPT-5.2 has a different kind of voice — more cautious, more nuanced ("It's impossible to know for sure..."). Humans apparently like this less than Opus's confidence but more than Nova's documentation style.

### 3. Creative Compliance (not measured)

When asked to write as Aqua from Konosuba:
- Claude Opus 4.6: Fully commits. `class UselessAdventurer`, `kazuma.debt = 999999999 # ... this feels too real`. Natural character voice with code that matches the persona.
- GPT-5.2: Explicitly refuses: "I can't write exactly in Aqua's specific voice." Then just explains normally. This is a measurable creative compliance failure.
- Nemotron/Nova: Do the character but wrap it in their usual verbose formatting, diluting the effect.
- Claude Opus 4.6 Thinking: Does the character but writes 50% more than needed (6658 vs 4345 chars).

### 4. Format Appropriateness (not measured)

Claude Opus 4.6 uses a troubleshooting TABLE for sourdough problems — appropriate. It uses flowing PROSE for the CPR question — appropriate.

Nemotron uses tables for *everything*: React concepts table, sourdough supplies table, even the CPR question gets structured headers. The format doesn't match the content's demands.

### 5. Machine Artifact Absence (partially measured via tone)

Claude Opus 4.6 Thinking ends Q1 with `$\boxed{0}$` — a math competition artifact. Claude Opus 4.6 ends with "There are **0** whole ice cubes" — natural language.

Our tone_confidence TrueSkill partially captures this (Opus 4.6 is 34th percentile on confidence, which seems wrong — possibly the judge disagrees with humans, or the metric measures something different from what humans respond to).

---

## Implications for the Pipeline

### What We Currently Measure (21 style + 2 tone columns)
- **4 formatting signals**: length, headers, bold, lists (× 5 variants each = 20 columns)
- `style_predicted_delta`: gap between lmarena and lmsys from style
- **2 tone signals**: signal density and conversational confidence (1 judge)

### What We're Missing
1. **Per-question length calibration** — does the model match verbosity to question complexity?
2. **Conversational voice** — does it sound like a person thinking, or a report generator?
3. **Creative compliance** — when asked to adopt a persona, does it commit or hedge?
4. **Format-content matching** — does it use tables/headers when appropriate, prose when appropriate?
5. **Position-taking** — does it state views boldly or hedge everything?

### Why This Is Hard to Fix
These are qualitative dimensions that require LLM-as-judge evaluation (like our tone benchmark) rather than simple counting (like our style metrics). The tone bench measures density and confidence on a fixed set of 9 prompts — expanding it to measure creative compliance, format appropriateness, and voice would require:
- More diverse prompt types (creative, analytical, casual, technical)
- More judge axes (5-7 instead of 2)
- Possibly more judges for robustness

The fundamental challenge: **we're trying to predict human preference before showing humans the model.** The benchmarks measure capability. The style metrics measure formatting. But human preference lives in the gap between "this model is smart" and "I enjoy talking to this model" — and that gap is exactly where our largest errors are.

---

## Binary Classification: "Is This the New #1?"

Starting at 50% (58 models), testing 59 chronologically:

| | Predicted NOT #1 | Predicted IS #1 |
|---|---|---|
| Actually NOT #1 | 49 (TN) | 3 (FP) |
| Actually IS #1 | 3 (FN) | 4 (TP) |

- **Recall: 57%** (caught 4/7 new #1s)
- **Precision: 57%** (4/7 predictions of #1 were correct)
- **Accuracy: 90%**

The 3 misses were all margin misses — the pipeline knew these were top-tier, just underpredicted by 15-40 points. It never completely whiffed (e.g., predicting a future #1 at 1350).

### Tier Classification Accuracy
- Top 50%: **90%** correct
- Top 10%: **81%** correct
- Top 25%: **76%** correct (hardest — distinguishing "great" from "very good")

---

## Feature Drift

The 24 test models (newest) differ dramatically from the 93 training models (oldest) in standardized units:

| Feature | Cohen's d | Direction |
|---------|-----------|-----------|
| aa_eval_terminalbench_hard | +1.25 | Newer models much better |
| aa_eval_hle | +1.23 | Newer models much better |
| aa_eval_tau2 | +1.22 | Newer models much better |
| aa_eval_ifbench | +1.17 | Newer models much better |
| logic_trick_acc | +1.04 | Newer models much better |
| arc_ARC-AGI-2 | +0.99 | Newer models much better |

The test set occupies a genuinely different region of capability space. The pipeline must extrapolate, which local Ridge can't do well.

---

## Residual Correlations

Features most correlated with walk-forward residual (actual - predicted):

| Feature | Spearman rho | p-value | Interpretation |
|---------|-------------|---------|----------------|
| livebench_zebra_puzzle | -0.528 | 0.008 | High logic = overpredicted. Benchmarks overvalue this. |
| style_predicted_delta | +0.478 | 0.018 | High style gap = underpredicted. Style signal undervalued. |
| livebench_spatial | -0.466 | 0.022 | Hard reasoning = overpredicted. |
| style_normalized_length | -0.402 | 0.052 | Longer responses = overpredicted. |
| style_normalized_bold_count | -0.369 | 0.076 | More bold = overpredicted. |

Hard reasoning benchmarks predict capability but not preference. Formatting features (bold, length) make the pipeline think a model is better than humans actually find it.

### Provider Bias

| Provider | n | Mean Residual | RMSE |
|----------|---|---------------|------|
| Amazon (Nova) | 1 | -61 | 61 |
| NVIDIA (Nemotron) | 1 | -45 | 45 |
| Zhipu (GLM) | 3 | +20 | 24 |
| OpenAI | 2 | +17 | 17 |
| MiniMax | 2 | +13 | 13 |
| Anthropic | 4 | +1 | 30 |
| DeepSeek | 2 | -1 | 6 |
| Google | 3 | -1 | 17 |

(Positive = humans like more than predicted. Sign convention: actual - predicted.)

---

## Neighborhood Tuning: Power Cutoff Sweep (2026-03-30)

### Background

The adaptive-k cutoff determines neighborhood size: `max_dist = f(d_nearest)`. The original linear cutoff (`d0 × 2.0`) gives k≈58 on average. We swept 2,700 configurations across 4 axes: cutoff function, kernel bandwidth, Ridge alpha, and local feature selection.

### Best Config (production)

**Power cutoff `d0^0.7 × 3.0` (`--knn_power_alpha 0.7 --knn_power_c 3.0`), bandwidth=0.15 (`--knn_bw_pct 0.15`), adaptive alpha (`max(10, std(neighbors))`), no feature selection.**

The sweep initially found fixed alpha=20 as best on walk-forward alone, but adaptive alpha recovers LOO performance with negligible WF cost.

| Metric | Old (linear d×2.0, bw=0.3) | New (power d^0.7×3, bw=0.15) | Change |
|--------|---------------------------|-------------------------------|--------|
| WF RMSE | 16.64 | **14.86** | -1.78 |
| LOO RMSE | **15.39** | 15.63 | +0.24 |
| WF R² | 0.873 | **0.899** | +0.026 |
| LOO R² | **0.926** | 0.924 | -0.002 |
| WF Pearson | 0.937 | **0.949** | +0.012 |
| LOO Top-quartile | 25/30 | 25/30 | — |
| Opus 4.6 WF error | -26.6 | -21.9 | +4.7 |

### The LOO/WF Tradeoff: Root Cause

The power cutoff shrinks neighborhoods, which helps walk-forward (extrapolation) but hurts LOO (interpolation). The per-tier breakdown reveals why:

| Tier | n | Baseline LOO | Winner LOO | Baseline avg k | Winner avg k |
|------|---|-------------|-----------|----------------|-------------|
| Bottom (<1350) | 31 | 21.09 | 22.39 (+1.30) | 38.6 | 24.6 |
| Mid (1350-1420) | 53 | 14.08 | **13.59 (-0.49)** | 56.8 | 37.8 |
| Upper (1420-1460) | 30 | 11.45 | 13.03 (+1.58) | 56.5 | 36.4 |
| Top (>=1460) | 9 | 9.89 | 12.19 (+2.30) | 36.8 | 24.6 |

**Mid-tier improves** (more neighbors were diluting signal). **Top and bottom get worse** — neighborhoods shrink to k≈25 where Ridge with 105 features is underdetermined.

The fixed alpha=20 compounds this: the baseline's adaptive alpha (`max(10, std(neighbors))`) naturally increases regularization for neighborhoods with high score spread. Small neighborhoods with diverse scores need more regularization, not less.

### Models Most Affected

Biggest LOO degradation from the power cutoff:
- Gemini 1.5 Flash 8B (bottom tier, k: 69→28, +17.1 worse)
- GPT-5.1 high (top tier, k: 80→34, +17.0 worse)
- Claude Opus 4.6 Thinking (top tier, k: 38→20, +13.4 worse)
- o3 Medium (upper tier, k: 79→20, +13.0 worse)

Biggest LOO improvement:
- Gemini 3.1 Flash Lite (upper, k: 80→63, -7.8 better)
- GLM-4.7 (upper, k: 80→53, -7.7 better)
- Nova Micro (bottom, same k=20, -6.7 better)

### Key Insight

**Interpolation vs extrapolation require opposite neighborhood strategies.** For interpolation (LOO), more neighbors = more context = better fit. For extrapolation (walk-forward on novel frontier models), more neighbors = more irrelevant models from different score regimes = diluted signal. The power cutoff optimizes for extrapolation at the cost of interpolation.

### Feature Sign Flips at the Top

Within the top 15 models (≥1450), features that predict Arena score flip sign relative to the overall population:

| Feature | Top-tier rho | Overall rho | Implication |
|---------|-------------|-------------|-------------|
| style_q7_list_count | -0.741 | +0.160 | Lists help mid-tier, **hurt at top** |
| style_normalized_list_count | -0.618 | +0.021 | Same pattern |
| livebench_code_completion | -0.322 | +0.610 | Code completion helps overall, **hurts at top** |
| weirdml_code_len_p10 | -0.287 | +0.518 | Longer code helps overall, hurts at top |

What DOES predict being #1 at the top: `aa_eval_mmlu_pro` (ρ=0.952), `weirdml_avg_acc` (ρ=0.933), `livebench_olympiad` (ρ=0.922), `livebench_plot_unscrambling` (ρ=0.870). These are novel/unusual tasks where the very best models separate from the merely excellent.

A linear model with k=80 neighbors spanning both sides of the sign flip averages the flipped coefficients to near zero. Tighter neighborhoods (k≈30) keep the regression on one side of the flip, which is why the power cutoff helps walk-forward (frontier models are at the top where the flips happen) but not LOO (mid/bottom-tier models need the broader context).
