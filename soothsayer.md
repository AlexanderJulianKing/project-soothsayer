# Project Soothsayer — Technical Profile

> **Purpose of this document:** This is a comprehensive technical description of Project Soothsayer, written for an AI resume/cover-letter agent. It contains deep detail about what the project does, how it works, what's novel, and what skills it demonstrates. Use it to generate tailored descriptions for job applications.

---

## 1. Project Overview

**Project Soothsayer** is a solo-built end-to-end system for evaluating and predicting the quality of large language models. It combines 4 custom-designed LLM benchmarks with the 17 public/input benchmark families and 6 custom/derived CSV families consumed by the shipped predictor, then uses a custom imputation and adaptive machine learning pipeline to predict Arena ELO scores — the gold-standard measure of LLM quality based on millions of human preference votes. ([arena.ai](https://arena.ai), formerly LMSYS / Chatbot Arena / lmarena.ai.)

**The core problem:** Arena scores are the most trusted measure of LLM quality, but they require massive-scale human voting and many models lack scores. Soothsayer predicts these scores from benchmark data alone, enabling evaluation of new models without waiting for human preference data to accumulate.

**The core challenge:** Only 127 training examples, 119 KNN features in the current shipped run after `--drop_style_tone`, and severe missing data. 37 of the base features come from custom benchmarks I built (2.3% missing); the other ~50 come from public sources (40% missing). Standard ML breaks on data like this — global linear regression with median imputation achieves RMSE 26.3. Soothsayer achieves **RMSE 13.61** (R² = 0.941) on 10×5-fold CV, **RMSE 14.69** (R² = 0.900) on a fully-honest walk-forward that re-fits imputation, PCA, and PLS at every step.

**Scale:**
- 163 models tracked, 127 with known Arena scores used for training
- Benchmark columns drawn from the 17 public/input benchmark families plus 6 custom/derived CSV families consumed by the shipped predictor, plus 32 response-embedding dims (`sem_f*`) derived from the raw responses of the 4 custom benchmarks via bge-small → PCA-32
- 50+ controlled experiments across multiple improvement cycles, plus a full column-level LOBO analysis
- Best result: **RMSE 13.61** (10×5-fold CV), **R² = 0.941**, **Spearman ρ = 0.971**; honest temporal walk-forward **RMSE 14.69** (R² = 0.900, Spearman 0.940) on the newest 23 models

---

## 2. System Architecture

The system is organized as a three-stage pipeline with a shared core package:

```
Stage 1: Benchmark Execution (4 custom benchmarks, parallel)
    |
Stage 2: Data Collection & Combination (17 public/input + 6 custom/derived CSV families)
    |
Stage 3: Imputation & Prediction (missing value imputation -> adaptive KNN prediction)
    |
Post-hoc Analysis (diagnostic suite)
```

### Shared Core Package (`core/`)
A reusable Python package providing:
- **`llm_client.py`** — Unified API abstraction for all LLM interactions via OpenRouter. Handles provider routing, reasoning effort mapping, retry with exponential backoff.
- **`trueskill_arena.py`** — Generic TrueSkill pairwise comparison engine with information-gain match selection, used by EQ and Writing benchmarks.
- **`benchmark.py`** — Abstract base class that all benchmarks implement, enabling a unified CLI orchestrator.
- **`cli.py`** — Parallel benchmark orchestrator using `ThreadPoolExecutor`.
- **`utils.py`** / **`config.py`** — Shared utilities and configuration.

### Key Engineering Patterns
- **Resume/idempotency:** Every benchmark checks existing outputs before running. Interrupted runs resume seamlessly.
- **Parallel execution:** `ThreadPoolExecutor` throughout (10-40 workers depending on stage).
- **Modular refactoring:** The codebase went through 9 structured refactoring phases — from monolithic scripts to a proper Python package with abstract base classes, shared engines, and a CLI orchestrator.

---

## 3. Custom Benchmarks

A key insight of Soothsayer is that **custom benchmarks provide near-complete coverage** (2.3% missing) compared to public benchmarks (40% missing). This dramatically reduces reliance on imputation for the most predictive features. Further, because the custom benchmarks are the only place where the raw text responses are in hand, they double as the input to the `sem_f*` response-embedding feature set (see §6 Prediction Pipeline — embedding fingerprints alone cut OOF RMSE by 1.29 points).

### Soothsayer EQ — Emotional Intelligence
Evaluates LLMs on emotional intelligence, empathy, and interpersonal reasoning through challenging role-play scenarios. Uses a **TrueSkill pairwise tournament** where an LLM judge compares pairs of model responses:
- Paired A/B evaluation cancels positional bias (each pair tested in both orientations)
- Information-gain match selection: prioritizes pairs with high `sigma x match_quality` (uncertainty x closeness)
- Batch parallel execution with iterative rating recomputation
- Output: TrueSkill ELO ratings per judge model

### Soothsayer Writing — Creative Writing Quality
Evaluates storytelling quality across diverse creative writing prompts using the same TrueSkill pairwise engine:
- Story generation from parameterized creative prompts (character + object + action + setting templates)
- Two-stage evaluation: direct rubric scoring + pairwise comparison tournament
- Multiple LLM judges for robustness
- Output: Per-judge TrueSkill ratings and rubric scores

### Soothsayer Logic — Commonsense Reasoning
Tests common-sense reasoning with "trick" questions — riddles with critical modifications that change the expected answer, exposing pattern-matching vs. genuine understanding:
- Multi-run answer collection (4 runs per model) to sample behavioral variance
- LLM grading with per-question accuracy extraction
- PCA decomposition of per-question scores into latent reasoning style components (PC2-PC4)
- Output: Accuracy, weighted accuracy, per-category accuracy (physics, trick), PCA components, token usage statistics

### Soothsayer Style — Writing Style Analysis
Quantifies how models format their responses — length, headers, bold text, lists — and how consistently they apply formatting across different types of prompts:
- Collects responses to 9 diverse questions testing different cognitive loads, 3 runs each for consistency
- Per-question metrics: response length, headers, bold text, list items, coefficient of variation (measuring adaptability)
- `style_predicted_delta`: predicts the gap between raw and style-debiased Arena scores (r = -0.900 with actual gap)
- 20+ style columns including aggregate stats, per-question metrics, minimums, and fraction-used metrics


---

## 4. Data Integration Pipeline

### Scraping Layer (12 grabber scripts, 11 currently active in `scrape.bash`)
The repo contains 12 scraper scripts. The current `scrape.bash` enables 11 of them, while `benchmark_combiner/combine.py` consumes 23 CSV patterns total: 17 public/input families plus 6 custom/derived families.

### Model Name Resolution
The central challenge of combining many benchmark inputs is that every source uses different model names. The pipeline handles this with:
1. **Per-source JSON mappings** (20 mapping files) — manual source-to-canonical name translation
2. **LLM-assisted mapping** — For unmapped models, queries Google Gemini API to suggest matches against the known namespace
3. **Bad model filtering** — Known problematic entries excluded

### Combination Layer (`benchmark_combiner/`)
- `combine.py`: Multi-encoding CSV reading, full outer join aggregation across all sources, conflict detection, duplicate resolution
- `correlations.py`: Exploratory analysis — low-variance column filtering, correlation analysis, dimensionality reduction (PCA, t-SNE, UMAP), clustering
- Output: `clean_combined_all_benches.csv` — 163 models × base benchmark columns; sem-augmented variant `clean_combined_all_benches_with_sem_v4_d32.csv` adds 32 response-embedding dims for a total of 123 columns (127 rows have the `lmarena_Score` target)

---

## 5. Imputation Pipeline

The public benchmark data has 40% missing values (not every model appears in every benchmark). The custom `ModelBankImputer` fills these gaps with per-cell model selection and low-rank coherence projection.

### ModelBankImputer — Custom Per-Cell Imputation Architecture

**Key insight:** Standard imputation assigns one model per column. ModelBankImputer recognizes that the best predictor set varies by *which other benchmarks a model has been tested on*. A model tested on LiveBench + AiderBench but missing GPQA should use different predictors than one tested on MMLU + ARC but missing GPQA.

**Algorithm:**

1. **Candidate ranking:** For each target column, rank predictors by `|correlation| x sqrt(n_common_rows)` with redundancy filtering (skip predictors with |r| > 0.85 to already-selected ones)

2. **Pass 1 (observed-only):** For each missing cell, find the best predictor subset from columns actually observed in that row. Fit BayesianRidge (cached by predictor subset key). Single-proxy challenger: if a 1-predictor model has lower LOO variance than the multi-predictor model, prefer the simpler one. Adaptive k: k=1 for n<15 training rows, up to k=8 for n>80.

3. **Pass 2 (expansion):** Revisit high-uncertainty cells, allow one confidently-imputed value (sigma/sd < 0.4) as an additional predictor. Accept only if variance improves by >5%. Jacobi-style frozen inputs prevent oscillation.

4. **Coherence projection (λ = 8.0, exp shape):** SVD low-rank projection of the completed matrix. Blend each imputed cell toward the coherent estimate: `x' = (1-w)*x_imputed + w*SVD_x`, where `w = tau/(tau+lambda)` and `tau = cell_variance/column_variance`. High-uncertainty cells are pulled toward the SVD reconstruction; confident cells retain their model-bank value. This restores cross-column consistency — individually optimal per-cell predictions can create "Frankenstein" model profiles that are accurate per-column but incoherent as a row. The shipped λ = 8.0 was tuned after the PLS hybrid landed; at λ = 1.0 (the prior default), the PLS component over-regularized.

5. **LOO-forward predictor selection:** Rather than fixing a predictor subset per column, at each cell the imputer adds predictors one at a time and keeps the extension if it lowers raw leave-one-out MSE on the observed rows. Cheaper than full subset search; catches the common case where one more predictor moves the estimate into a different regime.

6. **Per-cell uncertainty tracking:** Analytical hat-matrix leave-one-out variance for every imputed cell, carried through the entire pipeline.

7. **Feature extraction from imputation:** 6 SVD latent factors (from the completed matrix decomposition), 6 squared terms (nonlinear transforms), 6 interaction terms (pairwise products of first 4 factors), and 3 trajectory features (mean delta, max delta, n_imputed across passes). These 21 derived features extend the imputed benchmark matrix into the KNN feature set.

### Why coherence projection matters
Without it, each cell is imputed by its own little regression model. The results are individually reasonable but the completed rows don't behave like real models — they combine traits that no actual model exhibits. SVD projection forces the completed matrix toward a low-rank approximation, which is the natural structure of benchmark data (models have a small number of latent capability dimensions). The blending weight ensures confident imputations aren't disturbed while uncertain ones snap to the coherent manifold.

---

## 6. Prediction Pipeline

### Why Adaptive KNN — The Feature Sign Flip Problem

Global linear regression fails because **different features predict Arena score at different score levels**. Feature correlations flip sign around ~1400 ELO:
- `style_list_count`: +0.16 overall, **-0.74** within top 15 models
- `livebench_code_completion`: +0.61 overall, **-0.32** within top 15

A global model averages these out. KNN with local Ridge regression recovers the local relationships.

### Adaptive KNN Algorithm

1. **Feature matrix:** imputed benchmark columns + 21 imputation-derived features (SVD / trajectory) + 32 sem_f* response-embedding dims. The `style_*` and `tone_*` columns are dropped pre-KNN (`--drop_style_tone`) — they were hurting once PLS was introduced in step 3. Standardized per fold.

2. **Response-embedding fingerprints (`sem_f*`):** Raw responses from all four custom benchmarks are embedded with bge-small (384-dim), pooled per (model, slot) across 5 slots (eq_t1 / eq_t3 / logic / style / writing), then compressed to 32 PCA dims. Mean-pooling across slots cancels prompt-topic variance while accumulating model-voice signal. Adds −1.29 RMSE over the base pipeline; an embedder + PCA + mode sweep (bge-small, bge-base, bge-large, nomic, gte-large at 2048 ctx; d=16/24/32/48/64; 5/6/13 slots) confirmed the shipped config (bge-small, d=32, 5-slot) is a global optimum for this training size.

3. **PLS hybrid (fold-internal PLS-3):** Inside every fold, a 3-component Partial-Least-Squares regression is fit on the training features against `lmarena_Score` and its transformed coordinates are appended to the feature matrix. PLS supervises the feature space with the target but never touches held-out rows — the appended PLS columns act as a low-rank summary of "the direction of ELO" which the KNN distance can use without the fragility of tree-based models at n = 127. Adds −0.97 RMSE on top of sem.

4. **Adaptive neighborhood via sublinear power cutoff:**
   - `max_dist = d_nearest^0.7 x 3.0`
   - Models with distinctive feature profiles (nearest neighbor is relatively far) get tight neighborhoods (k ~= 20, the floor)
   - Models in dense regions of feature space (many similar peers) get wider neighborhoods (k up to 80)
   - The sublinear exponent (0.7) prevents the cutoff from blowing up in sparse regions while keeping it tight in dense ones
   - This was selected from a sweep of 2,700 configurations as the best tradeoff between LOO accuracy and walk-forward (temporal) accuracy

5. **Gaussian kernel weighting:** `w = exp(-0.5 * (dist/bw)^2)` with bandwidth at the 15th percentile neighbor distance. Closer neighbors get exponentially more influence.

6. **Local Ridge regression with adaptive alpha:** `alpha = max(10, std(neighbor_scores))`. Tight, homogeneous neighborhoods get lower regularization (more flexible fit); diverse neighborhoods get higher regularization.

7. **Jackknife variance inflation (bias correction):** Ridge regression centers predictions toward the neighborhood mean. The jackknife estimates the compression factor by leaving each neighbor out, refitting, and measuring how much the model compresses actual variation. The correction factor `b` (clipped to [1.0, 1.5]) re-expands predictions: `pred' = mean(neighbors) + b * (raw_pred - mean(neighbors))`.

8. **Per-model uncertainty outputs:** The shipped pipeline writes grouped-conformal uncertainty artifacts (`conformal_diagnostics.csv`, `conformal_uncertainty_features.csv`) plus aggregate OOF outputs. One-off neighborhood introspection is available through analysis helpers and diagnostic sidecars, but `predict.py` does not currently emit `knn_neighborhood_diagnostics.csv` / `knn_oof_diagnostics.csv` as default outputs.

9. **Grouped conformal prediction intervals:** 4 groups by predicted score x missingness level. Empirical quantile calibration per group. ~92% honest coverage.

### Baseline Comparison

| Method | RMSE | R² | Spearman ρ |
|--------|------|-----|-----------|
| Predict mean (dummy) | 56.6 | 0.00 | — |
| Public benchmarks + median impute + Ridge | 30.0 | 0.72 | 0.85 |
| All benchmarks + median impute + Ridge | 26.3 | 0.78 | 0.90 |
| **Full Soothsayer pipeline** | **13.61** | **0.941** | **0.971** |

The full pipeline achieves a 49% reduction over the best simple baseline (26.3 → 13.61), through the combination of custom benchmarks (near-complete coverage), response-embedding fingerprints, per-cell imputation with coherence projection, fold-internal PLS supervision, and adaptive local regression.

---

---

## 7. Results

### Prediction Accuracy (10×5-fold repeated cross-validation, 2026-04-18 shipped config)
| Metric | Value |
|--------|-------|
| RMSE | 13.61 (on ~1100-1500 ELO scale) |
| R² | 0.941 |
| Spearman ρ | 0.971 |
| Top-15 RMSE | 14.26 |
| Honest walk-forward RMSE | 14.69 (n = 23 newest, re-fits imputation + PCA-32 + PLS-3 every step) |
| Honest walk-forward R² / ρ | 0.900 / 0.940 |
| Training set | 127 models with `lmarena_Score` (163 total in combined CSV) |
| Feature set | imputed benchmarks + 21 imputation-derived (SVD / trajectory) + 32 sem_f* + 3 fold-internal PLS dims; `style_*` and `tone_*` dropped |

### Per-Model Interpretability
Every prediction comes with:
- Neighborhood size (k) and neighbor identities
- Ridge coefficients within the local neighborhood
- Per-feature contributions (coefficient x standardized value) ranked by magnitude
- Jackknife correction factor
- Conformal prediction interval

Example: Claude Opus 4.6 (actual: 1497) — the shipped pipeline predicts it cleanly (honest walk-forward err −24.2, one of the three hardest cases alongside Sonnet 4.6 Thinking and Qwen3.5 27B Thinking). The default shipped outputs include the grouped-conformal interval and aggregate OOF predictions; deeper neighbor-level introspection currently lives in ad hoc diagnostics rather than default `predict.py` CSV outputs.

---

## 8. Engineering Quality

### Documentation
- **`ARCHITECTURE.md`** (~380 lines) — System design, data flow, algorithm descriptions, configuration reference
- **`DATA_DICTIONARY.md`** (~350 lines) — Every column definition, file format, data type, naming conventions
- **`FINDINGS.md`** (~1000 lines) — All experiment results, leakage analysis, error decomposition
- **`WALKFORWARD_ANALYSIS.md`** (~330 lines) — Temporal CV methodology and the 2026-04-18 honest walk-forward result
- **`JUDGE_FINDINGS.md`** (~660 lines) — Parallel line of work on LLM-judge preference biases (tangential to the RMSE pipeline)

### Codebase Structure
- ~200 Python files (excluding scratch experiments), ~88k lines total
- 112 current collected test cases across `tests/`
- 12 automated scraper scripts
- 20 JSON model name mapping files
- Shell orchestration scripts for each pipeline stage (`scrape.bash`, `combine.bash`, `predict.sh`, `run_all_benches.bash`)
- `pyproject.toml` packaging with `pip install -e .` dev mode



---

## 9. Technologies & Skills Demonstrated

### Machine Learning & Statistics
- Adaptive KNN with kernel weighting and sublinear distance cutoffs
- Ridge regression with per-neighborhood adaptive regularization
- Partial Least Squares as a fold-internal supervised-feature extractor (PLS hybrid)
- Jackknife variance inflation (bias correction for shrinkage estimators)
- Conformal prediction intervals (calibrated uncertainty quantification)
- Bayesian imputation (BayesianRidge with per-cell LOO-forward predictor selection)
- Low-rank matrix completion (SVD coherence projection)
- Response-embedding fingerprints (bge-small → per-slot mean-pool → PCA-32) as a sibling feature family to explicit style metrics
- TrueSkill rating systems (information-gain match selection, paired evaluation)
- PCA, SVD, factor analysis for dimensionality reduction
- Cross-validation methodology (repeated K-fold, fully-honest temporal walk-forward with per-step refitting, LOO)
- Leakage detection and prevention
- Experimental design (50+ controlled experiments, ablation studies, hyperparameter sweeps)
- Constraint programming (CP-SAT for puzzle generation with guaranteed unique solutions)

### Software Engineering
- End-to-end system design (scraping -> combination -> imputation -> prediction -> analysis)
- Python package architecture (abstract base classes, shared engines, CLI orchestrator)
- Systematic refactoring (9-phase modernization of a growing codebase)
- Testing (112 current collected test cases, edge cases, integration tests)
- Parallel programming (ThreadPoolExecutor, Jacobi-style iterative updates)
- Rate-limit-aware concurrent API systems (per-model semaphores, adaptive backoff, resumable ETL)
- API integration (OpenRouter, Gemini — with retry logic, rate limiting, error handling)
- Data pipeline design (idempotent, resumable, cached)
- Remote compute orchestration (rsync-based code sync, parallel experiment scheduling on multi-core server)

### Data Engineering
- Multi-source data integration (16+ benchmark sources with inconsistent schemas)
- Entity resolution (model name mapping with LLM-assisted fuzzy matching)
- Missing data handling (40% missingness in public benchmarks, custom benchmarks providing near-complete coverage)
- Per-cell data quality tracking through the full pipeline

### Research & Analysis
- Scientific methodology (controlled experiments, baselines, ablation studies)
- Error analysis (provider-level, model-type, tier-dependent diagnostic breakdowns)
- Custom algorithm design (per-cell imputer with coherence projection, adaptive KNN with power cutoff)
- Domain expertise in LLM evaluation (Arena scoring mechanics, style bias, benchmark saturation)
