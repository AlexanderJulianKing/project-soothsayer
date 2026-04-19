# Project Soothsayer — Technical Profile

> **Purpose of this document:** This is a comprehensive technical description of Project Soothsayer, written for an AI resume/cover-letter agent. It contains deep detail about what the project does, how it works, what's novel, and what skills it demonstrates. Use it to generate tailored descriptions for job applications.

---

## 1. Project Overview

**Project Soothsayer** is a solo-built end-to-end system for evaluating and predicting the quality of large language models. It combines 4 custom-designed LLM benchmarks with the 17 public/input benchmark families and 6 custom/derived CSV families consumed by the shipped predictor, then uses a custom imputation pipeline plus adaptive local regression to predict Arena ELO scores — the gold-standard measure of LLM quality based on millions of human preference votes. ([arena.ai](https://arena.ai), formerly LMSYS / Chatbot Arena / lmarena.ai.)

**The core problem:** Arena scores are the most trusted measure of LLM quality, but they require massive-scale human voting, so many models go months without a score. Soothsayer predicts these scores from benchmark data alone, enabling evaluation of new models without waiting for human preference data to accumulate.

**Scale:**
- ~88,000 lines of Python across ~200 files (excluding scratch experiments)
- 112 current collected test cases
- 163 models tracked, 127 with known Arena scores used for training
- Benchmark columns from 16+ sources (37 custom + ~50 public), plus 32 response-embedding dims derived from the raw responses of the 4 custom benchmarks (via bge-small → PCA-32)
- 50+ controlled experiments across multiple improvement cycles, plus a full column-level LOBO analysis
- Best result: **RMSE 13.61** on Arena ELO prediction (scale ~1100–1500), **R² = 0.941**, **Spearman ρ = 0.971** on 10×5-fold CV; **RMSE 14.69, R² 0.900, ρ 0.940** on a fully-honest temporal walk-forward that re-fits imputation, PCA, and PLS at every step.

---

## 2. System Architecture

The system is organized as a three-stage pipeline with a shared core package:

```
Stage 1: Benchmark Execution (4 custom benchmarks, parallel)
    ↓
Stage 2: Data Collection & Combination (17 public/input + 6 custom/derived CSV families)
    ↓
Stage 3: Imputation & Prediction (missing value imputation → adaptive KNN + local Ridge)
    ↓
Post-hoc Analysis (15-section diagnostic suite)
```

### Shared Core Package (`core/`)
A reusable Python package providing:
- **`llm_client.py`** — Unified API abstraction for all LLM interactions via OpenRouter. Handles provider routing, reasoning effort mapping, retry with exponential backoff.
- **`trueskill_arena.py`** — Generic TrueSkill pairwise comparison engine with information-gain match selection, used by two benchmarks.
- **`benchmark.py`** — Abstract base class that all benchmarks implement, enabling a unified CLI orchestrator.
- **`cli.py`** — Parallel benchmark orchestrator using `ThreadPoolExecutor`.
- **`utils.py`** / **`config.py`** — Shared utilities and configuration.

### Key Engineering Patterns
- **Resume/idempotency:** Every benchmark checks existing outputs before running. Interrupted runs resume seamlessly. The embedding chain is likewise idempotent — adding a new model incurs only its own ~100–200 rows of embedding work.
- **Parallel execution:** `ThreadPoolExecutor` throughout (10–40 workers depending on stage).
- **Leakage-safe CV:** Every feature that touches the target is computed fold-internally. Three major pre-CV leakage bugs were caught and diagnosed (delta head, pairwise rank, in-sample style predictions).
- **Modular refactoring:** The codebase went through 9 structured refactoring phases — from monolithic scripts to a proper Python package with abstract base classes, shared engines, and a CLI orchestrator.

---

## 3. Custom Benchmarks

### Soothsayer EQ — Emotional Intelligence
Evaluates LLMs on emotional intelligence, empathy, and interpersonal reasoning through challenging role-play scenarios. Uses a **TrueSkill pairwise tournament** where an LLM judge compares pairs of model responses:
- Paired A/B evaluation cancels positional bias (each pair tested in both orientations)
- Information-gain match selection: prioritizes pairs with high `sigma × match_quality` (uncertainty × closeness)
- Batch parallel execution with iterative rating recomputation
- Output: TrueSkill ELO ratings per judge model

### Soothsayer Writing — Creative Writing Quality
Evaluates storytelling quality across diverse creative writing prompts using the same TrueSkill pairwise engine:
- Two-stage evaluation: Stage 1 (rubric-based scoring by LLM judges), Stage 2 (pairwise comparison tournament)
- Multiple LLM judges for robustness (results stored per-judge)
- Output: Per-judge TrueSkill ratings and rubric scores

### Soothsayer Logic — Commonsense Reasoning
Tests common-sense reasoning with "trick" questions — riddles with critical modifications that change the expected answer, exposing pattern-matching vs. genuine understanding:
- Multi-run answer collection (4 runs per model) to sample behavioral variance
- LLM grading with per-question accuracy extraction
- PCA decomposition of per-question scores into latent reasoning style components
- Output: Accuracy, weighted accuracy, per-category accuracy, PCA components, token usage

### Soothsayer Style — Writing Style Analysis
Quantifies how models format their responses — length, headers, bold text, lists — and how consistently they apply formatting across different types of prompts:
- Collects responses to 9 diverse questions across cognitive loads, 3 runs each
- Per-question metrics: response length, headers, bold text, list items, coefficient of variation
- `style_predicted_delta`: predicts the gap between raw and style-debiased Arena scores (r = −0.900 with actual gap)
- 20+ style columns including aggregate stats, per-question metrics, minimums, and fraction-used metrics
- Note: in the *shipped* 2026-04-18 config, `style_*` and `tone_*` columns are dropped before the KNN predictor (`--drop_style_tone`). They still inform the style_delta feature and the judge-bias analysis, but hurt KNN distance once the PLS hybrid was introduced.

---

## 4. Response-Embedding Feature Set (`sem_f*`)

A sibling feature family to `style_*`/`tone_*`, built from the raw text responses of the 4 custom benchmarks:

1. **Collect:** `embeddings/collect_responses.py` gathers every raw response (model × benchmark × prompt × run) into one parquet file.
2. **Embed:** `embeddings/embed_responses.py` runs bge-small (384-dim sentence-transformer, MPS on Apple Silicon) with atomic-rename checkpointing every 500 pooled responses — load-bearing for longer-context embedders where the O(L²) attention memory blows up at 8192 ctx.
3. **Pool:** each model gets a fingerprint by mean-pooling embeddings within 5 slots — `eq_t1`, `eq_t3`, `logic`, `style`, `writing` — chosen because splits on per-model axes (EQ turn escalation) add signal while splits on per-prompt axes (topic mix) add topic variance that PCA can't separate from voice variance.
4. **Compress:** PCA-32 across pooled fingerprints, yielding `sem_f01 … sem_f32`.

**Result:** −1.29 RMSE over the non-sem baseline. An embedder + PCA + slot sweep (bge-small, bge-base, bge-large, nomic-embed-v1.5, gte-large at 2048 ctx; d ∈ {16, 24, 32, 48, 64}; 5- vs 6- vs 13-slot) confirmed the shipped **bge-small / 5-slot / d=32** config is a global optimum at the current training size. Bigger embedders (bge-large, gte-large) each lose at their own optimum — their extra dimensions encode finer **content** distinctions, not finer **voice** distinctions, which is what the averaging trick extracts.

---

## 5. Data Integration Pipeline

### Scraping Layer (12 grabber scripts, 8 currently active in `scrape.bash`)
The repo contains 12 scraper scripts. The current `scrape.bash` enables 11 of them, while `benchmark_combiner/combine.py` consumes 23 CSV patterns total: 17 public/input families plus 6 custom/derived families.

### Model Name Resolution
The central challenge of combining 16+ benchmark sources is that every source uses different model names. The pipeline handles this with:
1. **Per-source JSON mappings** (20 mapping files) — manual source-to-canonical translation
2. **LLM-assisted mapping** — unmapped models routed through Google Gemini to suggest matches against the known namespace
3. **Bad-model filtering** — known problematic entries excluded

### Combination Layer (`benchmark_combiner/`)
- `combine.py`: multi-encoding CSV reading, full outer join across all sources, conflict detection, duplicate resolution
- `correlations.py`: exploratory analysis — low-variance column filtering, correlation analysis, PCA/t-SNE/UMAP, clustering
- Output: `clean_combined_all_benches.csv` (base, 163 models) and `clean_combined_all_benches_with_sem_v4_d32.csv` (sem-augmented, 123 columns)

---

## 6. Imputation Pipeline

The public benchmark data has ~40% missing values. The custom `ModelBankImputer` fills these gaps with per-cell model selection and low-rank coherence projection.

### ModelBankImputer — Custom Per-Cell Architecture

**Key insight:** Standard imputation assigns one model per column. ModelBankImputer recognizes that the best predictor set varies by *which other benchmarks a model has been tested on*. A model tested on LiveBench + AiderBench but missing GPQA should use different predictors than one tested on MMLU + ARC but missing GPQA.

**Algorithm:**

1. **Candidate ranking:** For each target column, rank predictors by `|correlation| × sqrt(n_common_rows)` with redundancy filtering (skip predictors with |r| > 0.85 to already-selected ones).
2. **Pass 1 (observed-only):** For each missing cell, find the best predictor subset from columns actually observed in that row. Fit BayesianRidge (cached by predictor subset key). LOO-forward selection: predictors are added one at a time and kept only if they lower raw LOO-MSE on the observed rows.
3. **Pass 2 (expansion):** Revisit high-uncertainty cells, allow one confidently-imputed value (sigma/sd < 0.4) as an additional predictor. Accept only if variance improves by >5%. Jacobi-style frozen inputs prevent oscillation.
4. **Coherence projection (λ = 8.0, exp shape):** SVD low-rank projection of the completed matrix. Blend each imputed cell toward the coherent estimate: `x' = (1-w)*x_imputed + w*SVD_x`, where `w = tau/(tau+lambda)` and `tau = cell_variance/column_variance`. High-uncertainty cells pulled toward the SVD reconstruction; confident cells retain their model-bank value. Restores cross-column consistency — individually optimal per-cell predictions can create "Frankenstein" profiles that are accurate per-column but incoherent as a row.
5. **Per-cell uncertainty tracking:** Analytical hat-matrix leave-one-out variance for every imputed cell, carried through the entire pipeline.
6. **Feature extraction from imputation:** 6 SVD latent factors, 6 squared terms, 6 interaction terms (pairwise products of first 4 factors), 3 trajectory features (`mean_delta`, `max_delta`, `n_imputed` across passes). These 21 derived features extend the imputed benchmark matrix into the KNN feature set.

### Why coherence projection matters
Without it, each cell is imputed by its own little regression model. Individually reasonable, but the completed rows don't behave like real models — they combine traits that no actual model exhibits. SVD projection forces the completed matrix toward a low-rank approximation, which is the natural structure of benchmark data (models have a small number of latent capability dimensions). The blending weight ensures confident imputations aren't disturbed while uncertain ones snap to the coherent manifold.

---

## 7. Prediction Pipeline

### Why Adaptive KNN — The Feature Sign Flip Problem

Global linear regression fails because **different features predict Arena score at different score levels**. Feature correlations flip sign around ~1400 ELO:
- `style_list_count`: +0.16 overall, **−0.74** within top 15 models
- `livebench_code_completion`: +0.61 overall, **−0.32** within top 15

A global model averages these out. KNN with local Ridge regression recovers the local relationships.

### Adaptive KNN Algorithm (shipped 2026-04-18)

1. **Feature matrix:** imputed benchmark columns + 21 imputation-derived features (SVD / trajectory) + 32 sem_f* response-embedding dims. The `style_*` and `tone_*` columns are **dropped** pre-KNN (`--drop_style_tone`) — they were hurting once PLS landed. Standardized per fold.
2. **PLS hybrid (fold-internal PLS-3):** Within each fold, a 3-component Partial-Least-Squares regression is fit on the training features against `lmarena_Score`. Its components are appended to both train and test feature matrices, giving the KNN distance a supervised, low-rank "direction of ELO" summary without any of the overfitting that tree-based alternatives suffer at n = 126. Adds −0.97 RMSE on top of sem.
3. **Adaptive neighborhood via sublinear power cutoff:** `max_dist = d_nearest^0.7 × 3.0`. Models with distinctive feature profiles get tight neighborhoods (k ≈ 20, the floor); models in dense regions of feature space get wider neighborhoods (k up to 80). The 0.7 exponent was selected from a sweep of 2,700 configurations as the best tradeoff between LOO and walk-forward accuracy.
4. **Gaussian kernel weighting:** `w = exp(−0.5 * (dist/bw)²)` with bandwidth at the 15th percentile neighbor distance. Closer neighbors get exponentially more influence.
5. **Local Ridge regression with adaptive alpha:** `alpha = max(10, std(neighbor_scores))`. Tight, homogeneous neighborhoods get lower regularization; diverse neighborhoods get higher regularization.
6. **Jackknife variance inflation (bias correction):** Ridge regression centers predictions toward the neighborhood mean. The jackknife estimates the compression factor by leaving each neighbor out, refitting, and measuring how much the model compresses actual variation. Correction factor `b` (clipped to [1.0, 1.5]) re-expands predictions.
7. **Per-model neighborhood diagnostics:** every prediction logs neighborhood size, neighbor names, Ridge coefficients, per-feature contributions (coef × standardized value), jackknife correction factor, top contributing features.
8. **Grouped conformal prediction intervals:** 4 groups by predicted score × missingness level. Empirical quantile calibration per group. ~92% honest coverage.

### Baseline Comparison

| Method | RMSE | R² | Spearman ρ |
|--------|------|-----|-----------|
| Predict mean (dummy) | 56.6 | 0.00 | — |
| Public benchmarks + median impute + Ridge | 30.0 | 0.72 | 0.85 |
| All benchmarks + median impute + Ridge | 26.3 | 0.78 | 0.90 |
| **Full Soothsayer pipeline** | **13.61** | **0.941** | **0.971** |

The full pipeline achieves a 49% reduction over the best simple baseline (26.3 → 13.61).

---

## 8. Experimental Methodology

### Systematic Experimentation
50+ controlled experiments across multiple improvement cycles, each with a fixed baseline and isolated single-variable changes. Categories explored:

- **Imputation:** SVD, specialized, model bank, coherence (λ, shape, rank penalty, capping)
- **Feature engineering:** trajectory, delta head (LEAKED), pairwise rank (LEAKED), LOBO residuals, alias archaeology, sigma² features
- **Feature transforms:** quantile normalize (hurt), reliability-weighted PCA (neutral), orthogonalize to ALT (hurt)
- **Regularization:** EB parent (initial winner, later removed when PLS rendered it over-shrinking), EB provider (neutral), prediction shrinkage (hurt)
- **Semi-supervised:** self-training with pseudo-labels (hurt)
- **Top-tier optimization:** top-tier boost (+gain in earlier regime, removed in cleanup), continuous ELO weighting (hurt), top-50 residual correction (hurt)
- **Model variants:** partial-linear ALT (neutral), target-aware coherence (neutral), pairwise anchor head (hurt)
- **IRT retest:** IRT ALT RMSE 22.71 — dead end at current data size

### Leakage Detection and Correction
Caught and diagnosed 3 major pre-CV leakage bugs:

1. **`style_predicted_delta` leakage** — in-sample ExtraTrees predictions gave r = 1.000; honest OOF r = 0.870. Fixed with `cross_val_predict`.
2. **Pairwise rank feature** — global ALT prediction rank computed before the CV loop leaked full-data ALT signal through feature selection. RMSE dropped 21.48 → 15.57 (artificially). Fixed by removing the feature.
3. **Delta head** — BayesianRidge trained on `TARGET − ALT` gap using all known-target rows before CV. Train r = 0.937 (leaked) vs 0.625 (honest). Confirmed pure leakage when fold-internal computation was +1.90 RMSE *worse*.

**Leakage rule:** any feature computed globally before the CV loop that uses (a) the full-data ALT model or (b) TARGET values will leak. Must be computed fold-internally.

### LOBO Analysis (Leave-One-Benchmark-Out)
For each benchmark column, drop it, re-impute from scratch, re-run the entire pipeline, measure RMSE impact. Key findings:
- Coding benchmarks (LiveBench code generation, typescript, python) and instruction-following (IFBench) are most valuable
- Some benchmarks hurt overall RMSE but help top-tier prediction — tier-dependent value
- Partial correlations are unreliable for imputation pipelines (LOBO is the only honest method)

### Walk-Forward Validation
Two variants:

- **Fast walk-forward** (2026-03-30, historical): fits imputation and PCA on the full pool, walks only the KNN fit → RMSE 14.86.
- **Honest walk-forward** (2026-04-18, current): re-fits ModelBankImputer + PCA-32 + PLS-3 at every step on `[0..i]` / `[0..i-1]`; `predict_adaptive_knn` sees only older models. On the oldest-80% → newest-20% split (n = 23): **RMSE 14.69, R² 0.900, Spearman ρ 0.940**. The +1.21 RMSE gap vs OOF is the honest cost of not having ~20% of the data in imputation-/PCA-/PLS-fitting.

### Judge-Bias Parallel Line of Work
Orthogonal to the main RMSE pipeline, built a judge-preference probe over TrueSkill battles. Finding: Gemini 3.0 Flash and Grok 4 Fast agree on preferred response shape at **r = +1.000 on EQ** (n=4,805 vs 5,868) and **r = +0.998 on Writing** — i.e. LLM judges agree with each other more than they agree with themselves across tasks (r ≈ +0.28 same-judge across-task). Shared preferences: reward em-dashes, first/second-person voice; penalize hedging (t = −17 on EQ), bullet points, bold/numbered formatting. Framing: "shared-bias is structural, not provider-specific." See `docs/JUDGE_FINDINGS.md`.

---

## 9. Post-hoc Analysis Suite

A 15-section automated diagnostic suite (`posthoc_suite.py`) generating cost-vs-performance Pareto frontiers, residual analysis, radar capability profiles, benchmark clustering dendrograms, reasoning/non-reasoning violin comparisons, calibration diagrams, residual bias decomposition by vendor × reasoning, rank-stability intervals, redundancy heatmaps, and per-model capability breakdowns. Some sections emit multiple figure files.

---

## 10. Engineering Quality

### Testing
- **112 current collected test cases** via pytest, covering utility functions, configuration, the TrueSkill engine, ModelBankImputer, joint prediction utilities, zebra puzzle generation, and integration tests
- Tests validate imputer coherence projection, per-cell uncertainty tracking, edge cases (single-column, all-missing, etc.)

### Documentation
- **`ARCHITECTURE.md`** (~380 lines) — full system design, data flow, algorithm descriptions, configuration reference
- **`DATA_DICTIONARY.md`** (~350 lines) — every column definition, file format, data type, naming convention
- **`FINDINGS.md`** (~1,000 lines) — all experiment results, leakage analysis, error decomposition
- **`WALKFORWARD_ANALYSIS.md`** (~330 lines) — temporal CV methodology, per-model failure-mode diagnosis
- **`JUDGE_FINDINGS.md`** (~660 lines) — judge-bias parallel line of work

### Codebase Structure
- ~200 Python files (excluding scratch experiments), ~88k lines total
- 12 automated scraper scripts
- 20 JSON model name mapping files
- Shell orchestration scripts for each pipeline stage (`scrape.bash`, `combine.bash`, `predict.sh`, `run_all_benches.bash`)
- `pyproject.toml` packaging with `pip install -e .` dev mode

---

## 11. Key Results

### Prediction Accuracy (10×5-fold repeated cross-validation)
| Metric | Value |
|--------|-------|
| Overall OOF RMSE | **13.61** (on ~1100–1500 ELO scale) |
| R² | **0.941** |
| Spearman ρ | **0.971** |
| Top-15 RMSE | 14.26 |
| Honest walk-forward RMSE | **14.69** (n = 23, re-fits imputation + PCA-32 + PLS-3 every step) |
| Honest walk-forward R² / ρ | 0.900 / 0.940 |
| Training set | 127 models |

### Technique Contributions (approximate RMSE deltas)
| Technique | Impact |
|-----------|--------|
| ModelBankImputer + coherence (vs. baseline imputer) | −0.72 |
| Sem_f* response-embedding fingerprints (bge-small, 5-slot, PCA-32) | −1.29 |
| PLS hybrid (fold-internal PLS-3) + drop_style_tone | −0.97 |
| Sublinear power cutoff KNN (vs linear cutoff) | −1.78 walk-forward (+0.24 LOO — worth it) |
| Jackknife variance inflation (bias correction) | modest but structural |
| Grouped conformal intervals | — (calibration, not point accuracy) |

### What Makes This Hard
- Only **127 training rows** with **~100 features** and **~40% missing values** in the base matrix
- Nonlinear / tree-based global models catastrophically overfit at this sample size — Ridge-in-neighborhood + PLS wins
- The primary target (Arena ELO) is itself noisy — based on human preference votes with inherent variance
- The vast majority of experiments failed to beat the baseline, demonstrating the difficulty of improving beyond the statistical floor

---

## 12. Technologies Used

**Languages:** Python, Bash

**ML/Statistics:**
- scikit-learn (Ridge, BayesianRidge, PLSRegression, PCA, StandardScaler, cross-validation, NearestNeighbors)
- TrueSkill (Bayesian skill rating for pairwise comparisons)
- SciPy (hierarchical clustering, statistical tests, optimization)
- Statsmodels (LOWESS smoothing, statistical analysis)
- sentence-transformers (bge-small for response embeddings; MPS on Apple Silicon)

**Data:**
- Pandas, NumPy (data manipulation)
- Conformal prediction (uncertainty quantification)
- SVD / low-rank matrix methods (coherence projection, warm-start imputation)

**Visualization:**
- Matplotlib, Seaborn (15-section diagnostic suite)
- adjustText (label placement)

**APIs & Infrastructure:**
- OpenRouter API (LLM inference for benchmarks)
- Google Gemini API (automated model name resolution)
- Shell orchestration (bash scripts for pipeline stages)

**Engineering:**
- pytest (112 current collected test cases)
- pyproject.toml packaging
- Git version control
- Remote compute orchestration (48-core Ubuntu tower for heavy pipeline runs via rsync + SSH; Tailscale hostnames)

---

## 13. Skills Demonstrated

### Machine Learning & Statistics
- Adaptive KNN with kernel weighting and sublinear distance cutoffs
- Local Ridge regression with per-neighborhood adaptive regularization
- Partial Least Squares as a fold-internal supervised-feature extractor (PLS hybrid)
- Matrix imputation (custom per-cell architecture, SVD coherence projection, LOO-forward predictor selection)
- Response-embedding fingerprints (sentence transformers → mean-pool → PCA) as a sibling feature family to explicit style metrics
- Jackknife variance inflation (bias correction for shrinkage estimators)
- Conformal prediction intervals (calibrated uncertainty quantification, grouped calibration)
- TrueSkill rating systems (information-gain match selection, paired evaluation)
- Cross-validation methodology (repeated K-fold, fully-honest temporal walk-forward with per-step refitting, LOO)
- Leakage detection and prevention (caught 3 major pre-CV leakage bugs through systematic diagnosis)
- Experimental design (50+ controlled experiments, ablation studies, error decomposition)
- Dimensionality reduction (PCA, Factor Analysis, SVD)

### Software Engineering
- End-to-end system design (scraping → combination → embeddings → imputation → prediction → analysis)
- Python package architecture (abstract base classes, shared engines, CLI orchestrator)
- Systematic refactoring (9-phase modernization of a growing codebase) and periodic dead-code sweeps (most recent: 14 CLI flags and ~100 lines of imputer code retired after the 2026-04-18 ablation)
- Testing (112 current collected test cases, edge cases, integration tests)
- Parallel programming (ThreadPoolExecutor, Jacobi-style updates)
- Rate-limit-aware concurrent API systems (per-model semaphores, adaptive 429 backoff, resumable ETL)
- API integration (OpenRouter, Gemini — with retry logic, rate limiting, error handling)
- Data pipeline design (idempotent, resumable, cached; embedding chain only re-embeds new responses)
- Remote compute orchestration (rsync-based code sync, parallel experiment scheduling on multi-core server)

### Data Engineering
- Multi-source data integration (16+ benchmark sources with inconsistent schemas)
- Entity resolution (model name mapping with LLM-assisted fuzzy matching)
- Missing data handling (~40% missingness across a heterogeneous matrix; custom benchmarks providing near-complete coverage)
- Per-cell data quality tracking through the full pipeline

### Research & Analysis
- Scientific methodology (controlled experiments, baselines, ablation studies)
- Error analysis (provider-level, model-type, tier-dependent diagnostic breakdowns)
- Custom algorithm design (per-cell imputer with coherence projection, PLS hybrid KNN)
- Domain expertise in LLM evaluation (Arena scoring mechanics, style bias, benchmark saturation)
- Judge-bias analysis as a parallel RAI-adjacent line of work

### Communication
- Technical writing (ARCHITECTURE.md, DATA_DICTIONARY.md, FINDINGS.md, WALKFORWARD_ANALYSIS.md, JUDGE_FINDINGS.md)
- Experiment logging (structured cycle-based documentation with clear verdicts)
- Visualization (15-section automated diagnostic suite)
