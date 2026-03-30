# Project Soothsayer — Architecture

## System Design Overview

This pipeline aggregates benchmark data from 20+ sources, performs model name unification, and uses adaptive KNN + kernel Ridge regression to predict lmarena scores (style-controlled Chatbot Arena ELO, R²=0.924) for large language models.

Missing benchmark values are filled by the **ModelBankImputer** (`column_imputer.py`), which selects the best predictor subset for each individual missing cell based on what that row actually has observed, with per-cell uncertainty tracking and low-rank coherence projection.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA COLLECTION LAYER                              │
│  scrape.bash → 12 grabber scripts → Raw benchmark CSVs                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA COMBINATION LAYER                             │
│  combine.py                                                                  │
│  • Multi-source benchmark aggregation (20+ sources)                         │
│  • LLM-assisted model name mapping (Gemini API)                             │
│  • Per-source mapping JSONs → OpenBench namespace unification               │
│  Output: benchmarks/combined_all_benches.csv                                │
│  → correlations.py cleans → benchmarks/clean_combined_all_benches.csv       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXPLORATORY ANALYSIS LAYER (Optional)                     │
│  correlations.py                                                             │
│  • Correlation analysis with low-variance filtering                          │
│  • PCA, t-SNE, UMAP dimensionality reduction                                │
│  • K-means clustering with silhouette scoring                               │
│  Output: Visualizations + analysis reports                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      IMPUTATION & PREDICTION LAYER                           │
│  predict.py (orchestrator)                                                   │
│  └── column_imputer.py (ModelBankImputer)                                   │
│                                                                              │
│  Key Components:                                                             │
│  • Imputation: ModelBankImputer (per-cell predictor selection + σ² tracking)│
│  • Prediction: Adaptive KNN + kernel Ridge + jackknife VI (--knn_predict)   │
│                                                                              │
│  Output: imputed_full.csv, predictions_best_model.csv, quality reports      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              INPUT FILES                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  External Benchmark Sources (scraped):                                       │
│  ├── lmarena_*.csv              (Style-controlled Arena ELO — prediction target) │
│  ├── lmsys_*.csv                (Raw Arena ELO — excluded as leakage)       │
│  ├── livebench_*.csv            (LiveBench multi-category scores)           │
│  ├── openbench_*.csv            (OpenBench suite — canonical namespace)     │
│  ├── artificialanalysis_*.csv   (AA quality + pricing metrics)              │
│  ├── aa_gdpval_*.csv            (AA evaluation benchmarks)                  │
│  ├── aa_omniscience_*.csv       (AA omniscience benchmarks)                 │
│  ├── aa_critpt_*.csv            (AA critical point benchmarks)              │
│  ├── aiderbench_*.csv           (Aider coding benchmark)                    │
│  ├── arc_*.csv                  (ARC-AGI reasoning)                         │
│  ├── contextarena_*.csv         (Long-context retrieval)                    │
│  ├── EQ-Bench_combined_*.csv    (EQ-Bench EQ + creative writing)           │
│  ├── simplebench_*.csv          (Commonsense trick questions)               │
│  ├── lechmazur_combined_*.csv   (Confabulations + generalization)           │
│  ├── weirdml_*.csv              (Unusual ML tasks)                          │
│  ├── yupp_text_coding_scores_*.csv (Yupp VIBEScore)                        │
│  └── UGI_Leaderboard_*.csv      (Writing quality)                          │
│                                                                              │
│  Soothsayer Benchmark Results:                                               │
│  ├── eq_*.csv                   (Soothsayer EQ TrueSkill ratings)           │
│  ├── writing_*.csv              (Soothsayer Writing TrueSkill ratings)      │
│  ├── writing_direct_*.csv       (Soothsayer Writing rubric scores, optional)│
│  ├── logic_*.csv                (Soothsayer Logic scores)                   │
│  ├── style_*.csv                (Soothsayer Style metrics)                  │
│  └── tone_*.csv                 (Soothsayer Tone scores)                    │
│                                                                              │
│  Mapping Files (per-source, in benchmark_combiner/mappings/):               │
│  ├── lmsys_to_openbench.json                                                │
│  ├── livebench_to_openbench.json                                            │
│  ├── aiderbench_to_openbench.json                                           │
│  ├── eqbench_to_openbench.json                                              │
│  ├── arc_to_openbench.json                                                  │
│  ├── aa_to_openbench.json                                                   │
│  ├── aa_evals_to_openbench.json                                             │
│  ├── weirdml_to_openbench.json                                              │
│  ├── yupp_to_openbench.json                                                 │
│  ├── ugi_to_openbench.json                                                  │
│  └── ... (20 mapping files total)                                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           INTERMEDIATE FILES                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  combined_all_benches.csv                                                    │
│  └── All benchmarks merged on Unified_Name                                   │
│      (~290 columns, ~1170 models)                                            │
│                                                                              │
│  clean_combined_all_benches.csv                                              │
│  └── Cleaned version (models with too few scores dropped,                   │
│      low-variance columns removed)                                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                             OUTPUT FILES                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Core Outputs:                                                               │
│  ├── imputed_full.csv                 (Complete imputed matrix)             │
│  ├── predictions_best_model.csv       (Final predictions + intervals)       │
│  ├── oof_predictions.csv              (Out-of-fold predictions)             │
│  ├── feature_ranking_gain.csv         (Feature importance rankings)         │
│  └── feature_matrix_used.csv          (Feature matrix fed to models)        │
│                                                                              │
│  Quality Reports:                                                            │
│  ├── imputation_quality_per_cell.csv  (Per-cell quality metrics)            │
│  ├── imputation_quality_per_column.csv (Per-column quality metrics)         │
│  ├── imputation_quality_by_extrapolation_bin.csv                            │
│  └── run_config.json                  (Configuration used)                  │
│                                                                              │
│  Model Diagnostics:                                                          │
│  ├── best_model_variance_contributions.csv                                  │
│  ├── conformal_diagnostics.csv                                              │
│  ├── model_eval_rmse.csv              (Per-model CV RMSE comparison)        │
│  └── column_dependency_graph.json     (Column dependency structure)          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

### combine.py — Benchmark Data Aggregator

**Purpose:** Aggregate 20+ benchmark sources into a unified namespace, handling model name variations across sources.

**Key Responsibilities:**
1. **Multi-encoding CSV reading** — Handle various encodings (UTF-8, Latin-1, CP1252)
2. **Model name mapping** — Per-source mapping JSONs map each benchmark's model names to OpenBench canonical names
3. **LLM-assisted mapping** — Uses Gemini API to suggest mappings for unmapped models
4. **Outer join aggregation** — Merge all sources on Unified_Name column

**Key Functions:**
- `load_existing_mappings()` — Load per-source JSON mapping dictionaries
- `get_llm_mapping_suggestions()` — Query Gemini for mapping suggestions
- `combine_benchmarks_with_auto_mapping()` — Main orchestration
- `find_mapping_issues()` — Detect duplicate/conflict mappings

---

### correlations.py — Exploratory Data Analysis

**Purpose:** Perform comprehensive exploratory analysis on the combined benchmark data.

**Key Responsibilities:**
1. **Data cleaning** — Remove low-variance columns, handle missing values
2. **Correlation analysis** — Compute pairwise correlations with significance testing
3. **Dimensionality reduction** — PCA, t-SNE, UMAP for visualization
4. **Clustering** — K-means with silhouette score optimization
5. **Visualization** — Scatter plots, radar charts, heatmaps

**Configuration Constants:**
- `MISSING_THRESHOLD = 0.508` — Max fraction of missing values per column
- `USE_UMAP = False` — Toggle UMAP (requires umap-learn)

**Key Functions:**
- `prepare_data_for_analysis()` — Missingness filtering and imputation
- `analyze_correlations()` — Pairwise correlation with variance filtering
- `perform_pca()` — PCA with explained variance analysis
- `perform_clustering()` — K-means with silhouette scoring

---

### predict.py — Prediction Pipeline Orchestrator

**Purpose:** Impute missing benchmark scores and predict lmarena_Score (style-controlled Arena ELO) from the combined benchmark matrix.

**Key Responsibilities:**
1. **Imputation orchestration** — Interface with ModelBankImputer for filling missing benchmark scores
2. **KNN prediction** — Adaptive neighbor selection, kernel weighting, local Ridge regression, jackknife bias correction
3. **Conformal prediction intervals** — Calibrated uncertainty quantification

**Key Functions:**
- `predict_adaptive_knn()` — Single-point prediction via adaptive KNN + kernel Ridge + jackknife VI
- `fit_and_predict_knn()` — Full KNN pipeline with CV evaluation and final predictions
- `compute_uncertainty_calibration_factor()` — Scale uncertainties for proper coverage
- `calibrate_prediction_intervals()` — Conformal calibration

---

## Key Algorithms

### 1. Per-Cell Model-Bank Imputation (ModelBankImputer)

The default imputation architecture. Instead of one model per column, it builds a **model bank** of cached models keyed by `(target_col, frozenset(predictor_subset))` and selects the best model for each individual missing cell.

**Algorithm:**

1. **Phase 0 — Preprocessing:** Log transforms, column classification, correlation matrices, SVD factor extraction (for row features only, no warm-start filling)
2. **Phase 1 — Candidate Rankings:** For each target column, rank predictors by `|corr| × sqrt(n_common_rows)` with redundancy filtering (skip predictor if `|r| > 0.85` with already-selected ones)
3. **Phase 2 — Pass 1 (observed-only):** For each missing cell, select the best predictor subset from columns observed in that row. Fit BayesianRidge (cached by predictor subset). Single-proxy challenger: if 1-predictor model has lower LOO σ² than multi-predictor, use the simpler model. Adaptive k: `k_max` scales with training support (k=1 for n<15, up to k=8 for n>80)
4. **Phase 3 — Pass 2 (expansion):** Revisit cells with high σ². Allow at most one confidently-imputed value (σ/sd < threshold) as an additional predictor. Accept only if σ²_new < 0.95 × σ²_old. Jacobi-style frozen inputs prevent oscillation
5. **Phase 4 — Fallback:** Remaining cells filled with column median
6. **Phase 4b — Coherence Projection:** SVD of the completed matrix at the same rank used for factor extraction. Blend each imputed cell toward the low-rank estimate with shrinkage weight `w = τ/(τ + λ)` where `τ = σ²_cell/sd²_col`. High-uncertainty cells are pulled more toward the coherent SVD; confident cells keep their model-bank value. Default `λ = 1.0`
7. **Phase 5 — Post-processing:** Inverse log transforms, σ²-based trajectory features, representative models for API compatibility

**Coherence Projection Rationale:** Per-cell imputation produces individually accurate values but they may be inconsistent across columns within a row (a "Frankenstein" profile). The low-rank projection restores cross-column coherence by blending toward the SVD estimate, weighted by uncertainty.

### 2. Conformal Prediction Intervals

For uncertainty quantification:
1. Compute OOF absolute residuals
2. Set half-width as `percentile(residuals, quantile)` (configurable)
3. Scale half-width by per-cell uncertainty signals (kNN distance, model std, range violation)

### 3. Adaptive KNN + Kernel Ridge + Jackknife VI (--knn_predict)

The primary prediction algorithm (R²=0.924 for lmarena_Score):

1. **Standardize** all ~105 features (benchmarks + SVD factors + trajectory)
2. **Adaptive k via sublinear power cutoff:** `max_dist = d_nearest^0.7 × 3.0`. This gives tighter neighborhoods in dense regions (top models, k≈20-30) while keeping adequate coverage in sparse regions (bottom models). The sublinear scaling naturally adapts: effective multiplier ≈1.5× for dense clusters, ≈1.35× for sparse ones. Min 20, max 80.
3. **Gaussian kernel weights:** `w = exp(-0.5 × (dist/bw)²)` where bandwidth = distance to 15th-percentile neighbor. Closer neighbors weighted more heavily.
4. **Ridge regression:** `alpha = max(10, std(neighbor_scores))`. Adaptive regularization — tight neighborhoods get lower alpha (more flexible).
5. **Jackknife variance inflation:** Leave each of the k neighbors out, refit Ridge on k-1, predict the left-out one. Estimate compression slope `b = cov(actual, jackknife_pred) / var(jackknife_pred)`. Clip to [1.0, 1.5]. Correct: `prediction = neighborhood_mean + b × (raw_prediction - neighborhood_mean)`. This reverses Ridge's centering bias.

**Why it works:** Different features predict lmarena at different score levels — feature correlations flip sign around ~1400 (e.g., list count is positive below, negative above). The sublinear power cutoff keeps each neighborhood within a consistent score regime, so the local Ridge learns the correct local relationship. A linear cutoff (d×2.0) gives k≈58, spanning both sides of the flip; the power cutoff (d^0.7×3) gives k≈25-35, staying on one side.

---

## Configuration Parameters

### predict.py CLI Arguments (key flags)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv_path` | `../benchmark_combiner/benchmarks/clean_combined_all_benches.csv` | Input data file |
| `--output_root` | `analysis_output` | Output directory |
| `--imputer_type` | `model_bank` | Imputer backend: `model_bank` or `specialized` |
| `--knn_predict` | on | Use adaptive KNN prediction (always on) |
| `--knn_power_alpha` | 0.7 | Exponent for sublinear distance cutoff (1.0 = linear) |
| `--knn_power_c` | 3.0 | Coefficient for distance cutoff: max_dist = d0^alpha × C |
| `--knn_max_k` | 80 | Maximum neighbors for KNN |
| `--knn_min_k` | 20 | Minimum neighbors for KNN |
| `--knn_bw_pct` | 0.15 | Kernel bandwidth at this percentile of neighbor distances |
| `--coherence_lambda` | 1.0 | ModelBankImputer: coherence projection shrinkage (0 = disable) |
| `--coherence_shape` | `exp` | Coherence shrinkage shape: linear, exp, squared, etc. |
| `--eb_parent` | off | Enable empirical-Bayes parent shrinkage |
| `--cv_repeats_outer` | 10 | CV repeats for OOF evaluation |
| `--margin` | 20.0 | Margin for top_by_margin_prob column |
| `--max_workers` | 0 (auto) | Parallel workers for imputation |

Run `python3 predict.py --help` for the full list.

### ModelBankImputer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.4 | Normalized σ/sd threshold for confident imputations in pass 2 |
| `coherence_lambda` | 1.0 | Shrinkage strength for low-rank coherence projection (0 = disable) |
| `redundancy_threshold` | 0.85 | Max \|r\| between selected predictors before filtering |
| `min_support` | 10 | Minimum training rows for a model to be fitted |

---

## Performance Considerations

### Parallelism

The pipeline uses a mixed parallelism strategy:

1. **Thread-based parallelism** (via `joblib` with `backend='threading'`):
   - Column model fitting (parallel across columns in same tier)
   - Column imputation (parallel predictions within a tier)

2. **Jacobi-style updates**:
   - All columns in a tier are updated from a snapshot
   - Prevents order-dependent results within a tier
   - Enables safe parallelization

### Caching

Imputation results are cached to avoid redundant computation:
- Cache key derived from data hash + config hash
- Stored as pickle files in `.imputation_cache/`
- Invalidated on data or config changes

---

## Error Handling

### Common Failure Modes

1. **Mapping conflicts in combiner**: Duplicate unified names
   - Check `find_mapping_issues()` output
   - Update mapping JSON files in `benchmark_combiner/mappings/`

2. **Imputation cache stale**: Data changed but cache wasn't invalidated
   - Delete `.imputation_cache/` and re-run
