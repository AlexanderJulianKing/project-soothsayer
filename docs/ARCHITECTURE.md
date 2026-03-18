# Project Soothsayer — Architecture

## System Design Overview

This pipeline aggregates benchmark data from 20+ sources, performs model name unification, and uses advanced imputation techniques to predict Chatbot Arena ELO ratings for large language models.

The system supports two imputation architectures: the **per-cell model-bank imputer** (`ModelBankImputer`, default) that selects the best predictor subset for each individual missing cell based on what that row actually has observed, with per-cell uncertainty tracking and low-rank coherence projection; and the legacy **per-column specialized imputer** (`SpecializedColumnImputer`) that auto-classifies columns by type and assigns one model per column.

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
│  Output: benchmarks/clean_combined_all_benches.csv                          │
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
│  └── column_imputer.py (ModelBankImputer + SpecializedColumnImputer)        │
│                                                                              │
│  Key Components:                                                             │
│  • Column type classification (categorical, linear, nonlinear,              │
│    extrapolation_prone) + distribution tags (bounded, floor_inflated)       │
│  • Two imputer backends: ModelBankImputer (default), SpecializedColumnImputer│
│  • ModelBankImputer: per-cell predictor selection + σ² tracking             │
│  • Low-rank coherence projection (ModelBankImputer)                         │
│  • ALT target imputation via OOF stacking                                   │
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
│  ├── lmsys_*.csv                (Chatbot Arena ELO ratings)                 │
│  ├── lmarena_*.csv              (Length-adjusted Arena scores)               │
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
│  ├── writing_direct_*.csv       (Soothsayer Writing rubric scores)          │
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
│      (~65 columns, ~160 models)                                              │
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
│  ├── alt_model_variance_contributions.csv                                   │
│  ├── conformal_diagnostics.csv                                              │
│  ├── model_eval_rmse.csv              (Per-model CV RMSE comparison)        │
│  └── dependency_graph.json            (Column dependency structure)          │
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

**Purpose:** Impute missing benchmark scores and predict Chatbot Arena ELO from the combined benchmark matrix.

**Key Responsibilities:**
1. **Feature selection** — Tree-based ranking (LightGBM/XGBoost) with 1-SE rule
2. **Collinearity pruning** — Remove redundant features via greedy de-correlation
3. **Imputation orchestration** — Interface with ModelBankImputer (default) or SpecializedColumnImputer
4. **ALT target handling** — Out-of-fold stacking for alternative targets (lmarena)
5. **Model comparison** — Compare BayesianRidge and ARDRegression via CV
6. **Polynomial interactions** — Configurable feature interactions with limit
7. **Conformal prediction intervals** — Calibrated uncertainty quantification
8. **Variance contribution analysis** — Per-feature importance via variance decomposition

**Key Functions:**
- `rank_features_tree()` — LightGBM/XGBoost gain-based ranking
- `select_features_tree()` — Full selection with pruning
- `impute_alt_for_all()` — ALT imputation via OOF stacking
- `cross_val_rmse_with_alt()` — CV with inner-loop ALT imputation
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

### 2. Per-Column Specialized Imputation (SpecializedColumnImputer)

Legacy imputer selected via `--imputer_type specialized`. Uses **per-column type classification** to assign specialized models.

**Column Type Classification:**
| Type | Criteria | Base Model |
|------|----------|------------|
| CATEGORICAL | Boolean, string, or low-cardinality integer (≤10 unique) | LogisticRegression (binary) or RandomForestClassifier (multi-class) |
| LINEAR | High Pearson correlation with predictors | BayesianRidge |
| NONLINEAR | Spearman >> Pearson (nonlinear monotonic relationships) | GP with Matérn kernel |
| EXTRAPOLATION_PRONE | High missingness + low correlation | GP with Linear+Matérn kernel |
| GP_LINEAR_MATERN | Default fallback | GP with Linear+Matérn kernel |

**Distribution Tags (applied independently, modify base model):**
| Tag | Criteria | Effect |
|-----|----------|--------|
| BOUNDED | Values in [0,1] or [0,100] | LINEAR columns get BoundedLinkModel wrapper (logit transform) |
| FLOOR_INFLATED | Bimodal floor cluster detected | Overrides base model with HurdleModel (LogisticRegression gate + BayesianRidge value) |

**Dependency-Aware Ordering:**
Columns are ordered by imputation difficulty (easy columns first):
```
difficulty = missingness_pct × (1 - max_correlation_with_observed)
```
Columns are grouped into tiers for iterative refinement.

**CorrelationWeightedImputer for Missing Predictors:**
Instead of filling missing predictor values with median (causing regression to mean), the imputer:
1. Computes the percentile of each observed value in the row
2. Weights those percentiles by R² correlation with the missing column
3. Fills missing values at the weighted-average percentile

This preserves the "performance tier" of a model across benchmarks.

### 3. Conformal Prediction Intervals

For uncertainty quantification:
1. Compute OOF absolute residuals
2. Set half-width as `percentile(residuals, quantile)` (configurable)
3. Scale half-width by per-cell uncertainty signals (kNN distance, model std, range violation)

### 4. 1-SE Rule for Feature Count

When selecting the number of features `k`:
1. Compute CV scores for `k = 1, 2, ..., max_features`
2. Find `k*` with minimum mean CV error
3. Select smallest `k` where `mean_error(k) <= mean_error(k*) + SE(k*)`

This gives a parsimonious model within one standard error of optimal.

### 5. ALT Target OOF Stacking

For alternative targets (e.g., lmarena) that have missing values:
1. **Problem**: Can't use imputed values as features (would leak information)
2. **Solution**: Out-of-fold predictions — split data into K folds, train on K-1, predict held-out fold, use OOF predictions as feature

---

## Configuration Parameters

### predict.py CLI Arguments (key flags)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv_path` | `../benchmark_combiner/benchmarks/clean_combined_all_benches.csv` | Input data file |
| `--output_root` | `analysis_output` | Output directory |
| `--imputer_type` | `model_bank` | Imputer backend: `model_bank` or `specialized` |
| `--passes` | 14 | Imputation passes (SpecializedColumnImputer) |
| `--alpha` | 0.9361 | Regularization parameter |
| `--coherence_lambda` | 1.0 | ModelBankImputer: coherence projection shrinkage (0 = disable) |
| `--coherence_shape` | `exp` | Coherence shrinkage shape: linear, exp, squared, etc. |
| `--confidence_threshold` | 0.4 | ModelBankImputer: σ/sd threshold for pass 2 expansion |
| `--eb_parent` | off | Enable empirical-Bayes parent shrinkage |
| `--poly_interactions` | off | Enable polynomial feature interactions |
| `--poly_limit` | 6 | Max polynomial interaction terms |
| `--top_tier_boost` | 1 (no boost) | Duplicate top-tier training rows N times |
| `--top_tier_threshold` | 1450 | ELO threshold for top-tier boost |
| `--style_only_final` | off | Restrict lmarena interactions to style columns only |
| `--margin` | 20.0 | Margin for top_by_margin_prob column |
| `--gp_selector_k_max` | 28 | Feature cap for GP models (mRMR selection) |
| `--selector_cv` | 5 | Feature selection CV folds |
| `--outer_cv` | None (uses `selector_cv`) | Outer evaluation CV folds |
| `--feature_selector` | `lgbm` | Feature selector: `none`, `lgbm`, `xgb` |
| `--max_workers` | 0 (auto) | Parallel workers for imputation |
| `--exclude_models` | (none) | Comma-separated model names to exclude |

Run `python3 predict.py --help` for the full list of ~80 arguments.

### ModelBankImputer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.4 | Normalized σ/sd threshold for confident imputations in pass 2 |
| `coherence_lambda` | 1.0 | Shrinkage strength for low-rank coherence projection (0 = disable) |
| `redundancy_threshold` | 0.85 | Max \|r\| between selected predictors before filtering |
| `min_support` | 10 | Minimum training rows for a model to be fitted |

### SpecializedColumnImputer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `passes` | 14 | Number of imputation passes |
| `alpha` | 0.1 | Regularization strength |
| `selector_tau` | 0.8 | De-correlation threshold for predictor selection |
| `selector_k_max` | 30 | Max predictors per column |
| `gp_selector_k_max` | 10 | Feature cap for GP models (mRMR) |
| `categorical_threshold` | 10 | Max distinct values for auto-categorical |
| `calibrate_tolerances` | False | Enable per-column tolerance calibration |
| `calibration_target_rmse_ratio` | 0.5 | Target RMSE/std ratio for calibration |
| `recalibrate_every_n_passes` | 0 | Recalibrate every N passes (0 = only at start) |

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

1. **Feature selection degeneracy**: No valid predictors found
   - Lower `selector_tau` or increase `selector_k_max`

2. **GP fitting issues**: Convergence failures or memory issues
   - Reduce `gp_selector_k_max` to limit features

3. **Mapping conflicts in combiner**: Duplicate unified names
   - Check `find_mapping_issues()` output
   - Update mapping JSON files in `benchmark_combiner/mappings/`
