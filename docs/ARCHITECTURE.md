# LLM Benchmarks Pipeline - Architecture

## System Design Overview

This pipeline aggregates benchmark data from 13+ sources, performs statistical transformations, and uses advanced imputation techniques to predict missing benchmark scores (particularly LMSYS/Arena ELO ratings) for large language models.

The system is designed around a **per-column specialized imputation** architecture that auto-classifies columns by type (categorical, linear, nonlinear, bounded) and assigns appropriate models with native uncertainty quantification.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA COLLECTION LAYER                              │
│  scrape.bash → 10+ grabber scripts → Raw benchmark CSVs                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA COMBINATION LAYER                             │
│  combine.py                                                                  │
│  • Multi-source benchmark aggregation                                        │
│  • LLM-assisted model name mapping (Gemini API)                             │
│  • OpenBench namespace unification                                           │
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
│  • Radar charts and cluster profiling                                        │
│  Output: Visualizations + analysis reports                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      IMPUTATION & PREDICTION LAYER                           │
│  predict.py (orchestrator)                                                   │
│  └── column_imputer.py (core imputer with per-column models)                │
│                                                                              │
│  Key Components:                                                             │
│  • Column type classification (categorical, linear, nonlinear, bounded)     │
│  • Specialized models per type (BayesianRidge, GP, LogisticRegression)      │
│  • Dependency-aware ordering (easy columns first, organized into tiers)     │
│  • Native model uncertainty (no learned gate)                               │
│  • mRMR feature selection for GP models (Spearman correlation)              │
│  • Per-column tolerance calibration via masked evaluation                   │
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
│  Benchmark Sources (13+):                                                    │
│  ├── livebench.csv           (LiveBench scores)                             │
│  ├── open_llm_leaderboard.csv (HuggingFace Open LLM Leaderboard)           │
│  ├── lmsys.csv               (LMSYS/Chatbot Arena ELO ratings)              │
│  ├── openbench.csv           (OpenBench comprehensive suite)                │
│  ├── bigcode.csv             (BigCode models leaderboard)                   │
│  ├── aider.csv               (Aider coding benchmark)                       │
│  ├── openrouter.csv          (OpenRouter performance data)                  │
│  ├── wildbench.csv           (WildBench evaluation)                         │
│  ├── artificial_analysis.csv (Artificial Analysis metrics)                  │
│  ├── tab_bench.csv           (Tabular benchmark)                            │
│  └── ...                                                                     │
│                                                                              │
│  Mapping Files:                                                              │
│  ├── name_to_openbench.json           (Source → OpenBench name)             │
│  ├── openbench_to_unified.json        (OpenBench → Unified name)            │
│  └── bad_models.json                  (Models to exclude)                   │
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
│      (~100-200 columns, ~500-1000 models)                                    │
│                                                                              │
│  clean_combined_all_benches_transformed.csv                                  │
│  └── Transformed version with normalized distributions                       │
│      (Yeo-Johnson/Log1p applied, scaled 0-100)                              │
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
│  └── feature_ranking_gain.csv         (Feature importance rankings)         │
│                                                                              │
│  Quality Reports:                                                            │
│  ├── imputation_quality_summary.csv   (Per-column quality metrics)          │
│  ├── imputation_quality_by_model.csv  (Per-model quality metrics)           │
│  └── imputation_config.json           (Configuration used)                  │
│                                                                              │
│  Dependency Analysis:                                                        │
│  ├── column_dependency_matrix.csv     (Which columns predict which)         │
│  └── column_dependency_graph.json     (Graph structure for viz)             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

### combine.py - Benchmark Data Aggregator

**Purpose:** Aggregate 13+ benchmark sources into a unified namespace, handling model name variations across sources.

**Key Responsibilities:**
1. **Multi-encoding CSV reading** - Handle various encodings (UTF-8, Latin-1, CP1252)
2. **Model name mapping** - Three-tier mapping system:
   - Source name → OpenBench name (per-source mapping)
   - OpenBench name → Unified name (canonical form)
   - Bad model filtering (exclude known problematic entries)
3. **LLM-assisted mapping** - Uses Gemini API to suggest mappings for unmapped models
4. **Outer join aggregation** - Merge all sources on Unified_Name column

**Key Functions:**
- `load_existing_mappings()` - Load JSON mapping dictionaries
- `get_llm_mapping_suggestions()` - Query Gemini for mapping suggestions
- `combine_benchmarks_with_auto_mapping()` - Main orchestration
- `find_mapping_issues()` - Detect duplicate/conflict mappings

---

### correlations.py - Exploratory Data Analysis

**Purpose:** Perform comprehensive exploratory analysis on the combined benchmark data.

**Key Responsibilities:**
1. **Data cleaning** - Remove low-variance columns, handle missing values
2. **Correlation analysis** - Compute pairwise correlations with significance testing
3. **Dimensionality reduction** - PCA, t-SNE, UMAP for visualization
4. **Clustering** - K-means with silhouette score optimization
5. **Visualization** - Scatter plots, radar charts, heatmaps

**Configuration Constants:**
- `MISSING_THRESHOLD = 0.5` - Max fraction of missing values per column
- `USE_UMAP = False` - Toggle UMAP (requires umap-learn)

**Key Functions:**
- `prepare_data_for_analysis()` - Missingness filtering and imputation
- `analyze_correlations()` - Pairwise correlation with variance filtering
- `perform_pca()` - PCA with explained variance analysis
- `perform_clustering()` - K-means with silhouette scoring

---

### predict.py - Prediction Pipeline Orchestrator

**Purpose:** End-to-end pipeline for predicting LMSYS/Arena ELO scores using imputed benchmark data.

**Key Responsibilities:**
1. **Feature selection** - Tree-based ranking (LightGBM/XGBoost) with 1-SE rule
2. **GP-specific feature selection** - mRMR with Spearman correlation for GP models (v7.2)
3. **Collinearity pruning** - Remove redundant features via R² threshold
4. **Imputation orchestration** - Interface with SpecializedColumnImputer
5. **Per-column tolerance calibration** - Calibrate uncertainty thresholds via masked evaluation (v7.2)
6. **Periodic recalibration** - Optionally recalibrate tolerances every N passes (v7.2)
7. **ALT target handling** - Out-of-fold stacking for alternative targets
8. **Model comparison** - Compare BayesianRidge and ARDRegression (v7)
9. **Calibrated uncertainty scaling** - Scale model std for proper 95% coverage (v7.1)
10. **Variance contribution analysis** - Compute feature importance via variance decomposition (v7.1)
11. **Dependency graph** - Generate filtered column dependency structure (≥1% variance contribution)

**Key Components:**

| Component | Description |
|-----------|-------------|
| Feature Selection | Tree-based ranking, 1-SE rule, collinearity pruning, mRMR for GP |
| Tolerance Calibration | Per-column calibration via masked evaluation |
| ALT Target Handling | OOF stacking to prevent leakage |
| Model Specifications | ModelSpec dataclass, BayesianRidge + ARDRegression |
| Cross-Validation | CV with/without ALT imputation |
| Variance Analysis | Per-feature variance contribution computation |
| Main Pipeline | Full orchestration |

**Key Functions:**
- `rank_features_tree()` - LightGBM/XGBoost gain-based ranking
- `select_features_tree()` - Full selection with pruning
- `impute_alt_for_all()` - ALT imputation via OOF stacking
- `cross_val_rmse_with_alt()` - CV with inner-loop ALT imputation
- `compute_uncertainty_calibration_factor()` - Scale uncertainties for proper coverage (v7.1)
- `calibrate_prediction_intervals()` - Conformal calibration

**v7.2 New Outputs:**
- `best_model_variance_contributions.csv` - Per-feature variance contributions for best model
- `alt_model_variance_contributions.csv` - Per-feature variance contributions for ALT model


---

## Key Algorithms

### 1. Per-Column Specialized Imputation

The core architecture uses **per-column type classification** to assign specialized models:

**Column Type Classification:**
| Type | Criteria | Model |
|------|----------|-------|
| CATEGORICAL | Boolean, string, or low-cardinality integer (≤10 unique) | LogisticRegression or RandomForestClassifier |
| LINEAR | High Pearson correlation with predictors | BayesianRidge |
| NONLINEAR | Spearman >> Pearson (nonlinear monotonic relationships) | GP with Matérn kernel |
| BOUNDED | Values in [0,1] or [0,100] | GP with Linear+Matérn kernel |
| EXTRAPOLATION_PRONE | High missingness + low correlation | GP with Linear+Matérn kernel |
| GP_LINEAR_MATERN | Default fallback | GP with Linear+Matérn kernel |

**Dependency-Aware Ordering:**
Columns are ordered by imputation difficulty (easy columns first):
```
difficulty = missingness_pct × (1 - max_correlation_with_observed)
```
Columns are grouped into tiers (easy, medium, hard) for iterative refinement.

**Tolerance-Based Acceptance:**
Instead of a learned gate, predictions are accepted based on uncertainty thresholds:
1. Each model provides native uncertainty estimates (GP std, Bayesian Ridge std, etc.)
2. Predictions are accepted if uncertainty ≤ tolerance threshold
3. If nothing passes tolerance, the single best prediction (lowest uncertainty) is used
4. Tolerances are relaxed if no progress is made in a pass

**CorrelationWeightedImputer for Missing Predictors:**
Instead of filling missing predictor values with median (causing regression to mean), the imputer:
1. Computes the percentile of each observed value in the row
2. Weights those percentiles by R² correlation with the missing column
3. Fills missing values at the weighted-average percentile
This preserves the "performance tier" of a model across benchmarks.

### 2. Per-Column Feature Selection

For each column to impute, select predictors using:

1. **Greedy de-correlation**: Start with best single predictor, iteratively add predictors that aren't too correlated with already-selected ones
2. **R² threshold**: Skip predictors that have R² > 0.95 with any selected predictor
3. **Residual rescue**: After initial selection, check if excluded high-importance predictors would significantly reduce residual variance

**Formula for residual rescue:**
```
Include if: corr(predictor, residual)² > 0.1 AND residual_reduction > 0.02
```

### 3. Conformal-Style Half-Width Calibration

For uncertainty quantification:

1. Compute OOF absolute residuals
2. Set half-width as `percentile(residuals, 90)` (or configurable quantile)
3. Scale half-width by per-cell uncertainty signals:
   - Higher kNN distance → wider interval
   - Higher BR std → wider interval
   - Higher range violation → wider interval

### 4. 1-SE Rule for Feature Count

When selecting the number of features `k`:

1. Compute CV scores for `k = 1, 2, ..., max_features`
2. Find `k*` with minimum mean CV error
3. Select smallest `k` where `mean_error(k) <= mean_error(k*) + SE(k*)`

This gives a parsimonious model within one standard error of optimal.

### 5. ALT Target OOF Stacking

For alternative targets (e.g., LMSYS) that have missing values:

1. **Problem**: Can't use imputed values as features (would leak information)
2. **Solution**: Out-of-fold predictions
   - Split data into K folds
   - For each fold, train on other K-1 folds, predict on held-out fold
   - Use OOF predictions as feature (no leakage)

### 6. mRMR Feature Selection for GP Models (v7.2)

For Gaussian Process models, use mRMR (Minimum Redundancy Maximum Relevance) with Spearman correlation:

1. **Spearman vs Pearson**: Spearman captures nonlinear monotonic relationships
2. **mRMR Algorithm**:
   - Rank features by relevance (correlation with target)
   - Penalize redundancy (correlation with already-selected features)
   - Score = relevance - mean(redundancy with selected features)
3. **Feature Cap**: Limited by `gp_selector_k_max` (default: 21)

**Intuition:** GP models benefit from features that are diverse (low redundancy) while still predictive (high relevance).

### 7. Per-Column Tolerance Calibration (v7.2)

Calibrate uncertainty thresholds for each column individually:

1. **Masked Evaluation**: Hold out a fraction of known values (`calibration_holdout_frac`)
2. **Prediction**: Predict held-out values with uncertainty estimates
3. **Threshold Search**: Find the largest uncertainty threshold where cumulative RMSE ≤ target RMSE
4. **Target RMSE**: Computed as `calibration_target_rmse_ratio × std(column)`
5. **Fallback**: Use global tolerance if calibration fails

**Formula:**
```
For each threshold t in sorted(uncertainties):
    mask = uncertainty <= t
    rmse = sqrt(mean((pred[mask] - true[mask])^2))
    if rmse <= target_rmse:
        tolerance = t
```

### 8. Periodic Recalibration (v7.2)

Optionally recalibrate tolerances every N imputation passes:

1. **Rationale**: As imputation progresses, predictions improve
2. **Trigger**: Every `recalibrate_every_n_passes` passes (default: 1)
3. **Benefit**: Adaptive tolerances based on current imputation quality
4. **Disable**: Set to 0 for single calibration at start only

---

## Configuration Parameters

### predict.py CLI Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | clean_combined...csv | Input data file |
| `--target` | lmsys_Score | Target column to predict |
| `--max-features` | 20 | Max features in selection |
| `--n-folds` | 5 | CV folds |
| `--use-alt` | False | Enable ALT target imputation |
| `--alt-cols` | None | Alternative target columns |
| `--cache-dir` | .impute_cache | Imputation cache directory |
| `--force-reimpute` | False | Ignore cache |
| `--margin` | 20.0 | Margin for top_by_margin_prob column |
| `--gp_selector_k_max` | 21 | Feature cap for GP models (mRMR) |
| `--calibrate_tolerances` | False | Enable per-column tolerance calibration |
| `--calibration_target_rmse_ratio` | 0.69 | Target RMSE/std ratio for calibration |
| `--calibration_n_rounds` | 3 | Monte Carlo rounds for calibration |
| `--calibration_holdout_frac` | 0.2 | Fraction of known values to hold out |
| `--recalibrate_every_n_passes` | 1 | Recalibrate every N passes (0 = only at start) |

### column_imputer.py Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iter` | 10 | Maximum imputation passes |
| `tol` | 0.01 | Convergence tolerance (relative change) |
| `n_nearest` | 5 | Neighbors for kNN ratio |
| `quality_quantile` | 0.9 | Quantile for half-width calibration |
| `selector_r2_cap` | 0.95 | Max R² for predictor correlation |
| `selector_max_predictors` | 15 | Max predictors per column |
| `gp_selector_k_max` | 21 | Feature cap for GP models (mRMR) |
| `calibrate_tolerances` | False | Enable per-column tolerance calibration |
| `calibration_target_rmse_ratio` | 0.69 | Target RMSE/std ratio |
| `recalibrate_every_n_passes` | 1 | Recalibrate every N passes |
| `n_jobs_oof` | -1 | Parallel jobs for OOF fitting |

### Model-Specific Parameters (SpecializedColumnImputer)

**Gaussian Process Models:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_type` | linear_matern | Kernel: "matern" or "linear_matern" |
| `n_restarts_optimizer` | 3 | GP hyperparameter optimization restarts |
| `normalize_y` | True | Normalize target in GP |

**Categorical Models:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 300 | RandomForest trees (if n_classes > 2) |
| `max_depth` | 6 | Maximum tree depth |
| `min_samples_leaf` | 2 | Minimum samples per leaf |

---

## Performance Considerations

### Parallelism

The pipeline uses a mixed parallelism strategy:

1. **Thread-based parallelism** (via `joblib` with `backend='threading'`):
   - Column model fitting (parallel across columns in same tier)
   - Column imputation (parallel predictions within a tier)
   - Reason: Lower overhead, shared memory for large arrays

2. **Jacobi-style updates**:
   - All columns in a tier are updated from a snapshot
   - Prevents order-dependent results within a tier
   - Enables safe parallelization

### Caching

Imputation results are cached to avoid redundant computation:

```
cache_key = hash(data_hash + config_hash)
cache_path = cache_dir / f"{cache_key}.parquet"
```

Cache invalidation triggers:
- Data changes (hash mismatch)
- Config changes (parameters differ)
- `--force-reimpute` flag

### Memory Management

For large datasets:
- Use `dtype=np.float32` where possible
- GP models use StandardScaler for numerical stability
- Feature selection limits predictor count per column (especially `gp_selector_k_max`)
- CorrelationWeightedImputer stores sorted arrays per column for percentile lookups

### Bottlenecks

| Stage | Complexity | Notes |
|-------|------------|-------|
| Column classification | O(n_cols × n_features) | Correlation analysis per column |
| Feature selection (mRMR) | O(k² × n_features) | k = gp_selector_k_max, pairwise redundancy |
| GP model fitting | O(n³) per column | Cubic in training samples, major bottleneck |
| Iterative passes | O(n_passes × n_cols × predict) | Usually 3-14 passes |
| Quality evaluation | O(n_monte_carlo × impute) | Optional, set n_mc_samples |

---

## Error Handling

### Common Failure Modes

1. **Convergence failure**: Imputer doesn't converge within max_iter
   - Increase `max_iter` or `tol`
   - Check for adversarial column relationships

2. **Feature selection degeneracy**: No valid predictors found
   - Lower `selector_r2_cap`
   - Increase `selector_max_predictors`

3. **GP fitting issues**: Convergence failures or memory issues
   - Fallback: Reduce `gp_selector_k_max` to limit features
   - Or increase `n_restarts_optimizer` for better hyperparameter search

4. **Mapping conflicts in combiner**: Duplicate unified names
   - Check `find_mapping_issues()` output
   - Update mapping JSON files

### Logging

All modules use Python `logging`:
```python
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

Key log levels:
- `INFO`: Progress updates, summary statistics
- `WARNING`: Non-fatal issues (convergence slow, fallbacks)
- `ERROR`: Fatal issues requiring intervention
