# Project Soothsayer — Architecture

## System Design Overview

This pipeline aggregates 23 CSV patterns (17 public/input benchmark families plus 6 custom/derived internal families), performs model name unification, and uses adaptive KNN + fold-internal PLS hybrid + kernel Ridge regression to predict lmarena scores (style-controlled Arena ELO (arena.ai), 10×5-fold OOF R²=0.941) for large language models.

Missing benchmark values are filled by the **ModelBankImputer** (`column_imputer.py`), which selects the best predictor subset for each individual missing cell based on what that row actually has observed, with per-cell uncertainty tracking and low-rank coherence projection.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA COLLECTION LAYER                              │
│  scrape.bash → 11 active scraper runs (12 scripts in scrapers/)            │
│  run_all_benches.bash → 4 Soothsayer benchmarks → scored CSVs + raw        │
│     responses (EQ JSONs, Writing .txt, Logic multi-run CSV, Style CSV)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RESPONSE-EMBEDDING LAYER (Optional)                     │
│  embeddings/collect_responses.py → cache/all_responses.parquet              │
│  embeddings/embed_responses.py   → cache/response_embeddings.parquet        │
│  embeddings/build_fingerprints.py → cache/model_fingerprints_v4_d32.csv    │
│  • bge-small (BAAI/bge-small-en-v1.5, 384-dim) on MPS/CUDA/CPU              │
│  • Per-model fingerprint: 5-slot concat (eq_t1, eq_t3, logic, style,       │
│    writing) → 1920 raw dims → 32 PCA components (`sem_f01`..`sem_f32`)     │
│  • Joins into combined CSV on model_name (adds ΔRMSE ≈ −1.29 vs pre-sem)    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA COMBINATION LAYER                             │
│  combine.py                                                                  │
│  • 23 CSV patterns consumed (17 public/input + 6 custom/derived)           │
│  • LLM-assisted model name mapping (Gemini API)                             │
│  • Per-source mapping JSONs → OpenBench namespace unification               │
│  Output: benchmarks/combined_all_benches.csv                                │
│  → correlations.py cleans → benchmarks/clean_combined_all_benches.csv       │
│  → merge sem_ features → clean_combined_all_benches_with_sem_v4_d32.csv     │
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
│  • Prediction: Adaptive KNN + optional fold-internal PLS hybrid +           │
│    kernel Ridge + jackknife VI                                              │
│                                                                              │
│  Output: imputed_full.csv, predictions_best_model.csv, diagnostics          │
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
│  Soothsayer Benchmark Results and Derived Internal Features:                 │
│  ├── eq_*.csv                   (Soothsayer EQ TrueSkill ratings)           │
│  ├── writing_*.csv              (Soothsayer Writing TrueSkill ratings)      │
│  ├── logic_*.csv                (Soothsayer Logic scores)                   │
│  ├── style_*.csv                (Soothsayer Style metrics)                  │
│  ├── tone_*.csv                 (Soothsayer Tone scores)                    │
│  └── eq_multiturn_*.csv         (Derived EQ multi-turn behavioral features) │
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
│      (296 columns, 1261 models in the current snapshot)                      │
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
│  Current predict.py Outputs:                                                 │
│  ├── imputed_full.csv                 (Complete imputed matrix)             │
│  ├── predictions_best_model.csv       (Final predictions + intervals)       │
│  ├── oof_predictions.csv              (Out-of-fold predictions)             │
│  ├── metadata.json                    (OOF summary + run metadata)          │
│  └── run_config.json                  (Configuration used)                  │
│                                                                              │
│  Current Diagnostics:                                                        │
│  ├── imputation_quality_per_cell.csv                                       │
│  ├── imputation_quality_per_column.csv                                     │
│  ├── imputation_quality_by_extrapolation_bin.csv                           │
│  ├── imputation_importance.csv        (when available)                     │
│  ├── conformal_diagnostics.csv                                              │
│  ├── conformal_uncertainty_features.csv                                     │
│  ├── model_eval_rmse.csv             (Per-model CV RMSE comparison)        │
│  ├── column_dependency_graph.json                                          │
│  ├── column_dependency_summary.json                                         │
│  ├── column_dependency_summary.csv                                          │
│  └── column_degrees_of_separation.csv                                      │
│                                                                              │
│  Historical older-run artifacts may also appear in existing analysis dirs:   │
│  feature_ranking_gain.csv, feature_matrix_used.csv,                          │
│  best_model_variance_contributions.csv, and related sweep artifacts.         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

### combine.py — Benchmark Data Aggregator

**Purpose:** Aggregate the 23 CSV patterns consumed by the shipped pipeline into a unified namespace, handling model name variations across sources.

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

### embeddings/ — Response-Embedding Pipeline

**Purpose:** Extract learned response-character features from raw benchmark responses, compress them into 32 per-model coordinates, and merge into the predictor CSV.

**Feature family placement:** The `sem_f01`..`sem_f32` columns are a *derived feature set*, sibling to `style_` / `tone_` under the "response-character features" family — same genre of question ("how does this model respond?") answered via learned embeddings instead of engineered metrics. They are NOT a sub-benchmark of Style: the fingerprint draws from all 4 Soothsayer benchmarks' raw responses (EQ + Writing + Logic + Style). Style's responses are only 1 of 5 slots.

**Pipeline stages:**

1. **`collect_responses.py`** — walks all 4 benchmark response sources, normalizes to long-format parquet (`cache/all_responses.parquet`, ~23k rows × [model, benchmark, prompt_id, run_id, response_text]). Strips `<think>` / `<thinking>` / `<reasoning>` wrappers.
2. **`embed_responses.py`** — BAAI/bge-small-en-v1.5 (384-dim, 512-token context) on MPS/CUDA/CPU, unit-normalized, resumable. Writes `cache/response_embeddings.parquet`. Uses first-512-token truncation (chunk-and-pool was tested and regressed — see FINDINGS.md).
3. **`build_fingerprints.py`** — multiple pooling modes. Champion is `per_bench_eq_split` (5 slots: eq_t1, eq_t3, logic, style, writing — EQ middle turn t2 dropped). Missing slots imputed with per-slot centroid. Unsupervised PCA to **32 components** (`sem_f01`..`sem_f32`) gives the best OOF RMSE in the 24/32/48 sweep. Output: `cache/model_fingerprints_v4_d32.csv`. See *Fingerprint construction* below for the full dimensional walk-through.
4. **Merge into combined CSV** — join on `model_name` to produce `clean_combined_all_benches_with_sem_v4_d32.csv`, which is the input to `predict.py` when using sem features.

**Fingerprint construction (champion `per_bench_eq_split` mode, v4 @ 32 PCA dims):**

For each model, v4 produces a single 32-number coordinate vector (`sem_f01`..`sem_f32`) via the following dimensional progression:

1. **Embed each response individually** → each response becomes a 384-dim vector from bge-small. Per model: ~100 embeddings (~5 EQ-t1 + ~5 EQ-t3 + ~48 logic + ~38 style + ~5 writing; the parquet has 23,527 rows across 236 models, ~100/model mean). EQ middle turn (t2) is discarded at slot-pooling time. Runs of the same prompt are treated as separate rows at this stage.
2. **Group responses into 5 slots** by source:

   | Slot | What it contains | Per-model avg count |
   |---|---|---|
   | `eq_t1` | All EQ first-turn responses for this model | ~5 |
   | `eq_t3` | All EQ last-turn responses for this model | ~5 |
   | `logic` | All Logic responses for this model | ~48 |
   | `style` | All Style responses for this model | ~38 |
   | `writing` | All Writing responses for this model | ~5 |

3. **Average within each slot** → one mean 384-dim vector per slot, re-normalized to unit length. Models with no responses for a slot get the per-slot centroid (mean across all models that do have it) — light imputation to keep fingerprint shapes aligned. Produces **5 vectors × 384 dims = 1920 raw numbers per model.**
4. **Concatenate** the 5 slot vectors end-to-end → a single 1920-dim "raw fingerprint" per model. Fingerprint matrix across 236 models is 236 × 1920.
5. **PCA compress** across all models' fingerprints (unsupervised, no target leakage) → top **32 components** kept. These 32 numbers per model are the `sem_f01`..`sem_f32` columns merged into the predictor CSV.

**Why 5 slots instead of pooling everything into one 384-dim mean:** a single pooled fingerprint (v1, cross-bench mode) averaged all ~160 responses per model into one vector and achieved only Δ −0.52 RMSE. The 5-slot structure preserves *how a model's response-character varies across benchmark types* — e.g. "warm on EQ but terse on logic" is a real axis of model variation that single-pool averaging destroyed. EQ is split into first/last turns because multi-turn escalation is itself an ELO-relevant axis (confirmed by the prior `first_person_delta_t3` feature's validation). See FINDINGS.md for the ablation ladder showing each design decision's contribution.

**Dimensional summary:**
```
  1 response             → 384 numbers (bge-small embedding dim)
  ~100 responses/model   → grouped into 5 slots
  1 slot (per model)     → 384-dim mean (after averaging within slot)
  5 slots concat         → 1920-dim raw fingerprint per model
  PCA on 236 × 1920      → 32 components per model (sem_f01..sem_f32)
```

**Design rule discovered empirically (FINDINGS.md):** split a benchmark into sub-slots only when the split preserves a shared signal *plus* a structural delta (EQ turn escalation: Mantel r=0.69 between t1/t3 distance matrices). Don't split when the sub-slots are decorrelated noisy views (Style tech/casual Mantel r=0.39) — pooling across prompts does free pre-PCA denoising of nuisance topic variance that PCA can't replicate.

**Install pins (torch 2.1.1 compat):** `sentence-transformers==2.7.0 transformers==4.40.2 tokenizers<0.20 huggingface-hub<0.24`.

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
- `compute_grouped_conformal_intervals()` — Grouped conformal calibration for prediction intervals
- `main()` — Orchestrates imputation, feature construction, KNN prediction, conformal calibration, and output writing

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
6. **Phase 4b — Coherence Projection:** SVD of the completed matrix at the same rank used for factor extraction. Blend each imputed cell toward the low-rank estimate with shrinkage weight `w = τ/(τ + λ)` where `τ = σ²_cell/sd²_col`. High-uncertainty cells are pulled more toward the coherent SVD; confident cells keep their model-bank value. The shipped config uses `λ = 8.0`
7. **Phase 5 — Post-processing:** Inverse log transforms, σ²-based trajectory features, representative models for API compatibility

**Coherence Projection Rationale:** Per-cell imputation produces individually accurate values but they may be inconsistent across columns within a row (a "Frankenstein" profile). The low-rank projection restores cross-column coherence by blending toward the SVD estimate, weighted by uncertainty.

### 2. Conformal Prediction Intervals

For uncertainty quantification:
1. Compute OOF absolute residuals
2. Set half-width as `percentile(residuals, quantile)` (configurable)
3. Scale half-width by per-cell uncertainty signals (kNN distance, model std, range violation)

### 3. Adaptive KNN + PLS Hybrid + Kernel Ridge + Jackknife VI (shipped pipeline)

The primary prediction algorithm (10×5-fold OOF R²=0.941 for `lmarena_Score`):

1. **Drop `style_*` / `tone_*` columns (`--drop_style_tone`)** — once PLS is introduced in step 3, these columns redundantly dilute the KNN distance. They still inform the imputer's predictor selection and the style_delta feature.
2. **Standardize** the remaining feature matrix (imputed benchmarks + 21 imputation-derived SVD/trajectory features + 32 sem_f* response-embedding dims).
3. **Fold-internal PLS-3 hybrid (`--pls_hybrid_k 3`)** — within each CV fold, fit `PLSRegression(n_components=3)` on `(Xtr, y[tr])` and append the 3 transformed components to both `Xtr` and `Xte` before the KNN call. No leakage: PLS never sees held-out rows. PLS gives the distance a supervised, low-rank "direction of ELO" summary that the raw KNN couldn't see.
4. **Adaptive k via sublinear power cutoff:** `max_dist = d_nearest^0.7 × 3.0`. Tighter neighborhoods in dense regions (top models, k≈20-30) while keeping adequate coverage in sparse regions. Min 20, max 80.
5. **Gaussian kernel weights:** `w = exp(-0.5 × (dist/bw)²)` where bandwidth = distance to 15th-percentile neighbor.
6. **Ridge regression:** `alpha = max(10, std(neighbor_scores))`. Adaptive regularization.
7. **Jackknife variance inflation:** Leave each of the k neighbors out, refit Ridge on k-1, predict the left-out one. Estimate compression slope `b = cov(actual, jackknife_pred) / var(jackknife_pred)`. Clip to [1.0, 1.5]. Correct: `prediction = neighborhood_mean + b × (raw_prediction - neighborhood_mean)`. Reverses Ridge's centering bias.

**Why it works:** Different features predict lmarena at different score levels — feature correlations flip sign around ~1400 (e.g., list count is positive below, negative above). The sublinear power cutoff keeps each neighborhood within a consistent score regime. PLS supervises the feature space with the target but pays the n=127 tax cheaply (3 components, fold-internal). Adding PLS hybrid + drop_style_tone on 2026-04-18 cut OOF RMSE 14.45 → 13.48 at ship time; the post-CritPT rerun (n=127) lands at RMSE 13.61.

---

## Configuration Parameters

### predict.py CLI Arguments (shipped 2026-04-18, 46 flags total)

The **shipped 7-flag config** (from `predict.sh`):

```bash
python3 predict.py \
    --csv_path ../benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_v4_d32.csv \
    --imputer_type model_bank \
    --coherence_lambda 8.0 --coherence_shape exp \
    --predictor_selection loo_forward \
    --drop_style_tone \
    --pls_hybrid_k 3 \
    --cv_repeats_outer 10
```

Key flags:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv_path` | (base CSV) | Input data file; shipped config uses the sem-augmented variant |
| `--output_root` | `analysis_output` | Output directory |
| `--imputer_type` | `model_bank` | Imputer backend: `model_bank` or `specialized` |
| `--knn_power_alpha` | 0.7 | Exponent for sublinear distance cutoff (1.0 = linear) |
| `--knn_power_c` | 3.0 | Coefficient for distance cutoff: max_dist = d0^alpha × C |
| `--knn_max_k` | 80 | Maximum neighbors for KNN |
| `--knn_min_k` | 20 | Minimum neighbors for KNN |
| `--knn_bw_pct` | 0.15 | Kernel bandwidth at this percentile of neighbor distances |
| `--coherence_lambda` | 8.0 | ModelBankImputer: coherence projection shrinkage (0 = disable; 8.0 shipped) |
| `--coherence_shape` | `exp` | Coherence shrinkage shape: linear, exp, squared, etc. |
| `--predictor_selection` | `loo_forward` | Cell-level predictor selection mode |
| `--drop_style_tone` | off | Drop `style_*`/`tone_*` columns before KNN (ON in shipped config) |
| `--pls_hybrid_k` | 0 | Append K fold-internal PLS components to the KNN feature matrix (3 in shipped config) |
| `--cv_repeats_outer` | 10 | CV repeats for OOF evaluation |
| `--margin` | 20.0 | Margin for top_by_margin_prob column |
| `--max_workers` | 0 (auto) | Parallel workers for imputation |

Run `python3 predict.py --help` for the full 46-flag list.

**Flags retired in the 2026-04-18 cleanup:** `--eb_parent`, `--eb_parent_tier_n`, `--eb_parent_sigma_mult`, `--redundancy_threshold`, `--rank_by_corr_only`, `--local_corr_k`, `--learned_dist`, `--resid_weight`, `--coherence_fade_pct`, `--coherence_disagreement_beta`, `--coherence_tau_cap_pct`, `--relax_support_escalator`, `--loo_size_fair`, plus older retirees (`--poly_interactions`, `--poly_limit`, `--no_residual_head`, `--no_traj_in_alt`, `--top_tier_boost`, `--lmarena_style_restriction`, `--cv_repeats_inner`, `--feature_cv_repeats`, `--alt_cv_repeats`).

### ModelBankImputer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.4 | Normalized σ/sd threshold for confident imputations in pass 2 |
| `coherence_lambda` | 8.0 (shipped) | Shrinkage strength for low-rank coherence projection (0 = disable) |
| `predictor_selection` | `loo_forward` | Cell-level predictor selection: `loo_forward` (shipped) or `corr` (legacy/default CLI mode) |
| `min_support` | 10 | Minimum training rows for a model to be fitted |

`redundancy_threshold` is now an internal constant (0.85) — the filter is applied always; the CLI knob was retired in the 2026-04-18 cleanup.

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
