# LLM Benchmarks Pipeline - Data Dictionary

This document defines all data schemas, column definitions, and file formats used throughout the pipeline.

---

## Table of Contents

1. [Input Data](#input-data)
   - [Benchmark CSV Files](#benchmark-csv-files)
   - [Mapping JSON Files](#mapping-json-files)
2. [Combined Data](#combined-data)
   - [clean_combined_all_benches.csv](#benchmarksclean_combined_all_benchescsv)
   - [Combined Column Descriptions](#combined-column-descriptions)
3. [Output Data](#output-data)
   - [imputed_full.csv](#imputed_fullcsv)
   - [predictions_best_model.csv](#predictions_best_modelcsv)
   - [Feature Ranking Files](#feature-ranking-files)
   - [Quality Reports](#quality-reports)
   - [Dependency Analysis](#dependency-analysis)

---

## Input Data

### Benchmark CSV Files

Each benchmark source provides a CSV with model performance scores. All sources share a common pattern: one row per model, with benchmark-specific score columns. See [Combined Column Descriptions](#combined-column-descriptions) for detailed descriptions of individual columns as they appear in the combined CSV.

---

#### LMSYS/Chatbot Arena (lmsys.csv)

**What it measures:** Human preference-based ELO ratings from blind A/B comparisons where users vote on which model response they prefer.

**Why it's useful:** Gold standard for general-purpose model quality; reflects real user preferences across diverse tasks; primary prediction target in this pipeline.

| Column | Type | Description |
|--------|------|-------------|
| `Model` | string | Model name as displayed on Arena |
| `Arena Score` | float | ELO rating (~800-2000 range) from human preference voting |
| `95% CI` | string | 95% confidence interval (e.g., "+5/-5") |
| `Votes` | int | Number of votes received |
| `Organization` | string | Model provider/company |
| `License` | string | Model license type (Proprietary, Open, etc.) |

---

#### LiveBench (livebench.csv)

**What it measures:** Continuously updated benchmark across 6 categories: Reasoning, Coding, Math, Data Analysis, Language, and Instruction Following. Tasks are refreshed regularly to prevent contamination.

**Why it's useful:** Comprehensive multi-dimensional evaluation; actively maintained; captures reasoning (zebra puzzles, spatial), coding (code generation, agentic JS/TS/Python), math (olympiad, integrals), and language (plot unscrambling, typos).

| Column | Type | Description |
|--------|------|-------------|
| `Model` | string | Model identifier |
| `Global Average` | float | Overall average across all tasks |
| `Reasoning Average` | float | Average on reasoning tasks |
| `Coding Average` | float | Average on coding tasks |
| `Mathematics Average` | float | Average on math tasks |
| `Data Analysis Average` | float | Average on data analysis tasks |
| `Language Average` | float | Average on language tasks |
| `IF Average` | float | Average on instruction-following tasks |
| `zebra_puzzle` | float | Einstein's riddles with logical deduction |
| `spatial` | float | 2D/3D shape reasoning |
| `code_generation` | float | Real-world library usage scenarios |
| `olympiad` | float | USAMO/IMO math competition problems |
| `plot_unscrambling` | float | Reorder shuffled movie plot sentences |
| `connections` | float | NYT-style word grouping puzzle |

---

#### AiderBench (aider.csv)

**What it measures:** Code editing capability using diff-based and whole-file editing modes across 225 Exercism exercises in C++, Go, Java, JavaScript, Python, and Rust.

**Why it's useful:** Tests practical code modification (not just generation); evaluates real-world code editing workflows.

| Column | Type | Description |
|--------|------|-------------|
| `Model` | string | Model identifier |
| `Percent correct` | float | Accuracy rate for code edits |
| `Cost` | float | API cost per operation |
| `Edit Format` | string | Format category (diff, whole, etc.) |

---

#### ContextArena (contextarena.csv)

**What it measures:** Long-context retrieval accuracy from 8k to 1M tokens using OpenAI's MRCR (Multi-round Co-reference Resolution) dataset. Tests needle-in-haystack with 2, 4, or 8 needles.

**Why it's useful:** Critical for modern long-context applications; evaluates whether models can distinguish between multiple similar items hidden in very long contexts.

| Column | Type | Description |
|--------|------|-------------|
| `Model` | string | Model identifier |
| `Max Ctx` | int | Maximum context window supported |
| `8k (%) 2 needles` | float | Accuracy at 8k context with 2 needles |
| `8k (%) 8 needles` | float | Accuracy at 8k context with 8 needles |
| `16k (%)` through `1M (%)` | float | Accuracy at various context sizes |
| `AUC @128k (%)` | float | Area under curve up to 128k |
| `AUC @1M (%)` | float | Area under curve up to 1M |

---

#### EQ-Bench (eqbench.csv)

**What it measures:** Two benchmarks - (1) Emotional Intelligence (EQ): evaluates understanding, empathy, and interpersonal skills via challenging role-plays; (2) Creative Writing: evaluates storytelling quality via pairwise comparisons.

**Why it's useful:** Captures soft skills and creativity beyond STEM performance; measures human-like qualities.

| Column | Type | Description |
|--------|------|-------------|
| `model` | string | Model identifier |
| `eq_elo` | float | EQ-Bench ELO rating |
| `eq_rubric_score` | float | Rubric-based score (0-100) |
| `eq_humanlike`, `eq_safe`, `eq_empathy`, etc. | float | Individual personality/trait scores |
| `creative_elo` | float | Creative writing ELO rating |
| `creative_rubric_score` | float | Output quality (0-100) |
| `creative_vocab_complexity` | float | Vocabulary sophistication metric |

---

#### OpenBench (openbench.csv)

**What it measures:** Aggregation of major benchmarks including competitive math (AIME, AMC), science (GPQA), coding (Codeforces, SWE-Bench), and general knowledge (MMLU). Also includes cost/latency metrics.

**Why it's useful:** Acts as canonical namespace for models; comprehensive coverage; includes pricing and speed data alongside performance.

| Column | Type | Description |
|--------|------|-------------|
| `Model` | string | OpenBench canonical model name |
| `Organization` | string | Model provider |
| `GPQA Diamond` | float | Graduate-level science Q&A |
| `AIME 2024` | float | AIME 2024 math competition |
| `MATH-500` | float | MATH benchmark (500 problems) |
| `SWE-Bench Lite` | float | Software engineering benchmark |
| `MMLU` | float | Massive Multitask Language Understanding |
| `Input Cost Per Million Tokens ($)` | float | Pricing |
| `Reasoning` | bool | Whether model is a reasoning model |

---

#### SimpleBench (simplebench.csv)

**What it measures:** Common-sense reasoning and "trick" questions designed to expose pattern-matching vs genuine understanding. Questions are phrased like classic riddles but with critical modifications that change the answer.

**Why it's useful:** Tests brittleness of frontier models; reveals stochastic parrot behavior where models regurgitate memorized answers.

| Column | Type | Description |
|--------|------|-------------|
| `Model` | string | Model identifier |
| `Score (AVG@5)` | float | Average score across 5 attempts |
| `Organization` | string | Creator organization |

---

#### Lechmazur Combined (lechmazur.csv)

**What it measures:** 5 specialized benchmarks - (1) Confabulations: hallucination rate on misleading questions; (2) Generalization: out-of-domain generalization; (3) Step Game: multi-step reasoning; (4) Elimination Game: process of elimination logic; (5) NYT Connections: word association puzzles.

**Why it's useful:** Tests factual reliability and reasoning depth; NYT Connections is trending metric for semantic understanding.

| Column | Type | Description |
|--------|------|-------------|
| `confab_Confab %` | float | Hallucination rate on misleading questions |
| `confab_Non-Resp %` | float | Non-response rate on answerable questions |
| `gen_Avg Rank` | float | Generalization ability rank |
| `nytcon_Score %` | float | NYT Connections puzzle performance |
| `step_mu`, `step_sigma` | float | Multi-step reasoning stats |

---

#### Artificial Analysis (artificial_analysis.csv)

**What it measures:** Quality index, speed, pricing, and evaluation benchmarks including AIME '25, GPQA, HLE (hard questions replacing saturated MMLU), IFBench, LiveCodeBench, and more.

**Why it's useful:** Comprehensive cost-performance analysis; includes difficult new benchmarks that frontier models haven't saturated.

| Column | Type | Description |
|--------|------|-------------|
| `Model` | string | Model identifier |
| `Quality Index` | float | Overall quality score |
| `Speed (tokens/s)` | float | Generation speed |
| `price_1m_input_tokens` | float | Cost per million input tokens |
| `price_1m_output_tokens` | float | Cost per million output tokens |
| `aa_eval_aime_25` | float | AIME 2025 performance |
| `aa_eval_gpqa` | float | Graduate-level science Q&A |
| `aa_eval_hle` | float | Hard questions benchmark (replaces MMLU) |
| `aa_eval_livecodebench` | float | Contamination-free coding benchmark |

---

#### WeirdML (weirdml.csv)

**What it measures:** Unusual ML tasks requiring actual understanding: shape classification from noisy coordinates, chess position evaluation, XOR logic, pattern recognition. Tests LLM's ability to write working PyTorch code to solve weird problems.

**Why it's useful:** Tests robustness to unusual problems; 30+ edge-case tasks that require genuine understanding rather than pattern matching.

| Column | Type | Description |
|--------|------|-------------|
| `model` | string | Model identifier |
| `shapes_hard_acc` | float | Shape classification from noisy 2D points |
| `chess_winners_acc` | float | Chess position evaluation |
| `xor_easy_acc`, `xor_hard_acc` | float | XOR logic tasks |
| `number_patterns_acc` | float | Number sequence patterns |
| `avg_acc` | float | Overall accuracy across all tasks |
| `cost_per_run_usd` | float | Cost metric |

---

#### ARC Prize (arc.csv)

**What it measures:** Abstract reasoning via novel grid-based puzzles (ARC-AGI-1 and ARC-AGI-2). Tests "fluid intelligence" - the ability to acquire new skills from minimal examples.

**Why it's useful:** Focuses on pure reasoning without domain knowledge; trending AGI benchmark; ARC-AGI-2 adds complex compositional reasoning.

| Column | Type | Description |
|--------|------|-------------|
| `AI System` | string | Model/system identifier |
| `Organization` | string | Creator organization |
| `ARC-AGI-1` | float | Performance on task set 1 (%) |
| `ARC-AGI-2` | float | Performance on task set 2 (%) |
| `Cost/Task` | float | USD cost per task |

---

#### Yupp Leaderboard (yupp.csv)

**What it measures:** VIBEScore for text and coding tasks based on user preferences in side-by-side comparisons on Yupp.ai.

**Why it's useful:** Alternative quality metric from real user preferences; separates text vs coding performance.

| Column | Type | Description |
|--------|------|-------------|
| `model` | string | Model identifier |
| `Text_Score` | float | VIBEScore for text tasks |
| `Coding_Score` | float | VIBEScore for coding tasks |

---

#### UGI Leaderboard (ugi.csv)

**What it measures:** Writing quality score based on human evaluation, factoring in intelligence, style, repetition, and adherence to output length requests.

**Why it's useful:** Specialized prose quality metric; focuses on user-facing generation quality.

| Column | Type | Description |
|--------|------|-------------|
| `author/model_name` | string | Model identifier (HF-style) |
| `Writing` | float | Writing quality score |

---

#### Vectara Hallucination (vectara.csv)

**What it measures:** Hallucination rate and factual consistency in retrieval-augmented generation (RAG) scenarios.

**Why it's useful:** Critical for production deployment; evaluates factual accuracy when models have access to source documents.

| Column | Type | Description |
|--------|------|-------------|
| `Model` | string | Model identifier |
| `Hallucination Rate` | float | Percentage of outputs with factual errors |
| `Factual Consistency Rate` | float | Percentage of factually correct outputs |
| `Answer Rate` | float | Percentage of questions answered |

---

#### StyleBench / ToneBench

**What they measure:** Document formatting quality (headers, lists, bold text, length) and tone appropriateness in generated text.

**Why useful:** Evaluates output presentation beyond content accuracy; important for user-facing applications.

| Column | Type | Description |
|--------|------|-------------|
| `normalized_length` | float | Response length consistency |
| `normalized_header_count` | float | Header/structure usage |
| `normalized_bold_count` | float | Emphasis usage |
| `normalized_list_count` | float | List formatting usage |
| `scaled_avg_score` | float | Tone appropriateness score |

---

### Mapping JSON Files

#### name_to_openbench.json

Maps source-specific model names to OpenBench canonical names.

```json
{
  "source_name": {
    "Source Model Name": "OpenBench Model Name",
    "gpt-4-turbo-2024-04-09": "GPT-4 Turbo (2024-04-09)",
    "claude-3-opus-20240229": "Claude 3 Opus"
  }
}
```

**Structure:**
- Top-level keys: Source identifiers (e.g., "lmsys", "livebench", "aider")
- Nested keys: Original model names from that source
- Values: Corresponding OpenBench canonical names

#### openbench_to_unified.json

Maps OpenBench names to unified names (final canonical form).

```json
{
  "GPT-4 Turbo (2024-04-09)": "gpt-4-turbo-2024-04-09",
  "Claude 3 Opus": "claude-3-opus-20240229",
  "Claude 3.5 Sonnet (Oct 2024)": "claude-3-5-sonnet-20241022"
}
```

**Structure:**
- Keys: OpenBench model names
- Values: Unified model identifiers (typically lowercase, hyphenated)

#### bad_models.json

List of models to exclude from analysis.

```json
[
  "Unknown Model",
  "Test Model",
  "Deprecated Model v0.1"
]
```

**Usage:** Models in this list are filtered out during combination.

---

## Combined Data

### benchmarks/clean_combined_all_benches.csv

Combined benchmark data from all sources, merged on Unified_Name. This is the primary input to the prediction pipeline.

| Column | Type | Description |
|--------|------|-------------|
| `Unified_Name` | string | **Primary key** - Unified model identifier |
| `openbench_*` | float | OpenBench benchmark scores |
| `livebench_*` | float | LiveBench task scores |
| `lmsys_Score` | float | LMSYS/Arena ELO rating (primary target) |
| `lmarena_Score` | float | Length-adjusted Arena score |
| `aiderbench_*` | float | Aider coding scores |
| `contextarena_*` | float | Long-context retrieval scores |
| `eqbench_*` | float | EQ-Bench and creative writing scores |
| `arc_*` | float | ARC-AGI reasoning scores |
| `weirdml_*` | float | WeirdML task accuracies |
| `aa_*` | float | Artificial Analysis metrics |
| `... (~70+ columns)` | float | Additional benchmark columns |

**Notes:**
- Column names are prefixed with source identifier to avoid collisions
- Missing values are represented as `NaN` (model not in that benchmark)

### Combined Column Descriptions

Detailed descriptions of each column as it appears in the combined CSV.

#### OpenBench columns
| Column | Description |
|--------|-------------|
| `openbench_Reasoning` | Whether or not a model is a reasoning model |
| `openbench_Reasoning Output Tokens Used to Run Artificial Analysis Intelligence Index (million)` | Millions of reasoning tokens used to run the AA Intelligence Index |
| `openbench_Answer Output Tokens Used to Run Artificial Analysis Intelligence Index (million)` | Millions of output tokens used to run the AA Intelligence Index |

#### SimpleBench columns
| Column | Description |
|--------|-------------|
| `simplebench_Score (AVG@5)` | Common-sense reasoning "trick" questions that expose pattern-matching vs genuine understanding. Phrased like classic riddles but with critical modifications that change the answer. Average across 5 attempts. |

#### LiveBench columns
| Column | Description |
|--------|-------------|
| `livebench_theory_of_mind` | Evaluates reasoning about internal states of other people (replaced web_of_lies, Nov 2025) |
| `livebench_zebra_puzzle` | Einstein's riddles with logical deduction constraints (updated Nov 2025 with larger boards) |
| `livebench_spatial` | 2D/3D shape reasoning — intersections, orientations, cuts |
| `livebench_logic_with_navigation` | Logic problem solving + 2D space navigation (added Dec 2025) |
| `livebench_code_generation` | Real-world library usage scenarios (overhauled Apr 2025) |
| `livebench_code_completion` | Completing partial coding solutions (refreshed Apr 2025) |
| `livebench_javascript` | Agentic coding: resolve real JS GitHub issues via Mini-SWE-Agent (May 2025) |
| `livebench_typescript` | Agentic coding: resolve real TS GitHub issues in multi-turn environment |
| `livebench_python` | Agentic coding: resolve real Python GitHub issues via Mini-SWE-Agent |
| `livebench_AMPS_Hard` | Synthetic hard math (integration, derivation, algebra) |
| `livebench_integrals_with_game` | Math integrating game theory with integral calculations (Jan 2026) |
| `livebench_olympiad` | USAMO/IMO competition problems (fill-in-the-blank, updated annually) |
| `livebench_tablejoin` | Identify valid join columns between two partially-overlapping tables |
| `livebench_tablereformat` | Reformat a table between formats (JSON, XML, CSV, TSV) |
| `livebench_connections` | NYT-style word grouping puzzle |
| `livebench_plot_unscrambling` | Reorder shuffled movie plot sentences |
| `livebench_typos` | Fix injected typos in arXiv abstracts while preserving style |
| `livebench_IF Average` | Instruction Following aggregate — paraphrasing, summarizing, story generation with verifiable constraints |

#### Arena columns
| Column | Description |
|--------|-------------|
| `lmsys_Score` | Human preference ELO score from blind A/B tests (primary prediction target) |
| `lmarena_Score` | `lmsys_Score` adjusted for human length bias in response selection |

#### Lechmazur columns
| Column | Description |
|--------|-------------|
| `lechmazur_confab_Confab %` | Frequency of confabulated/hallucinated answers to misleading questions |
| `lechmazur_confab_Non-Resp %` | Frequency of non-responses to answerable questions |
| `lechmazur_gen_Avg Rank` | Generalization: infer a narrow theme from examples + anti-examples, detect true match among distractors |
| `lechmazur_nytcon_Score %` | 759 NYT Connections puzzles with added difficulty (extra words) |

#### AiderBench columns
| Column | Description |
|--------|-------------|
| `aiderbench_Percent correct` | Accuracy on 225 Exercism coding exercises across C++, Go, Java, JS, Python, Rust |

#### ContextArena columns
| Column | Description |
|--------|-------------|
| `contextarena_8k (%) 2 needles` | MRCR long-context retrieval: distinguish 2 needles hidden in 8k-token conversation |
| `contextarena_8k (%) 8 needles` | Same with 8 needles at 8k tokens |

#### EQ-Bench columns
| Column | Description |
|--------|-------------|
| `eqbench_eq_elo` | EQ-Bench 3 ELO: emotional intelligence via challenging role-plays (judged by Claude Sonnet 3.7) |
| `eqbench_creative_elo` | Creative Writing v3 ELO: short-form storytelling via pairwise comparisons (32 prompts) |

#### ARC columns
| Column | Description |
|--------|-------------|
| `arc_ARC-AGI-1` | "Fluid intelligence" via novel grid-based puzzles requiring skill acquisition from minimal examples |
| `arc_ARC-AGI-2` | Same + complex compositional reasoning and symbolic interpretation |

#### Artificial Analysis columns
| Column | Description |
|--------|-------------|
| `aa_pricing_price_1m_input_tokens` | Price per 1M input tokens |
| `aa_pricing_price_1m_output_tokens` | Price per 1M output tokens |
| `aa_eval_aime_25` | 2025 AIME math competition problems |
| `aa_eval_gpqa` | PhD-level "Google-proof" multiple-choice science questions |
| `aa_eval_hle` | Hard questions replacing saturated MMLU (too difficult for frontier models) |
| `aa_eval_ifbench` | Instruction following with specific formatting constraints |
| `aa_eval_lcr` | Hard text-based questions requiring reasoning across ~100k-token documents |
| `aa_eval_livecodebench` | Contamination-free coding from recent contests (LeetCode, AtCoder, Codeforces) |
| `aa_eval_mmlu_pro` | Enhanced MMLU: 10 choices, no trivial questions, reasoning-heavy |
| `aa_eval_scicode` | Write code to solve realistic research problems in physics, chemistry, biology, math |
| `aa_eval_tau2` | Agentic benchmark: tool-using conversations for user problem-solving (Dual-Control) |
| `aa_eval_terminalbench_hard` | Operate in a real Linux terminal: compile, debug, install, manage files |

#### WeirdML columns
| Column | Description |
|--------|-------------|
| `weirdml_shapes_hard_acc` | Classify shapes from 512 noisy 2D coordinates (random position/orientation/size) |
| `weirdml_classify_shuffled_acc` | Undisclosed ML classification task |
| `weirdml_number_patterns_acc` | Undisclosed number pattern task |
| `weirdml_avg_acc` | Overall accuracy across all WeirdML tasks (write working PyTorch code, debug over 5 iterations) |

#### Yupp columns
| Column | Description |
|--------|-------------|
| `yupp_Text_Score` | VIBE score from user side-by-side preferences (Bradley-Terry ELO variant, text tasks) |
| `yupp_Coding_Score` | VIBE score for coding tasks |

#### UGI columns
| Column | Description |
|--------|-------------|
| `ugileaderboard_Writing` | Writing quality: intelligence, style, repetition, output length adherence |

#### Soothsayer Style columns
| Column | Description |
|--------|-------------|
| `style_normalized_length` | Response length to a small set of prompts |
| `style_log_normalized_length` | Log-transformed normalized_length (reduces right-skew) |
| `style_normalized_header_count` | Header count in responses |
| `style_normalized_bold_count` | Bold text count in responses |
| `style_normalized_list_count` | List item count in responses |
| `style_predicted_delta` | Predicted difference between lmarena_Score and lmsys_Score based on style |

#### Soothsayer Logic columns
| Column | Description |
|--------|-------------|
| `logic_accuracy` | Like simplebench but smaller question set, ran in-house |
| `logic_weighted_accuracy` | BayesianRidge prediction of simplebench from per-question results (CV R²=0.8) |
| `logic_PC1` | 1st PC of 12 per-question logic scores (general reasoning, 36.6% variance) |
| `logic_PC2` | 2nd PC (reasoning style contrast, 12.2% variance) |
| `logic_PC3` | 3rd PC (Q4 vs Q5/Q2 contrast, 9.8% variance) |
| `logic_PC4` | 4th PC (Q11 vs Q6 contrast, 8.0% variance) |

#### Soothsayer Tone columns
| Column | Description |
|--------|-------------|
| `tone_scaled_avg_score_by_*` | Qualitative response quality (engaging, clear, user-friendly) as judged by each LLM judge |

#### Soothsayer Writing columns
| Column | Description |
|--------|-------------|
| `writing_*_score` | Story quality judged via rubric (craft/plot/world/originality/element use) by named LLM judge |
| `writing_* TrueSkill` | TrueSkill rating from pairwise creative writing comparisons by named judge |
| `writing_* Sigma` | TrueSkill uncertainty for that judge (lower = more confident) |

#### Soothsayer EQ columns
| Column | Description |
|--------|-------------|
| `eq_Grok 4 Fast TrueSkill` | Replication of eqbench EQ judged by Grok 4 Fast (R²=0.94 vs original) |

---

## Output Data

### imputed_full.csv

Complete imputed benchmark matrix.

| Column | Type | Description |
|--------|------|-------------|
| `Unified_Name` | string | **Primary key** |
| `{Benchmark}_imputed` | float | Imputed score (original if observed, imputed if missing) |
| `{Benchmark}_was_missing` | bool | True if value was imputed |
| `{Benchmark}_lower` | float | Lower bound of prediction interval (optional) |
| `{Benchmark}_upper` | float | Upper bound of prediction interval (optional) |
| `{Benchmark}_uncertainty` | float | Uncertainty estimate (half-width) |

**Imputation Details:**
- Uses Gated Iterative Imputation algorithm
- Missing values filled iteratively, refining estimates each pass
- Uncertainty quantified via conformal-style calibration

### predictions_best_model.csv

Final predictions for the target column (typically LMSYS).

| Column | Type | Description |
|--------|------|-------------|
| `Unified_Name` | string | Model identifier |
| `{Target}_actual` | float | Actual value (if observed) |
| `{Target}_predicted` | float | Model prediction |
| `{Target}_residual` | float | Actual - Predicted (if observed) |
| `{Target}_lower` | float | Lower prediction interval |
| `{Target}_upper` | float | Upper prediction interval |
| `{Target}_was_missing` | bool | Whether original value was missing |
| `fold` | int | CV fold (for OOF predictions) |
| `model` | string | Model name that made prediction |

### Feature Ranking Files

#### feature_ranking_gain.csv

Features ranked by importance for target prediction.

| Column | Type | Description |
|--------|------|-------------|
| `feature` | string | Feature/column name |
| `gain` | float | Total gain from tree splits |
| `split_count` | int | Number of times used in splits |
| `rank` | int | Rank (1 = most important) |
| `selected` | bool | Whether included in final model |

#### feature_ranking_correlation.csv

Features ranked by correlation with target.

| Column | Type | Description |
|--------|------|-------------|
| `feature` | string | Feature name |
| `correlation` | float | Pearson correlation with target |
| `abs_correlation` | float | Absolute correlation |
| `p_value` | float | Significance of correlation |
| `rank` | int | Rank by absolute correlation |

### Quality Reports

#### imputation_quality_summary.csv

Per-column imputation quality metrics.

| Column | Type | Description |
|--------|------|-------------|
| `column` | string | Column that was imputed |
| `n_missing` | int | Number of missing values |
| `n_observed` | int | Number of observed values |
| `missing_frac` | float | Fraction missing |
| `cv_rmse` | float | Cross-validated RMSE on observed |
| `cv_mae` | float | Cross-validated MAE on observed |
| `cv_r2` | float | Cross-validated R² on observed |
| `monte_carlo_rmse` | float | Monte Carlo estimated RMSE (optional) |
| `n_predictors` | int | Number of predictors used |
| `predictor_list` | string | Comma-separated predictor names |

#### imputation_quality_by_model.csv

Per-model imputation quality (how well each model's values were imputed).

| Column | Type | Description |
|--------|------|-------------|
| `Unified_Name` | string | Model identifier |
| `n_missing` | int | Columns missing for this model |
| `n_observed` | int | Columns observed for this model |
| `avg_uncertainty` | float | Average uncertainty across imputations |
| `max_uncertainty` | float | Maximum uncertainty |
| `avg_knn_ratio` | float | Average kNN ratio (extrapolation risk) |

#### imputation_config.json

Configuration used for imputation run (v7.2 format).

```json
{
  "max_iter": 10,
  "tol": 0.01,
  "n_nearest": 5,
  "quality_quantile": 0.9,
  "selector_r2_cap": 0.95,
  "selector_max_predictors": 15,
  "gp_selector_k_max": 21,
  "calibrate_tolerances": true,
  "calibration_target_rmse_ratio": 0.69,
  "calibration_n_rounds": 3,
  "calibration_holdout_frac": 0.2,
  "recalibrate_every_n_passes": 1,
  "use_catboost": true,
  "catboost_iters": 500,
  "random_state": 0,
  "timestamp": "2026-01-12T10:30:00Z",
  "data_hash": "abc123..."
}
```

#### variance_contributions.csv (v7.1+)

Per-feature variance contributions for model interpretation.

| Column | Type | Description |
|--------|------|-------------|
| `feature` | string | Feature name (or base feature for polynomial terms) |
| `variance_contribution` | float | (beta × std(X))² / var(pred) |
| `percentage` | float | Contribution as percentage of total |
| `is_polynomial` | bool | Whether this is a grouped polynomial term |

### Dependency Analysis

#### column_dependency_matrix.csv

Square matrix showing which columns predict which.

| Column | Type | Description |
|--------|------|-------------|
| `column` | string | Column being predicted |
| `{predictor}` | float | Importance of predictor for this column |

**Interpretation:**
- Row = target column
- Column = predictor column
- Value = importance weight (higher = more important)
- Zero = not used as predictor

#### column_dependency_graph.json

Graph structure for visualization.

```json
{
  "nodes": [
    {"id": "LMSYS", "group": 1, "missing_frac": 0.3},
    {"id": "OpenBench_Average", "group": 2, "missing_frac": 0.1}
  ],
  "links": [
    {"source": "OpenBench_Average", "target": "LMSYS", "weight": 0.8},
    {"source": "Aider_Polyglot", "target": "LMSYS", "weight": 0.5}
  ]
}
```

**Structure:**
- `nodes`: List of columns with metadata
  - `id`: Column name
  - `group`: Cluster assignment
  - `missing_frac`: Fraction of missing values
- `links`: Directed edges (predictor → target)
  - `source`: Predictor column
  - `target`: Target column
  - `weight`: Importance of this predictor

---

## Column Naming Conventions

### Prefix Conventions

| Prefix | Source |
|--------|--------|
| `LiveBench_` | LiveBench benchmark |
| `OpenLLM_` | Open LLM Leaderboard |
| `OpenBench_` | OpenBench suite |
| `Aider_` | Aider coding benchmark |
| `WildBench_` | WildBench evaluation |
| `BigCode_` | BigCode leaderboard |
| `AA_` | Artificial Analysis |

### Suffix Conventions

| Suffix | Meaning |
|--------|---------|
| `_transformed` | After normalization transformation |
| `_original` | Original untransformed value |
| `_imputed` | Value after imputation |
| `_was_missing` | Boolean flag for imputed values |
| `_lower` | Lower prediction interval bound |
| `_upper` | Upper prediction interval bound |
| `_uncertainty` | Uncertainty estimate (half-width) |
| `_log1p` | Log1p transformation applied |
| `_yeojohnson` | Yeo-Johnson transformation applied |
| `_boxcox` | Box-Cox transformation applied |

---

## Data Types and Constraints

### Numeric Columns

| Type | Range | Missing Representation |
|------|-------|------------------------|
| ELO ratings | ~800-2000 | `NaN` |
| Percentage scores | 0-100 | `NaN` |
| Transformed scores | 0-100 | `NaN` |
| Counts | ≥0 (integer) | `NaN` or 0 |

### String Columns

| Column | Format | Constraints |
|--------|--------|-------------|
| `Unified_Name` | lowercase-hyphenated | Unique, non-null |
| `OpenBench_Name` | Mixed case | May contain special chars |
| `Original_Name_*` | Source-specific | May be null if not in source |

### Boolean Columns

| Column Pattern | Values |
|----------------|--------|
| `*_was_missing` | True/False |
| `selected` | True/False |

---

## Data Quality Notes

### Common Issues

1. **Duplicate model names**: Same model appears under different names
   - Handled by mapping files
   - Check `find_mapping_issues()` output

2. **Inconsistent scales**: Different benchmarks use different scales
   - Addressed by transformation layer
   - All transformed scores on 0-100 scale

3. **High missingness**: Some benchmarks have >50% missing
   - Imputer handles gracefully
   - Quality metrics indicate reliability

4. **Outliers**: Some scores may be erroneous
   - Robust statistics used in imputation
   - Winsorization applied where appropriate

### Validation Checks

Run these checks on output data:

```python
# Check for NaN in imputed columns
assert df['*_imputed'].notna().all()

# Check prediction intervals are valid
assert (df['*_lower'] <= df['*_imputed']).all()
assert (df['*_imputed'] <= df['*_upper']).all()

# Check Unified_Name uniqueness
assert df['Unified_Name'].is_unique
```
