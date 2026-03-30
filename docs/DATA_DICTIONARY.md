# Project Soothsayer — Data Dictionary

This document defines the data schemas, column definitions, and file formats used throughout the pipeline.

---

## Table of Contents

1. [Input Data](#input-data)
   - [Benchmark CSV Files](#benchmark-csv-files)
   - [Mapping JSON Files](#mapping-json-files)
2. [Combined Data](#combined-data)
   - [clean_combined_all_benches.csv](#clean_combined_all_benchescsv)
   - [Combined Column Descriptions](#combined-column-descriptions)
3. [Output Data](#output-data)
   - [imputed_full.csv](#imputed_fullcsv)
   - [predictions_best_model.csv](#predictions_best_modelcsv)
   - [Other Output Files](#other-output-files)

---

## Input Data

### Benchmark CSV Files

Each benchmark source provides a dated CSV (e.g., `livebench_20260315.csv`) with model performance scores. All sources share a common pattern: one row per model, with benchmark-specific score columns. The pipeline uses glob patterns to find the latest file per source.

See [Combined Column Descriptions](#combined-column-descriptions) for detailed descriptions of individual columns as they appear in the combined CSV.

#### External Sources

| Source | File Pattern | What It Measures |
|--------|-------------|-----------------|
| LMArena | `lmarena_*.csv` | Style-controlled Arena ELO (**prediction target**) |
| Chatbot Arena | `lmsys_*.csv` | Raw Arena ELO (excluded from clean CSV — leakage) |
| LiveBench | `livebench_*.csv` | Multi-category: reasoning, coding, math, language, IF |
| OpenBench | `openbench_*.csv` | Canonical model namespace + AIME, GPQA, SWE-Bench, MMLU |
| Artificial Analysis | `artificialanalysis_*.csv` | Quality index, pricing, speed |
| AA Evaluations | `aa_gdpval_*.csv`, `aa_omniscience_*.csv`, `aa_critpt_*.csv` | AIME'25, GPQA, HLE, LiveCodeBench, etc. |
| AiderBench | `aiderbench_*.csv` | Code editing across 6 languages |
| ARC Prize | `arc_*.csv` | Abstract reasoning (ARC-AGI-1, ARC-AGI-2) |
| ContextArena | `contextarena_*.csv` | Long-context retrieval (8k–1M tokens) |
| EQ-Bench | `EQ-Bench_combined_*.csv` | Emotional intelligence + creative writing ELO |
| SimpleBench | `simplebench_*.csv` | Commonsense trick questions |
| Lechmazur | `lechmazur_combined_*.csv` | Confabulations, generalization, NYT Connections |
| WeirdML | `weirdml_*.csv` | Unusual ML tasks (shapes, chess, XOR) |
| Yupp | `yupp_text_coding_scores_*.csv` | VIBEScore for text + coding |
| UGI | `UGI_Leaderboard_*.csv` | Writing quality |

#### Soothsayer Sources

| Source | File Pattern | What It Measures |
|--------|-------------|-----------------|
| Soothsayer EQ | `eq_*.csv` | Emotional intelligence (TrueSkill pairwise) |
| Soothsayer Writing | `writing_*.csv` | Creative writing (TrueSkill pairwise) |
| Soothsayer Logic | `logic_*.csv` | Commonsense reasoning (ML-predicted SimpleBench scores) |
| Soothsayer Style | `style_*.csv` | Writing style metrics (length, formatting, bold, lists) |
| Soothsayer Tone | `tone_*.csv` | Response quality/tone (TrueSkill pairwise) |

---

### Mapping JSON Files

Model names differ across benchmark sources. Per-source JSON mapping files in `benchmark_combiner/mappings/` translate each source's model names to OpenBench canonical names.

**Structure:** Each file maps source model names → OpenBench names:

```json
{
  "gpt-4-turbo-2024-04-09": "GPT-4 Turbo (2024-04-09)",
  "claude-3-opus-20240229": "Claude 3 Opus"
}
```

**Mapping files (20 total):** `lmsys_to_openbench.json`, `livebench_to_openbench.json`, `aiderbench_to_openbench.json`, `eqbench_to_openbench.json`, `arc_to_openbench.json`, `aa_to_openbench.json`, `aa_evals_to_openbench.json`, `weirdml_to_openbench.json`, `yupp_to_openbench.json`, `ugi_to_openbench.json`, and others.

Unmapped models are flagged for LLM-assisted suggestion via the Gemini API, but suggestions require manual review before being committed.

---

## Combined Data

### clean_combined_all_benches.csv

Combined benchmark data from all sources, merged on model name. Models with too few benchmark scores are dropped. This is the primary input to the prediction pipeline.

**Primary key:** `model_name`

**~90 columns** — each prefixed by source to avoid collisions. Missing values are `NaN` (model not in that benchmark).

### Combined Column Descriptions

#### OpenBench columns
| Column | Description |
|--------|-------------|
| `openbench_Reasoning` | Whether the model is a reasoning model (boolean) |
| `openbench_Reasoning Output Tokens Used to Run Artificial Analysis Intelligence Index (million)` | Millions of reasoning tokens used for AA Intelligence Index |
| `openbench_Answer Output Tokens Used to Run Artificial Analysis Intelligence Index (million)` | Millions of output tokens used for AA Intelligence Index |

#### SimpleBench columns
| Column | Description |
|--------|-------------|
| `simplebench_Score (AVG@5)` | Commonsense trick questions — average across 5 attempts |

#### LiveBench columns
| Column | Description |
|--------|-------------|
| `livebench_theory_of_mind` | Reasoning about internal states of other people |
| `livebench_zebra_puzzle` | Einstein's riddles with logical deduction constraints |
| `livebench_spatial` | 2D/3D shape reasoning — intersections, orientations, cuts |
| `livebench_logic_with_navigation` | Logic problem solving + 2D space navigation |
| `livebench_code_generation` | Real-world library usage scenarios |
| `livebench_code_completion` | Completing partial coding solutions |
| `livebench_javascript` | Agentic coding: resolve real JS GitHub issues |
| `livebench_typescript` | Agentic coding: resolve real TS GitHub issues |
| `livebench_python` | Agentic coding: resolve real Python GitHub issues |
| `livebench_integrals_with_game` | Math integrating game theory with integral calculations |
| `livebench_olympiad` | USAMO/IMO competition problems |
| `livebench_consecutive_events` | Ordering of consecutive events |
| `livebench_tablejoin` | Identify valid join columns between overlapping tables |
| `livebench_connections` | NYT-style word grouping puzzle |
| `livebench_plot_unscrambling` | Reorder shuffled movie plot sentences |
| `livebench_typos` | Fix injected typos in arXiv abstracts |
| `livebench_IF Average` | Instruction Following aggregate |

#### Arena columns
| Column | Description |
|--------|-------------|
| `lmarena_Score` | Style-controlled Arena ELO — adjusted for human length/formatting bias (**primary prediction target**) |
| `lmsys_Score` | Raw Arena ELO from blind A/B tests (excluded from clean CSV as leakage — derived from same Arena voting as lmarena) |

#### Lechmazur columns
| Column | Description |
|--------|-------------|
| `lechmazur_confab_Confab %` | Frequency of confabulated/hallucinated answers to misleading questions |
| `lechmazur_gen_Avg Rank` | Generalization: infer a narrow theme from examples + anti-examples |
| `lechmazur_nytcon_Score %` | NYT Connections puzzles with added difficulty |

#### ContextArena columns
| Column | Description |
|--------|-------------|
| `contextarena_8k (%) 2 needles` | MRCR long-context retrieval: 2 needles in 8k-token conversation |

#### EQ-Bench columns
| Column | Description |
|--------|-------------|
| `eqbench_eq_elo` | EQ-Bench 3 ELO: emotional intelligence via role-plays |

#### ARC columns
| Column | Description |
|--------|-------------|
| `arc_ARC-AGI-1` | Fluid intelligence via novel grid-based puzzles |
| `arc_ARC-AGI-2` | Same + complex compositional reasoning |

#### Artificial Analysis columns
| Column | Description |
|--------|-------------|
| `aa_pricing_price_1m_input_tokens` | Price per 1M input tokens |
| `aa_pricing_price_1m_output_tokens` | Price per 1M output tokens |
| `aa_eval_aime_25` | 2025 AIME math competition problems |
| `aa_eval_gpqa` | PhD-level "Google-proof" multiple-choice science questions |
| `aa_eval_hle` | Hard questions replacing saturated MMLU |
| `aa_eval_ifbench` | Instruction following with formatting constraints |
| `aa_eval_lcr` | Hard text-based questions requiring reasoning across ~100k-token documents |
| `aa_eval_livecodebench` | Contamination-free coding from recent contests |
| `aa_eval_mmlu_pro` | Enhanced MMLU: 10 choices, reasoning-heavy |
| `aa_eval_scicode` | Code for research problems in physics, chemistry, biology, math |
| `aa_eval_tau2` | Agentic tool-using conversations |
| `aa_eval_terminalbench_hard` | Operate in a real Linux terminal |
| `aagdpval_ELO` | AA GDP-value ELO rating |
| `aaomniscience_OmniscienceAccuracy` | AA omniscience accuracy |
| `aaomniscience_OmniscienceIndex` | AA omniscience index |
| `aacritpt_CritPtScore` | AA critical point score |

#### WeirdML columns
| Column | Description |
|--------|-------------|
| `weirdml_splash_hard_acc` | Unusual ML classification tasks |
| `weirdml_code_len_p10` | Code length 10th percentile |
| `weirdml_avg_acc` | Overall accuracy across WeirdML tasks |

#### Yupp columns
| Column | Description |
|--------|-------------|
| `yupp_Text_Score` | VIBE score from user preferences (text tasks) |
| `yupp_Coding_Score` | VIBE score for coding tasks |

#### UGI columns
| Column | Description |
|--------|-------------|
| `ugileaderboard_Writing` | Writing quality: intelligence, style, repetition, length adherence |

#### Soothsayer Style columns
| Column | Description |
|--------|-------------|
| `style_normalized_length` | Response length to a set of prompts |
| `style_log_normalized_length` | Log-transformed normalized_length |
| `style_normalized_header_count` | Header count in responses |
| `style_normalized_bold_count` | Bold text count in responses |
| `style_normalized_list_count` | List item count in responses |
| `style_predicted_delta` | Predicted difference between lmarena and lmsys scores based on style |
| `style_cv_*` | Coefficient of variation for each style metric |
| `style_min_*` | Minimum value for each style metric |
| `style_frac_used_*` | Fraction of responses that used each formatting element |
| `style_q7_*` | Style metrics for question 7 specifically |

#### Soothsayer Logic columns
| Column | Description |
|--------|-------------|
| `logic_accuracy` | Raw accuracy on in-house commonsense question set |
| `logic_weighted_accuracy` | ExtraTrees prediction of SimpleBench score from per-question results |
| `logic_PC2` through `logic_PC4` | Principal components of per-question scores (PC1 dropped due to collinearity) |
| `logic_physics_acc` | Accuracy on physics/applied math questions |
| `logic_trick_acc` | Accuracy on lateral thinking questions |
| `logic_avg_reasoning_tokens` | Average reasoning tokens per response |
| `logic_avg_answer_tokens` | Average answer tokens per response |

#### Soothsayer Tone columns
| Column | Description |
|--------|-------------|
| `tone_* TrueSkill` | Response quality TrueSkill rating by named LLM judge |

#### Soothsayer Writing columns
| Column | Description |
|--------|-------------|
| `writing_*_score` | Story quality scored via rubric by named LLM judge |
| `writing_* TrueSkill` | TrueSkill rating from pairwise comparisons by named judge |

#### Soothsayer EQ columns
| Column | Description |
|--------|-------------|
| `eq_* TrueSkill` | EQ TrueSkill rating by named LLM judge |

---

## Output Data

All outputs are written to a timestamped subdirectory under `analysis_output/` (relative to where `predict.py` is run, typically `arena_predictor/analysis_output/`).

### imputed_full.csv

Complete imputed benchmark matrix.

| Column | Type | Description |
|--------|------|-------------|
| `model_name` | string | Model identifier |
| All benchmark columns | float | Imputed score (original if observed, imputed if missing) |

### predictions_best_model.csv

Final ELO predictions with confidence intervals.

| Column | Type | Description |
|--------|------|-------------|
| `model_name` | string | Model identifier |
| `predicted_score` | float | Predicted Arena ELO |
| `actual_score` | float | Actual ELO (if known) |
| `sigma_hat` | float | Estimated prediction uncertainty |
| `lower_bound` | float | Lower prediction interval |
| `upper_bound` | float | Upper prediction interval |
| `num_one_prob` | float | Predicted probability of being #1 |
| `top_by_margin_prob` | float | Probability of being top by margin |

### Other Output Files

| File | Description |
|------|-------------|
| `oof_predictions.csv` | Out-of-fold predictions for all training models |
| `feature_ranking_gain.csv` | Features ranked by tree-based importance |
| `feature_matrix_used.csv` | Feature matrix fed to the final models |
| `run_config.json` | Full configuration used for this run |
| `imputation_quality_per_cell.csv` | Per-cell imputation quality metrics |
| `imputation_quality_per_column.csv` | Per-column imputation quality metrics |
| `best_model_variance_contributions.csv` | Per-feature variance contributions |
| `alt_model_variance_contributions.csv` | Per-feature variance contributions (ALT model) |
| `conformal_diagnostics.csv` | Conformal interval calibration diagnostics |
| `model_eval_rmse.csv` | Per-model CV RMSE comparison |
| `column_dependency_graph.json` | Column dependency structure |
| `metadata.json` | Run metadata including OOF RMSE |

---

## Column Naming Conventions

### Prefix Conventions

| Prefix | Source |
|--------|--------|
| `openbench_` | OpenBench suite |
| `livebench_` | LiveBench benchmark |
| `lmsys_` | Chatbot Arena |
| `lmarena_` | Length-adjusted Arena |
| `simplebench_` | SimpleBench |
| `lechmazur_` | Lechmazur benchmarks |
| `aiderbench_` | Aider coding benchmark |
| `contextarena_` | ContextArena long-context |
| `eqbench_` | EQ-Bench (external) |
| `arc_` | ARC Prize |
| `aa_eval_`, `aa_pricing_` | Artificial Analysis |
| `aagdpval_`, `aaomniscience_`, `aacritpt_` | AA sub-benchmarks |
| `weirdml_` | WeirdML |
| `yupp_` | Yupp leaderboard |
| `ugileaderboard_` | UGI leaderboard |
| `style_` | Soothsayer Style |
| `logic_` | Soothsayer Logic |
| `tone_` | Soothsayer Tone |
| `writing_` | Soothsayer Writing |
| `eq_` | Soothsayer EQ |

### Data Types

| Type | Range | Missing Representation |
|------|-------|------------------------|
| ELO ratings | ~800–2000 | `NaN` |
| Percentage scores | 0–100 | `NaN` |
| Pricing | ≥0 (float) | `NaN` |
| Boolean | 0/1 | `NaN` |
| TrueSkill ratings | ~15–35 | `NaN` |
