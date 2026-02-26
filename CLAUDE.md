# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Project Soothsayer is a multi-benchmark LLM evaluation suite. It runs 4 custom benchmarks (emotional intelligence, creative writing, commonsense reasoning, writing style) against LLMs via the OpenRouter API, then feeds scores into a three-stage prediction pipeline (scrape → combine → predict) that combines them with 13+ public benchmarks to predict Chatbot Arena ELO scores.

## Commands

### Run all 4 benchmarks in parallel
```bash
./run_all_benches.bash                          # Full run with preflight checks
./run_all_benches.bash --skip-preflight         # Skip API smoke tests
./run_all_benches.bash eq writing               # Run only named benchmarks
```

### Run individual benchmarks
```bash
# Soothsayer EQ (emotional intelligence)
cd soothsayer_eq && python3 main.py && python3 super_bench.py

# Soothsayer Writing (creative writing)
cd soothsayer_writing && python3 main.py && python3 super_bench.py

# Soothsayer Logic (commonsense reasoning)
cd soothsayer_logic && python3 collect_and_grade.py && python3 score.py

# Soothsayer Style (writing style analysis)
cd soothsayer_style && python3 collect.py && python3 style_analysis.py && python3 process_analysis.py
```

### super_bench.py options (eq & writing)
```bash
python3 super_bench.py --max-battles 100 --workers 20 --judge "Grok 4 Fast"
# eq also accepts: --scenarios "1,2,6"
# writing also accepts: --prompt-ids "0,3,7"
```

### CLI orchestrator
```bash
python -m core.cli                        # Run all 4 benchmarks
python -m core.cli eq writing             # Run specific benchmarks
python -m core.cli --list-completed       # Show completed models per benchmark
python -m core.cli --skip-preflight       # Skip API checks
```

### Preflight only
```bash
python3 preflight.py    # Validates API access for all untested models + judge models
```

### Prediction pipeline (run from project root)
```bash
./scrape.bash            # Collect benchmark data from all 13+ sources
./combine.bash           # Combine + clean -> clean_combined_all_benches.csv
./predict.sh             # Create venv, install deps, impute + predict Arena ELO
python3 posthoc_suite.py # 12-chart post-hoc analysis
```

### Tests
```bash
pip install -e .                 # Install package in dev mode
python3 -m pytest tests/ -v     # Run test suite (62 tests)
```

### Environment
```bash
cp .env.example .env             # Then edit with your API keys
# Required: OPENROUTER_API_KEY (benchmarks), GEMINI_API_KEY (model name mapping)
# Key deps: pandas, numpy, scikit-learn, trueskill, requests
# Prediction pipeline additionally needs: catboost, lightgbm, xgboost, scipy, statsmodels
```

## Architecture

### Directory Layout

| Directory | Purpose | Key Files |
|---|---|---|
| `core/` | Shared Python package | `llm_client.py`, `trueskill_arena.py`, `benchmark.py`, `cli.py`, `utils.py`, `config.py` |
| `soothsayer_eq/` | Emotional intelligence eval | `main.py` (generate), `super_bench.py` (judge), `benchmark.py` (adapter) |
| `soothsayer_writing/` | Creative writing quality | `main.py` (generate), `super_bench.py` (judge), `benchmark.py` (adapter) |
| `soothsayer_logic/` | Commonsense/trick questions | `collect_and_grade.py` (collect + grade), `score.py` (aggregate), `benchmark.py` (adapter) |
| `soothsayer_style/` | Writing style tendencies | `collect.py` (collect), `style_analysis.py` + `process_analysis.py`, `benchmark.py` (adapter) |
| `scrapers/` | External benchmark scrapers | 12 grabbers (arena, livebench, aider, etc.) + `aa_models_grabber.py` |
| `benchmark_combiner/` | Merge + clean all benchmarks | `combine.py`, `correlations.py`, `benchmarks/`, `mappings/` |
| `arena_predictor/` | Impute + predict Arena ELO | `predict.py`, `column_imputer.py`, `analysis_output/` |
| `docs/` | Documentation | `ARCHITECTURE.md`, `DATA_DICTIONARY.md`, `FINDINGS.md` |
| `tests/` | Test suite | `test_utils.py`, `test_config.py`, `test_trueskill_arena.py` |

### Shared Package (`core/`)

- **`llm_client.py`**: Single API abstraction for all benchmarks. Handles provider routing, reasoning effort mapping, retry logic with exponential backoff. API key via `OPENROUTER_API_KEY` env var.
- **`trueskill_arena.py`**: Shared TrueSkill pairwise comparison engine used by EQ and Writing benchmarks. Handles match selection, battle history, rating computation.
- **`benchmark.py`**: Abstract `Benchmark` base class that all 4 benchmarks implement. Provides `get_completed_models()`, `run_stage()`, `run_all()`.
- **`cli.py`**: CLI orchestrator that runs benchmarks in parallel via `ThreadPoolExecutor`.
- **`utils.py`**: Shared utilities (`get_latest_file`, `load_models`, `discover_openbench_csv`, `extract_json_payload`).
- **`config.py`**: Central `BenchmarkConfig` dataclass with per-benchmark defaults.

Each benchmark dir has a `llm_client.py` shim that re-exports from `core/llm_client.py` for backward compatibility.

### Model Configuration

All benchmarks read from the same dated `openbench_*.csv` (in `benchmark_combiner/benchmarks/`). Key columns: `Model` (display name), `openbench_id` (OpenRouter model ID), `Reasoning` (boolean). Models without an `openbench_id` are skipped. `discover_openbench_csv()` finds the latest file automatically.

### TrueSkill Super Benchmarks (eq & writing)

Both `super_bench.py` files use the shared `TrueSkillArena` engine with benchmark-specific adapters:
- **Paired mode**: Each model pair is compared in both A/B orientations on the same scenario/prompt to cancel position bias
- **Info-gain selection**: Matches are prioritized by `combined_sigma x match_quality` (TrueSkill uncertainty x closeness)
- **Batch loop**: Selects batches of `workers * 2` battles, runs them in parallel, recomputes ratings, repeats until budget exhausted
- **Priority**: Completing pairs (missing reverse orientation) takes precedence over new matchups
- Results: `battle_history.csv` (raw battles), `battle_pairs.csv` (paired aggregates), dated TrueSkill CSV

### Key Patterns

- **Resume/idempotency**: Every benchmark checks existing outputs before running -- eq/writing by directory listing, style/logic by CSV model presence
- **Parallel execution**: `ThreadPoolExecutor` everywhere (10-40 workers depending on benchmark)
- **LLM-as-judge**: Logic and Writing stage 1 use LLMs for answer evaluation; EQ and Writing stage 2 use LLMs for pairwise comparison
- **Logs**: `run_all_benches.bash` writes to `logs/<bench>_YYYYMMDD_HHMMSS.log`

### Data Flow

```
openbench_*.csv (model list)
    |
    v
4 benchmarks run in parallel (each: generate responses -> judge/score)
    |
    v
soothsayer_eq/results/eq_*.csv
soothsayer_writing/results/writing_*.csv
soothsayer_logic/output/logic_*.csv
soothsayer_style/outputs/style_*.csv + tone_*.csv
    |
    v
scrape.bash -> combine.bash (combine.py + correlations.py)
    -> clean_combined_all_benches.csv
    |
    v
predict.sh (predict.py -> impute + predict Arena ELO)
    |
    v
posthoc_suite.py (12-chart analysis)
```
