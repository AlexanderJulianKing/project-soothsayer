# Project Soothsayer

Predict [Chatbot Arena](https://lmarena.ai) ELO scores for LLMs before they appear on the leaderboard.

[Chatbot Arena](https://lmarena.ai) (formerly LMSYS) is the most widely cited ranking of LLM quality — it's based on millions of blind head-to-head votes from real users. But new models can take weeks or months to accumulate enough votes for a stable rating. Soothsayer predicts those ratings early by running 4 custom benchmarks against LLMs via the OpenRouter API, combining those scores with 13+ public benchmark sources, and using iterative imputation + regression to predict Arena ELO.

## Benchmarks

| Benchmark | What it measures | Method |
|---|---|---|
| **Soothsayer EQ** | Emotional intelligence | Multi-turn scenario responses, TrueSkill pairwise judging |
| **Soothsayer Writing** | Creative writing quality | Story generation + direct scoring, TrueSkill pairwise judging |
| **Soothsayer Logic** | Commonsense / trick questions | Multi-run answer collection + LLM grading + ML regression |
| **Soothsayer Style** | Writing style tendencies | Style metric extraction + statistical analysis |

Soothsayer EQ and Soothsayer Writing are based on [EQ-Bench](https://eqbench.com) ([GitHub](https://github.com/EQ-bench)) — the scenario format and pairwise judging approach are adapted from their work.

Soothsayer Logic is based on [SimpleBench](https://simple-bench.com). Like SimpleBench, the question set and reference answers are intentionally kept private to prevent benchmark contamination. The evaluation code is included but the questions themselves are not.

## Architecture

```
openbench_*.csv (model list with OpenRouter IDs)
    |
    v
4 benchmarks run in parallel (generate responses -> judge/score)
    |
    v
Per-benchmark result CSVs
    |
    v
scrape.bash         -- scrape 13+ external benchmark sources
combine.bash        -- combine all sources -> clean_combined_all_benches.csv
predict.sh          -- impute missing scores + predict Arena ELO
    |
    v
analysis_output/    -- predictions, imputed matrices, diagnostics
posthoc_suite.py    -- 16-chart post-hoc analysis
```

### Directory Layout

```
core/                    Shared Python package
  llm_client.py            OpenRouter API client (retry, provider routing, reasoning effort)
  trueskill_arena.py       TrueSkill pairwise comparison engine
  benchmark.py             Abstract benchmark interface
  cli.py                   CLI orchestrator
  utils.py                 Shared utilities
  config.py                Central configuration

soothsayer_eq/           Emotional intelligence benchmark
soothsayer_writing/      Creative writing benchmark
soothsayer_logic/        Commonsense reasoning benchmark
soothsayer_style/        Writing style benchmark

scrapers/                External benchmark scrapers (13+ sources)
benchmark_combiner/      Merge + clean all benchmark CSVs
  combine.py               LLM-assisted model name mapping + merge
  correlations.py          Correlation analysis + cleaning
  benchmarks/              Central data store (all benchmark CSVs)
  mappings/                Model name mapping JSONs
arena_predictor/         Impute + predict Arena ELO scores
  predict.py               Orchestrator (feature selection, prediction, intervals)
  column_imputer.py        Per-cell model-bank imputation (default) + per-column specialized

docs/                    Documentation (architecture, data dictionary)
tests/                   Test suite (112 tests)
```

## Setup

```bash
# Clone and install
git clone https://github.com/AlexanderJulianKing/project-soothsayer.git
cd project-soothsayer
pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env with your OpenRouter and Gemini API keys

# Verify setup
python3 -m pytest tests/ -v
```

### Requirements

- Python 3.9+
- OpenRouter API key (for running benchmarks — a full run across all 178 models costs roughly $50-150 in API fees, dominated by frontier model pricing; running only budget/mid-tier models is ~$15-30)
- Gemini API key (for LLM-assisted model name mapping in the combiner)

## Usage

### Run all 4 benchmarks

```bash
./run_all_benches.bash                    # Full run with preflight API checks
./run_all_benches.bash --skip-preflight   # Skip smoke tests
./run_all_benches.bash eq style           # Run only named benchmarks
```

### Run individual benchmarks

```bash
# Soothsayer EQ
cd soothsayer_eq && python3 main.py && python3 super_bench.py

# Soothsayer Writing
cd soothsayer_writing && python3 main.py && python3 super_bench.py

# Soothsayer Logic
cd soothsayer_logic && python3 collect_and_grade.py && python3 score.py

# Soothsayer Style
cd soothsayer_style && python3 collect.py && python3 style_analysis.py && python3 process_analysis.py
```

### Prediction pipeline

```bash
./scrape.bash        # Collect external benchmark data
./combine.bash       # Combine + clean -> clean_combined_all_benches.csv
./predict.sh         # Impute missing scores + predict Arena ELO
python3 posthoc_suite.py  # Generate 16-chart analysis suite
```

### CLI

```bash
python -m core.cli                  # Run all benchmarks
python -m core.cli eq writing       # Run specific benchmarks
python -m core.cli --list-completed # Show progress
python -m core.cli --skip-preflight # Skip API checks
```

## How It Works

### Benchmark Phase

Each benchmark evaluates LLMs on a different capability. EQ and Writing use a two-stage pipeline: first generate responses, then run TrueSkill pairwise comparisons with information-gain-based match selection to efficiently rank models. Logic collects answers across multiple runs and uses ML regression to score. Style extracts writing style metrics and runs statistical analysis.

All benchmarks read from a shared `openbench_*.csv` that maps model display names to OpenRouter model IDs. The benchmarks are resume-safe -- they check existing outputs before running and skip already-evaluated models.

### Prediction Phase

The prediction pipeline (`scrapers/` → `benchmark_combiner/` → `arena_predictor/`) merges scores from the 4 custom benchmarks with 13+ external sources (LiveBench, Artificial Analysis, AiderBench, ARC, etc.) using a three-tier model name mapping system with LLM-assisted fuzzy matching. The mappings require significant human curation and pruning -- the LLM suggestions are a starting point, but incorrect matches (e.g. confusing model versions or sizes) must be manually reviewed and corrected in `benchmark_combiner/mappings/`.

The combined benchmark matrix is sparse -- most models are missing scores from several benchmarks. `ModelBankImputer` fills these gaps by selecting the best available predictors for each missing cell (based on what's actually observed in that row), fitting cached per-cell models, and applying a low-rank coherence projection to keep imputed values consistent across columns. See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for full details on the imputation algorithms.

Finally, `predict.py` trains on models that have Arena scores to predict scores for those that don't, using feature selection, polynomial interactions, and conformal prediction intervals.

A snapshot of benchmark data (as of March 2026) is included in `benchmark_combiner/benchmarks/` so the prediction pipeline can run from a fresh clone without needing to re-scrape.

## License

MIT License. See [LICENSE](LICENSE).
