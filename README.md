# Project Soothsayer

Predict [Arena](https://arena.ai) ELO scores for LLMs before they appear on the leaderboard.

**[→ Visual walkthrough: how it works](https://alexanderjulianking.github.io/soothsayer_explainer.html)**

[Arena](https://arena.ai) (formerly LMSYS / Chatbot Arena / lmarena.ai) is the most widely cited ranking of LLM quality — it's based on millions of blind head-to-head votes from real users. But new models can take weeks or months to accumulate enough votes for a stable rating. Soothsayer predicts those ratings early by running 4 custom benchmarks against LLMs via the OpenRouter API, then combining those scores with the 17 public/input benchmark families and 6 custom/derived CSV families consumed by the shipped predictor pipeline.

## Performance

Out-of-fold RMSE on n=127 labeled models (10× 5-fold CV, target = style-controlled Arena ELO):

| Method | RMSE | R² | Spearman ρ |
|---|---:|---:|---:|
| Predict mean (dummy) | 56.3 | 0.00 | — |
| Public benchmarks + median impute | 35.6 | 0.66 | 0.81 |
| All benchmarks + median impute | 25.7 | 0.82 | 0.90 |
| Full Soothsayer pipeline | **13.61** | **0.941** | **0.971** |

Honest walk-forward on the 23 newest models (re-fits imputation, PCA-32, and PLS-3 every step): RMSE **14.69**, R² **0.900**, Spearman ρ **0.940**.

Reproduce with `arena_predictor/baseline_comparison_lmarena.py` (first three rows) and `predict.sh` (final row).

## Benchmarks

| Benchmark | What it measures | Method |
|---|---|---|
| **Soothsayer EQ** | Emotional intelligence | Multi-turn scenario responses, TrueSkill pairwise judging |
| **Soothsayer Writing** | Creative writing quality | Story generation + direct scoring, TrueSkill pairwise judging |
| **Soothsayer Logic** | Commonsense / trick questions | Multi-run answer collection + LLM grading + ML regression |
| **Soothsayer Style** | Writing style + response voice | Response collection, pairwise tone judging, style metric extraction |

Soothsayer EQ and Soothsayer Writing are based on [EQ-Bench](https://eqbench.com) ([GitHub](https://github.com/EQ-bench)) — the scenario format and pairwise judging approach are adapted from their work.

Soothsayer Logic is based on [SimpleBench](https://simple-bench.com). The current repo includes the benchmark questions and reference answers in [`soothsayer_logic/questions.json`](/Users/alexanderking/Desktop/random_stuff/project_soothsayer/soothsayer_logic/questions.json). The task is still contamination-sensitive, so comparisons across forks or modified question sets should be treated carefully.

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
scrape.bash         -- run 11 active external scrapers (repo contains 12 grabber scripts)
combine.bash        -- combine 17 public/input + 6 custom/derived CSV families -> clean_combined_all_benches[_with_sem_v4_d32].csv
predict.sh          -- impute missing scores + predict Arena ELO
    |
    v
arena_predictor/analysis_output/ -- predictions, imputed matrices, diagnostics
posthoc_suite.py    -- 15-section post-hoc suite (some sections emit multiple figures)
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

scrapers/                External benchmark scrapers (12 grabber scripts; scrape.bash currently invokes 11)
benchmark_combiner/      Merge + clean all benchmark CSVs
  combine.py               LLM-assisted model name mapping + merge (23 CSV patterns: 17 public/input + 6 custom/derived)
  correlations.py          Correlation analysis + cleaning
  benchmarks/              Central data store (all benchmark CSVs)
  mappings/                Model name mapping JSONs
arena_predictor/         Impute + predict Arena ELO scores
  predict.py               Orchestrator (feature selection, prediction, intervals)
  column_imputer.py        Per-cell model-bank imputation (default) + per-column specialized

docs/                    Documentation (architecture, data dictionary)
tests/                   Test suite (112 current collected test cases)
```

## Setup

```bash
# Clone and install
git clone https://github.com/AlexanderJulianKing/project-soothsayer.git
cd project-soothsayer
pip install -e ".[benchmarks,predict,dev]"
pip install google-genai matplotlib seaborn adjustText factor-analyzer

# Optional: response-embedding branch used by combine.bash when available
pip install sentence-transformers torch

# Configure API keys
cp .env.example .env
# Edit .env with your OpenRouter and Gemini API keys
# Optional: add ANTHROPIC_API_KEY for native Claude 4.6/4.7 routing

# Verify setup
python3 -m pytest tests/ -v
```

### Requirements

- Python 3.9+
- OpenRouter API key (for running benchmarks — full-run API cost depends on the active OpenBench roster and chosen models)
- Gemini API key (for LLM-assisted model name mapping in the combiner)
- `google-genai`, `matplotlib`, `seaborn`, `adjustText`, and `factor-analyzer` for the full documented combine/post-hoc workflow
- Optional: `sentence-transformers` and `torch` for the sem-feature branch in `combine.bash`
- Optional: `ANTHROPIC_API_KEY` for direct Claude 4.6/4.7 routing in `core/llm_client.py`

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
cd soothsayer_style && python3 collect.py && python3 super_bench.py && python3 score.py
```

### Prediction pipeline

```bash
./scrape.bash        # Run 11 active external scrapers, plus collect latest custom results
./combine.bash       # Combine + clean -> clean_combined_all_benches[_with_sem_v4_d32].csv
./predict.sh         # Impute missing scores + predict Arena ELO
python3 posthoc_suite.py  # Generate the 15-section post-hoc suite
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

Each benchmark evaluates LLMs on a different capability. EQ and Writing use a two-stage pipeline: first generate responses, then run TrueSkill pairwise comparisons with information-gain-based match selection to efficiently rank models. Logic collects answers across multiple runs and uses ML regression to score. Style collects responses, runs pairwise tone judging, then aggregates style metrics and tone outputs via `score.py`.

All benchmarks read from a shared `openbench_*.csv` that maps model display names to OpenRouter model IDs. The benchmarks are resume-safe -- they check existing outputs before running and skip already-evaluated models.

### Prediction Phase

The prediction pipeline (`scrapers/` → `benchmark_combiner/` → `arena_predictor/`) merges scores from the 4 custom benchmarks plus `eq_multiturn_*.csv` behavioral features with 17 public/input benchmark families (LiveBench, Artificial Analysis, AiderBench, ARC, etc.) using per-source model name mapping JSONs with LLM-assisted suggestions for unmapped models. The repo contains 12 scraper scripts; the current `scrape.bash` enables 11 of them, and `combine.py` consumes 23 CSV patterns total (17 public/input + 6 custom/derived). The mappings require significant human curation and pruning -- the LLM suggestions are a starting point, but incorrect matches (e.g. confusing model versions or sizes) must be manually reviewed and corrected in `benchmark_combiner/mappings/`.

The combined benchmark matrix is sparse -- most models are missing scores from several benchmarks. `ModelBankImputer` fills these gaps by selecting the best available predictors for each missing cell (based on what's actually observed in that row), fitting cached per-cell models, and applying a low-rank coherence projection to keep imputed values consistent across columns. See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for full details on the imputation algorithms.

Finally, `predict.py` trains on models that have Arena scores to predict scores for those that don't, using fold-internal PLS-3 supervision appended to the KNN feature matrix, adaptive-neighborhood kernel Ridge with jackknife variance inflation, and grouped conformal prediction intervals.

A snapshot of benchmark data (as of April 2026 in the current tree) is included in `benchmark_combiner/benchmarks/` so the prediction pipeline can run from a fresh clone without needing to re-scrape.

## License

MIT License. See [LICENSE](LICENSE).
