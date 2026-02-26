# Golden Test Baselines

This directory contains reference output samples captured from actual benchmark runs.
They serve as regression baselines — if a refactoring changes the output format or
column names, a diff against these files will catch it.

## What's here

| File | Source | What it validates |
|------|--------|-------------------|
| `eqbench_scenario_1.json` | `soothsayer_eq/generated_responses/<model>/scenario_1.json` | Multi-turn response structure (turns, parsed sections, completed flag) |
| `writingbench_prompt_0.txt` | `soothsayer_writing/generated_stories/<model>/prompt_0.txt` | Story generation output (plain text) |
| `simplebench_first5.csv` | `soothsayer_logic/benchmark_results_multi_run.csv` | CSV schema: question_id, prompt, reference_answer, model_name, etc. |
| `style_first5.csv` | `soothsayer_style/model_outputs.csv` | CSV schema: model, code, Q1, Q2, ... |

## How to use

These are **reference files**, not automated test fixtures (yet). After a refactoring
pass, manually diff the current output against these baselines:

```bash
# Example: verify logic benchmark CSV schema hasn't changed
head -6 soothsayer_logic/benchmark_results_multi_run.csv | diff - tests/golden/simplebench_first5.csv
```

## Updating baselines

If a format change is intentional, re-capture the golden files:

```bash
cp soothsayer_eq/generated_responses/<model>/scenario_1.json tests/golden/eqbench_scenario_1.json
head -1 soothsayer_writing/generated_stories/<model>/prompt_0.txt > tests/golden/writingbench_prompt_0.txt
head -6 soothsayer_logic/benchmark_results_multi_run.csv > tests/golden/simplebench_first5.csv
head -6 soothsayer_style/model_outputs.csv > tests/golden/style_first5.csv
```
