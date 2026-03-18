# Golden Test Baselines

This directory contains reference output samples captured from actual benchmark runs.
They serve as regression baselines — if a refactoring changes the output format or
column names, a diff against these files will catch it.

## What's here

| File | Source | What it validates |
|------|--------|-------------------|
| `eqbench_scenario_1.json` | `soothsayer_eq/generated_responses/<model>/scenario_1.json` | Multi-turn response structure (turns, parsed sections, completed flag) |
| `writingbench_prompt_0.txt` | `soothsayer_writing/generated_stories/<model>/prompt_0.txt` | Story generation output (plain text) |
| `style_first5.csv` | `soothsayer_style/outputs/style_*.csv` | CSV schema: model, style metrics |

## Updating baselines

If a format change is intentional, re-capture the golden files:

```bash
cp soothsayer_eq/generated_responses/<model>/scenario_1.json tests/golden/eqbench_scenario_1.json
head -1 soothsayer_writing/generated_stories/<model>/prompt_0.txt > tests/golden/writingbench_prompt_0.txt
```
