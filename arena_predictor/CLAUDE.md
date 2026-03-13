# Arena Predictor — Autonomous Experiment Framework

## Current State
- **Best RMSE: 21.48** (with `--eb_parent` flag)
- **Training size: n=90** labeled models with Arena ELO scores
- **54 experiments** tried across 9 cycles; only 1 winner (EB parent, -0.22 RMSE)
- Full history in `../docs/FINDINGS.md`

## How Experiments Work

Each experiment adds a new flag to `predict.py` that can be toggled on/off independently.

### Step 1: Add Global Flag (~line 930 area, near other experiment flags)
```python
MY_EXPERIMENT_ENABLED = False
```

### Step 2: Add Argparse Argument (in the argparse section)
```python
parser.add_argument('--my_experiment', action='store_true',
                    help='Enable my experiment')
```

### Step 3: Wire in main() (in the main function where other flags are wired)
```python
global MY_EXPERIMENT_ENABLED
MY_EXPERIMENT_ENABLED = args.my_experiment
```

### Step 4: Implement the Logic

**CRITICAL LEAKAGE RULES:**
- ALL features that depend on target values or ALT predictions MUST be computed INSIDE `_precompute_single_fold()` to prevent data leakage
- The fold provides: `Xtr_df` (train features), `Xva_df` (val features), `y[tr]` (train targets), `y[va]` (val targets), `tr`/`va` indices
- NEVER compute features globally (before the CV loop) that use `y` (target) or full-data ALT predictions
- **Leakage test:** If RMSE drops > 1.0 points, it's almost certainly leakage
- Feature transforms applied to all features (like scaling) can be done pre-CV if they don't use target/ALT
- But anything that creates NEW features from target or ALT must be fold-internal

**Where to add fold-internal code:**
- In `_precompute_single_fold()` — this function receives `Xtr_df, Xva_df, y, tr, va` and runs inside each CV fold
- Look for the pattern `if SOME_FLAG_ENABLED:` blocks already in this function for examples
- You can also modify `fit_and_predict_all_with_alt()` if you need to change the final model path

**Where to add pre-CV transforms (safe transforms that don't use target/ALT):**
- In the feature pipeline, after imputation but before the CV loop
- Look for where `safe_features` is constructed

### Step 5: Create Bash Script

Template for `run_cycle{N}_experiments.bash`:
```bash
#!/usr/bin/env bash
# Cycle N: Description
# Baseline: EB parent ON (21.48)
# 3 experiments at a time on 48-core tower

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
BASE_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt --eb_parent"
CV_FLAGS="--cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
PARALLEL_FLAGS="--max_workers 16"

run_exp() {
    local exp_id=$1 name=$2 flags=$3
    local out_root="analysis_output/_cN_exp${exp_id}"
    mkdir -p "$out_root"
    python3 -u predict.py $BASE_FLAGS $CV_FLAGS $PARALLEL_FLAGS \
        --output_root "$out_root" $flags > "/tmp/cycleN_exp${exp_id}.log" 2>&1
    local rmse
    rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
    echo "Exp $exp_id ($name): RMSE=$rmse"
}

# Run 3 at a time (48 cores / 16 workers = 3 slots)
run_exp 1 "Name1" "--flag1" &
run_exp 2 "Name2" "--flag2" &
run_exp 3 "Name3" "--flag3" &
wait

run_exp 4 "Name4" "--flag4" &
run_exp 5 "Name5" "--flag5" &
run_exp 6 "Name6" "--flag6" &
wait
```

## What Has Been Tried (Categories)

Read `../docs/FINDINGS.md` for the full list. Key categories already explored:
- **Imputation:** SVD, specialized, model bank, coherence (lambda, shape, rank penalty, capping)
- **Feature engineering:** trajectory, delta head (LEAKED), pairwise rank (LEAKED), LOBO residuals, alias archaeology, sigma2 features
- **Feature transforms:** quantile normalize (hurt), reliability-weighted PCA (neutral), orthogonalize to ALT (hurt)
- **Regularization:** EB parent (WINNER, -0.22), EB provider (neutral), prediction shrinkage (hurt)
- **Semi-supervised:** self-training with pseudo-labels (hurt)
- **Model variants:** partial-linear ALT (neutral), target-aware coherence (neutral), pairwise anchor head (hurt)

## Key Findings
- The 21.48 wall is real at n=90. Only EB parent shrinkage ever improved.
- Coherence projection is the key innovation (-0.72 RMSE)
- Trajectory in target helps (-0.48 RMSE), trajectory in ALT is irrelevant
- Quantile transforms DESTROY signal (+3.06) because raw scales carry distance info
- ALT-correlated features are useful, not redundant (orthogonalizing hurts)
- Self-training with pseudo-labels from a 21.48 model is too noisy (+0.57)
- Alias features (model name parsing) create spurious correlations at n=90 (+2.87)
- Any feature computed globally (before CV) that uses target or ALT WILL leak

## Using Codex for Brainstorming

Use the `codex_start` MCP tool with `model="gpt-5.4"`. Share the experiment history and ask for novel ideas. Example:

```
codex_start(
    prompt="I'm trying to predict Chatbot Arena ELO from benchmark scores. Current best RMSE=21.48 with n=90 training samples. Here's what's been tried: [summary]. Give me 3 novel experiment ideas that haven't been tried.",
    model="gpt-5.4"
)
```
