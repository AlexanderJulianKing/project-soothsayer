#!/usr/bin/env bash
# ALT regressor experiments with style-restricted interactions
# Test whether better ALT prediction helps now that stage-2 is restricted
# Baseline: bayes ALT + style-restricted = 19.16

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
BASE_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --eb_parent --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt --top_tier_boost 2 --top_tier_threshold 1400"
CV_FLAGS="--cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PARALLEL_FLAGS="--max_workers 16"

run_exp() {
    local exp_id=$1 name=$2 flags=$3
    local out_root="analysis_output/_alt_reg_exp${exp_id}"
    mkdir -p "$out_root"
    echo "Starting exp $exp_id ($name)..."
    python3 -u predict.py $BASE_FLAGS $CV_FLAGS $PARALLEL_FLAGS \
        --output_root "$out_root" $flags > "/tmp/alt_reg_exp${exp_id}.log" 2>&1
    local rmse
    rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")

    python3 -c "
import pandas as pd, numpy as np, glob
fs = sorted(glob.glob('${out_root}/output_*/oof_predictions.csv'))
df = pd.read_csv(fs[-1])
top = df.nlargest(50, 'actual_score')
top50 = np.sqrt(((top.oof_predicted_score - top.actual_score)**2).mean())
fs2 = sorted(glob.glob('${out_root}/output_*/predictions_best_model.csv'))
pred = pd.read_csv(fs2[-1])
gpt = pred[pred['model_name'] == 'GPT-5.4 Thinking']
gpt_pred = f'{gpt.iloc[0][\"predicted_score\"]:.1f}' if len(gpt) else '?'
print(f'  top50={top50:.2f}  GPT-5.4={gpt_pred}')
"
    # Get ALT RMSE from log
    alt_rmse=$(grep "ALT nested-CV RMSE" "/tmp/alt_reg_exp${exp_id}.log" | head -1 | grep -oP '[\d.]+' | head -1)
    echo "Exp $exp_id ($name): RMSE=$rmse  ALT_RMSE=${alt_rmse:-?}"
}

# Run 3 in parallel
run_exp 1 "bayes_control" "--alt_regressor bayes" &
run_exp 2 "lgbm" "--alt_regressor lgbm" &
run_exp 3 "stack" "--alt_regressor stack" &
wait

echo ""
echo "=== SUMMARY ==="
echo "Baseline (style-restricted, bayes ALT): RMSE=19.16, top50=15.25, GPT-5.4=1467.9"
