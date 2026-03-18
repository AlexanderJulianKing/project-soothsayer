#!/usr/bin/env bash
# Style bridge experiments
# Exp 1: Pure style bridge (final model = lmarena + style only)
# Exp 2: Current best (lmarena restricted interactions, all features in final)
# Baseline: 19.16 RMSE (lmarena style-restricted interactions)

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
BASE_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --eb_parent --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt --top_tier_boost 2 --top_tier_threshold 1400"
CV_FLAGS="--cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PARALLEL_FLAGS="--max_workers 16"

report() {
    local out_root=$1 name=$2
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
gpt_pred = gpt.iloc[0]['predicted_score'] if len(gpt) else '?'
fs3 = sorted(glob.glob('${out_root}/output_*/best_model_feature_importance.csv'))
fi = pd.read_csv(fs3[-1])
fi_sorted = fi.reindex(fi['importance'].abs().sort_values(ascending=False).index)
print(f'  RMSE={${rmse}}  top50={top50:.2f}  GPT-5.4={gpt_pred}')
print(f'  Top 5 features:')
for _, r in fi_sorted.head(5).iterrows():
    print(f'    {r[\"feature\"]:55s} imp={r[\"importance\"]:+.3f}')
"
    echo "$name: RMSE=$rmse"
}

run_exp() {
    local exp_id=$1 name=$2 flags=$3
    local out_root="analysis_output/_style_bridge_exp${exp_id}"
    mkdir -p "$out_root"
    echo "Starting exp $exp_id ($name)..."
    python3 -u predict.py $BASE_FLAGS $CV_FLAGS $PARALLEL_FLAGS \
        --output_root "$out_root" $flags > "/tmp/style_bridge_exp${exp_id}.log" 2>&1
    echo ""
    echo "=== Exp $exp_id: $name ==="
    report "$out_root" "$name"
}

# Run in parallel
run_exp 1 "pure_style_bridge" "--style_only_final" &
run_exp 2 "lmarena_restricted_control" "" &
wait

echo ""
echo "=== SUMMARY ==="
echo "Previous baseline (unrestricted): RMSE=20.11, GPT-5.4=1536.3"
echo "lmarena style-restricted (last run): RMSE=19.16, GPT-5.4=1467.9"
