#!/usr/bin/env bash
# Top-tier optimization experiments
# Baseline: 20.11 RMSE (with top_tier_boost=2, threshold=1400)

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
BASE_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --eb_parent --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt"
CV_FLAGS="--cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PARALLEL_FLAGS="--max_workers 16"

run_exp() {
    local exp_id=$1 name=$2 flags=$3
    local out_root="analysis_output/_top_exp${exp_id}"
    mkdir -p "$out_root"
    echo "Starting exp $exp_id ($name)..."
    python3 -u predict.py $BASE_FLAGS $CV_FLAGS $PARALLEL_FLAGS \
        --output_root "$out_root" $flags > "/tmp/top_exp${exp_id}.log" 2>&1
    local rmse
    rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
    echo "Exp $exp_id ($name): RMSE=$rmse"
}

# Run 3 experiments in parallel (48 cores / 16 workers = 3 slots)

# Exp 1: Continuous ELO weighting (replaces binary boost)
run_exp 1 "elo_weight" "--elo_weight --elo_weight_center 1300 --elo_weight_scale 100" &

# Exp 2: ELO weight + binary boost combined
run_exp 2 "elo_weight+boost" "--elo_weight --elo_weight_center 1300 --elo_weight_scale 100 --top_tier_boost 2 --top_tier_threshold 1400" &

# Exp 3: Top correction (with binary boost baseline)
run_exp 3 "top_correction+boost" "--top_tier_boost 2 --top_tier_threshold 1400 --top_correction --top_correction_n 50" &

wait
echo ""
echo "=== ALL DONE ==="

# Print comparison
echo ""
echo "=== RESULTS SUMMARY ==="
for i in 1 2 3; do
    out_root="analysis_output/_top_exp${i}"
    rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))" 2>/dev/null || echo "FAILED")

    # Get top-50 RMSE
    top50=$(python3 -c "
import pandas as pd, numpy as np, glob
fs = sorted(glob.glob('${out_root}/output_*/oof_predictions.csv'))
if fs:
    df = pd.read_csv(fs[-1])
    top = df.nlargest(50, 'actual_score')
    print(f'{np.sqrt(((top.oof_predicted_score - top.actual_score)**2).mean()):.2f}')
else:
    print('?')
" 2>/dev/null || echo "?")

    name=$([ $i -eq 1 ] && echo "elo_weight" || ([ $i -eq 2 ] && echo "elo_weight+boost" || echo "top_correction+boost"))
    echo "Exp $i ($name): overall=$rmse  top50=$top50"
done
echo "Baseline: overall=20.11  top50=15.10"
