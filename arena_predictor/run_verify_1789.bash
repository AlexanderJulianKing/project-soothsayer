#!/usr/bin/env bash
# Verify 17.89 RMSE (top-50 LOO) with 10x5-fold repeated K-fold on all 112 models
# Also run top-50 LOO to confirm the original number

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
BASE_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --no_residual_head --no_traj_in_alt --eb_parent --poly_interactions --poly_limit 7"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
PARALLEL_FLAGS="--max_workers 16"

run_exp() {
    local exp_id=$1 name=$2 flags=$3
    local out_root="analysis_output/_verify_exp${exp_id}"
    mkdir -p "$out_root"
    python3 -u predict.py $BASE_FLAGS $PARALLEL_FLAGS \
        --output_root "$out_root" $flags > "/tmp/verify_exp${exp_id}.log" 2>&1
    local rmse
    rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
    echo "Exp $exp_id ($name): RMSE=$rmse"
}

echo "=== Verification: 17.89 RMSE ==="
echo ""

# Wave 1: Full repeated K-fold (all 112 models) + top-50 LOO rerun
echo "--- Wave 1: 10x5-fold + top-50 LOO ---"
run_exp 1 "10x5fold_all112" "--cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1" &
run_exp 2 "top50_LOO_rerun" "--top_k_loo 50 --cv_repeats_inner 1 --feature_cv_repeats 1 --alt_cv_repeats 1" &
wait
echo ""

# Wave 2: Also run the old baseline (without shape features = --poly_interactions --poly_limit 7 should still work)
# and 10x10-fold for extra rigor
echo "--- Wave 2: 10x10-fold ---"
run_exp 3 "10x10fold_all112" "--outer_cv 10 --cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1" &
wait
echo ""

echo "=== All verification experiments complete ==="
