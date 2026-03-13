#!/usr/bin/env bash
# Cycle 10: Feature selection stability + top-tier boost
# Baseline: EB parent ON (21.48), feature_selector=lgbm (default)
# Diagnosis: tree-based feature selection is unstable across LOO folds,
# causing tail shrinkage calibration to fail. Two interventions:
# 1) Skip feature selection entirely, let ARD handle sparsity
# 2) Duplicate top-tier training rows to counteract shrinkage

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
    local exp_id=$1 name=$2; shift 2; local flags="$*"
    local out_root="analysis_output/_c10_exp${exp_id}"
    mkdir -p "$out_root"
    python3 -u predict.py $BASE_FLAGS $CV_FLAGS $PARALLEL_FLAGS \
        --output_root "$out_root" $flags > "/tmp/cycle10_exp${exp_id}.log" 2>&1
    local rmse
    rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
    echo "Exp $exp_id ($name): RMSE=$rmse"
}

echo "=== Cycle 10: Feature Selection Stability + Top-Tier Boost ==="

# Wave 1: Standalone interventions
echo "--- Wave 1: Individual interventions ---"
run_exp 1 "no_feat_sel"         --feature_selector none &
run_exp 2 "boost_2x"            --top_tier_boost 2 --top_tier_threshold 1450 &
run_exp 3 "boost_3x"            --top_tier_boost 3 --top_tier_threshold 1450 &
wait
echo ""

# Wave 2: Combined + variants
echo "--- Wave 2: Combined + threshold variants ---"
run_exp 4 "no_sel_boost_2x"     --feature_selector none --top_tier_boost 2 --top_tier_threshold 1450 &
run_exp 5 "no_sel_boost_3x"     --feature_selector none --top_tier_boost 3 --top_tier_threshold 1450 &
run_exp 6 "boost_2x_thr1400"    --top_tier_boost 2 --top_tier_threshold 1400 &
wait
echo ""

echo "=== Cycle 10 complete ==="
