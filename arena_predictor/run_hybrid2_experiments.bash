#!/usr/bin/env bash
# Hybrid2: CD ALT + greedy residual with poly option, sweep GR 1-5
# New baseline: CD ALT + PCA-10 + GR5 = 19.20 (hybrid_exp3)
# Also: CD ALT + PCA-10 + GR1-5 (without poly, to isolate GR count effect)

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
BASE_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --no_residual_head --no_traj_in_alt --eb_parent"
CV_FLAGS="--top_k_loo 50 --cv_repeats_inner 1 --feature_cv_repeats 1 --alt_cv_repeats 1"
CD_FLAGS="--cross_domain_alt --cd_min_obs 50"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
PARALLEL_FLAGS="--max_workers 16"

run_exp() {
    local exp_id=$1 name=$2 flags=$3
    local out_root="analysis_output/_hybrid2_exp${exp_id}"
    mkdir -p "$out_root"
    python3 -u predict.py $BASE_FLAGS $CV_FLAGS $PARALLEL_FLAGS \
        --output_root "$out_root" $flags > "/tmp/hybrid2_exp${exp_id}.log" 2>&1
    local rmse
    rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
    echo "Exp $exp_id ($name): RMSE=$rmse"
}

echo "=== Hybrid2: GR with poly + GR count sweep ==="
echo ""

# Wave 1: CD ALT + GR1-3 with poly (no PCA)
echo "--- Wave 1: CD ALT + GR_poly (no PCA) ---"
run_exp 1  "CD_GR1_poly"  "$CD_FLAGS --greedy_residual_k 1 --greedy_residual_poly" &
run_exp 2  "CD_GR2_poly"  "$CD_FLAGS --greedy_residual_k 2 --greedy_residual_poly" &
run_exp 3  "CD_GR3_poly"  "$CD_FLAGS --greedy_residual_k 3 --greedy_residual_poly" &
wait
echo ""

# Wave 2: CD ALT + GR4-5 with poly (no PCA) + CD ALT + PCA-10 + GR1
echo "--- Wave 2: CD ALT + GR_poly (no PCA) + PCA sweep ---"
run_exp 4  "CD_GR4_poly"  "$CD_FLAGS --greedy_residual_k 4 --greedy_residual_poly" &
run_exp 5  "CD_GR5_poly"  "$CD_FLAGS --greedy_residual_k 5 --greedy_residual_poly" &
run_exp 6  "CD_PCA10_GR1" "$CD_FLAGS --pca_in_target 10 --greedy_residual_k 1" &
wait
echo ""

# Wave 3: CD ALT + PCA-10 + GR2-4
echo "--- Wave 3: CD ALT + PCA-10 + GR2-4 ---"
run_exp 7  "CD_PCA10_GR2" "$CD_FLAGS --pca_in_target 10 --greedy_residual_k 2" &
run_exp 8  "CD_PCA10_GR3" "$CD_FLAGS --pca_in_target 10 --greedy_residual_k 3" &
run_exp 9  "CD_PCA10_GR4" "$CD_FLAGS --pca_in_target 10 --greedy_residual_k 4" &
wait
echo ""

# Wave 4: CD ALT + PCA-10 + GR5 (rerun for consistency) + reference
echo "--- Wave 4: PCA-10 + GR5 rerun + ref ---"
run_exp 10 "CD_PCA10_GR5" "$CD_FLAGS --pca_in_target 10 --greedy_residual_k 5" &
wait
echo ""

echo "=== All experiments complete ==="
