#!/usr/bin/env bash
# Cycle 6: ALT feature representation experiments
# Baseline: EB parent ON, PCA(10) feature mode (21.48)
# Runs 3 experiments in parallel on 48-core tower

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
BASE_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt --eb_parent"
CV_FLAGS="--cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1"

# Parallel safety: cap threads per instance, isolate output dirs
# Hard cap: 16 threads total per instance (3 instances × 16 = 48 cores)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
PARALLEL_FLAGS="--max_workers 16"

run_exp() {
    local exp_id=$1 name=$2 flags=$3
    local out_root="analysis_output/_exp${exp_id}"
    mkdir -p "$out_root"
    python3 -u predict.py $BASE_FLAGS $CV_FLAGS $PARALLEL_FLAGS \
        --output_root "$out_root" $flags > "/tmp/cycle6_exp${exp_id}.log" 2>&1
    local rmse
    rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
    echo "Exp $exp_id ($name): RMSE=$rmse"
}

echo "=== Cycle 6: ALT feature representation ==="
echo "Started: $(date)"

# Batch 1: Exp 0, 1, 2 in parallel
echo ""
echo "--- Batch 1: Exp 0 (PCA baseline), 1 (Raw), 2 (FA) ---"
run_exp 0 "PCA-10 baseline"   "--alt_feature_mode pca" &
run_exp 1 "Raw columns"       "--alt_feature_mode raw" &
run_exp 2 "Factor Analysis"   "--alt_feature_mode fa"  &
wait
echo "Batch 1 done: $(date)"

# Batch 2: Exp 3, 4, 5 in parallel
echo ""
echo "--- Batch 2: Exp 3 (Hybrid), 4 (Raw no int), 5 (FA no int) ---"
run_exp 3 "Hybrid top-15+PCA-5"  "--alt_feature_mode hybrid"                          &
run_exp 4 "Raw no interactions"   "--alt_feature_mode raw --alt_interaction_max_pairs 0" &
run_exp 5 "FA no interactions"    "--alt_feature_mode fa --alt_interaction_max_pairs 0"  &
wait
echo "Batch 2 done: $(date)"

# Collect results
echo ""
echo "=== CYCLE 6 RESULTS ==="
for i in 0 1 2 3 4 5; do
    grep "^Exp $i " /tmp/cycle6_exp${i}.log 2>/dev/null || \
    python3 -c "
import json,glob
fs=sorted(glob.glob('analysis_output/_exp${i}/output_*/metadata.json'))
if fs:
    d=json.load(open(fs[-1]))
    print('Exp $i: RMSE=' + str(d.get('oof_rmse','?')))
else:
    print('Exp $i: NO RESULT')
"
done
echo ""
echo "Finished: $(date)"
