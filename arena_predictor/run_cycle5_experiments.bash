#!/usr/bin/env bash
# Cycle 5: Problem framing and regularization experiments
# Baseline: EB parent ON (21.48)
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
        --output_root "$out_root" $flags > "/tmp/cycle5_exp${exp_id}.log" 2>&1
    local rmse
    rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
    echo "Exp $exp_id ($name): RMSE=$rmse"
}

echo "=== Cycle 5: Problem framing and regularization ==="
echo "Started: $(date)"

# Batch 1: Exp 0, 1, 2 in parallel
echo ""
echo "--- Batch 1: Exp 0 (Baseline), 1 (No weighting), 2 (Weight power=4) ---"
run_exp 0 "Baseline"                ""                           &
run_exp 1 "No completeness weight"  "--completeness_weight_power 0" &
run_exp 2 "Weight power=4"         "--completeness_weight_power 4" &
wait
echo "Batch 1 done: $(date)"

# Batch 2: Exp 3, 4, 5 in parallel
echo ""
echo "--- Batch 2: Exp 3 (Pairwise rank), 4 (Drop sparse), 5 (Winsorize) ---"
run_exp 3 "Pairwise rank"     "--pairwise_rank_feature"  &
run_exp 4 "Drop sparse >70%"  "--drop_sparse_cols 0.7"   &
run_exp 5 "Winsorize 2%"      "--winsorize_imputed 0.02" &
wait
echo "Batch 2 done: $(date)"

# Batch 3: Exp 6 alone
echo ""
echo "--- Batch 3: Exp 6 (Feature noise) ---"
run_exp 6 "Feature noise 5%"  "--feature_noise 0.05"
echo "Batch 3 done: $(date)"

# Collect results
echo ""
echo "=== CYCLE 5 RESULTS ==="
for i in 0 1 2 3 4 5 6; do
    grep "^Exp $i " /tmp/cycle5_exp${i}.log 2>/dev/null || \
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
