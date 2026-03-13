#!/usr/bin/env bash
# Cycle 8: Feature transform experiments
# Baseline: EB parent ON (21.48)
# 3 experiments in parallel on 48-core tower

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
    local out_root="analysis_output/_c8_exp${exp_id}"
    mkdir -p "$out_root"
    python3 -u predict.py $BASE_FLAGS $CV_FLAGS $PARALLEL_FLAGS \
        --output_root "$out_root" $flags > "/tmp/cycle8_exp${exp_id}.log" 2>&1
    local rmse
    rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
    echo "Exp $exp_id ($name): RMSE=$rmse"
}

echo "=== Cycle 8: Feature transforms ==="
echo "Started: $(date)"

# All 3 in parallel
echo ""
echo "--- Exp 1 (Quantile transform), 2 (Reliability-weighted PCA), 3 (Orthogonalize to ALT) ---"
run_exp 1 "Quantile transform"         "--quantile_transform"       &
run_exp 2 "Reliability-weighted PCA"    "--alt_reliability_scale"    &
run_exp 3 "Orthogonalize to ALT"        "--orthogonalize_to_alt"     &
wait
echo "All done: $(date)"

# Collect results
echo ""
echo "=== CYCLE 8 RESULTS ==="
for i in 1 2 3; do
    grep "^Exp $i " /tmp/cycle8_exp${i}.log 2>/dev/null || \
    python3 -c "
import json,glob
fs=sorted(glob.glob('analysis_output/_c8_exp${i}/output_*/metadata.json'))
if fs:
    d=json.load(open(fs[-1]))
    print('Exp $i: RMSE=' + str(d.get('oof_rmse','?')))
else:
    print('Exp $i: NO RESULT')
"
done
echo ""
echo "Finished: $(date)"
