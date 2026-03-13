#!/usr/bin/env bash
# Cross-domain interaction ALT experiments — Top-50 LOO, observability filters
# Baseline: EB parent ON (21.48)

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
BASE_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt --eb_parent"
CV_FLAGS="--top_k_loo 50 --cv_repeats_inner 1 --feature_cv_repeats 1 --alt_cv_repeats 1"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
PARALLEL_FLAGS="--max_workers 16"

run_exp() {
    local exp_id=$1 name=$2 flags=$3
    local out_root="analysis_output/_cdobs_exp${exp_id}"
    mkdir -p "$out_root"
    python3 -u predict.py $BASE_FLAGS $CV_FLAGS $PARALLEL_FLAGS \
        --output_root "$out_root" $flags > "/tmp/cdobs_exp${exp_id}.log" 2>&1
    local rmse
    rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
    echo "Exp $exp_id ($name): RMSE=$rmse"
}

echo "=== Cross-domain Top-50 LOO experiments ==="
echo ""

# Wave 1: Baseline + min_obs=50 + min_obs=80
echo "--- Wave 1 ---"
run_exp 0 "Baseline_top50" "" &
run_exp 1 "CD_min50" "--cross_domain_alt --cd_min_obs 50" &
run_exp 2 "CD_min80" "--cross_domain_alt --cd_min_obs 80" &
wait
echo ""

# Wave 2: more variants
echo "--- Wave 2 ---"
run_exp 3 "CD_min50_k30" "--cross_domain_alt --cd_min_obs 50 --cd_top_k 30" &
run_exp 4 "CD_min80_k30" "--cross_domain_alt --cd_min_obs 80 --cd_top_k 30" &
run_exp 5 "CD_min100" "--cross_domain_alt --cd_min_obs 100" &
wait
echo ""

echo "=== All experiments complete ==="
