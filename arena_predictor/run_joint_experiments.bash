#!/usr/bin/env bash
# Joint prediction experiments: SCMF and BHLT
set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

run_exp() {
    local exp_id=$1 name=$2; shift 2; local flags="$*"
    local out_root="analysis_output/_joint_exp${exp_id}"
    mkdir -p "$out_root"
    python3 -u joint_predict.py --csv_path $CSV --output_root "$out_root" \
        $flags > "/tmp/joint_exp${exp_id}.log" 2>&1
    local rmse
    rmse=$(python3 -c "import json; d=json.load(open('${out_root}/metadata.json')); print(f'{d[\"oof_rmse\"]:.2f}')")
    echo "Exp $exp_id ($name): RMSE=$rmse"
}

echo "=== Joint Prediction Experiments ==="

# Wave 1: Standalone baselines
echo "--- Wave 1: Standalone baselines ---"
run_exp 1 "SCMF_k6_lt5"     --approach scmf --rank 6 --lambda_target 5.0 &
run_exp 2 "BHLT_k6_fp1"     --approach bhlt --n_factors 6 --family_prior 1.0 &
run_exp 3 "SCMF_k6_lt10"    --approach scmf --rank 6 --lambda_target 10.0 &
wait
echo ""

# Wave 2: Rank sweep
echo "--- Wave 2: Rank sweep ---"
run_exp 4 "SCMF_k4_lt5"     --approach scmf --rank 4 --lambda_target 5.0 &
run_exp 5 "SCMF_k8_lt5"     --approach scmf --rank 8 --lambda_target 5.0 &
run_exp 6 "BHLT_k8_fp05"    --approach bhlt --n_factors 8 --family_prior 0.5 &
wait
echo ""

# Wave 3: Lambda sweep + inductive
echo "--- Wave 3: Lambda + inductive ---"
run_exp 7 "SCMF_k6_lt1"     --approach scmf --rank 6 --lambda_target 1.0 &
run_exp 8 "SCMF_k6_lt20"    --approach scmf --rank 6 --lambda_target 20.0 &
run_exp 9 "SCMF_ind_k6"     --approach scmf --rank 6 --lambda_target 5.0 --mode inductive &
wait
echo ""

# Wave 4: BHLT variants
echo "--- Wave 4: BHLT variants ---"
run_exp 10 "BHLT_k4_fp1"    --approach bhlt --n_factors 4 --family_prior 1.0 &
run_exp 11 "BHLT_corr_k6"   --approach bhlt --n_factors 6 --family_prior 1.0 --clustering correlation &
run_exp 12 "BHLT_ind_k6"    --approach bhlt --n_factors 6 --family_prior 1.0 --mode inductive &
wait
echo ""

echo "=== All joint experiments complete ==="
