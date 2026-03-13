#!/usr/bin/env bash
# Hybrid architecture: Cross-domain ALT + PCA components in target + greedy residual correlates
# Baseline: EB parent ON (21.48 with PCA-10 ALT, standard feature selection)
# Previous: CD ALT min_obs=50 → 19.62 top-50 LOO RMSE

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
BASE_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --no_residual_head --no_traj_in_alt --eb_parent"
CV_FLAGS="--top_k_loo 50 --cv_repeats_inner 1 --feature_cv_repeats 1 --alt_cv_repeats 1"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
PARALLEL_FLAGS="--max_workers 16"

run_exp() {
    local exp_id=$1 name=$2 flags=$3
    local out_root="analysis_output/_hybrid_exp${exp_id}"
    mkdir -p "$out_root"
    python3 -u predict.py $BASE_FLAGS $CV_FLAGS $PARALLEL_FLAGS \
        --output_root "$out_root" $flags > "/tmp/hybrid_exp${exp_id}.log" 2>&1
    local rmse
    rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
    echo "Exp $exp_id ($name): RMSE=$rmse"
}

echo "=== Hybrid Architecture Experiments ==="
echo "CD ALT + PCA target + greedy residual correlates"
echo ""

# Wave 1: Baselines and simple hybrid
echo "--- Wave 1: Baselines ---"
# Exp 0: Standard baseline (PCA-10 ALT, tree feature selection, poly interactions)
run_exp 0 "Baseline_PCA10_poly" "--poly_interactions --poly_limit 7" &
# Exp 1: CD ALT baseline (CD for ALT, tree feature selection, poly)
run_exp 1 "CD_ALT_poly" "--cross_domain_alt --cd_min_obs 50 --poly_interactions --poly_limit 7" &
# Exp 2: CD ALT + PCA-10 in target (no poly, no greedy yet)
run_exp 2 "CD_ALT_PCA10" "--cross_domain_alt --cd_min_obs 50 --pca_in_target 10" &
wait
echo ""

# Wave 2: Greedy residual correlates
echo "--- Wave 2: Greedy residual ---"
# Exp 3: CD ALT + PCA-10 + 5 greedy residual columns
run_exp 3 "CD_PCA10_GR5" "--cross_domain_alt --cd_min_obs 50 --pca_in_target 10 --greedy_residual_k 5" &
# Exp 4: CD ALT + PCA-10 + 10 greedy residual columns
run_exp 4 "CD_PCA10_GR10" "--cross_domain_alt --cd_min_obs 50 --pca_in_target 10 --greedy_residual_k 10" &
# Exp 5: CD ALT + PCA-5 + 5 greedy residual (fewer PCs)
run_exp 5 "CD_PCA5_GR5" "--cross_domain_alt --cd_min_obs 50 --pca_in_target 5 --greedy_residual_k 5" &
wait
echo ""

# Wave 3: With poly interactions on top of hybrid
echo "--- Wave 3: Hybrid + poly ---"
# Exp 6: CD ALT + PCA-10 + GR5 + poly
run_exp 6 "CD_PCA10_GR5_poly" "--cross_domain_alt --cd_min_obs 50 --pca_in_target 10 --greedy_residual_k 5 --poly_interactions --poly_limit 7" &
# Exp 7: PCA-10 in target only (no CD ALT, standard ALT), greedy residual
run_exp 7 "PCA10_GR5_std" "--pca_in_target 10 --greedy_residual_k 5" &
# Exp 8: CD ALT + PCA-10 + GR15 (more greedy columns)
run_exp 8 "CD_PCA10_GR15" "--cross_domain_alt --cd_min_obs 50 --pca_in_target 10 --greedy_residual_k 15" &
wait
echo ""

echo "=== All experiments complete ==="
