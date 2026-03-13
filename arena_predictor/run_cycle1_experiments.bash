#!/usr/bin/env bash
# Cycle 1 imputation experiments — run sequentially on tower
# Baseline: 21.41 OOF RMSE (ModelBank + coherence + trajectory in target, improved style delta)

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
COMMON_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt --cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1"

echo "=== Cycle 1: 7 experiments (1 baseline + 6 ideas) ==="
echo "Started: $(date)"

# 0. Baseline (re-run to establish current baseline with updated style delta)
echo ""
echo "--- Exp 0: BASELINE ---"
python3 -u predict.py $COMMON_FLAGS > /tmp/cycle1_exp0_baseline.log 2>&1
RMSE0=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 0 (Baseline): RMSE=$RMSE0"

# Clear cache between experiments
rm -rf analysis_output/_cache/

# 1. Spearman-augmented ranking
echo ""
echo "--- Exp 1: Spearman ranking ---"
python3 -u predict.py $COMMON_FLAGS --spearman_ranking > /tmp/cycle1_exp1_spearman.log 2>&1
RMSE1=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 1 (Spearman ranking): RMSE=$RMSE1"

rm -rf analysis_output/_cache/

# 2. KNN fallback (k=5)
echo ""
echo "--- Exp 2: KNN fallback k=5 ---"
python3 -u predict.py $COMMON_FLAGS --knn_fallback_k 5 > /tmp/cycle1_exp2_knn.log 2>&1
RMSE2=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 2 (KNN fallback k=5): RMSE=$RMSE2"

rm -rf analysis_output/_cache/

# 3. Adaptive per-column lambda
echo ""
echo "--- Exp 3: Adaptive col lambda ---"
python3 -u predict.py $COMMON_FLAGS --adaptive_col_lambda > /tmp/cycle1_exp3_adaptive_lambda.log 2>&1
RMSE3=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 3 (Adaptive col lambda): RMSE=$RMSE3"

rm -rf analysis_output/_cache/

# 4. Empirical-Bayes parent model
echo ""
echo "--- Exp 4: EB parent ---"
python3 -u predict.py $COMMON_FLAGS --eb_parent > /tmp/cycle1_exp4_eb_parent.log 2>&1
RMSE4=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 4 (EB parent): RMSE=$RMSE4"

rm -rf analysis_output/_cache/

# 5. Masked-cell calibration
echo ""
echo "--- Exp 5: Masked calibration ---"
python3 -u predict.py $COMMON_FLAGS --masked_calibration > /tmp/cycle1_exp5_masked_cal.log 2>&1
RMSE5=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 5 (Masked calibration): RMSE=$RMSE5"

rm -rf analysis_output/_cache/

# 6. Graph-Laplacian coherence
echo ""
echo "--- Exp 6: Graph Laplacian ---"
python3 -u predict.py $COMMON_FLAGS --graph_laplacian > /tmp/cycle1_exp6_graph_laplacian.log 2>&1
RMSE6=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 6 (Graph Laplacian): RMSE=$RMSE6"

echo ""
echo "=== CYCLE 1 RESULTS ==="
echo "| # | Experiment | RMSE |"
echo "|---|-----------|------|"
echo "| 0 | Baseline | $RMSE0 |"
echo "| 1 | Spearman ranking | $RMSE1 |"
echo "| 2 | KNN fallback (k=5) | $RMSE2 |"
echo "| 3 | Adaptive col lambda | $RMSE3 |"
echo "| 4 | EB parent | $RMSE4 |"
echo "| 5 | Masked calibration | $RMSE5 |"
echo "| 6 | Graph Laplacian | $RMSE6 |"
echo ""
echo "Finished: $(date)"
