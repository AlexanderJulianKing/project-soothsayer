#!/usr/bin/env bash
# Cycle 3: Beyond imputation — feature selection, model architecture, ensemble
# Baseline: EB parent ON (21.48)
# All experiments keep EB parent ON since it's the established winner

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
BASE_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt --eb_parent"
CV_FLAGS="--cv_repeats_outer 10 --cv_repeats_inner 5 --alt_cv_repeats 1"

echo "=== Cycle 3: Pipeline-level experiments ==="
echo "Started: $(date)"

# 0. Baseline (EB parent, feature_cv_repeats=1)
echo ""
echo "--- Exp 0: BASELINE (EB parent, feature_cv_repeats=1) ---"
rm -rf analysis_output/_cache/
python3 -u predict.py $BASE_FLAGS $CV_FLAGS --feature_cv_repeats 1 > /tmp/cycle3_exp0.log 2>&1
RMSE0=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 0 (Baseline): RMSE=$RMSE0"

# 1. More feature selection CV repeats (5 instead of 1)
echo ""
echo "--- Exp 1: Feature CV repeats=5 ---"
rm -rf analysis_output/_cache/
python3 -u predict.py $BASE_FLAGS $CV_FLAGS --feature_cv_repeats 5 > /tmp/cycle3_exp1.log 2>&1
RMSE1=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 1 (Feature CV repeats=5): RMSE=$RMSE1"

# 2. More feature selection CV repeats (10)
echo ""
echo "--- Exp 2: Feature CV repeats=10 ---"
rm -rf analysis_output/_cache/
python3 -u predict.py $BASE_FLAGS $CV_FLAGS --feature_cv_repeats 10 > /tmp/cycle3_exp2.log 2>&1
RMSE2=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 2 (Feature CV repeats=10): RMSE=$RMSE2"

# 3. More ALT CV repeats (5 instead of 1)
echo ""
echo "--- Exp 3: ALT CV repeats=5 ---"
rm -rf analysis_output/_cache/
python3 -u predict.py $BASE_FLAGS --cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 5 > /tmp/cycle3_exp3.log 2>&1
RMSE3=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 3 (ALT CV repeats=5): RMSE=$RMSE3"

# 4. No polynomial interactions (simpler model)
echo ""
echo "--- Exp 4: No poly interactions ---"
rm -rf analysis_output/_cache/
python3 -u predict.py --csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --no_residual_head --no_traj_in_alt --eb_parent $CV_FLAGS --feature_cv_repeats 1 > /tmp/cycle3_exp4.log 2>&1
RMSE4=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 4 (No poly interactions): RMSE=$RMSE4"

# 5. Higher poly limit (10 instead of 7)
echo ""
echo "--- Exp 5: Poly limit=10 ---"
rm -rf analysis_output/_cache/
python3 -u predict.py --csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --poly_interactions --poly_limit 10 --no_residual_head --no_traj_in_alt --eb_parent $CV_FLAGS --feature_cv_repeats 1 > /tmp/cycle3_exp5.log 2>&1
RMSE5=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 5 (Poly limit=10): RMSE=$RMSE5"

# 6. All CV repeats set to 5 (most stable evaluation)
echo ""
echo "--- Exp 6: All CV repeats=5 ---"
rm -rf analysis_output/_cache/
python3 -u predict.py $BASE_FLAGS --cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 5 --alt_cv_repeats 5 > /tmp/cycle3_exp6.log 2>&1
RMSE6=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 6 (All CV repeats=5): RMSE=$RMSE6"

echo ""
echo "=== CYCLE 3 RESULTS ==="
echo "| # | Experiment | RMSE |"
echo "|---|-----------|------|"
echo "| 0 | Baseline (EB parent) | $RMSE0 |"
echo "| 1 | Feature CV repeats=5 | $RMSE1 |"
echo "| 2 | Feature CV repeats=10 | $RMSE2 |"
echo "| 3 | ALT CV repeats=5 | $RMSE3 |"
echo "| 4 | No poly interactions | $RMSE4 |"
echo "| 5 | Poly limit=10 | $RMSE5 |"
echo "| 6 | All CV repeats=5 | $RMSE6 |"
echo ""
echo "Finished: $(date)"
