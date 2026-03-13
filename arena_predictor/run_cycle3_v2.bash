#!/usr/bin/env bash
# Cycle 3 v2: Pipeline-level experiments (practical runtime)
# Baseline: EB parent ON (21.48)
# Skip expensive feature_cv_repeats, focus on model/structure changes

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
BASE_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt --eb_parent"
CV_FLAGS="--cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1"

echo "=== Cycle 3 v2: Pipeline experiments ==="
echo "Started: $(date)"

# 0. Baseline
echo ""
echo "--- Exp 0: BASELINE ---"
rm -rf analysis_output/_cache/
python3 -u predict.py $BASE_FLAGS $CV_FLAGS > /tmp/cycle3v2_exp0.log 2>&1
RMSE0=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 0 (Baseline): RMSE=$RMSE0"

# 1. No poly interactions (simpler model, less overfitting)
echo ""
echo "--- Exp 1: No poly interactions ---"
rm -rf analysis_output/_cache/
python3 -u predict.py --csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --no_residual_head --no_traj_in_alt --eb_parent $CV_FLAGS > /tmp/cycle3v2_exp1.log 2>&1
RMSE1=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 1 (No poly): RMSE=$RMSE1"

# 2. Higher poly limit (10 instead of 7)
echo ""
echo "--- Exp 2: Poly limit=10 ---"
rm -rf analysis_output/_cache/
python3 -u predict.py --csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --poly_interactions --poly_limit 10 --no_residual_head --no_traj_in_alt --eb_parent $CV_FLAGS > /tmp/cycle3v2_exp2.log 2>&1
RMSE2=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 2 (Poly limit=10): RMSE=$RMSE2"

# 3. Lower poly limit (5)
echo ""
echo "--- Exp 3: Poly limit=5 ---"
rm -rf analysis_output/_cache/
python3 -u predict.py --csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --poly_interactions --poly_limit 5 --no_residual_head --no_traj_in_alt --eb_parent $CV_FLAGS > /tmp/cycle3v2_exp3.log 2>&1
RMSE3=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 3 (Poly limit=5): RMSE=$RMSE3"

# 4. With residual head (currently --no_residual_head)
echo ""
echo "--- Exp 4: With residual head ---"
rm -rf analysis_output/_cache/
python3 -u predict.py --csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --poly_interactions --poly_limit 7 --no_traj_in_alt --eb_parent $CV_FLAGS > /tmp/cycle3v2_exp4.log 2>&1
RMSE4=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 4 (With residual head): RMSE=$RMSE4"

# 5. ALT trajectory ON (currently off)
echo ""
echo "--- Exp 5: ALT trajectory ON ---"
rm -rf analysis_output/_cache/
python3 -u predict.py --csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --poly_interactions --poly_limit 7 --no_residual_head --eb_parent $CV_FLAGS > /tmp/cycle3v2_exp5.log 2>&1
RMSE5=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 5 (ALT trajectory ON): RMSE=$RMSE5"

# 6. Higher outer CV repeats (20 instead of 10) for more stable eval
echo ""
echo "--- Exp 6: Outer CV repeats=20 ---"
rm -rf analysis_output/_cache/
python3 -u predict.py $BASE_FLAGS --cv_repeats_outer 20 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1 > /tmp/cycle3v2_exp6.log 2>&1
RMSE6=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 6 (Outer CV=20): RMSE=$RMSE6"

echo ""
echo "=== CYCLE 3 v2 RESULTS ==="
echo "| # | Experiment | RMSE |"
echo "|---|-----------|------|"
echo "| 0 | Baseline (EB parent) | $RMSE0 |"
echo "| 1 | No poly interactions | $RMSE1 |"
echo "| 2 | Poly limit=10 | $RMSE2 |"
echo "| 3 | Poly limit=5 | $RMSE3 |"
echo "| 4 | With residual head | $RMSE4 |"
echo "| 5 | ALT trajectory ON | $RMSE5 |"
echo "| 6 | Outer CV=20 | $RMSE6 |"
echo ""
echo "Finished: $(date)"
