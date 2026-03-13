#!/usr/bin/env bash
# Cycle 2 imputation experiments — run sequentially on tower
# Baseline: EB parent ON (21.48 from Cycle 1)

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
COMMON_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt --cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1"

echo "=== Cycle 2: 7 experiments (1 baseline + 6 ideas) ==="
echo "Started: $(date)"

# 0. Baseline = EB parent ON
echo ""
echo "--- Exp 0: BASELINE (EB parent) ---"
rm -rf analysis_output/_cache/
python3 -u predict.py $COMMON_FLAGS --eb_parent > /tmp/cycle2_exp0_baseline.log 2>&1
RMSE0=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 0 (EB parent baseline): RMSE=$RMSE0"

# 1. EB residual (Codex idea 1): blend parent into per-cell during pass1
echo ""
echo "--- Exp 1: EB residual ---"
rm -rf analysis_output/_cache/
python3 -u predict.py $COMMON_FLAGS --eb_residual > /tmp/cycle2_exp1_eb_residual.log 2>&1
RMSE1=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 1 (EB residual): RMSE=$RMSE1"

# 2. Exact joint-support selection (Codex idea 2)
echo ""
echo "--- Exp 2: Exact joint support + EB parent ---"
rm -rf analysis_output/_cache/
python3 -u predict.py $COMMON_FLAGS --eb_parent --exact_joint_support > /tmp/cycle2_exp2_exact_support.log 2>&1
RMSE2=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 2 (Exact joint support + EB): RMSE=$RMSE2"

# 3. SPD graph smoother post-SVD (Codex idea 3) + EB parent
echo ""
echo "--- Exp 3: SPD graph smoother + EB parent ---"
rm -rf analysis_output/_cache/
python3 -u predict.py $COMMON_FLAGS --eb_parent --spd_graph_smoother > /tmp/cycle2_exp3_spd_graph.log 2>&1
RMSE3=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 3 (SPD graph smoother + EB): RMSE=$RMSE3"

# 4. EB after coherence (Claude idea 1)
echo ""
echo "--- Exp 4: EB after coherence ---"
rm -rf analysis_output/_cache/
python3 -u predict.py $COMMON_FLAGS --eb_after_coherence > /tmp/cycle2_exp4_eb_after.log 2>&1
RMSE4=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 4 (EB after coherence): RMSE=$RMSE4"

# 5. Double EB (Claude idea 2): EB before AND after coherence
echo ""
echo "--- Exp 5: Double EB ---"
rm -rf analysis_output/_cache/
python3 -u predict.py $COMMON_FLAGS --eb_double > /tmp/cycle2_exp5_double_eb.log 2>&1
RMSE5=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 5 (Double EB): RMSE=$RMSE5"

# 6. EB with Bayesian predictive sigma (Claude idea 3)
echo ""
echo "--- Exp 6: EB Bayesian sigma ---"
rm -rf analysis_output/_cache/
python3 -u predict.py $COMMON_FLAGS --eb_parent --eb_bayesian_sigma > /tmp/cycle2_exp6_eb_bayesian.log 2>&1
RMSE6=$(python3 -c "import json,glob; fs=sorted(glob.glob('analysis_output/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Exp 6 (EB Bayesian sigma): RMSE=$RMSE6"

echo ""
echo "=== CYCLE 2 RESULTS ==="
echo "| # | Experiment | RMSE |"
echo "|---|-----------|------|"
echo "| 0 | EB parent baseline | $RMSE0 |"
echo "| 1 | EB residual | $RMSE1 |"
echo "| 2 | Exact joint support + EB | $RMSE2 |"
echo "| 3 | SPD graph smoother + EB | $RMSE3 |"
echo "| 4 | EB after coherence | $RMSE4 |"
echo "| 5 | Double EB | $RMSE5 |"
echo "| 6 | EB Bayesian sigma | $RMSE6 |"
echo ""
echo "Finished: $(date)"
