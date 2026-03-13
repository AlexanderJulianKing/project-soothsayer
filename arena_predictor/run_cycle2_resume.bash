#!/usr/bin/env bash
# Cycle 2 resume from Exp 3 (after SPD graph smoother crash fix)

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
COMMON_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt --cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1"

echo "=== Cycle 2 RESUME (Exp 3-6) ==="
echo "Started: $(date)"

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
echo "=== CYCLE 2 RESULTS (Exp 3-6) ==="
echo "| # | Experiment | RMSE |"
echo "|---|-----------|------|"
echo "| 3 | SPD graph smoother + EB | $RMSE3 |"
echo "| 4 | EB after coherence | $RMSE4 |"
echo "| 5 | Double EB | $RMSE5 |"
echo "| 6 | EB Bayesian sigma | $RMSE6 |"
echo ""
echo "Finished: $(date)"
