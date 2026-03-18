#!/usr/bin/env bash
# Threshold experiments v2: min_group_size=10 (was 25)
# Re-test the thresholds that collapsed the top group last time

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
BASE_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --eb_parent --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt"
CV_FLAGS="--cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PARALLEL_FLAGS="--max_workers 16"

run_exp() {
    local exp_id=$1 name=$2 boost=$3 threshold=$4
    local out_root="analysis_output/_thresh2_exp${exp_id}"
    mkdir -p "$out_root"
    echo "Starting exp $exp_id ($name)..."
    python3 -u predict.py $BASE_FLAGS $CV_FLAGS $PARALLEL_FLAGS \
        --top_tier_boost "$boost" --top_tier_threshold "$threshold" \
        --output_root "$out_root" > "/tmp/thresh2_exp${exp_id}.log" 2>&1

    python3 -c "
import pandas as pd, numpy as np, glob, json

out = '$out_root'
fs = sorted(glob.glob(f'{out}/output_*/metadata.json'))
meta = json.load(open(fs[-1]))
rmse = meta.get('oof_rmse', '?')

fs2 = sorted(glob.glob(f'{out}/output_*/oof_predictions.csv'))
df = pd.read_csv(fs2[-1])
for n in [10, 20, 50]:
    top = df.nlargest(n, 'actual_score')
    r = np.sqrt(((top.oof_predicted_score - top.actual_score)**2).mean())
    print(f'  top{n}_rmse={r:.2f}')

fs3 = sorted(glob.glob(f'{out}/output_*/predictions_best_model.csv'))
pred = pd.read_csv(fs3[-1])
pred_with_actual = pred.dropna(subset=['actual_score']).copy()
pred_with_actual['covered'] = (pred_with_actual['actual_score'] >= pred_with_actual['lower_bound']) & \
                               (pred_with_actual['actual_score'] <= pred_with_actual['upper_bound'])
pred_with_actual['ci_width'] = pred_with_actual['upper_bound'] - pred_with_actual['lower_bound']

for n in [13, 25]:
    topn = pred_with_actual.nlargest(n, 'actual_score')
    cov = topn['covered'].mean() * 100
    width = topn['ci_width'].mean()
    print(f'  top{n}_cov={cov:.1f}%  ci_width={width:.0f}')

overall_cov = pred_with_actual['covered'].mean() * 100
print(f'  overall_rmse={rmse}  overall_cov={overall_cov:.1f}%')

gpt = pred[pred['model_name'] == 'GPT-5.4 Thinking']
if len(gpt):
    r = gpt.iloc[0]
    print(f'  GPT-5.4: pred={r[\"predicted_score\"]:.1f}  CI=[{r[\"lower_bound\"]:.0f}, {r[\"upper_bound\"]:.0f}]')
"
    # Show conformal groups
    grep "Grouped conformal\|halfwidth" "/tmp/thresh2_exp${exp_id}.log" | head -8
    echo "=== Exp $exp_id ($name) done ==="
    echo ""
}

# All 6 re-run with min_group_size=10
run_exp 1 "boost2_t1350" 2 1350 &
run_exp 2 "boost2_t1400" 2 1400 &
run_exp 3 "boost2_t1425" 2 1425 &
wait

run_exp 4 "boost2_t1450" 2 1450 &
run_exp 5 "boost3_t1400" 3 1400 &
run_exp 6 "boost3_t1425" 3 1425 &
wait

echo "=== ALL DONE ==="
