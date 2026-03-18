#!/usr/bin/env bash
# Threshold experiments for grouped conformal CIs
# Goal: honest coverage for top models specifically

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
    local out_root="analysis_output/_thresh_exp${exp_id}"
    mkdir -p "$out_root"
    echo "Starting exp $exp_id ($name)..."
    python3 -u predict.py $BASE_FLAGS $CV_FLAGS $PARALLEL_FLAGS \
        --top_tier_boost "$boost" --top_tier_threshold "$threshold" \
        --output_root "$out_root" > "/tmp/thresh_exp${exp_id}.log" 2>&1

    python3 -c "
import pandas as pd, numpy as np, glob, json

out = '$out_root'
fs = sorted(glob.glob(f'{out}/output_*/metadata.json'))
meta = json.load(open(fs[-1]))
rmse = meta.get('oof_rmse', '?')

# OOF predictions
fs2 = sorted(glob.glob(f'{out}/output_*/oof_predictions.csv'))
df = pd.read_csv(fs2[-1])
df['abs_err'] = (df['oof_predicted_score'] - df['actual_score']).abs()

# Top-N RMSEs
for n in [10, 20, 50]:
    top = df.nlargest(n, 'actual_score')
    r = np.sqrt(((top.oof_predicted_score - top.actual_score)**2).mean())
    print(f'  top{n}_rmse={r:.2f}')

# Predictions
fs3 = sorted(glob.glob(f'{out}/output_*/predictions_best_model.csv'))
pred = pd.read_csv(fs3[-1])

# Coverage by decile
pred_with_actual = pred.dropna(subset=['actual_score']).copy()
pred_with_actual['covered'] = (pred_with_actual['actual_score'] >= pred_with_actual['lower_bound']) & \
                               (pred_with_actual['actual_score'] <= pred_with_actual['upper_bound'])
pred_with_actual['ci_width'] = pred_with_actual['upper_bound'] - pred_with_actual['lower_bound']

# Top 13 (bin 9 equivalent)
top13 = pred_with_actual.nlargest(13, 'actual_score')
top13_cov = top13['covered'].mean() * 100
top13_width = top13['ci_width'].mean()

# Top 25
top25 = pred_with_actual.nlargest(25, 'actual_score')
top25_cov = top25['covered'].mean() * 100

overall_cov = pred_with_actual['covered'].mean() * 100

print(f'  overall_rmse={rmse}')
print(f'  overall_cov={overall_cov:.1f}%  top13_cov={top13_cov:.1f}%  top25_cov={top25_cov:.1f}%')
print(f'  top13_avg_ci_width={top13_width:.0f}')

# GPT-5.4
gpt = pred[pred['model_name'] == 'GPT-5.4 Thinking']
if len(gpt):
    r = gpt.iloc[0]
    print(f'  GPT-5.4: pred={r[\"predicted_score\"]:.1f}  CI=[{r[\"lower_bound\"]:.0f}, {r[\"upper_bound\"]:.0f}]')
"
    echo "=== Exp $exp_id ($name) done ==="
    echo ""
}

# Wave 1: vary threshold with boost=2
run_exp 1 "boost2_t1350" 2 1350 &
run_exp 2 "boost2_t1400" 2 1400 &
run_exp 3 "boost2_t1425" 2 1425 &
wait

run_exp 4 "boost2_t1450" 2 1450 &
run_exp 5 "boost3_t1400" 3 1400 &
run_exp 6 "boost3_t1425" 3 1425 &
wait

echo "=== ALL DONE ==="
