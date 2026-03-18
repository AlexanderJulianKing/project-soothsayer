#!/usr/bin/env bash
# lmarena style-only interaction experiment
# Change: lmarena_Score can only interact with style_* features
# Also adds: frac_used_length, combined_length/header/bold/list to pipeline
# Baseline: 20.11 RMSE, GPT-5.4 Thinking predicted at 1536.3

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
BASE_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --eb_parent --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt --top_tier_boost 2 --top_tier_threshold 1400"
CV_FLAGS="--cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PARALLEL_FLAGS="--max_workers 16"

out_root="analysis_output/_lmarena_style_only"
mkdir -p "$out_root"

echo "Running: lmarena style-only interactions + extra style features..."
python3 -u predict.py $BASE_FLAGS $CV_FLAGS $PARALLEL_FLAGS \
    --output_root "$out_root" > /tmp/lmarena_style.log 2>&1

rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "RMSE: $rmse"

python3 -c "
import pandas as pd, numpy as np, glob

# Top-50 RMSE
fs = sorted(glob.glob('${out_root}/output_*/oof_predictions.csv'))
df = pd.read_csv(fs[-1])
top = df.nlargest(50, 'actual_score')
top50 = np.sqrt(((top.oof_predicted_score - top.actual_score)**2).mean())
print(f'Top-50 RMSE: {top50:.2f}')

# GPT-5.4 prediction
fs2 = sorted(glob.glob('${out_root}/output_*/predictions_best_model.csv'))
pred = pd.read_csv(fs2[-1])
gpt = pred[pred['model_name'] == 'GPT-5.4 Thinking']
if len(gpt):
    r = gpt.iloc[0]
    print(f'GPT-5.4 Thinking: pred={r[\"predicted_score\"]:.1f}  CI=[{r[\"lower_bound\"]:.0f}, {r[\"upper_bound\"]:.0f}]')

# lmarena features
fs3 = sorted(glob.glob('${out_root}/output_*/best_model_feature_importance.csv'))
fi = pd.read_csv(fs3[-1])
fi_sorted = fi.reindex(fi['importance'].abs().sort_values(ascending=False).index)
lm = fi_sorted[fi_sorted['feature'].str.contains('lmarena', case=False)]
print(f'\\nlmarena features in model: {len(lm)}')
for _, r in lm.head(10).iterrows():
    print(f'  {r[\"feature\"]:55s} imp={r[\"importance\"]:+.3f}')

print(f'\\nTop 10 features overall:')
for _, r in fi_sorted.head(10).iterrows():
    print(f'  {r[\"feature\"]:55s} imp={r[\"importance\"]:+.3f}')
"

echo ""
echo "Baseline comparison: RMSE=20.11, top50=15.10, GPT-5.4=1536.3"
