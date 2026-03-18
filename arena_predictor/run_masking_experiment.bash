#!/usr/bin/env bash
# Masking experiment: hide the same 21 columns that Grok 4.2 beta is missing
# from 5 nearby high-ELO models and rerun prediction.
# If they also drop 50-70 points, the miss is a coverage-pattern artifact.

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
BASE_FLAGS="--csv_path $CSV --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --eb_parent --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt --top_tier_boost 2 --top_tier_threshold 1400"
CV_FLAGS="--cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PARALLEL_FLAGS="--max_workers 16"

# Run 1: baseline (no masking) — with conformal fix
echo "=== Running baseline with conformal fix ==="
out_root="analysis_output/_masking_baseline"
mkdir -p "$out_root"
python3 -u predict.py $BASE_FLAGS $CV_FLAGS $PARALLEL_FLAGS \
    --output_root "$out_root" > /tmp/masking_baseline.log 2>&1
baseline_rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Baseline RMSE: $baseline_rmse"

# Run 2: mask Grok's missing columns on 5 peer models
echo ""
echo "=== Running masking experiment ==="
out_root="analysis_output/_masking_grok_pattern"
mkdir -p "$out_root"

# Create a Python script that masks the columns and saves a modified CSV
python3 -c "
import pandas as pd
import numpy as np

df = pd.read_csv('$CSV')

# Grok 4.2 beta's missing columns
grok = df[df['model_name'] == 'Grok 4.2 beta'].iloc[0]
missing_cols = [c for c in df.columns if c != 'model_name' and pd.isna(grok[c])]
print(f'Masking {len(missing_cols)} columns: {missing_cols}')

# Target models: similar Arena score to Grok (1467), with good coverage
targets = [
    'Claude Sonnet 4.5',
    'GLM-4.5',
    'GPT-5.1 (high)',
    'Gemini 3.0 Pro Preview (2025-11-18)',
    'Claude Opus 4.5',
]

for model in targets:
    row = df[df['model_name'] == model]
    if len(row) == 0:
        print(f'  WARNING: {model} not found')
        continue
    orig_missing = row.iloc[0][missing_cols].isna().sum()
    print(f'  {model}: originally missing {orig_missing}/{len(missing_cols)} of these cols')
    for col in missing_cols:
        df.loc[df['model_name'] == model, col] = np.nan

df.to_csv('/tmp/masked_combined.csv', index=False)
print('Saved masked CSV to /tmp/masked_combined.csv')
"

python3 -u predict.py --csv_path /tmp/masked_combined.csv \
    --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp \
    --eb_parent --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt \
    --top_tier_boost 2 --top_tier_threshold 1400 \
    $CV_FLAGS $PARALLEL_FLAGS \
    --output_root "$out_root" > /tmp/masking_experiment.log 2>&1
masked_rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(d.get('oof_rmse','?'))")
echo "Masked RMSE: $masked_rmse"

# Compare predictions for masked models
echo ""
echo "=== Impact on masked models ==="
python3 -c "
import pandas as pd
import glob, json

baseline_dir = sorted(glob.glob('analysis_output/_masking_baseline/output_*'))[-1]
masked_dir = sorted(glob.glob('analysis_output/_masking_grok_pattern/output_*'))[-1]

b_oof = pd.read_csv(f'{baseline_dir}/oof_predictions.csv')
m_oof = pd.read_csv(f'{masked_dir}/oof_predictions.csv')

targets = ['Claude Sonnet 4.5', 'GLM-4.5', 'GPT-5.1 (high)',
           'Gemini 3.0 Pro Preview (2025-11-18)', 'Claude Opus 4.5']

merged = b_oof.merge(m_oof, on='model_name', suffixes=('_base', '_masked'))

print(f'{\"Model\":45s} {\"Actual\":>8s} {\"Base\":>8s} {\"Masked\":>8s} {\"Delta\":>8s}')
print('-' * 80)
for model in targets + ['Grok 4.2 beta']:
    row = merged[merged['model_name'] == model]
    if len(row) == 0:
        print(f'{model:45s} NOT FOUND')
        continue
    r = row.iloc[0]
    actual = r['actual_score_base']
    base_pred = r['oof_predicted_score_base']
    mask_pred = r['oof_predicted_score_masked']
    delta = mask_pred - base_pred
    print(f'{model:45s} {actual:8.0f} {base_pred:8.1f} {mask_pred:8.1f} {delta:+8.1f}')
"
