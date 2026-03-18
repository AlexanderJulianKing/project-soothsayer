#!/usr/bin/env bash
# Full LOBO on ALL non-style, non-target columns (new baseline without eqbench_creative_elo)
set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
FLAGS="--imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --eb_parent --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt --top_tier_boost 2 --top_tier_threshold 1400 --cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1 --max_workers 16"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RESULTS="/tmp/lobo_full_results.csv"
echo "column,rmse,t20_rmse,t10_rmse" > "$RESULTS"

# Get all non-style, non-target columns
COLUMNS=$(python3 -c "
import pandas as pd
df = pd.read_csv('$CSV')
skip = {'model_name', 'lmsys_Score', 'lmarena_Score'}
# Skip style/tone (they feed the final model directly, not through imputation)
cols = [c for c in df.columns if c not in skip and not c.startswith('style_') and not c.startswith('tone_')]
print('\n'.join(cols))
")

total=$(echo "$COLUMNS" | wc -l | tr -d ' ')
echo "Testing $total columns"

run_lobo() {
    local col="$1"
    local idx="$2"
    local safe=$(echo "$col" | tr ' /%()@' '______')
    local tmp="/tmp/lobo2_${safe}.csv"
    local out="analysis_output/_lobo2_${safe}"

    python3 -c "
import pandas as pd
df = pd.read_csv('$CSV')
if '$col' in df.columns:
    df = df.drop(columns=['$col'])
df.to_csv('$tmp', index=False)
"
    rm -rf "$out/_cache" 2>/dev/null
    mkdir -p "$out"
    python3 -u predict.py --csv_path "$tmp" $FLAGS --output_root "$out" > "/tmp/lobo2_${safe}.log" 2>&1

    python3 -c "
import pandas as pd, numpy as np, glob
fs = sorted(glob.glob('${out}/output_*/oof_predictions.csv'))
df = pd.read_csv(fs[-1])
df['err'] = df['oof_predicted_score'] - df['actual_score']
r_all = np.sqrt((df['err']**2).mean())
r_t20 = np.sqrt((df.nlargest(20,'actual_score')['err']**2).mean())
r_t10 = np.sqrt((df.nlargest(10,'actual_score')['err']**2).mean())
print(f'$col,{r_all:.4f},{r_t20:.4f},{r_t10:.4f}')
" >> "$RESULTS"

    rm -f "$tmp"
    echo "[$idx/$total] $col done"
}

i=0
while IFS= read -r col; do
    i=$((i+1))
    run_lobo "$col" "$i" &
    if (( i % 3 == 0 )); then
        wait
        echo "--- Batch done ($i/$total) ---"
    fi
done <<< "$COLUMNS"
wait
echo "--- All done ---"

echo ""
echo "=== RESULTS (sorted by delta from baseline 18.84) ==="
python3 -c "
import pandas as pd
df = pd.read_csv('$RESULTS')
df['d_all'] = df['rmse'] - 18.84
df['d_t20'] = df['t20_rmse'] - 14.82
df['d_t10'] = df['t10_rmse'] - 11.40
df = df.sort_values('d_all')
print(df.to_string(index=False))
"
