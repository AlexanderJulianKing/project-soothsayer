#!/usr/bin/env bash
# LOBO (Leave One Benchmark Out) experiments
# Drop each sparse column individually, re-impute from scratch, measure RMSE
# Baseline: 19.04 RMSE with all columns

set -euo pipefail

CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
BASE_FLAGS="--imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --eb_parent --poly_interactions --poly_limit 7 --no_residual_head --no_traj_in_alt --top_tier_boost 2 --top_tier_threshold 1400"
CV_FLAGS="--cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PARALLEL_FLAGS="--max_workers 16"

RESULTS_FILE="/tmp/lobo_results.csv"
echo "column,rmse,delta" > "$RESULTS_FILE"

run_lobo() {
    local col="$1"
    local safe_name=$(echo "$col" | tr ' /%()' '_____')
    local out_root="analysis_output/_lobo_${safe_name}"

    # Create modified CSV with this column dropped
    local tmp_csv="/tmp/lobo_${safe_name}.csv"
    python3 -c "
import pandas as pd
df = pd.read_csv('$CSV')
if '$col' in df.columns:
    df = df.drop(columns=['$col'])
df.to_csv('$tmp_csv', index=False)
print('Dropped $col, ${tmp_csv} has', len(df.columns), 'columns')
"

    # Clear cache and run
    rm -rf "$out_root/_cache" 2>/dev/null
    mkdir -p "$out_root"
    python3 -u predict.py --csv_path "$tmp_csv" $BASE_FLAGS $CV_FLAGS $PARALLEL_FLAGS \
        --output_root "$out_root" > "/tmp/lobo_${safe_name}.log" 2>&1

    local rmse
    rmse=$(python3 -c "import json,glob; fs=sorted(glob.glob('${out_root}/output_*/metadata.json')); d=json.load(open(fs[-1])); print(f'{d.get(\"oof_rmse\",\"?\"):.4f}')")
    local delta
    delta=$(python3 -c "print(f'{float(${rmse}) - 19.04:+.4f}')")

    echo "${col},${rmse},${delta}" >> "$RESULTS_FILE"
    echo "LOBO $col: RMSE=$rmse (delta=$delta)"

    # Cleanup tmp CSV
    rm -f "$tmp_csv"
}

# All 24 sparse columns to test
COLUMNS=(
    "simplebench_Score (AVG@5)"
    "eqbench_eq_elo"
    "eqbench_creative_elo"
    "livebench_theory_of_mind"
    "livebench_zebra_puzzle"
    "livebench_spatial"
    "livebench_logic_with_navigation"
    "livebench_code_generation"
    "livebench_code_completion"
    "livebench_integrals_with_game"
    "livebench_olympiad"
    "livebench_javascript"
    "livebench_typescript"
    "livebench_python"
    "livebench_consecutive_events"
    "livebench_tablejoin"
    "livebench_connections"
    "livebench_plot_unscrambling"
    "livebench_typos"
    "livebench_IF Average"
    "arc_ARC-AGI-1"
    "lechmazur_confab_Confab %"
    "lechmazur_gen_Avg Rank"
    "lechmazur_nytcon_Score %"
)

# Run 3 at a time
total=${#COLUMNS[@]}
for ((i=0; i<total; i+=3)); do
    batch_pids=()
    for ((j=i; j<i+3 && j<total; j++)); do
        col="${COLUMNS[$j]}"
        echo "Starting LOBO: $col ($(( j+1 ))/$total)"
        run_lobo "$col" &
        batch_pids+=($!)
    done
    # Wait for this batch
    for pid in "${batch_pids[@]}"; do
        wait "$pid"
    done
    echo "--- Batch done ($(( i+3 < total ? i+3 : total ))/$total) ---"
done

echo ""
echo "=== ALL DONE ==="
echo ""
echo "=== RESULTS (sorted by delta) ==="
sort -t',' -k3 -n "$RESULTS_FILE" | column -t -s','
echo ""
echo "Baseline: 19.04"
