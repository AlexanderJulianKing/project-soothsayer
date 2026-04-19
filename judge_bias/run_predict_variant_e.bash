#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

run() {
    local tag=$1 csv=$2
    local out_root="arena_predictor/analysis_output/_stylecontrol_${tag}"
    mkdir -p "$out_root"
    echo "--- ${tag} ---"
    ( cd arena_predictor && \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 \
      python3 -u predict.py --csv_path "../$csv" \
        --imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --eb_parent \
        --cv_repeats_outer 10 --max_workers 8 \
        --output_root "analysis_output/_stylecontrol_${tag}" \
        > "/tmp/stylecontrol_${tag}.log" 2>&1 )
    local rmse
    rmse=$(python3 -c "
import json, glob
fs = sorted(glob.glob('${out_root}/output_*/metadata.json'))
d = json.load(open(fs[-1]))
print(d.get('oof_rmse', '?'))
")
    echo "=== ${tag}: OOF RMSE ${rmse} ==="
}

run "E_baseline_variantE"       "benchmark_combiner/benchmarks/clean_combined_all_benches_variantE.csv"
run "E_sem_champion_variantE"   "benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_v4_d32_variantE.csv"

echo
echo "=============== VARIANT E COMPLETE ==============="
