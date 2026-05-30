#!/usr/bin/env bash
# Stage 5: cv10 confirmation of the sweep winners vs champion (14.03).
# pls6/pls8 reuse the _sweep_main champion imputation cache (KNN-only knobs).
# selk25 / d40 combos re-impute into their own roots.
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY="${PY:-python3}"
D32="../benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_v4_d32.csv"
D40="../benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_v4_d40.csv"
BASE="--imputer_type model_bank --coherence_lambda 8.0 --coherence_shape exp --predictor_selection loo_forward --drop_style_tone --max_workers 16"
RES=/tmp/stage5_results.csv; echo "label,oof_rmse,ci_lo,ci_hi,extra" > "$RES"

run5(){ local label="$1" outroot="$2" csv="$3"; shift 3; local extra="$*"
  mkdir -p "$outroot"
  if $PY -u predict.py --csv_path "$csv" $BASE --cv_repeats_outer 10 --output_root "$outroot" $extra \
        > "/tmp/s5_${label}.log" 2>&1; then
    local d r; d=$(ls -td "$outroot"/output_* | head -1)
    r=$($PY -c "import json;m=json.load(open('$d/metadata.json'));print('%s,%s,%s'%(m['oof_rmse'],m['oof_rmse_ci_lo'],m['oof_rmse_ci_hi']))" 2>/dev/null || echo ",,")
    echo "$label,$r,\"$extra\"" >> "$RES"; echo "[s5/$label] $r"
  else echo "$label,FAIL,,,\"$extra\"" >> "$RES"; echo "[s5/$label] FAILED"; fi
}

printf 'livebench_connections\n' > /tmp/dropconn.txt   # top LOO single-column win (-0.21 cv5)
run5 pls6                  analysis_output/_sweep_main "$D32" --pls_hybrid_k 6
run5 pls8                  analysis_output/_sweep_main "$D32" --pls_hybrid_k 8
run5 pls6_dropconn         analysis_output/_sweep_main "$D32" --pls_hybrid_k 6 --drop_cols_file /tmp/dropconn.txt
run5 pls6_selk25           analysis_output/_sweep5     "$D32" --pls_hybrid_k 6 --selector_k_max 25
run5 pls8_selk25           analysis_output/_sweep5     "$D32" --pls_hybrid_k 8 --selector_k_max 25
run5 pls6_selk25_dropconn  analysis_output/_sweep5     "$D32" --pls_hybrid_k 6 --selector_k_max 25 --drop_cols_file /tmp/dropconn.txt
if [ -f "$D40" ]; then
  run5 pls6_selk25_d40 analysis_output/_sweep5d40 "$D40" --pls_hybrid_k 6 --selector_k_max 25
fi
echo "===== STAGE 5 cv10 (champion baseline = 14.0301) ====="
( head -1 "$RES"; tail -n +2 "$RES" | sort -t, -k2 -g ) | column -t -s,
