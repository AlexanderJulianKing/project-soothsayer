#!/usr/bin/env bash
# Stage 6: per-column leave-one-out ablation (cv5).
# Reuses the primed _sweep_main imputation cache via per-worker copies, so all
# runs are cache-hits (no re-impute) and there are no output-dir/cache races.
# Reports columns whose drop changes cv5 RMSE by >= 0.1 (rest = noise).
# Reads the KNN column list from /tmp/loo_cols.txt (one exact column per line).
set -uo pipefail
cd "$(dirname "$0")"                      # arena_predictor/
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY="${PY:-python3}"
CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_v4_d32.csv"
BASE_IMP="--imputer_type model_bank --coherence_lambda 8.0 --coherence_shape exp --predictor_selection loo_forward --drop_style_tone --pls_hybrid_k 3 --max_workers 8"
COLS=/tmp/loo_cols.txt
BASE_CV5="${BASE_CV5:-14.1843}"
NW=3
RESDIR=/tmp/loo_res; mkdir -p "$RESDIR"; rm -f "$RESDIR"/* 2>/dev/null || true

# prime worker caches from the main sweep's primed champion imputation
for w in $(seq 1 $NW); do
  mkdir -p "analysis_output/_loo_w$w"
  rm -rf "analysis_output/_loo_w$w/_cache"
  cp -r analysis_output/_sweep_main/_cache "analysis_output/_loo_w$w/_cache"
done

one() {  # idx col worker
  local idx="$1" col="$2" w="$3"
  local cf="/tmp/loo_col_w$w.txt"; printf '%s\n' "$col" > "$cf"
  if $PY -u predict.py --csv_path "$CSV" $BASE_IMP --cv_repeats_outer 5 \
        --output_root "analysis_output/_loo_w$w" --drop_cols_file "$cf" \
        > "/tmp/loo_log_$idx.log" 2>&1; then
    local d r
    d=$(ls -td "analysis_output/_loo_w$w"/output_* | head -1)
    r=$($PY -c "import json;print(json.load(open('$d/metadata.json'))['oof_rmse'])" 2>/dev/null || echo NA)
    printf '%s\t%s\n' "$r" "$col" > "$RESDIR/$idx.txt"
  else
    printf 'FAIL\t%s\n' "$col" > "$RESDIR/$idx.txt"
  fi
  echo "[$idx] $r  drop=$col"
}

i=0
while IFS= read -r col; do
  [ -z "$col" ] && continue
  w=$(( i % NW + 1 ))
  one "$i" "$col" "$w" &
  i=$((i+1))
  (( i % NW == 0 )) && wait
done < "$COLS"
wait

echo "===== LOO complete: $i columns ====="
echo "=== columns whose drop changes cv5 RMSE by >= 0.1 (base_cv5=$BASE_CV5) ==="
cat "$RESDIR"/*.txt | awk -F'\t' -v b="$BASE_CV5" '$1!="FAIL" && $1!="NA"{d=$1-b; if(d<=-0.1||d>=0.1) printf "%+.3f\t%.4f\t%s\n", d, $1, $2}' | sort -g
echo "=== full sorted: best 12 (drop helps most) ==="
cat "$RESDIR"/*.txt | awk -F'\t' '$1!="FAIL"&&$1!="NA"' | sort -g | head -12
echo "=== worst 8 (drop hurts most = most load-bearing) ==="
cat "$RESDIR"/*.txt | awk -F'\t' '$1!="FAIL"&&$1!="NA"' | sort -g | tail -8
echo "=== any FAILs ==="; grep -l FAIL "$RESDIR"/*.txt 2>/dev/null | wc -l
