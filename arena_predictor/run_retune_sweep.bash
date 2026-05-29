#!/usr/bin/env bash
# Retune sweep (2026-05-29): claw OOF RMSE 14.03 -> ~13.5 on n=175.
# Sequential + robust: one launch, no cache races. Each run uses --max_workers 16
# (the tower's 48 cores parallelize CV folds within a run). Screen at cv5;
# finalists get a cv10 confirm in Stage 5 (done manually after reading results).
# Results -> /tmp/retune_results.csv  (stage,label,oof_rmse,ci_lo,ci_hi,extra)
set -uo pipefail
cd "$(dirname "$0")"                      # arena_predictor/
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY="${PY:-python3}"
ROOT=..
CSV="../benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_v4_d32.csv"
BASE_IMP="--imputer_type model_bank --coherence_lambda 8.0 --coherence_shape exp --predictor_selection loo_forward --drop_style_tone --pls_hybrid_k 3 --max_workers 16"
RES=/tmp/retune_results.csv
echo "stage,label,oof_rmse,ci_lo,ci_hi,extra" > "$RES"

run() {  # stage label outroot cvr csv extra...
  local stage="$1" label="$2" outroot="$3" cvr="$4" csv="$5"; shift 5
  local extra="$*"
  mkdir -p "$outroot"
  if $PY -u predict.py --csv_path "$csv" $BASE_IMP --cv_repeats_outer "$cvr" \
        --output_root "$outroot" $extra > "/tmp/sw_${stage}_${label}.log" 2>&1; then
    local d r
    d=$(ls -td "$outroot"/output_* 2>/dev/null | head -1)
    r=$($PY -c "import json;m=json.load(open('$d/metadata.json'));print('%s,%s,%s'%(m['oof_rmse'],m['oof_rmse_ci_lo'],m['oof_rmse_ci_hi']))" 2>/dev/null || echo ",,")
    echo "$stage,$label,$r,\"$extra\"" >> "$RES"
    echo "[$stage/$label] $r"
  else
    echo "$stage,$label,FAIL,,,\"$extra\"" >> "$RES"
    echo "[$stage/$label] FAILED (see /tmp/sw_${stage}_${label}.log)"
  fi
}

echo "===== STAGE 0: baseline (primes _sweep_main cache) ====="
run s0 base_cv5  analysis_output/_sweep_main 5  "$CSV"
run s0 base_cv10 analysis_output/_sweep_main 10 "$CSV"

echo "===== STAGE 1: KNN knobs (cv5, cache reuse) ====="
for v in 0.5 0.6 0.8 0.9;        do run s1 "palpha_$v" analysis_output/_sweep_main 5 "$CSV" --knn_power_alpha "$v"; done
for v in 2.0 2.5 3.5 4.0;        do run s1 "powc_$v"   analysis_output/_sweep_main 5 "$CSV" --knn_power_c "$v"; done
for v in 50 65 100 120;          do run s1 "maxk_$v"   analysis_output/_sweep_main 5 "$CSV" --knn_max_k "$v"; done
for v in 12 16 25 30;            do run s1 "mink_$v"   analysis_output/_sweep_main 5 "$CSV" --knn_min_k "$v"; done
for v in 0.08 0.12 0.20 0.30;    do run s1 "bw_$v"     analysis_output/_sweep_main 5 "$CSV" --knn_bw_pct "$v"; done
for v in 0 2 4 6;                do run s1 "pls_$v"    analysis_output/_sweep_main 5 "$CSV" --pls_hybrid_k "$v"; done
for v in 1.25 1.75 2.0;          do run s1 "vclip_$v"  analysis_output/_sweep_main 5 "$CSV" --vi_clip_hi "$v"; done

echo "===== STAGE 2: feature-family ablation (cv5, cache reuse) ====="
for fam in tone_ sem_ writing_ logic_ aa_eval_ livebench_ weirdml_ arc_ ugileaderboard_ lechmazur_ contextarena_ yupp_ openbench_ "eqmt_,eqbench_,eq_" "aagdpval,aaomniscience,aacritpt"; do
  lbl=$(echo "$fam" | tr ',' '+' | tr -d ' ')
  run s2 "drop_$lbl" analysis_output/_sweep_main 5 "$CSV" --drop_families "$fam"
done
run s2 nosvd analysis_output/_sweep_main 5 "$CSV" --no_svd_in_features

echo "===== STAGE 3: imputation knobs (cv5, re-impute -> _sweep_imp) ====="
for v in 4 6 12 16;    do run s3 "cohl_$v" analysis_output/_sweep_imp 5 "$CSV" --coherence_lambda "$v"; done
run s3 cohshape_linear analysis_output/_sweep_imp 5 "$CSV" --coherence_shape linear
for v in 25 31 45 55;  do run s3 "selk_$v" analysis_output/_sweep_imp 5 "$CSV" --selector_k_max "$v"; done
for v in 0.90 0.96;    do run s3 "alpha_$v" analysis_output/_sweep_imp 5 "$CSV" --alpha "$v"; done
for v in 0.3 0.5;      do run s3 "conf_$v" analysis_output/_sweep_imp 5 "$CSV" --confidence_threshold "$v"; done

echo "===== STAGE 4: sem PCA dim (build + merge + predict) ====="
for N in 24 40 48; do
  ( cd "$ROOT/embeddings" && $PY build_fingerprints.py --mode per_bench_eq_split --n_components "$N" \
        --out "cache/model_fingerprints_v4_d${N}.csv" ) > "/tmp/sw_s4_build_d${N}.log" 2>&1
  ( cd "$ROOT" && $PY -c "
import pandas as pd
base=pd.read_csv('benchmark_combiner/benchmarks/clean_combined_all_benches.csv')
fp=pd.read_csv('embeddings/cache/model_fingerprints_v4_d${N}.csv')
base=base.drop(columns=[c for c in base.columns if c.startswith('sem_')],errors='ignore')
base.merge(fp,on='model_name',how='left').to_csv('benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_v4_d${N}.csv',index=False)
" ) >> "/tmp/sw_s4_build_d${N}.log" 2>&1
  run s4 "sem_d${N}" analysis_output/_sweep_sem 5 "../benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_v4_d${N}.csv"
done

echo "===== SWEEP COMPLETE. Results sorted by oof_rmse: ====="
( head -1 "$RES"; tail -n +2 "$RES" | sort -t, -k3 -g ) | column -t -s,
echo "(baseline cv10 is the s0/base_cv10 row; champion = 14.03)"
