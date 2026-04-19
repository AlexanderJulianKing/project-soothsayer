#!/usr/bin/env bash
# Pre-run: combine external + custom benchmarks, run correlations, build
# response-embedding fingerprints, and merge them in to produce the champion
# predictor CSV at clean_combined_all_benches_with_sem_v4_d32.csv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$ROOT/.env" ]]; then
    set -a
    source "$ROOT/.env"
    set +a
fi
bash "$ROOT/benchmark_combiner/collect_benchmarks.sh"

START_TIME=$SECONDS

# ~~~ 1. External-benchmark combine + correlations ~~~
cd "$ROOT/benchmark_combiner"
python3 combine.py
python3 correlations.py

CLEAN_CSV="benchmarks/clean_combined_all_benches.csv"
if [[ ! -f "$CLEAN_CSV" ]]; then
    echo "ERROR: combine.py did not produce $CLEAN_CSV; aborting before embedding step." >&2
    exit 1
fi

# ~~~ 2. Response-embedding fingerprint chain (champion: bge-small, 5-slot, d=32) ~~~
#
# collect_responses.py and embed_responses.py are both idempotent / resumable:
#   - collect rebuilds all_responses.parquet from the 4 benchmark dirs
#   - embed compares keys vs existing response_embeddings.parquet and only
#     embeds rows whose (model, benchmark, prompt_id, run_id) is new
# So incremental runs (a handful of new models) add seconds, not minutes.
#
# If the embedding stack isn't installed (sentence-transformers, torch), we skip
# the embed chain and predict.sh falls back to the base CSV without sem_*.
cd "$ROOT"
FINGERPRINTS="embeddings/cache/model_fingerprints_v4_d32.csv"
AUGMENTED_CSV="benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_v4_d32.csv"

if python3 -c "import sentence_transformers, torch" 2>/dev/null; then
    echo ""
    echo "Building response-embedding fingerprints (champion: bge-small, 5-slot, d=32)..."
    python3 embeddings/collect_responses.py
    python3 embeddings/embed_responses.py
    python3 embeddings/build_fingerprints.py \
        --mode per_bench_eq_split --n_components 32 \
        --out "$FINGERPRINTS"

    BASE_CSV_PATH="$ROOT/benchmark_combiner/$CLEAN_CSV" \
    FINGERPRINTS_PATH="$ROOT/$FINGERPRINTS" \
    AUGMENTED_CSV_PATH="$ROOT/$AUGMENTED_CSV" \
    python3 - <<'PY'
import os
import pandas as pd
base = pd.read_csv(os.environ["BASE_CSV_PATH"])
fp = pd.read_csv(os.environ["FINGERPRINTS_PATH"])
base = base.drop(columns=[c for c in base.columns if c.startswith("sem_")], errors="ignore")
merged = base.merge(fp, on="model_name", how="left")
merged.to_csv(os.environ["AUGMENTED_CSV_PATH"], index=False)
sem_cols = [c for c in merged.columns if c.startswith("sem_")]
n_matched = merged[sem_cols].notna().all(axis=1).sum()
print(f"merged: {merged.shape[0]} models × {merged.shape[1]} cols; "
      f"{n_matched} with sem_* populated")
PY
else
    echo ""
    echo "WARN: sentence-transformers/torch not importable — skipping sem_ fingerprints."
    echo "      predict.sh will fall back to the base CSV (no sem_* features)."
fi

ELAPSED=$(( SECONDS - START_TIME ))
FINAL_CSV="$ROOT/benchmark_combiner/$CLEAN_CSV"
if [[ -f "$ROOT/$AUGMENTED_CSV" ]]; then
    FINAL_CSV="$ROOT/$AUGMENTED_CSV"
fi
ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$FINAL_CSV')))")
COLS=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$FINAL_CSV').columns))")
echo ""
echo "Combine complete: ${ROWS} models, ${COLS} columns in $(basename "$FINAL_CSV"), ${ELAPSED}s elapsed."
