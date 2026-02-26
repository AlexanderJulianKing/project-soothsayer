#!/usr/bin/env bash
# Pre-run: combine benchmarks + run correlations to produce clean_combined_all_benches.csv
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$ROOT/.env" ]]; then
    set -a
    source "$ROOT/.env"
    set +a
fi

cd "$ROOT/benchmark_combiner"

START_TIME=$SECONDS

python3 combine.py
python3 correlations.py

ELAPSED=$(( SECONDS - START_TIME ))
CLEAN_CSV="benchmarks/clean_combined_all_benches.csv"
if [[ -f "$CLEAN_CSV" ]]; then
    ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$CLEAN_CSV')))")
    COLS=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$CLEAN_CSV').columns))")
    echo ""
    echo "Combine complete: ${ROWS} models, ${COLS} columns in clean CSV, ${ELAPSED}s elapsed."
fi
