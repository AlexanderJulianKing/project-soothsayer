#!/usr/bin/env bash
# Collect benchmark data from all 13+ sources, then combine.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$ROOT/.env" ]]; then
    set -a
    source "$ROOT/.env"
    set +a
fi
bash benchmark_combiner/collect_benchmarks.sh
cd "$ROOT/benchmark_combiner"

START_TIME=$SECONDS


python3 ../scrapers/arena_ai_grabber.py
# python3 ../scrapers/lmarena_grabber.py  # replaced by arena_ai_grabber.py
# python3 ../scrapers/vectara_grabber.py
# python3 ../scrapers/lechmazur_grabber.py
python3 ../scrapers/livebench_grabber.py
python3 ../scrapers/aa_models_grabber.py
python3 ../scrapers/aa_evaluations_grabber.py
python3 ../scrapers/aiderbench_grabber.py
python3 ../scrapers/context_arena_grabber.py
python3 ../scrapers/arc_grabber.py
python3 ../scrapers/eqbench_grabber.py
python3 ../scrapers/ugi_leaderboard_grabber.py
python3 ../scrapers/weirdml_grabber.py
python3 ../scrapers/yupp_grabber.py
python3 combine.py

ELAPSED=$(( SECONDS - START_TIME ))
CSV_COUNT=$(ls benchmarks/*.csv 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "Scrape complete: ${CSV_COUNT} CSV files in benchmarks/, ${ELAPSED}s elapsed."
