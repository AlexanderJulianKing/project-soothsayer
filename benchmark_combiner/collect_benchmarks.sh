#!/usr/bin/env bash
# collect_benchmarks.sh — Copy latest custom benchmark result CSVs into benchmarks/
#
# Copies only the latest dated file per prefix from each custom benchmark runner.
# Run from the benchmark_combiner/ directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/benchmarks"

mkdir -p "$DEST"

copied=0
skipped=0

# Copy only the latest file matching a glob pattern (sorted lexically, last = newest date)
copy_latest() {
    local src_pattern="$1"
    local latest=""
    for src in $src_pattern; do
        [ -f "$src" ] || continue
        latest="$src"
    done
    [ -z "$latest" ] && return
    local basename
    basename="$(basename "$latest")"
    if [ -f "$DEST/$basename" ] && [ ! "$latest" -nt "$DEST/$basename" ]; then
        skipped=$((skipped + 1))
    else
        cp "$latest" "$DEST/$basename"
        echo "  Copied: $basename"
        copied=$((copied + 1))
    fi
}

echo "Collecting custom benchmark results into benchmarks/..."

# soothsayer_eq
copy_latest "$SCRIPT_DIR/../soothsayer_eq/results/eq_*.csv"

# soothsayer_writing
copy_latest "$SCRIPT_DIR/../soothsayer_writing/results/writing_direct_*.csv"
copy_latest "$SCRIPT_DIR/../soothsayer_writing/results/writing_[0-9]*.csv"

# soothsayer_style
copy_latest "$SCRIPT_DIR/../soothsayer_style/outputs/style_[0-9]*.csv"
copy_latest "$SCRIPT_DIR/../soothsayer_style/outputs/tone_*.csv"

# soothsayer_logic
copy_latest "$SCRIPT_DIR/../soothsayer_logic/output/logic_*.csv"

echo "Done. Copied $copied new file(s), skipped $skipped already present."
