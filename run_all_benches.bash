#!/usr/bin/env bash
# Run all 4 homemade benchmarks in parallel.
# Each benchmark's multi-step pipeline runs sequentially within its own
# background subshell, but all 4 subshells run concurrently.
#
# Usage:  ./run_all_benches.bash [--skip-preflight] [bench ...]
#   No args  → run all 4
#   With args → run only the named ones (eq, logic, style, writing)
#
# Logs go to  logs/<bench>_YYYYMMDD_HHMMSS.log

set -euo pipefail

# ── parse flags ──────────────────────────────────────────────────────────
SKIP_PREFLIGHT=false
BENCH_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --skip-preflight) SKIP_PREFLIGHT=true ;;
        *) BENCH_ARGS+=("$arg") ;;
    esac
done
set -- "${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}"

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Load .env if present (provides OPENROUTER_API_KEY)
if [[ -f "$ROOT/.env" ]]; then
    set -a
    source "$ROOT/.env"
    set +a
fi

LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)

# ── benchmark definitions ───────────────────────────────────────────────
# bench_dir <name>  → prints the working directory
bench_dir() {
    case "$1" in
        eq)           echo "$ROOT/soothsayer_eq" ;;
        logic)        echo "$ROOT/soothsayer_logic" ;;
        style)        echo "$ROOT/soothsayer_style" ;;
        writing)      echo "$ROOT/soothsayer_writing" ;;
        *) return 1 ;;
    esac
}

# bench_cmds <name>  → prints '|'-separated commands
bench_cmds() {
    case "$1" in
        eq)           echo "python3 main.py|python3 super_bench.py" ;;
        logic)        echo "python3 collect_and_grade.py|python3 score.py" ;;
        style)        echo "python3 collect.py|python3 super_bench.py|python3 score.py" ;;
        writing)      echo "python3 main.py|python3 super_bench.py" ;;
        *) return 1 ;;
    esac
}

ALL_BENCHES="eq logic style writing"

# ── helpers ──────────────────────────────────────────────────────────────
run_bench() {
    local name="$1"
    local dir="$2"
    local logfile="$LOGDIR/${name}_${STAMP}.log"
    shift 2
    echo "[$(date +%H:%M:%S)] Starting $name  →  $logfile"
    (
        cd "$dir"
        for cmd in "$@"; do
            echo "=== $(date +%H:%M:%S) Running: $cmd ===" >> "$logfile"
            eval "$cmd" >> "$logfile" 2>&1
        done
        echo "=== $(date +%H:%M:%S) $name DONE ===" >> "$logfile"
    ) &
}

# ── preflight ────────────────────────────────────────────────────────────
if [[ "$SKIP_PREFLIGHT" == false ]]; then
    echo "Running preflight checks..."
    echo ""
    if ! python3 "$ROOT/preflight.py"; then
        echo ""
        echo "Aborting. Fix the issues above or re-run with --skip-preflight to bypass."
        exit 1
    fi
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo ""
fi

# ── decide which to run ─────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
    SELECTED=("$@")
else
    SELECTED=($ALL_BENCHES)
fi

# ── launch ───────────────────────────────────────────────────────────────
PIDS=()
for bench in "${SELECTED[@]}"; do
    dir=$(bench_dir "$bench" 2>/dev/null) || {
        echo "Unknown benchmark: $bench"
        echo "Valid names: $ALL_BENCHES"
        exit 1
    }

    IFS='|' read -ra cmds <<< "$(bench_cmds "$bench")"
    run_bench "$bench" "$dir" "${cmds[@]}"
    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} benchmarks launched.  PIDs: ${PIDS[*]}"
echo "Logs:  $LOGDIR/*_${STAMP}.log"
echo ""
echo "Waiting for all to finish (Ctrl-C to cancel)..."

# ── wait & report ────────────────────────────────────────────────────────
FAILURES=0
for i in "${!SELECTED[@]}"; do
    bench="${SELECTED[$i]}"
    pid="${PIDS[$i]}"
    if wait "$pid"; then
        echo "  ✓ $bench finished successfully"
    else
        echo "  ✗ $bench FAILED (exit code $?)"
        ((FAILURES++))
    fi
done

echo ""
if [[ $FAILURES -eq 0 ]]; then
    echo "All benchmarks completed successfully."
else
    echo "$FAILURES benchmark(s) failed. Check logs for details."
    exit 1
fi
