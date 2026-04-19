#!/usr/bin/env python3
"""
Preflight smoke-test for the benchmark suite.

Checks each benchmark's output files to find models that still need evaluation,
then fires one cheap API call per untested model (in parallel) to verify the
model responds correctly with the right reasoning config.

Also validates the judge models used by eq and writing benchmarks.

Exit 0  → all good, safe to launch
Exit 1  → something's broken, prints what went wrong
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── locate openbench CSV ────────────────────────────────────────────────
from core.utils import discover_openbench_csv
try:
    csv_path = discover_openbench_csv(SCRIPT_DIR)
except ValueError:
    print("PREFLIGHT FAIL: No openbench_*.csv found.")
    sys.exit(1)

# ── imports ─────────────────────────────────────────────────────────────
import pandas as pd

from core.llm_client import get_llm_response, API_KEY

# ── load models ─────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
df = df.dropna(subset=["openbench_id"])
df["Model"] = df["Model"].str.strip()

PROMPT = (
    "A farmer has 17 sheep. All but 9 die. How many sheep are left? "
    "Think through this carefully and explain your reasoning before giving the final number."
)
MAX_WORKERS = 30

# Judge models used by the actual benchmarks
JUDGE_MODELS = {
    "writing": "Grok 4 Fast",
    "eq":      "Gemini 3.0 Flash Preview (2025-12-17)",
}


# ── per-benchmark completion checks (delegated to benchmark adapters) ─
from soothsayer_eq.benchmark import EQBenchmark
from soothsayer_writing.benchmark import WritingBenchmark
from soothsayer_style.benchmark import StyleBenchmark
from soothsayer_logic.benchmark import LogicBenchmark

_BENCHMARKS = {
    "eq":       EQBenchmark(),
    "writing":  WritingBenchmark(),
    "logic":    LogicBenchmark(),
    "style":    StyleBenchmark(),
}


def get_models_needing_testing():
    """Return list of (name, openbench_id, reasoning) for models not yet
    fully evaluated across all 4 benchmarks."""
    completed = {name: bench.get_completed_models() for name, bench in _BENCHMARKS.items()}

    print("Existing results:")
    for bench, done in completed.items():
        print(f"  {bench:15s}  {len(done):3d} models done")
    print()

    needed = []
    skipped = 0
    for _, row in df.iterrows():
        name = row["Model"]
        mid = row["openbench_id"]
        reasoning = bool(row["Reasoning"])

        in_style     = name in completed["style"]
        in_simple    = name in completed["logic"]
        in_eq        = name in completed["eq"]
        in_writing   = name in completed["writing"]

        if in_style and in_simple and in_eq and in_writing:
            skipped += 1
        else:
            missing_in = []
            if not in_style:    missing_in.append("style")
            if not in_simple:   missing_in.append("logic")
            if not in_eq:       missing_in.append("eq")
            if not in_writing:  missing_in.append("writing")
            needed.append((name, mid, reasoning, missing_in))

    print(f"Skipping {skipped} models already evaluated by all benchmarks.")
    print(f"Need to test {len(needed)} models.\n")
    return needed


# ── single-model test ──────────────────────────────────────────────────
def test_call(name, model_id, reasoning):
    """Fire one API call, return (ok, elapsed, detail)."""
    t0 = time.time()
    try:
        resp, usage = get_llm_response(
            prompt=PROMPT,
            model=model_id,
            name=name,
            reasoning=reasoning,
            include_usage=True,
        )
    except Exception as e:
        return False, time.time() - t0, f"{type(e).__name__}: {e}"

    elapsed = time.time() - t0

    if not resp or not resp.strip():
        return False, elapsed, "empty response"

    if len(resp) > 5000:
        return False, elapsed, f"suspiciously long ({len(resp)} chars)"

    reasoning_toks = usage.get("reasoning_tokens", 0) or 0
    completion_toks = usage.get("completion_tokens", 0) or 0
    tok_str = f"reasoning={reasoning_toks}, completion={completion_toks}"

    return True, elapsed, f"{resp.strip()[:80]}\"  [{tok_str}]"


# ── main ────────────────────────────────────────────────────────────────
def main():
    print(f"Preflight using: {os.path.basename(csv_path)}")
    print(f"API key: {API_KEY[:12]}...{API_KEY[-4:]}")
    print()

    needed = get_models_needing_testing()

    # Build task list: (label, name, model_id, reasoning)
    tasks = []
    for name, mid, reasoning, missing_in in needed:
        tag = "R" if reasoning else " "
        missing_str = ",".join(missing_in)
        tasks.append((f"[{tag}] {name}  (needs: {missing_str})", name, mid, reasoning))

    # Add judge models
    tested_names = {t[1] for t in tasks}
    for bench_label, judge_name in JUDGE_MODELS.items():
        if judge_name in tested_names:
            continue
        match = df[df["Model"] == judge_name]
        if match.empty:
            tasks.append((f"[J] {judge_name}", judge_name, None, False))
        else:
            r = match.iloc[0]
            tasks.append((f"[J] {judge_name}", r["Model"], r["openbench_id"], bool(r["Reasoning"])))

    total = len(tasks)
    if total == 0:
        print("Nothing to test — all models already evaluated everywhere.")
        return 0

    print(f"Testing {total} models with {MAX_WORKERS} parallel workers...\n")

    passed = []
    failed = []

    def run_one(task):
        label, name, model_id, reasoning = task
        if model_id is None:
            return task, (False, 0.0, "NOT FOUND in openbench CSV")
        return task, test_call(name, model_id, reasoning)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for future in as_completed(futures):
            task, (ok, elapsed, detail) = future.result()
            label = task[0]
            if ok:
                passed.append((label, elapsed, detail))
                print(f"  PASS  {label}  ({elapsed:.1f}s)  →  \"{detail}")
            else:
                failed.append((label, elapsed, detail))
                print(f"  FAIL  {label}  ({elapsed:.1f}s)  →  {detail}")

    # ── summary ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  {len(passed)}/{total} passed,  {len(failed)}/{total} failed")

    if failed:
        print(f"\nFailed models:")
        for label, elapsed, detail in failed:
            print(f"  {label}  →  {detail}")
        print(f"\nPreflight FAILED — fix the issues above before running benchmarks.")
        return 1
    else:
        print(f"\nAll preflight checks passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
