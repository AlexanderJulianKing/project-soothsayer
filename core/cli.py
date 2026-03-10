"""CLI entry point for running benchmarks.

Usage:
    python -m core.cli                          # Run all 4 benchmarks
    python -m core.cli eq writing               # Run named benchmarks
    python -m core.cli --skip-preflight         # Skip preflight checks
    python -m core.cli --list-completed         # Show completed models per benchmark
"""

import argparse
import datetime as dt
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

from core.benchmark import Benchmark, BenchmarkResult

# Lazy imports to avoid loading pandas/etc at CLI parse time
_REGISTRY: Dict[str, type] = {}


def _load_registry() -> Dict[str, Benchmark]:
    """Import and instantiate all benchmark classes."""
    if _REGISTRY:
        return {name: cls() for name, cls in _REGISTRY.items()}

    # Import here to avoid circular imports and slow startup
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from soothsayer_eq.benchmark import EQBenchmark
    from soothsayer_writing.benchmark import WritingBenchmark
    from soothsayer_style.benchmark import StyleBenchmark
    from soothsayer_logic.benchmark import LogicBenchmark
    from soothsayer_zebra.benchmark import ZebraBenchmark

    _REGISTRY["eq"] = EQBenchmark
    _REGISTRY["writing"] = WritingBenchmark
    _REGISTRY["logic"] = LogicBenchmark
    _REGISTRY["style"] = StyleBenchmark
    _REGISTRY["zebra"] = ZebraBenchmark

    return {name: cls() for name, cls in _REGISTRY.items()}


def run_preflight(root_dir: str) -> bool:
    """Run preflight.py, return True if it passes."""
    preflight_path = os.path.join(root_dir, "preflight.py")
    if not os.path.exists(preflight_path):
        print("Warning: preflight.py not found, skipping.")
        return True
    result = subprocess.run(
        [sys.executable, preflight_path],
        cwd=root_dir,
    )
    return result.returncode == 0


def list_completed():
    """Print completed model counts per benchmark."""
    benchmarks = _load_registry()
    for name, bench in sorted(benchmarks.items()):
        completed = bench.get_completed_models()
        print(f"  {name:15s}  {len(completed):3d} models done")


def run_benchmarks(selected_names, skip_preflight=False):
    """Run the selected benchmarks in parallel."""
    root_dir = os.path.join(os.path.dirname(__file__), '..')

    if not skip_preflight:
        print("Running preflight checks...")
        if not run_preflight(root_dir):
            print("\nPreflight FAILED. Fix the issues above or use --skip-preflight.")
            return 1
        print()

    benchmarks = _load_registry()

    # Validate names
    for name in selected_names:
        if name not in benchmarks:
            print(f"Unknown benchmark: {name}")
            print(f"Valid names: {', '.join(sorted(benchmarks.keys()))}")
            return 1

    selected = {name: benchmarks[name] for name in selected_names}

    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print(f"Launching {len(selected)} benchmarks: {', '.join(selected.keys())}")
    print(f"Logs: logs/*_{stamp}.log\n")

    def run_one(name, bench):
        log_path = os.path.join(log_dir, f"{name}_{stamp}.log")
        results = bench.run_all()
        # Write log
        with open(log_path, "w") as f:
            for r in results:
                f.write(f"=== Stage: {r.stage} (exit {r.exit_code}) ===\n")
                f.write(r.log_output)
                f.write("\n")
        return name, results, log_path

    failures = 0
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures = {pool.submit(run_one, name, bench): name for name, bench in selected.items()}
        for future in as_completed(futures):
            name, results, log_path = future.result()
            final = results[-1] if results else None
            if final and final.exit_code == 0:
                stages_done = len([r for r in results if r.exit_code == 0])
                print(f"  OK  {name} ({stages_done}/{len(results)} stages)")
            else:
                failures += 1
                failed_stage = next((r for r in results if r.exit_code != 0), None)
                stage_name = failed_stage.stage if failed_stage else "?"
                print(f"  FAIL  {name} (failed at stage: {stage_name})")
                print(f"        See: {log_path}")

    print()
    if failures == 0:
        print("All benchmarks completed successfully.")
        return 0
    else:
        print(f"{failures} benchmark(s) failed. Check logs for details.")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Soothsayer benchmark runner",
        prog="soothsayer",
    )
    parser.add_argument(
        "benchmarks", nargs="*",
        help="Benchmarks to run (default: all). Valid: eq, writing, logic, style, zebra",
    )
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip API smoke tests before running benchmarks",
    )
    parser.add_argument(
        "--list-completed", action="store_true",
        help="Show completed model counts per benchmark and exit",
    )
    args = parser.parse_args()

    if args.list_completed:
        list_completed()
        return 0

    if not args.benchmarks:
        args.benchmarks = ["eq", "writing", "logic", "style", "zebra"]

    return run_benchmarks(args.benchmarks, skip_preflight=args.skip_preflight)


if __name__ == "__main__":
    sys.exit(main())
