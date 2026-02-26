#!/usr/bin/env python3
"""
Sweep Feature Selector 3 (LMSYS Predictor v6) - Optuna Version

Parallel hyperparameter optimization using Optuna's TPE sampler.
Supports multiple concurrent trials and live dashboard monitoring.

Usage:
    python sweep.py --n_trials 200 --n_jobs 8

Dashboard:
    pip install optuna-dashboard
    optuna-dashboard sqlite:///sweep_optuna.db
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
from optuna.samplers import TPESampler

# --- Configuration & Constants ---

# Parameter space definition
PARAM_SPACE_DEF: Dict[str, Dict[str, Any]] = {
    "alpha": {"type": "float", "bounds": (0.90, 0.99)},
    "selector_tau": {"type": "float", "bounds": (0.60, 0.99)},
    "selector_k_max": {"type": "int", "bounds": (10, 70)},
    "gp_selector_k_max": {"type": "int", "bounds": (5, 25)},  # v7.2
    "categorical_threshold": {"type": "int", "bounds": (0, 12)},
    "tolerance_percentile": {"type": "float", "bounds": (88.0, 99.8)},
    "tolerance_relaxation_factor": {"type": "float", "bounds": (1.00, 3.00)},
    "tolerance_multiplier": {"type": "float", "bounds": (1.5, 8.0)},
    "selector_cv": {"type": "fixed", "value": 5},
    "alt_selector_cv": {"type": "fixed", "value": 5},
    "feature_selector": {"type": "choice", "values": ["lgbm", "xgb"]},
    "alt_feature_selector": {"type": "choice", "values": ["lgbm", "xgb"]},
    "poly_interactions": {"type": "choice", "values": [False, True]},
    "poly_include_squares": {"type": "choice", "values": [False, True]},
    "poly_limit": {"type": "int", "bounds": (0, 20)},
    # v7.2: Per-column tolerance calibration
    "calibrate_tolerances": {"type": "choice", "values": [False, True]},
    "calibration_target_rmse_ratio": {"type": "float", "bounds": (0.3, 0.8)},
    "recalibrate_every_n_passes": {"type": "int", "bounds": (0, 5)},
}

KEY_PARAMS = [k for k, v in PARAM_SPACE_DEF.items() if v["type"] != "fixed"]

TARGET_MODELS = {"BayesianRidge"}

# Defaults - reduced for parallel execution
DEFAULT_CSV_PATH = "../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
DEFAULT_OUTPUT_ROOT = "analysis_output"
DEFAULT_PASSES = 14
DEFAULT_CV_JOBS = 1
DEFAULT_MODEL_JOBS = 1
DEFAULT_SELECTOR_N_JOBS = 2
DEFAULT_IMPUTER_JOBS = 2
DEFAULT_MAX_WORKERS = 4  # Per-trial worker cap
DEFAULT_SELECTOR_K_GRID = "4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,all"
DEFAULT_ALT_SELECTOR_K_GRID = "4,5,6,7,10,15,20,all"
DEFAULT_TIER_QUANTILES = "0.33,0.67"
DEFAULT_TOP_K_FEATURES = "auto"
DEFAULT_ALT_TOP_K_FEATURES = "auto"
DEFAULT_CV_REPEATS_OUTER = 5
DEFAULT_CV_REPEATS_INNER = 1
DEFAULT_FEATURE_CV_REPEATS = 1
DEFAULT_ALT_CV_REPEATS = 1
DEFAULT_N_JOBS = 4  # Parallel trials
DEFAULT_DB_PATH = "sweep_optuna.db"


def compute_file_hash(path: str) -> Optional[str]:
    """Return SHA256 for the CSV so we can detect data changes."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return None


def extract_run_csv_hash(run_config: Dict[str, Any]) -> Optional[str]:
    """Pull the CSV hash embedded in run_config['cache_key'] if present."""
    key = run_config.get("cache_key")
    if isinstance(key, str):
        prefix = "imputed_full_"
        suffix = "_passes"
        if key.startswith(prefix) and suffix in key:
            start = len(prefix)
            end = key.find(suffix, start)
            if end > start:
                return key[start:end]
    return run_config.get("csv_hash")


def _coerce_choice(value: Any, choices: List[Any]) -> Optional[Any]:
    """Try to coerce a run_config value into one of the allowed choices."""
    for choice in choices:
        if value == choice:
            return choice
        if isinstance(choice, str) and isinstance(value, str) and value.lower() == choice.lower():
            return choice
        if isinstance(choice, bool):
            if isinstance(value, bool) and value == choice:
                return choice
            if isinstance(value, (int, float)):
                if bool(value) == choice:
                    return choice
            if isinstance(value, str):
                lv = value.strip().lower()
                if lv in {"1", "true", "t", "yes", "y", "on"} and choice is True:
                    return choice
                if lv in {"0", "false", "f", "no", "n", "off"} and choice is False:
                    return choice
        if isinstance(choice, (int, float)) and isinstance(value, (int, float)):
            if abs(float(value) - float(choice)) < 1e-9:
                return choice
    return None


def load_existing_runs(output_root: Path, current_csv_hash: Optional[str], csv_path: str) -> List[Dict[str, Any]]:
    """Load completed runs from output directories to seed Optuna."""
    scored: List[Dict[str, Any]] = []

    if not output_root.exists():
        return scored

    for run_dir in sorted(output_root.glob("output_*")):
        rc_path = run_dir / "run_config.json"
        if not rc_path.exists():
            continue
        try:
            with rc_path.open("r", encoding="utf-8") as fh:
                rc = json.load(fh)
        except Exception:
            continue

        run_csv_hash = extract_run_csv_hash(rc)
        if current_csv_hash and run_csv_hash and run_csv_hash != current_csv_hash:
            continue
        if current_csv_hash and run_csv_hash is None:
            rc_csv_path = rc.get("csv_path")
            if rc_csv_path and Path(rc_csv_path).resolve() != Path(csv_path).resolve():
                continue

        combo: Dict[str, Any] = {}
        missing = False
        for param in KEY_PARAMS:
            if param not in rc:
                missing = True
                break
            spec = PARAM_SPACE_DEF[param]
            ptype = spec["type"]
            val = rc[param]
            if ptype == "int":
                combo[param] = int(val)
            elif ptype == "float":
                combo[param] = float(val)
            elif ptype == "choice":
                coerced = _coerce_choice(val, spec["values"])
                if coerced is None:
                    missing = True
                    break
                combo[param] = coerced
            else:
                missing = True
                break
        if missing:
            continue

        rmse_path = run_dir / "model_eval_rmse.csv"
        best_rmse = None
        if rmse_path.exists():
            try:
                with rmse_path.open("r", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    values = [
                        float(row["rmse_mean"])
                        for row in reader
                        if row.get("model") in TARGET_MODELS and row.get("rmse_mean")
                    ]
                if values:
                    best_rmse = min(values)
            except Exception:
                best_rmse = None

        if best_rmse is not None:
            scored.append({
                "params": combo,
                "score": best_rmse,
                "dir": str(run_dir),
            })

    return scored


class ObjectiveRunner:
    """Callable objective function for Optuna that runs the predictor."""

    def __init__(self, args):
        self.args = args
        self.output_root = Path(args.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Build the common command parts
        self.common_cmd = [
            sys.executable,
            args.predictor_script,
            "--csv_path", args.csv_path,
            "--passes", str(args.passes),
            "--cv_n_jobs", str(args.cv_jobs),
            "--model_n_jobs", str(args.model_jobs),
            "--selector_n_jobs", str(args.selector_n_jobs),
            "--imputer_n_jobs", str(args.imputer_n_jobs),
            "--max_workers", str(args.max_workers),
            "--output_root", str(self.output_root),
            "--tier_quantiles", str(args.tier_quantiles),
            "--top_k_features", str(args.top_k_features),
            "--alt_top_k_features", str(args.alt_top_k_features),
            "--selector_k_grid", str(args.selector_k_grid),
            "--alt_selector_k_grid", str(args.alt_selector_k_grid),
            "--cv_splits_path", str(args.cv_splits_path),
            "--cv_repeats_outer", str(args.cv_repeats_outer),
            "--cv_repeats_inner", str(args.cv_repeats_inner),
            "--feature_cv_repeats", str(args.feature_cv_repeats),
            "--alt_cv_repeats", str(args.alt_cv_repeats),
        ]

    def __call__(self, trial: optuna.Trial) -> float:
        # Sample parameters
        alpha = trial.suggest_float("alpha", 0.90, 0.99)
        selector_tau = trial.suggest_float("selector_tau", 0.60, 0.99)
        selector_k_max = trial.suggest_int("selector_k_max", 10, 70)
        gp_selector_k_max = trial.suggest_int("gp_selector_k_max", 5, 25)  # v7.2
        categorical_threshold = trial.suggest_int("categorical_threshold", 0, 12)
        tolerance_percentile = trial.suggest_float("tolerance_percentile", 88.0, 99.8)
        tolerance_relaxation_factor = trial.suggest_float("tolerance_relaxation_factor", 1.00, 3.00)
        tolerance_multiplier = trial.suggest_float("tolerance_multiplier", 1.5, 8.0)
        feature_selector = trial.suggest_categorical("feature_selector", ["lgbm", "xgb"])
        alt_feature_selector = trial.suggest_categorical("alt_feature_selector", ["lgbm", "xgb"])
        poly_interactions = trial.suggest_categorical("poly_interactions", [False, True])
        poly_include_squares = trial.suggest_categorical("poly_include_squares", [False, True])
        # v7.2: Per-column tolerance calibration
        calibrate_tolerances = trial.suggest_categorical("calibrate_tolerances", [False, True])
        calibration_target_rmse_ratio = trial.suggest_float("calibration_target_rmse_ratio", 0.3, 0.8)
        recalibrate_every_n_passes = trial.suggest_int("recalibrate_every_n_passes", 0, 5)

        # Enforce constraints
        if poly_include_squares and not poly_interactions:
            poly_interactions = True
            trial.set_user_attr("poly_interactions_forced", True)

        # Only sample poly_limit if poly_interactions is enabled
        if poly_interactions:
            poly_limit = trial.suggest_int("poly_limit", 1, 20)
        else:
            poly_limit = 0

        # Build command
        cmd = self.common_cmd + [
            "--alpha", str(alpha),
            "--selector_tau", str(selector_tau),
            "--selector_k_max", str(selector_k_max),
            "--gp_selector_k_max", str(gp_selector_k_max),  # v7.2
            "--categorical_threshold", str(categorical_threshold),
            "--tolerance_percentile", str(tolerance_percentile),
            "--tolerance_relaxation_factor", str(tolerance_relaxation_factor),
            "--tolerance_multiplier", str(tolerance_multiplier),
            "--selector_cv", str(PARAM_SPACE_DEF["selector_cv"]["value"]),
            "--alt_selector_cv", str(PARAM_SPACE_DEF["alt_selector_cv"]["value"]),
            "--feature_selector", str(feature_selector),
            "--alt_feature_selector", str(alt_feature_selector),
            "--poly_limit", str(poly_limit),
            # v7.2: Per-column tolerance calibration
            "--calibration_target_rmse_ratio", str(calibration_target_rmse_ratio),
            "--recalibrate_every_n_passes", str(recalibrate_every_n_passes),
        ]

        if poly_interactions:
            cmd.append("--poly_interactions")
        if poly_include_squares:
            cmd.append("--poly_include_squares")
        if calibrate_tolerances:
            cmd.append("--calibrate_tolerances")

        # Log trial info
        trial_str = f"Trial {trial.number}: alpha={alpha:.4f}, tau={selector_tau:.4f}, k_max={selector_k_max}"
        print(f"[Starting] {trial_str}")
        sys.stdout.flush()

        # Run the predictor
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            print(f"[FAILED] Trial {trial.number} failed with exit code {exc.returncode}")
            if exc.stderr:
                # Show last 2000 chars of stderr to capture the actual error
                print(f"  stderr (last 2000 chars): ...{exc.stderr[-2000:]}")
            raise optuna.TrialPruned(f"Predictor failed with exit {exc.returncode}")

        # Find the output and extract RMSE
        best_rmse = self._find_trial_rmse()

        if best_rmse is None:
            print(f"[FAILED] Trial {trial.number}: Could not find RMSE output")
            raise optuna.TrialPruned("Could not find RMSE output")

        print(f"[DONE] Trial {trial.number}: RMSE = {best_rmse:.6f}")
        return best_rmse

    def _find_trial_rmse(self) -> Optional[float]:
        """Find the most recent output directory and extract RMSE."""
        output_dirs = sorted(
            self.output_root.glob("output_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for run_dir in output_dirs[:5]:  # Check recent dirs
            rmse_path = run_dir / "model_eval_rmse.csv"
            if rmse_path.exists():
                try:
                    with rmse_path.open("r", encoding="utf-8") as fh:
                        reader = csv.DictReader(fh)
                        values = [
                            float(row["rmse_mean"])
                            for row in reader
                            if row.get("model") in TARGET_MODELS and row.get("rmse_mean")
                        ]
                    if values:
                        return min(values)
                except Exception:
                    continue

        return None


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Feature Selector 3 (Optuna)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run 200 trials with 8 parallel jobs
    python sweep.py --n_trials 200 --n_jobs 8

    # View dashboard (in another terminal)
    pip install optuna-dashboard
    optuna-dashboard sqlite:///sweep_optuna.db
        """
    )

    # Optuna settings
    parser.add_argument("--n_trials", type=int, default=200, help="Number of trials to run")
    parser.add_argument("--n_jobs", type=int, default=DEFAULT_N_JOBS, help="Number of parallel trials")
    parser.add_argument("--db_path", type=str, default=DEFAULT_DB_PATH, help="SQLite database path for Optuna")
    parser.add_argument("--study_name", type=str, default="lmsys_sweep", help="Optuna study name")
    parser.add_argument("--seed_from_existing", action="store_true", help="Seed study with existing runs from output_root")

    # Predictor settings
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH, help="Path to the CSV file")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Root directory for output")
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES, help="Number of imputation passes")
    parser.add_argument("--predictor_script", type=str, default="predict.py", help="Script to run")

    # Per-trial resource limits (reduced for parallelism)
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS, help="Max workers per trial")
    parser.add_argument("--cv_jobs", type=int, default=DEFAULT_CV_JOBS, help="CV jobs per trial")
    parser.add_argument("--model_jobs", type=int, default=DEFAULT_MODEL_JOBS, help="Model jobs per trial")
    parser.add_argument("--selector_n_jobs", type=int, default=DEFAULT_SELECTOR_N_JOBS, help="Selector jobs per trial")
    parser.add_argument("--imputer_n_jobs", type=int, default=DEFAULT_IMPUTER_JOBS, help="Imputer jobs per trial")

    # Other predictor settings
    parser.add_argument("--tier_quantiles", type=str, default=DEFAULT_TIER_QUANTILES, help="Tier quantiles string")
    parser.add_argument("--top_k_features", type=str, default=DEFAULT_TOP_K_FEATURES, help="Target top-k mode")
    parser.add_argument("--alt_top_k_features", type=str, default=DEFAULT_ALT_TOP_K_FEATURES, help="ALT top-k mode")
    parser.add_argument("--selector_k_grid", type=str, default=DEFAULT_SELECTOR_K_GRID, help="Grid for target selector")
    parser.add_argument("--alt_selector_k_grid", type=str, default=DEFAULT_ALT_SELECTOR_K_GRID, help="Grid for ALT selector")
    parser.add_argument("--cv_splits_path", type=str, default="sweep_cv_splits.json", help="Path to reuse CV splits across trials")
    parser.add_argument("--cv_repeats_outer", type=int, default=DEFAULT_CV_REPEATS_OUTER,
                        help="Repeats for outer model CV + uncertainty calibration")
    parser.add_argument("--cv_repeats_inner", type=int, default=DEFAULT_CV_REPEATS_INNER,
                        help="Repeats for inner ALT OOF CV")
    parser.add_argument("--feature_cv_repeats", type=int, default=DEFAULT_FEATURE_CV_REPEATS,
                        help="Repeats for tree-based feature selection CV")
    parser.add_argument("--alt_cv_repeats", type=int, default=DEFAULT_ALT_CV_REPEATS,
                        help="Repeats for ALT imputation CV report")

    args = parser.parse_args()

    if args.n_trials <= 0:
        print("Nothing to do: n_trials <= 0")
        sys.exit(0)

    # Create/load the Optuna study with SQLite storage
    storage_url = f"sqlite:///{args.db_path}"

    sampler = TPESampler(
        seed=42,
        n_startup_trials=10,  # Random trials before TPE kicks in
        multivariate=True,    # Model parameter interactions
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    # Optionally seed with existing runs
    if args.seed_from_existing:
        output_root = Path(args.output_root)
        csv_hash = compute_file_hash(args.csv_path)
        existing = load_existing_runs(output_root, csv_hash, args.csv_path)
        if existing:
            print(f"Found {len(existing)} existing runs to potentially seed from")
            # Note: Optuna handles duplicates, so we just enqueue
            for run in existing:
                try:
                    study.enqueue_trial(run["params"])
                except Exception:
                    pass

    # Print study info
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Study '{args.study_name}' loaded with {n_complete} completed trials")
    print(f"Database: {args.db_path}")
    print(f"Running {args.n_trials} trials with {args.n_jobs} parallel jobs")
    print(f"Per-trial resources: max_workers={args.max_workers}, cv_jobs={args.cv_jobs}, "
          f"model_jobs={args.model_jobs}, selector_n_jobs={args.selector_n_jobs}")
    print()
    print("To view live dashboard:")
    print(f"    optuna-dashboard sqlite:///{args.db_path}")
    print()
    sys.stdout.flush()

    # Create the objective runner
    objective = ObjectiveRunner(args)

    # Run optimization
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
        catch=(Exception,),  # Don't crash on individual trial failures
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

    print(f"Completed: {n_complete}, Pruned: {n_pruned}, Failed: {n_failed}")

    if study.best_trial:
        print(f"\nBest RMSE: {study.best_value:.6f}")
        print("Best parameters:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print(f"\nFull results stored in: {args.db_path}")
    print(f"View with: optuna-dashboard sqlite:///{args.db_path}")


if __name__ == "__main__":
    main()
