#!/usr/bin/env python3
"""Prediction-level stacking of model_bank and specialized imputer pipelines.

Runs both imputers through the full pipeline, collects OOF predictions,
and finds the optimal blend weight via leave-one-out cross-validation.

Usage:
    python3 stack_imputers.py [--freeze_alt_pairs PATH]
"""
import subprocess
import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def run_pipeline(imputer_type: str, extra_args: list[str] = None) -> str:
    """Run predict.py with given imputer type, return output directory path."""
    cmd = [
        sys.executable, "predict.py",
        "--imputer_type", imputer_type,
    ]
    if imputer_type == "model_bank":
        cmd += ["--coherence_lambda", "1.0", "--coherence_shape", "exp"]
    if extra_args:
        cmd += extra_args
    print(f"\n{'='*60}")
    print(f"Running {imputer_type} pipeline...")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".")
    # Extract output dir from stdout
    for line in result.stdout.splitlines():
        if "Results saved to:" in line:
            out_dir = line.split("Results saved to:")[-1].strip()
            print(f"  -> {out_dir}")
            return out_dir
        # Also print key metrics
        if any(k in line for k in ["OOF RMSE", "Model comparison", "impute:", "ALT nested"]):
            print(f"  {line.strip()}")
    # If we didn't find the output dir, print stderr
    if result.returncode != 0:
        print(f"ERROR: {result.stderr[-500:]}")
    return None


def load_oof(out_dir: str) -> pd.DataFrame:
    """Load OOF predictions from an output directory."""
    path = os.path.join(out_dir, "oof_predictions.csv")
    df = pd.read_csv(path)
    return df


def find_optimal_alpha(y_true, pred_a, pred_b, n_grid=101):
    """Find optimal blend weight via grid search on RMSE."""
    best_alpha, best_rmse = 0.0, float("inf")
    for alpha in np.linspace(0, 1, n_grid):
        blended = alpha * pred_a + (1 - alpha) * pred_b
        rmse = np.sqrt(mean_squared_error(y_true, blended))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    return best_alpha, best_rmse


def loo_blend(y_true, pred_a, pred_b):
    """Leave-one-out optimal blend to avoid overfitting alpha."""
    n = len(y_true)
    loo_preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        # Find alpha on n-1 points
        alpha, _ = find_optimal_alpha(y_true[mask], pred_a[mask], pred_b[mask])
        loo_preds[i] = alpha * pred_a[i] + (1 - alpha) * pred_b[i]
    loo_rmse = np.sqrt(mean_squared_error(y_true, loo_preds))
    # Also report the full-data alpha for reference
    full_alpha, full_rmse = find_optimal_alpha(y_true, pred_a, pred_b)
    return loo_rmse, loo_preds, full_alpha, full_rmse


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze_alt_pairs", type=str, default=None)
    args = ap.parse_args()

    extra = []
    if args.freeze_alt_pairs:
        extra += ["--freeze_alt_pairs", args.freeze_alt_pairs]

    # Run both pipelines
    mb_dir = run_pipeline("model_bank", extra)
    sp_dir = run_pipeline("specialized", extra)

    if not mb_dir or not sp_dir:
        print("ERROR: one or both pipelines failed")
        sys.exit(1)

    # Load OOF predictions
    mb_oof = load_oof(mb_dir)
    sp_oof = load_oof(sp_dir)

    # Merge on model name
    merged = mb_oof.merge(sp_oof, on="model_name", suffixes=("_mb", "_sp"))
    y_true = merged["actual_score_mb"].values
    pred_mb = merged["oof_predicted_score_mb"].values
    pred_sp = merged["oof_predicted_score_sp"].values

    # Individual RMSEs
    rmse_mb = np.sqrt(mean_squared_error(y_true, pred_mb))
    rmse_sp = np.sqrt(mean_squared_error(y_true, pred_sp))

    print(f"\n{'='*60}")
    print("STACKING RESULTS")
    print(f"{'='*60}")
    print(f"Model-bank RMSE:     {rmse_mb:.2f}")
    print(f"Specialized RMSE:    {rmse_sp:.2f}")

    # Correlation between predictions
    corr = np.corrcoef(pred_mb, pred_sp)[0, 1]
    print(f"Prediction corr:     {corr:.4f}")

    # Correlation between errors
    err_mb = y_true - pred_mb
    err_sp = y_true - pred_sp
    err_corr = np.corrcoef(err_mb, err_sp)[0, 1]
    print(f"Error correlation:   {err_corr:.4f}")

    # Grid search for optimal alpha (in-sample, for reference)
    full_alpha, full_rmse = find_optimal_alpha(y_true, pred_mb, pred_sp)
    print(f"\nFull-data optimal:   α={full_alpha:.2f} -> RMSE {full_rmse:.2f}")
    print(f"  (α=1 is pure model-bank, α=0 is pure specialized)")

    # LOO blend (honest evaluation)
    loo_rmse, loo_preds, _, _ = loo_blend(y_true, pred_mb, pred_sp)
    print(f"LOO blend RMSE:      {loo_rmse:.2f}")

    # Fixed alpha blends
    for alpha in [0.25, 0.33, 0.50, 0.67, 0.75]:
        blended = alpha * pred_mb + (1 - alpha) * pred_sp
        rmse = np.sqrt(mean_squared_error(y_true, blended))
        print(f"  α={alpha:.2f} blend:       {rmse:.2f}")

    # Bootstrap CI for LOO blend
    rng = np.random.RandomState(42)
    residuals_sq = (y_true - loo_preds) ** 2
    n = len(residuals_sq)
    boot_rmses = [np.sqrt(np.mean(residuals_sq[rng.randint(0, n, size=n)])) for _ in range(2000)]
    ci_lo, ci_hi = np.percentile(boot_rmses, [2.5, 97.5])
    print(f"  LOO blend 95% CI:  ({ci_lo:.2f} – {ci_hi:.2f})")

    # Save results
    results = {
        "rmse_model_bank": float(rmse_mb),
        "rmse_specialized": float(rmse_sp),
        "prediction_correlation": float(corr),
        "error_correlation": float(err_corr),
        "full_data_alpha": float(full_alpha),
        "full_data_blend_rmse": float(full_rmse),
        "loo_blend_rmse": float(loo_rmse),
        "loo_ci_lo": float(ci_lo),
        "loo_ci_hi": float(ci_hi),
        "model_bank_dir": mb_dir,
        "specialized_dir": sp_dir,
    }
    out_path = os.path.join(os.path.dirname(mb_dir), "stacking_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
