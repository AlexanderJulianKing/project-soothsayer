#!/usr/bin/env python3
"""
Analyze Optuna sweep results from sweep_optuna.db.

Usage:
    python analyze_sweep.py summary
    python analyze_sweep.py params [--top-pct 10]
    python analyze_sweep.py categorical
    python analyze_sweep.py correlations
    python analyze_sweep.py best [N]
    python analyze_sweep.py export [--output results.csv]
    python analyze_sweep.py recommend
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Parameter metadata for decoding categorical values
CATEGORICAL_PARAMS = {
    "feature_selector": {0: "lgbm", 1: "xgb"},
    "alt_feature_selector": {0: "lgbm", 1: "xgb"},
    "calibrate_tolerances": {0: False, 1: True},
    "poly_interactions": {0: False, 1: True},
    "poly_include_squares": {0: False, 1: True},
}

# All parameter names in the sweep
ALL_PARAMS = [
    "alpha",
    "selector_tau",
    "selector_k_max",
    "gp_selector_k_max",
    "categorical_threshold",
    "tolerance_percentile",
    "tolerance_relaxation_factor",
    "tolerance_multiplier",
    "feature_selector",
    "alt_feature_selector",
    "poly_interactions",
    "poly_include_squares",
    "poly_limit",
    "calibrate_tolerances",
    "calibration_target_rmse_ratio",
    "recalibrate_every_n_passes",
]


def get_connection(db_path: str) -> sqlite3.Connection:
    """Get a database connection."""
    if not Path(db_path).exists():
        print(f"Error: Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    return sqlite3.connect(db_path)


def get_trials_with_params(conn: sqlite3.Connection) -> List[Dict]:
    """Load all completed trials with their parameters."""
    # Build pivot query dynamically
    param_cols = ",\n        ".join(
        f"MAX(CASE WHEN p.param_name='{param}' THEN p.param_value END) as {param}"
        for param in ALL_PARAMS
    )

    query = f"""
    SELECT
        t.trial_id,
        t.number as trial_number,
        v.value as rmse,
        {param_cols}
    FROM trials t
    JOIN trial_params p ON t.trial_id = p.trial_id
    JOIN trial_values v ON t.trial_id = v.trial_id
    WHERE t.state='COMPLETE'
    GROUP BY t.trial_id
    ORDER BY v.value ASC
    """

    cursor = conn.execute(query)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    return [dict(zip(columns, row)) for row in rows]


def decode_categorical(param: str, value: float) -> str:
    """Decode categorical parameter value to human-readable string."""
    if param in CATEGORICAL_PARAMS and value is not None:
        int_val = int(value)
        decoded = CATEGORICAL_PARAMS[param].get(int_val, value)
        return str(decoded)
    return str(value) if value is not None else "N/A"


def cmd_summary(args):
    """Quick overview: counts, best/avg/worst, top 10."""
    conn = get_connection(args.db)

    # Get counts
    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN state='COMPLETE' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN state='FAIL' THEN 1 ELSE 0 END) as failed,
            SUM(CASE WHEN state='RUNNING' THEN 1 ELSE 0 END) as running
        FROM trials
    """).fetchone()

    total, completed, failed, running = stats

    # Get RMSE stats
    rmse_stats = conn.execute("""
        SELECT
            MIN(value) as best,
            AVG(value) as avg,
            MAX(value) as worst
        FROM trial_values v
        JOIN trials t ON v.trial_id = t.trial_id
        WHERE t.state='COMPLETE'
    """).fetchone()

    best, avg, worst = rmse_stats

    # Calculate std
    std_result = conn.execute("""
        SELECT AVG((value - ?) * (value - ?)) as variance
        FROM trial_values v
        JOIN trials t ON v.trial_id = t.trial_id
        WHERE t.state='COMPLETE'
    """, (avg, avg)).fetchone()
    std = (std_result[0] ** 0.5) if std_result[0] else 0

    print(f"\nSweep Summary ({args.db})")
    print("=" * 50)
    print(f"Total trials:  {total}")
    print(f"  Completed:   {completed} ({100*completed/total:.1f}%)")
    print(f"  Failed:      {failed}")
    print(f"  Running:     {running}")
    print()
    print("RMSE Statistics:")
    print(f"  Best:        {best:.2f}")
    print(f"  Average:     {avg:.2f}")
    print(f"  Worst:       {worst:.2f}")
    print(f"  Std Dev:     {std:.2f}")
    print()

    # Top 10 trials
    trials = get_trials_with_params(conn)[:10]

    print("Top 10 Trials:")
    print("-" * 80)
    for t in trials:
        feat_sel = decode_categorical("feature_selector", t.get("feature_selector"))
        alt_sel = decode_categorical("alt_feature_selector", t.get("alt_feature_selector"))
        cal = decode_categorical("calibrate_tolerances", t.get("calibrate_tolerances"))
        k = int(t.get("selector_k_max", 0)) if t.get("selector_k_max") else "?"
        gp_k = int(t.get("gp_selector_k_max", 0)) if t.get("gp_selector_k_max") else "?"

        print(f"  #{t['trial_id']:<5} RMSE={t['rmse']:.2f}  "
              f"sel={feat_sel}/{alt_sel}  cal={cal}  k={k}  gp_k={gp_k}")

    conn.close()


def cmd_params(args):
    """Compare param values in top vs bottom trials."""
    conn = get_connection(args.db)
    trials = get_trials_with_params(conn)

    if not trials:
        print("No completed trials found.")
        return

    n_trials = len(trials)
    top_n = max(1, int(n_trials * args.top_pct / 100))
    bottom_n = max(1, int(n_trials * args.top_pct / 100))

    top_trials = trials[:top_n]
    bottom_trials = trials[-bottom_n:]

    print(f"\nParameter Analysis (top {args.top_pct}% vs bottom {args.top_pct}%)")
    print(f"Top {top_n} trials (RMSE < {top_trials[-1]['rmse']:.2f})")
    print(f"Bottom {bottom_n} trials (RMSE > {bottom_trials[0]['rmse']:.2f})")
    print("=" * 70)
    print(f"{'Parameter':<30} {'Top Avg':>12} {'Bottom Avg':>12} {'Delta':>10}")
    print("-" * 70)

    for param in ALL_PARAMS:
        top_vals = [t[param] for t in top_trials if t.get(param) is not None]
        bot_vals = [t[param] for t in bottom_trials if t.get(param) is not None]

        if not top_vals or not bot_vals:
            continue

        top_avg = sum(top_vals) / len(top_vals)
        bot_avg = sum(bot_vals) / len(bot_vals)
        delta = top_avg - bot_avg

        # Format based on param type
        if param in CATEGORICAL_PARAMS:
            top_str = decode_categorical(param, round(top_avg))
            bot_str = decode_categorical(param, round(bot_avg))
            print(f"{param:<30} {top_str:>12} {bot_str:>12} {delta:>+10.2f}")
        else:
            print(f"{param:<30} {top_avg:>12.3f} {bot_avg:>12.3f} {delta:>+10.3f}")

    conn.close()


def cmd_categorical(args):
    """Breakdown by categorical params."""
    conn = get_connection(args.db)

    print(f"\nCategorical Parameter Breakdown ({args.db})")
    print("=" * 60)

    for param, mapping in CATEGORICAL_PARAMS.items():
        print(f"\n{param}:")
        print("-" * 50)

        query = f"""
        SELECT
            CAST(p.param_value AS INT) as value,
            COUNT(*) as n,
            AVG(v.value) as avg_rmse,
            MIN(v.value) as best_rmse
        FROM trial_params p
        JOIN trials t ON p.trial_id = t.trial_id
        JOIN trial_values v ON t.trial_id = v.trial_id
        WHERE p.param_name = ? AND t.state='COMPLETE'
        GROUP BY CAST(p.param_value AS INT)
        ORDER BY avg_rmse ASC
        """

        rows = conn.execute(query, (param,)).fetchall()

        for val, n, avg_rmse, best_rmse in rows:
            label = mapping.get(val, val)
            print(f"  {str(label):<10}  n={n:<5}  avg={avg_rmse:.2f}  best={best_rmse:.2f}")

    conn.close()


def cmd_correlations(args):
    """Spearman correlation of each param with RMSE."""
    conn = get_connection(args.db)
    trials = get_trials_with_params(conn)

    if not trials:
        print("No completed trials found.")
        return

    print(f"\nParameter-RMSE Correlations ({args.db})")
    print("=" * 50)
    print("(Positive = higher value → higher RMSE = bad)")
    print("(Negative = higher value → lower RMSE = good)")
    print("-" * 50)

    correlations = []

    for param in ALL_PARAMS:
        vals = [(t[param], t['rmse']) for t in trials if t.get(param) is not None]
        if len(vals) < 10:
            continue

        # Simple Pearson correlation (good enough for continuous params)
        x = [v[0] for v in vals]
        y = [v[1] for v in vals]
        n = len(vals)

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
        std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5

        if std_x > 0 and std_y > 0:
            corr = cov / (std_x * std_y)
            correlations.append((param, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Parameter':<35} {'Correlation':>12}")
    print("-" * 50)
    for param, corr in correlations:
        indicator = "***" if abs(corr) > 0.3 else "**" if abs(corr) > 0.2 else "*" if abs(corr) > 0.1 else ""
        print(f"{param:<35} {corr:>+12.3f} {indicator}")

    conn.close()


def cmd_bins(args):
    """Analyze parameter in bins to detect nonlinear patterns."""
    conn = get_connection(args.db)
    param = args.param
    n_bins = args.bins

    # Validate param name
    if param not in ALL_PARAMS:
        print(f"Error: Unknown parameter '{param}'")
        print(f"Available: {', '.join(ALL_PARAMS)}")
        return

    # Check if categorical
    if param in CATEGORICAL_PARAMS:
        print(f"Note: '{param}' is categorical. Use 'categorical' command instead.")
        return

    # Get min/max for this param
    range_query = """
    SELECT MIN(p.param_value), MAX(p.param_value)
    FROM trial_params p
    JOIN trials t ON p.trial_id = t.trial_id
    WHERE p.param_name = ? AND t.state='COMPLETE'
    """
    min_val, max_val = conn.execute(range_query, (param,)).fetchone()

    if min_val is None:
        print(f"No data found for parameter '{param}'")
        return

    # Create bins
    bin_width = (max_val - min_val) / n_bins
    bins = []
    for i in range(n_bins):
        lo = min_val + i * bin_width
        hi = min_val + (i + 1) * bin_width
        bins.append((lo, hi))

    print(f"\nBinned Analysis: {param} ({args.db})")
    print(f"Range: {min_val:.2f} to {max_val:.2f} in {n_bins} bins")
    print("=" * 70)
    print(f"{'Bin Range':<20} {'Count':>8} {'Avg RMSE':>12} {'Best RMSE':>12} {'Worst':>10}")
    print("-" * 70)

    results = []
    for lo, hi in bins:
        # Use <= for last bin to include max value
        if hi == max_val:
            query = """
            SELECT COUNT(*), AVG(v.value), MIN(v.value), MAX(v.value)
            FROM trial_params p
            JOIN trials t ON p.trial_id = t.trial_id
            JOIN trial_values v ON t.trial_id = v.trial_id
            WHERE p.param_name = ? AND t.state='COMPLETE'
              AND p.param_value >= ? AND p.param_value <= ?
            """
        else:
            query = """
            SELECT COUNT(*), AVG(v.value), MIN(v.value), MAX(v.value)
            FROM trial_params p
            JOIN trials t ON p.trial_id = t.trial_id
            JOIN trial_values v ON t.trial_id = v.trial_id
            WHERE p.param_name = ? AND t.state='COMPLETE'
              AND p.param_value >= ? AND p.param_value < ?
            """

        row = conn.execute(query, (param, lo, hi)).fetchone()
        count, avg_rmse, best_rmse, worst_rmse = row

        if count and count > 0:
            results.append((lo, hi, count, avg_rmse, best_rmse, worst_rmse))
            bin_label = f"{lo:.2f} - {hi:.2f}"
            print(f"{bin_label:<20} {count:>8} {avg_rmse:>12.2f} {best_rmse:>12.2f} {worst_rmse:>10.2f}")

    # Find sweet spot
    if results:
        best_bin = min(results, key=lambda x: x[3])  # lowest avg RMSE
        print("-" * 70)
        print(f"Sweet spot: {best_bin[0]:.2f} - {best_bin[1]:.2f} (avg RMSE: {best_bin[3]:.2f})")

        # Check for U-shape or other patterns
        if len(results) >= 3:
            avg_rmses = [r[3] for r in results]
            min_idx = avg_rmses.index(min(avg_rmses))

            # Check if minimum is in the interior (not at edges)
            if 0 < min_idx < len(avg_rmses) - 1:
                # Check if values increase on both sides
                left_higher = avg_rmses[0] > avg_rmses[min_idx]
                right_higher = avg_rmses[-1] > avg_rmses[min_idx]
                if left_higher and right_higher:
                    print(f"Pattern: U-shaped (sweet spot in bin {min_idx + 1}/{len(avg_rmses)})")
                else:
                    print(f"Pattern: Complex (minimum at bin {min_idx + 1})")
            elif min_idx == 0:
                print("Pattern: Monotonic increasing (lower param values are better)")
            elif min_idx == len(avg_rmses) - 1:
                print("Pattern: Monotonic decreasing (higher param values are better)")

            # Report effect size
            effect = max(avg_rmses) - min(avg_rmses)
            print(f"Effect size: {effect:.2f} RMSE between best and worst bins")

    conn.close()


def cmd_best(args):
    """Show top N trials with full params."""
    conn = get_connection(args.db)
    trials = get_trials_with_params(conn)[:args.n]

    if not trials:
        print("No completed trials found.")
        return

    print(f"\nTop {args.n} Trials ({args.db})")
    print("=" * 80)

    for i, t in enumerate(trials, 1):
        print(f"\n#{i} Trial {t['trial_id']} (RMSE: {t['rmse']:.4f})")
        print("-" * 40)

        for param in ALL_PARAMS:
            val = t.get(param)
            if val is None:
                continue

            if param in CATEGORICAL_PARAMS:
                decoded = decode_categorical(param, val)
                print(f"  {param:<35} {decoded}")
            elif isinstance(val, float) and val == int(val):
                print(f"  {param:<35} {int(val)}")
            else:
                print(f"  {param:<35} {val:.4f}" if isinstance(val, float) else f"  {param:<35} {val}")

    # Print CLI args for best trial
    if trials:
        best = trials[0]
        print("\n" + "=" * 80)
        print("CLI args for best trial:")
        print("-" * 80)

        cli_parts = []
        for param in ALL_PARAMS:
            val = best.get(param)
            if val is None:
                continue

            if param in CATEGORICAL_PARAMS:
                if param in ("poly_interactions", "poly_include_squares", "calibrate_tolerances"):
                    if int(val) == 1:
                        cli_parts.append(f"--{param}")
                else:
                    cli_parts.append(f"--{param} {decode_categorical(param, val)}")
            elif isinstance(val, float) and val == int(val):
                cli_parts.append(f"--{param} {int(val)}")
            else:
                cli_parts.append(f"--{param} {val:.4f}" if isinstance(val, float) else f"--{param} {val}")

        # Wrap at 100 chars
        line = "python predict.py "
        for part in cli_parts:
            if len(line) + len(part) > 100:
                print(line + "\\")
                line = "    "
            line += part + " "
        print(line)

    conn.close()


def cmd_export(args):
    """Export all trials to CSV."""
    conn = get_connection(args.db)
    trials = get_trials_with_params(conn)

    if not trials:
        print("No completed trials found.")
        return

    # Write CSV
    output_path = args.output
    with open(output_path, 'w') as f:
        # Header
        columns = ["trial_id", "trial_number", "rmse"] + ALL_PARAMS
        f.write(",".join(columns) + "\n")

        # Rows
        for t in trials:
            row = [str(t.get(col, "")) for col in columns]
            f.write(",".join(row) + "\n")

    print(f"Exported {len(trials)} trials to {output_path}")
    conn.close()


def cmd_recommend(args):
    """Suggest param ranges based on top performers."""
    conn = get_connection(args.db)
    trials = get_trials_with_params(conn)

    if not trials:
        print("No completed trials found.")
        return

    n_trials = len(trials)
    top_n = max(1, int(n_trials * args.top_pct / 100))
    top_trials = trials[:top_n]

    print(f"\nRecommended Parameters (from top {args.top_pct}% = {top_n} trials)")
    print(f"Best RMSE in top group: {top_trials[0]['rmse']:.2f}")
    print(f"Worst RMSE in top group: {top_trials[-1]['rmse']:.2f}")
    print("=" * 60)

    print("\nContinuous Parameters:")
    print("-" * 60)
    print(f"{'Parameter':<35} {'Min':>8} {'Max':>8} {'Median':>8}")
    print("-" * 60)

    continuous_params = [p for p in ALL_PARAMS if p not in CATEGORICAL_PARAMS]

    for param in continuous_params:
        vals = sorted([t[param] for t in top_trials if t.get(param) is not None])
        if not vals:
            continue

        min_val = vals[0]
        max_val = vals[-1]
        median_val = vals[len(vals) // 2]

        if isinstance(min_val, float) and min_val == int(min_val):
            print(f"{param:<35} {int(min_val):>8} {int(max_val):>8} {int(median_val):>8}")
        else:
            print(f"{param:<35} {min_val:>8.2f} {max_val:>8.2f} {median_val:>8.2f}")

    print("\nCategorical Parameters:")
    print("-" * 60)

    for param, mapping in CATEGORICAL_PARAMS.items():
        vals = [t[param] for t in top_trials if t.get(param) is not None]
        if not vals:
            continue

        # Count occurrences
        counts = {}
        for v in vals:
            int_v = int(v)
            label = str(mapping.get(int_v, int_v))
            counts[label] = counts.get(label, 0) + 1

        # Sort by count
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        best_choice = sorted_counts[0][0]
        pct = sorted_counts[0][1] / len(vals) * 100

        print(f"  {param:<33} {best_choice} ({pct:.0f}% of top trials)")

    # Print recommended CLI
    print("\n" + "=" * 60)
    print("Recommended CLI args (median values from top trials):")
    print("-" * 60)

    cli_parts = []
    for param in continuous_params:
        vals = sorted([t[param] for t in top_trials if t.get(param) is not None])
        if not vals:
            continue
        median_val = vals[len(vals) // 2]

        if isinstance(median_val, float) and median_val == int(median_val):
            cli_parts.append(f"--{param} {int(median_val)}")
        else:
            cli_parts.append(f"--{param} {median_val:.2f}")

    for param, mapping in CATEGORICAL_PARAMS.items():
        vals = [t[param] for t in top_trials if t.get(param) is not None]
        if not vals:
            continue

        counts = {}
        for v in vals:
            int_v = int(v)
            counts[int_v] = counts.get(int_v, 0) + 1

        best_int = max(counts, key=counts.get)

        if param in ("poly_interactions", "poly_include_squares", "calibrate_tolerances"):
            if best_int == 1:
                cli_parts.append(f"--{param}")
        else:
            cli_parts.append(f"--{param} {mapping.get(best_int, best_int)}")

    print("python predict.py \\")
    for i, part in enumerate(cli_parts):
        if i < len(cli_parts) - 1:
            print(f"    {part} \\")
        else:
            print(f"    {part}")

    conn.close()


# Key params for binned analysis
KEY_CONTINUOUS_PARAMS = [
    "selector_k_max",
    "tolerance_percentile",
    "gp_selector_k_max",
    "calibration_target_rmse_ratio",
]


def run_full_report(db_path: str, top_pct: float = 5.0):
    """Run all analyses, print to stdout, save to file."""
    from datetime import datetime
    from io import StringIO

    output = StringIO()

    def pr(s=""):
        print(s, file=output)
        print(s)

    conn = get_connection(db_path)
    trials = get_trials_with_params(conn)

    if not trials:
        pr("No completed trials found.")
        return

    n_trials = len(trials)
    top_n = max(1, int(n_trials * top_pct / 100))
    top_trials = trials[:top_n]

    # Header
    pr("=" * 80)
    pr("                         SWEEP ANALYSIS REPORT")
    pr(f"                         {db_path}")
    pr(f"                         Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    pr("=" * 80)

    # [1/6] Summary
    pr("\n[1/6] SUMMARY")
    pr("-" * 60)

    stats = conn.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN state='COMPLETE' THEN 1 ELSE 0 END),
               SUM(CASE WHEN state='FAIL' THEN 1 ELSE 0 END)
        FROM trials
    """).fetchone()
    total, completed, failed = stats

    rmse_stats = conn.execute("""
        SELECT MIN(value), AVG(value), MAX(value)
        FROM trial_values v JOIN trials t ON v.trial_id = t.trial_id
        WHERE t.state='COMPLETE'
    """).fetchone()
    best, avg, worst = rmse_stats

    std_result = conn.execute("""
        SELECT AVG((value - ?) * (value - ?))
        FROM trial_values v JOIN trials t ON v.trial_id = t.trial_id
        WHERE t.state='COMPLETE'
    """, (avg, avg)).fetchone()
    std = (std_result[0] ** 0.5) if std_result[0] else 0

    pr(f"Total: {total} | Completed: {completed} | Failed: {failed}")
    pr(f"RMSE: Best={best:.2f} | Avg={avg:.2f} | Worst={worst:.2f} | Std={std:.2f}")

    # [2/6] Categorical
    pr("\n[2/6] CATEGORICAL PARAMETERS")
    pr("-" * 60)

    for param, mapping in CATEGORICAL_PARAMS.items():
        query = """
        SELECT CAST(p.param_value AS INT) as value, COUNT(*), AVG(v.value), MIN(v.value)
        FROM trial_params p
        JOIN trials t ON p.trial_id = t.trial_id
        JOIN trial_values v ON t.trial_id = v.trial_id
        WHERE p.param_name = ? AND t.state='COMPLETE'
        GROUP BY CAST(p.param_value AS INT)
        ORDER BY AVG(v.value) ASC
        """
        rows = conn.execute(query, (param,)).fetchall()

        if len(rows) >= 2:
            best_row = rows[0]
            worst_row = rows[-1]
            best_label = mapping.get(best_row[0], best_row[0])
            worst_label = mapping.get(worst_row[0], worst_row[0])
            pr(f"{param:<28} {best_label} (n={best_row[1]}, avg={best_row[2]:.1f}) "
               f"vs {worst_label} (n={worst_row[1]}, avg={worst_row[2]:.1f})")

    # [3/6] Correlations
    pr("\n[3/6] CORRELATIONS (sorted by impact)")
    pr("-" * 60)

    correlations = []
    for param in ALL_PARAMS:
        vals = [(t[param], t['rmse']) for t in trials if t.get(param) is not None]
        if len(vals) < 10:
            continue
        x = [v[0] for v in vals]
        y = [v[1] for v in vals]
        n = len(vals)
        mean_x, mean_y = sum(x)/n, sum(y)/n
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
        std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5
        if std_x > 0 and std_y > 0:
            correlations.append((param, cov / (std_x * std_y)))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for param, corr in correlations[:8]:
        stars = "***" if abs(corr) > 0.3 else "**" if abs(corr) > 0.2 else "*" if abs(corr) > 0.1 else ""
        direction = "(higher hurts)" if corr > 0 else "(higher helps)"
        pr(f"{param:<35} {corr:>+.3f} {stars:<3} {direction}")

    # [4/6] Nonlinear patterns
    pr("\n[4/6] NONLINEAR PATTERNS (key params)")
    pr("-" * 60)

    for param in KEY_CONTINUOUS_PARAMS:
        range_q = """
        SELECT MIN(p.param_value), MAX(p.param_value)
        FROM trial_params p JOIN trials t ON p.trial_id = t.trial_id
        WHERE p.param_name = ? AND t.state='COMPLETE'
        """
        min_val, max_val = conn.execute(range_q, (param,)).fetchone()
        if min_val is None:
            continue

        n_bins = 5
        bin_width = (max_val - min_val) / n_bins
        results = []

        for i in range(n_bins):
            lo = min_val + i * bin_width
            hi = min_val + (i + 1) * bin_width
            op = "<=" if i == n_bins - 1 else "<"
            query = f"""
            SELECT COUNT(*), AVG(v.value), MIN(v.value)
            FROM trial_params p
            JOIN trials t ON p.trial_id = t.trial_id
            JOIN trial_values v ON t.trial_id = v.trial_id
            WHERE p.param_name = ? AND t.state='COMPLETE'
              AND p.param_value >= ? AND p.param_value {op} ?
            """
            row = conn.execute(query, (param, lo, hi)).fetchone()
            if row[0] and row[0] > 0:
                results.append((lo, hi, row[0], row[1], row[2]))

        if results:
            avg_rmses = [r[3] for r in results]
            min_idx = avg_rmses.index(min(avg_rmses))
            best_bin = results[min_idx]
            effect = max(avg_rmses) - min(avg_rmses)

            if 0 < min_idx < len(results) - 1:
                pattern = f"U-shaped, sweet spot {best_bin[0]:.0f}-{best_bin[1]:.0f}"
            elif min_idx == 0:
                pattern = "Monotonic (lower is better)"
            else:
                pattern = "Monotonic (higher is better)"

            pr(f"{param:<32} {pattern}, effect={effect:.2f}")

    # [5/6] Recommendations
    pr(f"\n[5/6] RECOMMENDATIONS (from top {top_pct}% = {top_n} trials)")
    pr("-" * 60)

    continuous_params = [p for p in ALL_PARAMS if p not in CATEGORICAL_PARAMS]

    # Continuous recommendations
    rec_parts = []
    for param in continuous_params:
        vals = sorted([t[param] for t in top_trials if t.get(param) is not None])
        if not vals:
            continue
        median_val = vals[len(vals) // 2]
        if isinstance(median_val, float) and median_val == int(median_val):
            rec_parts.append(f"{param}={int(median_val)}")
        else:
            rec_parts.append(f"{param}={median_val:.2f}")

    pr("Continuous: " + ", ".join(rec_parts[:6]))
    if len(rec_parts) > 6:
        pr("            " + ", ".join(rec_parts[6:]))

    # Categorical recommendations
    cat_parts = []
    for param, mapping in CATEGORICAL_PARAMS.items():
        vals = [t[param] for t in top_trials if t.get(param) is not None]
        if not vals:
            continue
        counts = {}
        for v in vals:
            int_v = int(v)
            counts[int_v] = counts.get(int_v, 0) + 1
        best_int = max(counts, key=counts.get)
        cat_parts.append(f"{param}={mapping.get(best_int, best_int)}")

    pr("Categorical: " + ", ".join(cat_parts))

    # [6/6] Top trials
    pr(f"\n[6/6] TOP 5 TRIALS")
    pr("-" * 60)

    for t in trials[:5]:
        feat_sel = decode_categorical("feature_selector", t.get("feature_selector"))
        cal = decode_categorical("calibrate_tolerances", t.get("calibrate_tolerances"))
        k = int(t.get("selector_k_max", 0)) if t.get("selector_k_max") else "?"
        gp_k = int(t.get("gp_selector_k_max", 0)) if t.get("gp_selector_k_max") else "?"
        tol = t.get("tolerance_percentile", 0)

        pr(f"#{t['trial_id']:<5} RMSE={t['rmse']:.2f}  k={k} gp_k={gp_k} tol={tol:.1f} "
           f"sel={feat_sel} cal={cal}")

    # CLI for best trial
    pr("\n" + "=" * 80)
    pr("CLI for best trial:")
    pr("-" * 80)

    best = trials[0]
    cli_parts = []
    for param in continuous_params:
        val = best.get(param)
        if val is None:
            continue
        if isinstance(val, float) and val == int(val):
            cli_parts.append(f"--{param} {int(val)}")
        else:
            cli_parts.append(f"--{param} {val:.4f}")

    for param, mapping in CATEGORICAL_PARAMS.items():
        val = best.get(param)
        if val is None:
            continue
        if param in ("poly_interactions", "poly_include_squares", "calibrate_tolerances"):
            if int(val) == 1:
                cli_parts.append(f"--{param}")
        else:
            cli_parts.append(f"--{param} {mapping.get(int(val), int(val))}")

    pr("python predict.py \\")
    for i, part in enumerate(cli_parts):
        if i < len(cli_parts) - 1:
            pr(f"    {part} \\")
        else:
            pr(f"    {part}")

    pr("=" * 80)

    conn.close()

    # Save to file
    report_path = Path("analysis_output/sweep_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(output.getvalue())
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Optuna sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
    python analyze_sweep.py              # Full report (prints + saves to file)
    python analyze_sweep.py summary      # Quick overview only
    python analyze_sweep.py categorical  # Categorical breakdown only
    python analyze_sweep.py bins <param> # Nonlinear pattern for one param
    python analyze_sweep.py recommend    # Recommendations only
    python analyze_sweep.py best 20      # Top 20 trials
    python analyze_sweep.py export -o out.csv  # Export all to CSV
        """
    )

    parser.add_argument("--db", type=str, default="sweep_optuna.db",
                        help="Path to Optuna database (default: sweep_optuna.db)")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # summary
    p_summary = subparsers.add_parser("summary", help="Quick overview")

    # params
    p_params = subparsers.add_parser("params", help="Parameter importance analysis")
    p_params.add_argument("--top-pct", type=float, default=10,
                          help="Define 'top' as best N%% of trials (default: 10)")

    # categorical
    p_cat = subparsers.add_parser("categorical", help="Breakdown by categorical params")

    # correlations
    p_corr = subparsers.add_parser("correlations", help="Param-RMSE correlations")

    # bins
    p_bins = subparsers.add_parser("bins", help="Analyze param in bins for nonlinear patterns")
    p_bins.add_argument("param", type=str, help="Parameter name to analyze")
    p_bins.add_argument("--bins", "-n", type=int, default=5,
                        help="Number of bins (default: 5)")

    # best
    p_best = subparsers.add_parser("best", help="Show top N trials with full params")
    p_best.add_argument("n", type=int, nargs="?", default=10,
                        help="Number of trials to show (default: 10)")

    # export
    p_export = subparsers.add_parser("export", help="Export all trials to CSV")
    p_export.add_argument("--output", "-o", type=str, default="sweep_results.csv",
                          help="Output CSV path (default: sweep_results.csv)")

    # recommend
    p_recommend = subparsers.add_parser("recommend", help="Suggest param ranges")
    p_recommend.add_argument("--top-pct", type=float, default=10,
                             help="Base recommendations on top N%% (default: 10)")

    args = parser.parse_args()

    # If no command given, run full report
    if args.command is None:
        run_full_report(args.db)
        return

    # Dispatch to command
    commands = {
        "summary": cmd_summary,
        "params": cmd_params,
        "categorical": cmd_categorical,
        "correlations": cmd_correlations,
        "bins": cmd_bins,
        "best": cmd_best,
        "export": cmd_export,
        "recommend": cmd_recommend,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
