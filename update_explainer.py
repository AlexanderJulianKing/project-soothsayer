#!/usr/bin/env python3
"""Update soothsayer_explainer.html with latest prediction results.

Run after predict.py to refresh:
  - OOF scatter plot data (model names, actual/predicted scores)
  - Stats annotation (R², Spearman ρ, RMSE)
  - Stat cards (RMSE, R²)
  - Comparison table (RMSE, R², Spearman for full pipeline row)
  - Pipeline node stats
  - Imputer matrix model names (top 14 by ELO)

Usage:
    python3 update_explainer.py
    python3 update_explainer.py --output-dir arena_predictor/analysis_output/output_YYYYMMDD_HHMMSS
"""
import argparse
import glob
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats


def find_latest_output():
    pattern = "arena_predictor/analysis_output/output_*/metadata.json"
    files = sorted(glob.glob(pattern))
    if not files:
        print("ERROR: No output directories found.", file=sys.stderr)
        sys.exit(1)
    return files[-1].rsplit("/metadata.json", 1)[0]


def compute_metrics(oof_path):
    oof = pd.read_csv(oof_path)
    actual = oof["actual_score"].values
    pred = oof["oof_predicted_score"].values
    resid = actual - pred

    rmse = float(np.sqrt(np.mean(resid ** 2)))
    r2 = float(1 - np.sum(resid ** 2) / np.sum((actual - actual.mean()) ** 2))
    spearman = float(stats.spearmanr(actual, pred)[0])

    return {
        "rmse": round(rmse, 2),
        "r2": round(r2, 2),
        "spearman": round(spearman, 2),
    }


def build_oof_json(oof_path):
    oof = pd.read_csv(oof_path)
    data = []
    for _, row in oof.iterrows():
        data.append([
            row["model_name"],
            round(row["actual_score"], 1),
            round(row["oof_predicted_score"], 1),
        ])
    return json.dumps(data)


CSV_PATH = "benchmark_combiner/benchmarks/clean_combined_all_benches.csv"

BENCH_COLS = {
    "EQ": "eq_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill",
    "Writing": "writing_Grok 4 Fast TrueSkill",
    "Logic": "logic_accuracy",
    "Style": "style_predicted_delta",
    "ARC-AGI 2": "arc_ARC-AGI-2",
    "GPQA": "aa_eval_gpqa",
    "GDPVal": "aagdpval_ELO",
    "AIME": "aa_eval_aime_25",
    "Terminal": "aa_eval_terminalbench_hard",
    "Simple": "simplebench_Score (AVG@5)",
}


def shorten_name(name):
    short = name
    short = re.sub(r"\s*Preview\s*\(\d{4}-\d{2}-\d{2}\)", "", short)
    short = re.sub(r"\s*\(\d{4}-\d{2}-\d{2}\)", "", short)
    short = re.sub(r"\s*Preview$", "", short)
    short = (short
             .replace(" Thinking", " Th.")
             .replace("Instruct", "Inst.")
             .replace(" (Non-Reasoning)", " NR")
             .replace(" (Non-reasoning)", " NR")
             .replace(" (Reasoning)", " R")
             .replace(" (Minimal)", " Min.")
             .replace(" (high)", "")
             .replace("Sonnet", "Son.")
             .replace("Claude", "Cl.")
             )
    short = re.sub(r"\s+A\d+B", "", short)
    if len(short) > 20:
        short = short[:18] + ".."
    return short


def top_model_names(oof_path, n=14):
    oof = pd.read_csv(oof_path)
    top = oof.nlargest(n, "actual_score")
    return [shorten_name(name) for name in top["model_name"].values]


def build_raw_scores(oof_path, n=14):
    """Build the rawScores JS array and model names from actual benchmark data."""
    oof = pd.read_csv(oof_path)
    combined = pd.read_csv(CSV_PATH)
    top = oof.nlargest(n, "actual_score")
    top_names = top["model_name"].values

    # Compute per-column min/max for normalization
    col_ranges = {}
    for bench_name, col in BENCH_COLS.items():
        if col in combined.columns:
            vals = combined[col].dropna()
            col_ranges[bench_name] = (vals.min(), vals.max())

    rows_js = []
    for model_name in top_names:
        row = combined[combined["model_name"] == model_name]
        if len(row) == 0:
            rows_js.append("      " + ",".join(["null"] * len(BENCH_COLS)))
            continue
        row = row.iloc[0]
        vals = []
        for bench_name, col in BENCH_COLS.items():
            v = row.get(col, np.nan)
            if pd.isna(v):
                vals.append("null")
            else:
                lo, hi = col_ranges.get(bench_name, (0, 1))
                normed = (v - lo) / (hi - lo) if hi > lo else 0.5
                normed = min(max(normed, 0.0), 1.0)
                if normed >= 0.995:
                    vals.append("1.0")
                else:
                    vals.append(f"{normed:.2f}")
        rows_js.append("      [" + ",".join(vals) + "]")

    return "[\n" + ",\n".join(rows_js) + ",\n    ]"


def update_html(html, metrics, oof_json, model_names, raw_scores_js):
    # 1. OOF scatter data
    html = re.sub(
        r"const DATA = \[.*?\];",
        f"const DATA = {oof_json};",
        html,
        count=1,
        flags=re.DOTALL,
    )

    # 2. Stats annotation in scatter plot — use string find/replace to avoid regex \u issues
    stats_pattern = re.search(
        r"ctx\.fillText\('R.u00B2 = ([\d.]+) +Spearman .u03C1 = ([\d.]+) +RMSE = ([\d.]+)'",
        html,
    )
    if stats_pattern:
        old = stats_pattern.group(0)
        new = old.replace(stats_pattern.group(1), f"{metrics['r2']:.2f}")
        new = new.replace(stats_pattern.group(2), f"{metrics['spearman']:.2f}")
        new = new.replace(stats_pattern.group(3), f"{metrics['rmse']:.1f}")
        html = html.replace(old, new)

    # 3. Stat card R² target
    html = re.sub(
        r'data-target="[\d.]+"(>0</div><div class="stat-label">R)',
        f'data-target="{metrics["r2"]:.2f}"\\1',
        html,
    )

    # 4. Comparison table — full pipeline row
    html = re.sub(
        r"(<tr class=\"highlight\"><td>Full Soothsayer pipeline</td><td>)[\d.]+(</td><td>)[\d.]+(</td><td>)[\d.]+(</td></tr>)",
        f"\\g<1>{metrics['rmse']:.1f}\\g<2>{metrics['r2']:.2f}\\g<3>{metrics['spearman']:.2f}\\g<4>",
        html,
    )

    # 5. Pipeline node
    html = re.sub(
        r"(Spearman &rho; = )[\d.]+( &middot; RMSE )[\d.]+",
        f"\\g<1>{metrics['spearman']:.2f}\\g<2>{metrics['rmse']:.1f}",
        html,
    )

    # 6. Imputer matrix model names
    old_names_pattern = r"const modelNames = \[.*?\];"
    new_names = "const modelNames = [" + ",".join(f"'{n}'" for n in model_names) + "];"
    html = re.sub(old_names_pattern, new_names, html, count=1, flags=re.DOTALL)

    # 7. Imputer matrix rawScores
    html = re.sub(
        r"const rawScores = \[.*?\];",
        f"const rawScores = {raw_scores_js};",
        html,
        count=1,
        flags=re.DOTALL,
    )

    return html


def main():
    parser = argparse.ArgumentParser(description="Update explainer with latest results")
    parser.add_argument("--output-dir", default=None, help="Specific output directory to use")
    parser.add_argument("--html", default="soothsayer_explainer.html", help="Path to HTML file")
    args = parser.parse_args()

    output_dir = args.output_dir or find_latest_output()
    oof_path = f"{output_dir}/oof_predictions.csv"
    print(f"Using output: {output_dir}")

    if not pd.io.common.file_exists(oof_path):
        print(f"ERROR: {oof_path} not found", file=sys.stderr)
        sys.exit(1)

    metrics = compute_metrics(oof_path)
    print(f"  RMSE={metrics['rmse']}, R²={metrics['r2']}, Spearman={metrics['spearman']}")

    oof_json = build_oof_json(oof_path)
    model_names = top_model_names(oof_path, n=14)
    raw_scores_js = build_raw_scores(oof_path, n=14)
    print(f"  Top models: {', '.join(model_names[:5])}...")

    with open(args.html) as f:
        html = f.read()

    html = update_html(html, metrics, oof_json, model_names, raw_scores_js)

    with open(args.html, "w") as f:
        f.write(html)

    print(f"  Updated {args.html}")


if __name__ == "__main__":
    main()
