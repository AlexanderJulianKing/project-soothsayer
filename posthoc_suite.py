#!/usr/bin/env python3
"""Post-hoc analysis suite: 15 chart sections for benchmark/prediction diagnostics.

Some sections emit multiple figure files, so the output figure count is larger
than the number of chart sections listed here.

Charts produced:
  1. Cost vs Performance + Pareto Frontier
  2. Arena Residuals (actual - predicted)
  3. Multi-Factor Radar Profiles (top models + vendor averages)
  4. Benchmark Clustering Dendrogram
  5. Reasoning vs Non-Reasoning violin comparison
  6. Prediction Calibration (reliability diagram + interval coverage)
  7. Residual Bias Decomposition (vendor × reasoning heatmap)
  8. Rank Stability (Monte Carlo rank intervals)
  9. Benchmark Redundancy (partial correlation heatmap)
 10. Marginal Benchmark Value (incremental cluster contribution)
 11. Capability Archetype Map (PCA scatter by vendor)
 12. Token Productivity (tokens vs performance)
 13. Capability Over Time (release date vs performance)
 14. Capability Profiles (radar + bars)
 15. Predicted vs Actual Arena ELO (top 50 scatter)

All outputs land in arena_predictor/analysis_output/posthoc_suite/run_YYYYMMDD_HHMMSS/
"""

from __future__ import annotations

import glob
import os
import warnings
from datetime import datetime
from pathlib import Path

# Run from arena_predictor/ so all relative paths resolve correctly
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena_predictor"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from factor_analyzer import FactorAnalyzer
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*factor analysis model converges.*")
warnings.filterwarnings("ignore", message="invalid value encountered")

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

CLEAN_BENCH_FILE = "../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
COMBINED_BENCH_FILE = "../benchmark_combiner/benchmarks/combined_all_benches.csv"
IMPUTATION_DIR_PATTERN = "analysis_output/output_*"
IMPUTED_FILENAME = "imputed_full.csv"
PREDICTIONS_FILENAME = "predictions_best_model.csv"
OOF_PREDICTIONS_FILENAME = "oof_predictions.csv"
FEATURE_RANKING_FILENAME = "feature_ranking_gain.csv"
OUTPUT_ROOT = Path("analysis_output/posthoc_suite")

CUSTOM_SOURCE_STYLES = {
    "OpenAI":     ("o", "#AB47BC"),
    "Anthropic":  ("*", "#d4a37f"),
    "Meta":       ("^", "#0081FB"),
    "Google":     ("D", "#d62728"),
    "xAI":        ("X", "#13161a"),
    "Alibaba":    ("p", "#FF6701"),
    "DeepSeek":   (">", "#4CAF50"),
    "Other":      ("s", "#90A4AE"),
}
MARKER_CYCLE = ["o", "s", "^", "D", "P", "X", "v", ">", "<", "p", "h", "*"]

EXCLUDE_FROM_FA = [
    "model_name",
    "aa_pricing_price_1m_output_tokens",
    "openbench_EtE Response Time 500 Output Tokens",
    "openbench_Reasoning Output Tokens Used to Run Artificial Analysis Intelligence Index (million)",
    "openbench_Answer Output Tokens Used to Run Artificial Analysis Intelligence Index (million)",
    "logic_avg_reasoning_tokens",
    "logic_avg_answer_tokens",
]

COST_COLS_BASE = [
    "model_name",
    "aa_pricing_price_1m_output_tokens",
    "openbench_Reasoning Output Tokens Used to Run Artificial Analysis Intelligence Index (million)",
    "openbench_Answer Output Tokens Used to Run Artificial Analysis Intelligence Index (million)",
    "logic_avg_reasoning_tokens",
    "logic_avg_answer_tokens",
]
COST1_X_TOKENS = 2000.0
AAI_DIVISOR = 20000.0

LABEL_BBOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75)

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _is_missingness_flag(col: str) -> bool:
    name = str(col).lower()
    return name.endswith("__was_missing") or name.endswith("_was_missing") or "missingness" in name


def _is_transformed(col: str) -> bool:
    if not col or not col.endswith("~"):
        return False
    if "_" not in col:
        return False
    base, suffix = col.rsplit("_", 1)
    return bool(base and suffix[:-1])


def _find_latest_with(filename: str) -> Path:
    """Find the newest output_* dir that contains the given file."""
    dirs = [Path(p) for p in glob.glob(IMPUTATION_DIR_PATTERN) if Path(p).is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No dirs matching '{IMPUTATION_DIR_PATTERN}'")
    candidates = [d for d in dirs if (d / filename).exists()]
    if not candidates:
        raise FileNotFoundError(f"No output dir contains '{filename}'")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_data():
    """Return (df_clean, df_combined, df_imputed, df_predictions)."""
    df_clean = pd.read_csv(CLEAN_BENCH_FILE)
    df_combined = pd.read_csv(COMBINED_BENCH_FILE)

    imp_dir = _find_latest_with(IMPUTED_FILENAME)
    pred_dir = _find_latest_with(PREDICTIONS_FILENAME)

    df_imputed = pd.read_csv(imp_dir / IMPUTED_FILENAME)
    df_predictions = pd.read_csv(pred_dir / PREDICTIONS_FILENAME)

    print(f"Clean: {df_clean.shape}, Combined: {df_combined.shape}")
    print(f"Imputed: {df_imputed.shape} (from {imp_dir.name})")
    print(f"Predictions: {df_predictions.shape} (from {pred_dir.name})")
    return df_clean, df_combined, df_imputed, df_predictions


def load_extended_data():
    """Return (df_clean, df_combined, df_imputed, df_predictions, df_oof, df_feature_ranking)."""
    df_clean, df_combined, df_imputed, df_predictions = load_data()

    # OOF predictions (graceful fallback)
    df_oof = pd.DataFrame()
    try:
        oof_dir = _find_latest_with(OOF_PREDICTIONS_FILENAME)
        df_oof = pd.read_csv(oof_dir / OOF_PREDICTIONS_FILENAME)
        print(f"OOF predictions: {df_oof.shape} (from {oof_dir.name})")
    except FileNotFoundError:
        print("OOF predictions: not found (Chart 6 will use full-fit fallback)")

    # Feature ranking (graceful fallback)
    df_feature_ranking = pd.DataFrame()
    try:
        fr_dir = _find_latest_with(FEATURE_RANKING_FILENAME)
        df_feature_ranking = pd.read_csv(fr_dir / FEATURE_RANKING_FILENAME)
        print(f"Feature ranking: {df_feature_ranking.shape} (from {fr_dir.name})")
    except FileNotFoundError:
        print("Feature ranking: not found (Chart 10 will be skipped)")

    return df_clean, df_combined, df_imputed, df_predictions, df_oof, df_feature_ranking


def build_source_map(df_combined: pd.DataFrame) -> pd.DataFrame:
    """model_name -> openbench_Source lookup."""
    return (
        df_combined[["Unified_Name", "openbench_Source"]]
        .rename(columns={"Unified_Name": "model_name"})
        .drop_duplicates(subset=["model_name"], keep="first")
    )


def source_styles(sources: list[str]):
    """Return (source_to_color, source_to_marker) dicts honouring CUSTOM_SOURCE_STYLES."""
    palette = sns.color_palette("tab20" if len(sources) > 10 else "tab10", n_colors=len(sources))
    s2c = {s: palette[i] for i, s in enumerate(sources)}
    s2m = {s: MARKER_CYCLE[i % len(MARKER_CYCLE)] for i, s in enumerate(sources)}
    for s in sources:
        if s in CUSTOM_SOURCE_STYLES:
            m, c = CUSTOM_SOURCE_STYLES[s]
            s2m[s] = m
            s2c[s] = c
    return s2c, s2m


def override_from_imputed(base_df: pd.DataFrame, imp_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if imp_df.empty or "model_name" not in imp_df.columns:
        return base_df
    use = [c for c in cols if c in imp_df.columns]
    if not use:
        return base_df
    out = base_df.set_index("model_name")
    imp_sub = imp_df[["model_name"] + use].drop_duplicates("model_name").set_index("model_name")
    out.update(imp_sub)
    return out.reset_index()


def prepare_fa_matrix(df_clean: pd.DataFrame, df_imputed: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare FA input: impute + scale. Returns (df with model_name + fa_cols, fa_cols list)."""
    df = df_clean.copy()

    # Fill from imputed
    if not df_imputed.empty and "model_name" in df_imputed.columns:
        common = df.columns.intersection(df_imputed.columns).tolist()
        if "model_name" in common:
            common.remove("model_name")
        if common:
            idx = df.set_index("model_name")
            imp = df_imputed[["model_name"] + common].drop_duplicates("model_name").set_index("model_name")
            idx.update(imp)
            df = idx.reset_index()

    numeric = df.select_dtypes(include=np.number).columns.tolist()
    fa_cols = [c for c in numeric
               if c not in EXCLUDE_FROM_FA
               and not _is_missingness_flag(c)
               and not _is_transformed(c)]

    return df[["model_name"] + fa_cols], fa_cols


# ──────────────────────────────────────────────────────────────────
# Chart 1: Cost vs Performance + Pareto Frontier
# ──────────────────────────────────────────────────────────────────

def compute_pareto_frontier(costs: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points (lower cost, higher score)."""
    n = len(costs)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is cheaper AND better (or equal on both with strict on one)
            if costs[j] <= costs[i] and scores[j] >= scores[i]:
                if costs[j] < costs[i] or scores[j] > scores[i]:
                    is_pareto[i] = False
                    break
    return is_pareto


def plot_cost_vs_performance(df_clean, df_combined, df_imputed, run_dir):
    print("\n--- Chart 1: Cost vs Performance + Pareto Frontier ---")

    df_fa, fa_cols = prepare_fa_matrix(df_clean, df_imputed)

    # 1-factor FA
    fa_input = df_fa[fa_cols].copy()
    X_imp = SimpleImputer(strategy="median").fit_transform(fa_input)
    X_scaled = StandardScaler().fit_transform(X_imp)

    fa = FactorAnalyzer(n_factors=1, rotation=None, method="ml")
    fa.fit(X_scaled)
    factor1 = fa.transform(X_scaled).ravel()
    loadings = fa.loadings_.ravel()
    if np.nanmean(loadings) < 0:
        factor1 = -factor1

    df_scores = pd.DataFrame({"model_name": df_fa["model_name"].values, "factor1_score": factor1})

    # Cost scenarios
    need = [c for c in COST_COLS_BASE if c in df_clean.columns]
    df_cost = df_clean[need].copy()

    AAI_R = "openbench_Reasoning Output Tokens Used to Run Artificial Analysis Intelligence Index (million)"
    AAI_A = "openbench_Answer Output Tokens Used to Run Artificial Analysis Intelligence Index (million)"
    SIMP_R = "logic_avg_reasoning_tokens"
    SIMP_A = "logic_avg_answer_tokens"

    df_cost = override_from_imputed(df_cost, df_imputed, [AAI_R, AAI_A, SIMP_R, SIMP_A])
    df_cost["price_per_token"] = df_cost["aa_pricing_price_1m_output_tokens"] / 1_000_000.0
    df_cost["cost1"] = df_cost["price_per_token"] * COST1_X_TOKENS
    df_cost["cost2"] = df_cost["price_per_token"] * (
        df_cost[SIMP_R].fillna(0) + df_cost[SIMP_A].fillna(0)
    )
    aa_total = (df_cost[AAI_R].fillna(0) + df_cost[AAI_A].fillna(0)) * 1_000_000.0
    df_cost["cost3"] = (df_cost["price_per_token"] * aa_total) / max(AAI_DIVISOR, 1.0)

    cost_scen = ["cost1", "cost2", "cost3"]
    df_cost["cost_min"] = df_cost[cost_scen].min(axis=1)
    df_cost["cost_max"] = df_cost[cost_scen].max(axis=1)
    df_cost["cost_mid"] = df_cost[cost_scen].median(axis=1)
    df_cost = df_cost[df_cost["cost_mid"] > 0].copy()

    # Source
    df_source = build_source_map(df_combined)
    df_plot = (
        df_scores
        .merge(df_cost[["model_name", "cost_min", "cost_max", "cost_mid"]], on="model_name", how="inner")
        .merge(df_source, on="model_name", how="left")
    )
    df_plot["openbench_Source"] = df_plot["openbench_Source"].fillna("Unknown")

    sources = df_plot["openbench_Source"].unique().tolist()
    s2c, s2m = source_styles(sources)

    # Pareto frontier
    costs_arr = df_plot["cost_mid"].values.astype(float)
    scores_arr = df_plot["factor1_score"].values.astype(float)
    pareto_mask = compute_pareto_frontier(costs_arr, scores_arr)
    df_plot["pareto"] = pareto_mask

    # Save pareto CSV
    df_plot[df_plot["pareto"]].to_csv(run_dir / "pareto_frontier_models.csv", index=False)
    print(f"  Pareto frontier: {pareto_mask.sum()} models")

    # Plot
    fig, ax = plt.subplots(figsize=(20, 20))
    x_med = np.exp(np.median(np.log(df_plot["cost_mid"].astype(float))))

    texts = []
    for _, r in df_plot.iterrows():
        x = float(r["cost_mid"])
        y = float(r["factor1_score"])
        left = max(x - float(r["cost_min"]), 0.0)
        right = max(float(r["cost_max"]) - x, 0.0)
        c = s2c[r["openbench_Source"]]
        alpha = 0.95 if r["pareto"] else 0.4

        ax.errorbar(x, y, xerr=[[left], [right]],
                    fmt=s2m[r["openbench_Source"]], color=c, ecolor=c,
                    markersize=11, capsize=4, elinewidth=1.6, alpha=alpha)

        to_left = x <= x_med
        dx = -20 if to_left else 20
        ha = "right" if to_left else "left"
        ann = ax.annotate(
            r["model_name"], xy=(x, y), xycoords="data",
            xytext=(dx, -10), textcoords="offset points",
            ha=ha, va="center",
            bbox={**LABEL_BBOX, "pad": 0.25},
            arrowprops=dict(arrowstyle="-", lw=0.6, color="gray", alpha=0.6),
            fontsize=11 if r["pareto"] else 9,
            alpha=1.0 if r["pareto"] else 0.5,
            zorder=3,
        )
        texts.append(ann)

    # Pareto frontier stepped line
    frontier = df_plot[df_plot["pareto"]].sort_values("cost_mid")
    if len(frontier) >= 2:
        fx = frontier["cost_mid"].values
        fy = frontier["factor1_score"].values
        # Build stepped path: for each pair, go horizontal then vertical
        step_x, step_y = [fx[0]], [fy[0]]
        for i in range(1, len(fx)):
            step_x.extend([fx[i], fx[i]])
            step_y.extend([fy[i - 1], fy[i]])
        ax.plot(step_x, step_y, color="#E53935", linewidth=2.5, linestyle="--",
                alpha=0.7, zorder=2, label="Pareto frontier")

    adjust_text(texts, only_move={"text": "xy"}, expand_text=(10, 10),
                force_text=1, force_static=1, pull_threshold=5)

    ax.set_xscale("log")
    ax.set_xlabel(f"Cost (USD) — [1] {COST1_X_TOKENS:.0f} token response · [2] SimpleBenchfree avg · [3] AAI ÷ {AAI_DIVISOR:g}",
                  fontsize=14)
    ax.set_ylabel("Performance (Factor 1 score)", fontsize=14)
    ax.set_title("Cost Range vs Performance (1-Factor FA) + Pareto Frontier\nColored by openbench_Source",
                 fontsize=18)
    ax.tick_params(labelsize=12)

    handles = [Line2D([0], [0], marker=s2m[s], linestyle="", color=s2c[s],
                      label=s, markersize=11) for s in sources]
    handles.append(Line2D([0], [0], color="#E53935", linewidth=2.5, linestyle="--",
                          alpha=0.7, label="Pareto frontier"))
    ax.legend(handles=handles, title="openbench_Source", frameon=False, ncol=3, loc="lower right",
              fontsize=12)

    ax.grid(True, which="both", axis="x", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.margins(x=0.07, y=0.08)
    fig.tight_layout()
    fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.10)

    fig.savefig(run_dir / "cost_vs_performance_pareto.png", dpi=300)
    fig.savefig(run_dir / "cost_vs_performance_pareto.svg")
    plt.close(fig)
    print("  Saved cost_vs_performance_pareto.png + .svg")

    # Return factor1 scores + fa components for reuse
    return df_scores, fa_cols, X_scaled


# ──────────────────────────────────────────────────────────────────
# Chart 2: Arena Residuals
# ──────────────────────────────────────────────────────────────────

def plot_arena_residuals(df_predictions, df_combined, run_dir):
    print("\n--- Chart 2: Arena Residuals ---")

    df = df_predictions.dropna(subset=["actual_score"]).copy()
    df["residual"] = df["actual_score"] - df["predicted_score"]
    df = df.sort_values("residual")

    # Add source
    df_source = build_source_map(df_combined)
    df = df.merge(df_source, on="model_name", how="left")
    df["openbench_Source"] = df["openbench_Source"].fillna("Unknown")

    sources = df["openbench_Source"].unique().tolist()
    s2c, _ = source_styles(sources)

    df.to_csv(run_dir / "arena_residuals.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, max(10, len(df) * 0.22)))
    colors = [s2c[s] for s in df["openbench_Source"]]
    bars = ax.barh(range(len(df)), df["residual"].values, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["model_name"].values, fontsize=7)
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)

    # Label top/bottom 5
    for idx in list(range(5)) + list(range(len(df) - 5, len(df))):
        if 0 <= idx < len(df):
            val = df["residual"].iloc[idx]
            ha = "left" if val >= 0 else "right"
            offset = 0.5 if val >= 0 else -0.5
            ax.text(val + offset, idx, f"{val:+.1f}", va="center", ha=ha, fontsize=7, fontweight="bold")

    handles = [Line2D([0], [0], marker="s", linestyle="", color=s2c[s],
                      label=s, markersize=8) for s in sorted(set(df["openbench_Source"]))]
    ax.legend(handles=handles, title="Vendor", frameon=False, loc="lower right", fontsize=9)

    ax.set_xlabel("Residual (Actual − Predicted Arena Score)", fontsize=12)
    ax.set_title("Arena Score vs Benchmark Prediction", fontsize=16)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(run_dir / "arena_residuals.png", dpi=300)
    plt.close(fig)
    print(f"  Saved arena_residuals.png ({len(df)} models)")


def plot_arena_residuals_oof(df_oof, df_combined, run_dir):
    """Chart 2b: Arena residuals using out-of-fold predictions."""
    print("\n--- Chart 2b: Arena Residuals (OOF) ---")

    if df_oof is None or df_oof.empty or len(df_oof) < 5:
        print("  Skipped – OOF predictions not available")
        return

    df = df_oof.dropna(subset=["actual_score", "oof_predicted_score"]).copy()
    df["residual"] = df["actual_score"] - df["oof_predicted_score"]
    df = df.sort_values("residual")

    # Add source
    df_source = build_source_map(df_combined)
    df = df.merge(df_source, on="model_name", how="left")
    df["openbench_Source"] = df["openbench_Source"].fillna("Unknown")

    sources = df["openbench_Source"].unique().tolist()
    s2c, _ = source_styles(sources)

    df.to_csv(run_dir / "arena_residuals_oof.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, max(10, len(df) * 0.22)))
    colors = [s2c[s] for s in df["openbench_Source"]]
    ax.barh(range(len(df)), df["residual"].values, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["model_name"].values, fontsize=7)
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)

    # Label top/bottom 5
    for idx in list(range(5)) + list(range(len(df) - 5, len(df))):
        if 0 <= idx < len(df):
            val = df["residual"].iloc[idx]
            ha = "left" if val >= 0 else "right"
            offset = 0.5 if val >= 0 else -0.5
            ax.text(val + offset, idx, f"{val:+.1f}", va="center", ha=ha, fontsize=7, fontweight="bold")

    handles = [Line2D([0], [0], marker="s", linestyle="", color=s2c[s],
                      label=s, markersize=8) for s in sorted(set(df["openbench_Source"]))]
    ax.legend(handles=handles, title="Vendor", frameon=False, loc="lower right", fontsize=9)

    ax.set_xlabel("Residual (Actual − OOF Predicted Arena Score)", fontsize=12)
    ax.set_title("Arena Score vs Benchmark Prediction (Out-of-Fold)", fontsize=16)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(run_dir / "arena_residuals_oof.png", dpi=300)
    plt.close(fig)
    print(f"  Saved arena_residuals_oof.png ({len(df)} models)")


# ──────────────────────────────────────────────────────────────────
# Chart 3: Multi-Factor Radar Profiles
# ──────────────────────────────────────────────────────────────────

def _auto_name_factor(loadings_col: pd.Series, fa_cols: list[str], top_n: int = 3) -> str:
    """Name a factor from its top-N loadings by picking dominant benchmark source prefix."""
    abs_load = loadings_col.abs().sort_values(ascending=False)
    top_features = abs_load.head(top_n).index.tolist()

    # Extract prefix before first underscore as category hint
    prefixes = []
    for f in top_features:
        parts = f.split("_", 1)
        prefixes.append(parts[0] if len(parts) > 1 else f)

    # Most common prefix
    from collections import Counter
    counts = Counter(prefixes)
    dominant = counts.most_common(1)[0][0]

    # Clean up common prefixes
    name_map = {
        "livebench": "LiveBench",
        "aa": "ArtificialAnalysis",
        "openbench": "OpenBench",
        "eqbench": "EQBench",
        "eq": "EQ",
        "logic": "Logic",
        "simplebench": "SimpleBench",
        "style": "Style",
        "tone": "Tone",
        "writing": "Writing",
        "arc": "ARC",
        "lechmazur": "LechMazur",
        "aiderbench": "AiderBench",
        "contextarena": "ContextArena",
        "weirdml": "WeirdML",
        "yupp": "Yupp",
        "ugileaderboard": "UGI",
        "aagdpval": "GDP",
        "aaomniscience": "Omniscience",
        "aacritpt": "CritPt",
        "lmsys": "Arena",
        "lmarena": "Arena",
    }
    clean = name_map.get(dominant.lower(), dominant)

    # Add top feature hint
    short_features = [f.split("_", 1)[-1][:20] if "_" in f else f[:20] for f in top_features[:2]]
    return f"{clean} ({', '.join(short_features)})"


def extract_factors(df_clean, df_imputed, n_factors=4):
    """Run n-factor FA with varimax. Returns (factor_scores_df, loadings_df, factor_names)."""
    print("  Running 4-factor FA with varimax rotation...")
    df_fa, fa_cols = prepare_fa_matrix(df_clean, df_imputed)

    fa_input = df_fa[fa_cols].copy()
    X_imp = SimpleImputer(strategy="median").fit_transform(fa_input)
    X_scaled = StandardScaler().fit_transform(X_imp)

    fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax", method="ml")
    fa.fit(X_scaled)

    scores = fa.transform(X_scaled)
    loadings = pd.DataFrame(fa.loadings_, index=fa_cols,
                            columns=[f"Factor{i+1}" for i in range(n_factors)])

    # Auto-name factors
    factor_names = []
    for i in range(n_factors):
        col = loadings.iloc[:, i]
        name = _auto_name_factor(col, fa_cols)
        factor_names.append(name)
    print(f"  Factor names: {factor_names}")

    scores_df = pd.DataFrame(scores, columns=[f"Factor{i+1}" for i in range(n_factors)])
    scores_df.insert(0, "model_name", df_fa["model_name"].values)

    return scores_df, loadings, factor_names


def _radar_chart(ax, categories, values_dict, colors_dict, title=""):
    """Draw a radar/spider chart on the given axis."""
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=9)

    for label, vals in values_dict.items():
        vals_closed = vals.tolist() + [vals[0]]
        ax.plot(angles, vals_closed, linewidth=2, label=label, color=colors_dict.get(label, None))
        ax.fill(angles, vals_closed, alpha=0.1, color=colors_dict.get(label, None))

    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.1), fontsize=8)


def plot_radar_top_models(scores_df, factor_names, df_combined, run_dir, df_f1_scores=None, top_n=8):
    print("  Plotting radar: top models...")
    # Top models by 1-factor FA score (overall performance) if available,
    # otherwise fall back to multi-factor Factor1
    if df_f1_scores is not None:
        top_names = df_f1_scores.nlargest(top_n, "factor1_score")["model_name"].tolist()
        top = scores_df[scores_df["model_name"].isin(top_names)]
    else:
        top = scores_df.nlargest(top_n, "Factor1")

    # Source colors
    df_source = build_source_map(df_combined)
    top = top.merge(df_source, on="model_name", how="left")
    top["openbench_Source"] = top["openbench_Source"].fillna("Unknown")
    sources = top["openbench_Source"].unique().tolist()
    s2c, _ = source_styles(sources)

    factor_cols = [c for c in scores_df.columns if c.startswith("Factor")]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    vals_dict = {}
    colors_dict = {}
    for _, row in top.iterrows():
        name = row["model_name"]
        vals_dict[name] = row[factor_cols].values.astype(float)
        colors_dict[name] = s2c[row["openbench_Source"]]

    _radar_chart(ax, factor_names, vals_dict, colors_dict, title="Top Models: Multi-Factor Profile")
    fig.tight_layout()
    fig.savefig(run_dir / "factor_radar_top_models.png", dpi=300)
    plt.close(fig)
    print(f"  Saved factor_radar_top_models.png ({top_n} models)")


def plot_radar_by_vendor(scores_df, factor_names, df_combined, run_dir):
    print("  Plotting radar: vendor averages...")
    df_source = build_source_map(df_combined)
    df = scores_df.merge(df_source, on="model_name", how="left")
    df["openbench_Source"] = df["openbench_Source"].fillna("Unknown")

    factor_cols = [c for c in scores_df.columns if c.startswith("Factor")]
    vendor_avg = df.groupby("openbench_Source")[factor_cols].mean()

    sources = vendor_avg.index.tolist()
    s2c, _ = source_styles(sources)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    vals_dict = {}
    colors_dict = {}
    for vendor in sources:
        vals_dict[vendor] = vendor_avg.loc[vendor].values.astype(float)
        colors_dict[vendor] = s2c[vendor]

    _radar_chart(ax, factor_names, vals_dict, colors_dict, title="Vendor Averages: Multi-Factor Profile")
    fig.tight_layout()
    fig.savefig(run_dir / "factor_radar_by_vendor.png", dpi=300)
    plt.close(fig)
    print(f"  Saved factor_radar_by_vendor.png ({len(sources)} vendors)")


# ──────────────────────────────────────────────────────────────────
# Chart 4: Benchmark Clustering Dendrogram
# ──────────────────────────────────────────────────────────────────

def plot_benchmark_dendrogram(df_imputed, run_dir):
    print("\n--- Chart 4: Benchmark Clustering Dendrogram ---")

    if "model_name" in df_imputed.columns:
        numeric = df_imputed.set_index("model_name").select_dtypes(include=np.number)
    else:
        numeric = df_imputed.select_dtypes(include=np.number)

    # Drop missingness/transformed cols
    keep = [c for c in numeric.columns
            if not _is_missingness_flag(c) and not _is_transformed(c)]
    numeric = numeric[keep]

    # Drop constant columns
    nunique = numeric.nunique(dropna=False)
    numeric = numeric.loc[:, nunique > 1]

    # Fill NaN for correlation
    numeric = numeric.fillna(numeric.mean())

    print(f"  Computing correlations for {numeric.shape[1]} benchmarks...")
    corr = numeric.corr(method="pearson").abs()
    dist = 1.0 - corr

    # Condense to pairwise distance vector
    from scipy.spatial.distance import squareform
    dist_vec = squareform(dist.values, checks=False)
    dist_vec = np.nan_to_num(dist_vec, nan=1.0)

    Z = linkage(dist_vec, method="ward")
    clusters = fcluster(Z, t=5, criterion="maxclust")

    cluster_df = pd.DataFrame({"benchmark": corr.columns, "cluster": clusters})
    cluster_df.to_csv(run_dir / "benchmark_clusters.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, max(10, len(keep) * 0.25)))
    dendrogram(
        Z,
        labels=corr.columns.tolist(),
        orientation="left",
        leaf_font_size=7,
        color_threshold=Z[-4, 2] if len(Z) >= 4 else None,
        ax=ax,
    )
    ax.set_xlabel("Distance (1 − |r|)", fontsize=12)
    ax.set_title("Benchmark Clustering Dendrogram (Ward's linkage)", fontsize=16)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(run_dir / "benchmark_dendrogram.png", dpi=300)
    plt.close(fig)
    print(f"  Saved benchmark_dendrogram.png ({len(keep)} benchmarks, 5 clusters)")


# ──────────────────────────────────────────────────────────────────
# Chart 5: Reasoning vs Non-Reasoning
# ──────────────────────────────────────────────────────────────────

def plot_reasoning_comparison(scores_df, factor_names, df_clean, run_dir):
    print("\n--- Chart 5: Reasoning vs Non-Reasoning ---")

    # openbench_Reasoning is 1.0/0.0 in clean
    if "openbench_Reasoning" not in df_clean.columns:
        print("  WARNING: openbench_Reasoning column not found, skipping.")
        return

    reasoning_map = df_clean[["model_name", "openbench_Reasoning"]].copy()
    reasoning_map["reasoning_label"] = reasoning_map["openbench_Reasoning"].map(
        {1.0: "Reasoning", 0.0: "Non-Reasoning"}
    )
    reasoning_map = reasoning_map.dropna(subset=["reasoning_label"])

    df = scores_df.merge(reasoning_map[["model_name", "reasoning_label"]], on="model_name", how="inner")

    factor_cols = [c for c in scores_df.columns if c.startswith("Factor")]
    n_factors = len(factor_cols)

    # Stats
    stats_rows = []
    for i, fc in enumerate(factor_cols):
        reasoning = df.loc[df["reasoning_label"] == "Reasoning", fc].values
        non_reasoning = df.loc[df["reasoning_label"] == "Non-Reasoning", fc].values
        u_stat, p_val = mannwhitneyu(reasoning, non_reasoning, alternative="two-sided")
        stats_rows.append({
            "factor": factor_names[i],
            "factor_col": fc,
            "reasoning_median": float(np.median(reasoning)),
            "non_reasoning_median": float(np.median(non_reasoning)),
            "mann_whitney_U": float(u_stat),
            "p_value": float(p_val),
            "n_reasoning": len(reasoning),
            "n_non_reasoning": len(non_reasoning),
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(run_dir / "reasoning_comparison.csv", index=False)

    # Violin plots
    fig, axes = plt.subplots(1, n_factors, figsize=(4 * n_factors, 7), sharey=False)
    if n_factors == 1:
        axes = [axes]

    palette = {"Reasoning": "#2196F3", "Non-Reasoning": "#FF9800"}

    for i, (fc, ax) in enumerate(zip(factor_cols, axes)):
        sns.violinplot(data=df, x="reasoning_label", y=fc, ax=ax, palette=palette,
                       inner="box", cut=0, density_norm="width")

        p = stats_rows[i]["p_value"]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f"{factor_names[i]}\np={p:.2e} {stars}", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Factor Score" if i == 0 else "")
        ax.tick_params(labelsize=10)

    fig.suptitle("Reasoning vs Non-Reasoning: Factor Score Comparison", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(run_dir / "reasoning_vs_nonreasoning.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    n_r = stats_rows[0]["n_reasoning"]
    n_nr = stats_rows[0]["n_non_reasoning"]
    print(f"  Saved reasoning_vs_nonreasoning.png ({n_r} reasoning, {n_nr} non-reasoning)")


# ──────────────────────────────────────────────────────────────────
# Chart 6: Prediction Calibration
# ──────────────────────────────────────────────────────────────────

def plot_prediction_calibration(df_predictions, df_oof, run_dir):
    print("\n--- Chart 6: Prediction Calibration ---")

    # Use OOF predictions if available, else fall back to full-fit
    if not df_oof.empty and len(df_oof) > 10:
        actual = df_oof["actual_score"].values
        predicted = df_oof["oof_predicted_score"].values
        source_label = "OOF"
    else:
        df_fit = df_predictions.dropna(subset=["actual_score"]).copy()
        if len(df_fit) < 10:
            print("  WARNING: Not enough models with actual scores, skipping.")
            return
        actual = df_fit["actual_score"].values
        predicted = df_fit["predicted_score"].values
        source_label = "full-fit"

    print(f"  Using {source_label} predictions ({len(actual)} models)")

    # Panel 1: Reliability diagram — bin by predicted decile
    n_bins = min(10, max(3, len(actual) // 8))
    bin_edges = np.percentile(predicted, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6
    bin_idx = np.digitize(predicted, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bin_mean_pred = np.array([predicted[bin_idx == b].mean() for b in range(n_bins)])
    bin_mean_act = np.array([actual[bin_idx == b].mean() for b in range(n_bins)])
    bin_counts = np.array([np.sum(bin_idx == b) for b in range(n_bins)])

    # Bootstrap 95% CI for actual means
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_means = np.full((n_bins, n_boot), np.nan)
    for b in range(n_bins):
        vals = actual[bin_idx == b]
        if len(vals) >= 2:
            for j in range(n_boot):
                boot_means[b, j] = rng.choice(vals, size=len(vals), replace=True).mean()
    ci_lo = np.nanpercentile(boot_means, 2.5, axis=1)
    ci_hi = np.nanpercentile(boot_means, 97.5, axis=1)

    # Panel 2: Prediction interval coverage (if bounds available)
    # Use OOF predictions for training models to avoid refit overfit.
    # Merge OOF predictions with the CI bounds from df_predictions (which are
    # based on conformal groups, not the refit point prediction).
    has_bounds = "lower_bound" in df_predictions.columns and "upper_bound" in df_predictions.columns
    if has_bounds and not df_oof.empty:
        # Get sigma_hat (halfwidth) from predictions, apply to OOF point predictions
        df_cov = df_predictions.dropna(subset=["actual_score"]).copy()
        df_cov = df_cov.merge(
            df_oof[["model_name", "oof_predicted_score"]],
            on="model_name", how="inner"
        )
        cov_actual = df_cov["actual_score"].values
        cov_pred = df_cov["oof_predicted_score"].values
        halfwidth = df_cov["sigma_hat"].values
        cov_lower = cov_pred - halfwidth
        cov_upper = cov_pred + halfwidth
        in_interval = (cov_actual >= cov_lower) & (cov_actual <= cov_upper)

        cov_bin_edges = np.percentile(cov_pred, np.linspace(0, 100, n_bins + 1))
        cov_bin_edges[0] -= 1e-6
        cov_bin_edges[-1] += 1e-6
        cov_bin_idx = np.clip(np.digitize(cov_pred, cov_bin_edges) - 1, 0, n_bins - 1)

        cov_by_bin = np.array([in_interval[cov_bin_idx == b].mean() if np.sum(cov_bin_idx == b) > 0 else np.nan
                               for b in range(n_bins)])
        overall_cov = in_interval.mean()
    elif has_bounds:
        # Fallback: use refit predictions if no OOF available
        df_cov = df_predictions.dropna(subset=["actual_score"]).copy()
        cov_actual = df_cov["actual_score"].values
        cov_pred = df_cov["predicted_score"].values
        cov_lower = df_cov["lower_bound"].values
        cov_upper = df_cov["upper_bound"].values
        in_interval = (cov_actual >= cov_lower) & (cov_actual <= cov_upper)

        cov_bin_edges = np.percentile(cov_pred, np.linspace(0, 100, n_bins + 1))
        cov_bin_edges[0] -= 1e-6
        cov_bin_edges[-1] += 1e-6
        cov_bin_idx = np.clip(np.digitize(cov_pred, cov_bin_edges) - 1, 0, n_bins - 1)

        cov_by_bin = np.array([in_interval[cov_bin_idx == b].mean() if np.sum(cov_bin_idx == b) > 0 else np.nan
                               for b in range(n_bins)])
        overall_cov = in_interval.mean()

    n_panels = 2 if has_bounds else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: Reliability diagram
    ax = axes[0]
    rng_vals = [bin_mean_pred.min(), bin_mean_pred.max()]
    ax.plot(rng_vals, rng_vals, "k--", alpha=0.5, label="Perfect calibration")
    ax.errorbar(bin_mean_pred, bin_mean_act,
                yerr=[bin_mean_act - ci_lo, ci_hi - bin_mean_act],
                fmt="o-", color="#2196F3", capsize=4, label=f"Binned ({source_label})")
    for b in range(n_bins):
        ax.annotate(f"n={bin_counts[b]}", (bin_mean_pred[b], bin_mean_act[b]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.7)
    ax.set_xlabel("Mean Predicted Score (bin)", fontsize=12)
    ax.set_ylabel("Mean Actual Score (bin)", fontsize=12)
    ax.set_title("Reliability Diagram", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)

    # Panel 2: Coverage
    if has_bounds:
        ax2 = axes[1]
        bin_centers = np.arange(n_bins)
        bars = ax2.bar(bin_centers, cov_by_bin * 100, color="#4CAF50", edgecolor="white", alpha=0.8)
        ax2.axhline(95, color="red", linestyle="--", alpha=0.7, label="95% target")
        ax2.axhline(overall_cov * 100, color="#2196F3", linestyle="-.", alpha=0.7,
                     label=f"Overall: {overall_cov:.1%}")
        ax2.set_xlabel("Predicted Score Decile", fontsize=12)
        ax2.set_ylabel("Coverage (%)", fontsize=12)
        ax2.set_title("Prediction Interval Coverage", fontsize=14)
        ax2.set_xticks(bin_centers)
        ax2.set_xticklabels([f"D{b+1}" for b in bin_centers], fontsize=9)
        ax2.set_ylim(0, 105)
        ax2.legend(fontsize=9)
        ax2.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("Prediction Calibration Analysis", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(run_dir / "prediction_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save stats
    cal_df = pd.DataFrame({
        "bin": np.arange(n_bins),
        "mean_predicted": bin_mean_pred,
        "mean_actual": bin_mean_act,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "count": bin_counts,
    })
    if has_bounds:
        cal_df["coverage_pct"] = cov_by_bin * 100
    cal_df.to_csv(run_dir / "calibration_stats.csv", index=False)
    print(f"  Saved prediction_calibration.png ({len(actual)} models, {n_bins} bins)")


# ──────────────────────────────────────────────────────────────────
# Chart 7: Residual Bias Decomposition
# ──────────────────────────────────────────────────────────────────

def plot_residual_bias(df_predictions, df_combined, df_clean, run_dir, df_oof=None):
    print("\n--- Chart 7: Residual Bias Decomposition ---")

    # Prefer OOF predictions (proper held-out residuals) over full-fit
    if df_oof is not None and not df_oof.empty and len(df_oof) > 10:
        df = df_oof.copy()
        df["predicted_score"] = df["oof_predicted_score"]
        df["residual"] = df["actual_score"] - df["oof_predicted_score"]
        print("  Using OOF predictions for residuals")
    else:
        df = df_predictions.dropna(subset=["actual_score"]).copy()
        df["residual"] = df["actual_score"] - df["predicted_score"]
        print("  Using full-fit predictions for residuals (OOF not available)")

    # Merge vendor
    df_source = build_source_map(df_combined)
    df = df.merge(df_source, on="model_name", how="left")
    df["openbench_Source"] = df["openbench_Source"].fillna("Other")

    # Merge reasoning flag
    if "openbench_Reasoning" in df_clean.columns:
        reason_map = df_clean[["model_name", "openbench_Reasoning"]].drop_duplicates("model_name")
        df = df.merge(reason_map, on="model_name", how="left")
        df["reasoning_label"] = df["openbench_Reasoning"].map(
            {1.0: "Reasoning", 0.0: "Non-Reasoning"}
        ).fillna("Unknown")
    else:
        df["reasoning_label"] = "Unknown"

    if len(df) < 5:
        print("  WARNING: Not enough models, skipping.")
        return

    # Save CSV
    bias_df = df[["model_name", "openbench_Source", "reasoning_label", "residual",
                   "actual_score", "predicted_score"]].copy()
    bias_df.to_csv(run_dir / "residual_bias.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Heatmap of vendor × reasoning mean residuals
    ax1 = axes[0]
    pivot = df.pivot_table(values="residual", index="openbench_Source",
                           columns="reasoning_label", aggfunc="mean")
    if pivot.empty:
        ax1.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax1.transAxes)
    else:
        vmax = max(abs(np.nanmin(pivot.values)), abs(np.nanmax(pivot.values)), 1)
        sns.heatmap(pivot, ax=ax1, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                    annot=True, fmt=".1f", linewidths=0.5, cbar_kws={"label": "Mean Residual"})
        ax1.set_title("Mean Residual: Vendor × Reasoning", fontsize=13)
        ax1.set_ylabel("")
        ax1.set_xlabel("")

    # Panel 2: Vendor bar chart with 95% CI
    ax2 = axes[1]
    vendor_stats = df.groupby("openbench_Source")["residual"].agg(["mean", "std", "count"]).reset_index()
    vendor_stats["se"] = vendor_stats["std"] / np.sqrt(vendor_stats["count"])
    vendor_stats["ci95"] = 1.96 * vendor_stats["se"]
    vendor_stats = vendor_stats.sort_values("mean")

    s2c, _ = source_styles(vendor_stats["openbench_Source"].tolist())
    colors = [s2c.get(v, "#90A4AE") for v in vendor_stats["openbench_Source"]]
    bars = ax2.barh(range(len(vendor_stats)), vendor_stats["mean"].values,
                    xerr=vendor_stats["ci95"].values, color=colors,
                    edgecolor="white", capsize=3)
    ax2.set_yticks(range(len(vendor_stats)))
    ax2.set_yticklabels(vendor_stats["openbench_Source"].values, fontsize=10)
    ax2.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    for i, row in vendor_stats.reset_index(drop=True).iterrows():
        ax2.annotate(f"n={int(row['count'])}", (row["mean"], i),
                     textcoords="offset points", xytext=(5, 0), fontsize=8, alpha=0.7)
    ax2.set_xlabel("Mean Residual (Actual − Predicted)", fontsize=12)
    ax2.set_title("Residual by Vendor (±95% CI)", fontsize=13)
    ax2.grid(True, axis="x", linestyle=":", alpha=0.4)

    fig.suptitle("Residual Bias Decomposition", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(run_dir / "residual_bias_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved residual_bias_heatmap.png ({len(df)} models)")


# ──────────────────────────────────────────────────────────────────
# Chart 8: Rank Stability
# ──────────────────────────────────────────────────────────────────

def plot_rank_stability(df_predictions, df_combined, run_dir):
    print("\n--- Chart 8: Rank Stability ---")

    df = df_predictions.copy()
    if "lower_bound" not in df.columns or "upper_bound" not in df.columns:
        print("  WARNING: No prediction intervals available, skipping.")
        return

    # Estimate sigma from prediction intervals (assume ~95% = ±1.96σ)
    df["sigma"] = (df["upper_bound"] - df["lower_bound"]) / 3.92
    df["sigma"] = df["sigma"].clip(lower=1e-3)

    # Monte Carlo rank simulation
    n_models = len(df)
    n_draws = 1000
    rng = np.random.default_rng(42)
    predicted = df["predicted_score"].values
    sigma = df["sigma"].values

    rank_matrix = np.zeros((n_models, n_draws), dtype=int)
    for d in range(n_draws):
        draws = rng.normal(predicted, sigma)
        rank_matrix[:, d] = n_models - draws.argsort().argsort()  # rank 1 = best

    median_rank = np.median(rank_matrix, axis=1).astype(int)
    pct5 = np.percentile(rank_matrix, 5, axis=1).astype(int)
    pct95 = np.percentile(rank_matrix, 95, axis=1).astype(int)

    df["median_rank"] = median_rank
    df["rank_5pct"] = pct5
    df["rank_95pct"] = pct95
    df["rank_range"] = pct95 - pct5

    # Add source
    df_source = build_source_map(df_combined)
    df = df.merge(df_source, on="model_name", how="left")
    df["openbench_Source"] = df["openbench_Source"].fillna("Other")

    # Save full CSV
    rank_df = df[["model_name", "openbench_Source", "predicted_score", "sigma",
                   "median_rank", "rank_5pct", "rank_95pct", "rank_range"]].copy()
    rank_df = rank_df.sort_values("median_rank")
    rank_df.to_csv(run_dir / "rank_stability.csv", index=False)

    # Plot top 30
    top = df.nsmallest(30, "median_rank").sort_values("median_rank", ascending=False)
    s2c, _ = source_styles(top["openbench_Source"].unique().tolist())

    fig, ax = plt.subplots(figsize=(12, 10))
    y_pos = range(len(top))
    colors = [s2c.get(s, "#90A4AE") for s in top["openbench_Source"]]

    for i, (_, row) in enumerate(top.iterrows()):
        ax.barh(i, row["rank_95pct"] - row["rank_5pct"],
                left=row["rank_5pct"], height=0.6,
                color=colors[i], alpha=0.4, edgecolor="none")
        ax.plot(row["median_rank"], i, "o", color=colors[i], markersize=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["model_name"].values, fontsize=8)
    ax.set_xlabel("Rank (1 = best)", fontsize=12)
    ax.set_title("Rank Stability: Top 30 Models (MC 1000 draws, 5th–95th pct)", fontsize=14)
    ax.invert_xaxis()
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    # Legend
    unique_sources = sorted(set(top["openbench_Source"]))
    handles = [Line2D([0], [0], marker="o", linestyle="", color=s2c.get(s, "#90A4AE"),
                      label=s, markersize=7) for s in unique_sources]
    ax.legend(handles=handles, title="Vendor", frameon=False, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(run_dir / "rank_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved rank_stability.png (top 30 of {n_models} models)")


# ──────────────────────────────────────────────────────────────────
# Chart 9: Benchmark Redundancy
# ──────────────────────────────────────────────────────────────────

def plot_benchmark_redundancy(df_imputed, run_dir):
    print("\n--- Chart 9: Benchmark Redundancy ---")

    if "model_name" in df_imputed.columns:
        numeric = df_imputed.set_index("model_name").select_dtypes(include=np.number)
    else:
        numeric = df_imputed.select_dtypes(include=np.number)

    keep = [c for c in numeric.columns
            if not _is_missingness_flag(c) and not _is_transformed(c)]
    numeric = numeric[keep]
    nunique = numeric.nunique(dropna=False)
    numeric = numeric.loc[:, nunique > 1]
    numeric = numeric.fillna(numeric.mean())

    if numeric.shape[1] < 3:
        print("  WARNING: Too few benchmarks, skipping.")
        return

    # Correlation matrix
    corr = numeric.corr(method="pearson")

    # Partial correlation: pcorr = -inv(corr) / outer(sqrt(diag(inv(corr))))
    reg = 1e-4 * np.eye(corr.shape[0])
    try:
        corr_inv = np.linalg.inv(corr.values + reg)
    except np.linalg.LinAlgError:
        print("  WARNING: Correlation matrix not invertible, skipping.")
        return

    d = np.sqrt(np.abs(np.diag(corr_inv)))
    d[d == 0] = 1e-10
    pcorr = -corr_inv / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)
    pcorr_df = pd.DataFrame(pcorr, index=corr.index, columns=corr.columns)
    pcorr_df.to_csv(run_dir / "partial_correlations.csv")

    # Load cluster assignments from Chart 4
    cluster_file = run_dir / "benchmark_clusters.csv"
    if cluster_file.exists():
        cluster_df = pd.read_csv(cluster_file)
        cluster_map = dict(zip(cluster_df["benchmark"], cluster_df["cluster"]))
        # Reorder columns by cluster
        ordered_cols = sorted(pcorr_df.columns,
                              key=lambda c: (cluster_map.get(c, 999), c))
        pcorr_df = pcorr_df.loc[ordered_cols, ordered_cols]

        # Compute within-cluster vs between-cluster mean |pcorr|
        n = len(ordered_cols)
        within_vals, between_vals = [], []
        for i in range(n):
            for j in range(i + 1, n):
                ci = cluster_map.get(ordered_cols[i], -1)
                cj = cluster_map.get(ordered_cols[j], -1)
                val = abs(pcorr_df.iloc[i, j])
                if ci == cj and ci != -1:
                    within_vals.append(val)
                else:
                    between_vals.append(val)
        within_mean = np.mean(within_vals) if within_vals else 0
        between_mean = np.mean(between_vals) if between_vals else 0
        title_extra = f"\nWithin-cluster |pcorr|={within_mean:.3f}, Between={between_mean:.3f}"
    else:
        cluster_map = {}
        title_extra = ""

    # Plot
    fig, ax = plt.subplots(figsize=(max(14, len(pcorr_df) * 0.2), max(12, len(pcorr_df) * 0.2)))
    vmax = np.percentile(np.abs(pcorr_df.values[np.triu_indices(len(pcorr_df), k=1)]), 95)
    vmax = max(vmax, 0.1)
    sns.heatmap(pcorr_df, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                xticklabels=True, yticklabels=True,
                cbar_kws={"label": "Partial Correlation"})
    ax.tick_params(labelsize=5)

    # Draw cluster boundaries
    if cluster_map:
        clusters_ordered = [cluster_map.get(c, -1) for c in pcorr_df.columns]
        prev_cluster = clusters_ordered[0]
        for idx in range(1, len(clusters_ordered)):
            if clusters_ordered[idx] != prev_cluster:
                ax.add_patch(Rectangle((idx, 0), 0, len(pcorr_df),
                                       fill=False, edgecolor="black", linewidth=1.5))
                ax.axvline(idx, color="black", linewidth=1.2, alpha=0.7)
                ax.axhline(idx, color="black", linewidth=1.2, alpha=0.7)
                prev_cluster = clusters_ordered[idx]

    ax.set_title(f"Benchmark Partial Correlation (Redundancy){title_extra}", fontsize=14)
    fig.tight_layout()
    fig.savefig(run_dir / "benchmark_redundancy_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved benchmark_redundancy_heatmap.png ({len(pcorr_df)} benchmarks)")


# ──────────────────────────────────────────────────────────────────
# Chart 10: Marginal Benchmark Value
# ──────────────────────────────────────────────────────────────────

def plot_marginal_benchmark_value(df_imputed, df_predictions, df_feature_ranking, run_dir):
    print("\n--- Chart 10: Marginal Benchmark Value ---")

    if df_feature_ranking.empty:
        print("  WARNING: Feature ranking not available, skipping.")
        return

    cluster_file = run_dir / "benchmark_clusters.csv"
    if not cluster_file.exists():
        print("  WARNING: benchmark_clusters.csv not found, skipping.")
        return

    cluster_df = pd.read_csv(cluster_file)
    cluster_map = dict(zip(cluster_df["benchmark"], cluster_df["cluster"]))

    # Map features to clusters and compute per-cluster importance
    gain_col = "mean_gain" if "mean_gain" in df_feature_ranking.columns else df_feature_ranking.columns[1]
    feature_col = "feature" if "feature" in df_feature_ranking.columns else df_feature_ranking.columns[0]

    cluster_importance = {}
    for _, row in df_feature_ranking.iterrows():
        feat = row[feature_col]
        cl = cluster_map.get(feat, None)
        if cl is not None:
            cluster_importance[cl] = cluster_importance.get(cl, 0) + row[gain_col]

    if not cluster_importance:
        print("  WARNING: No feature-to-cluster mapping found, skipping.")
        return

    sorted_clusters = sorted(cluster_importance.keys(), key=lambda c: -cluster_importance[c])

    # Build data: models with actual scores
    df_actual = df_predictions.dropna(subset=["actual_score"]).copy()
    if len(df_actual) < 15:
        print("  WARNING: Too few models with scores, skipping.")
        return

    y = df_actual["actual_score"].values

    # Get benchmark features from imputed, matching models
    if "model_name" in df_imputed.columns:
        imp = df_imputed.set_index("model_name")
    else:
        imp = df_imputed.copy()

    common_models = df_actual["model_name"].values
    imp_sub = imp.loc[imp.index.isin(common_models)]
    imp_sub = imp_sub.loc[common_models[np.isin(common_models, imp_sub.index)]]
    y_sub = df_actual.set_index("model_name").loc[imp_sub.index, "actual_score"].values

    if len(imp_sub) < 15:
        print("  WARNING: Too few overlapping models, skipping.")
        return

    # Cluster → list of benchmark columns
    cluster_to_cols = {}
    for bench, cl in cluster_map.items():
        if bench in imp_sub.columns:
            cluster_to_cols.setdefault(cl, []).append(bench)

    # Sequentially add clusters, measure CV R² and Spearman ρ
    results = []
    used_cols = []
    for i, cl in enumerate(sorted_clusters):
        cols = cluster_to_cols.get(cl, [])
        if not cols:
            continue
        used_cols.extend(cols)
        X_subset = imp_sub[used_cols].fillna(imp_sub[used_cols].mean()).values
        if X_subset.shape[1] == 0:
            continue

        ridge = Ridge(alpha=1.0)
        cv_r2 = cross_val_score(ridge, X_subset, y_sub, cv=5, scoring="r2")
        ridge.fit(X_subset, y_sub)
        pred = ridge.predict(X_subset)
        rho, _ = spearmanr(y_sub, pred)

        results.append({
            "n_clusters": i + 1,
            "cluster_added": cl,
            "cluster_importance": cluster_importance.get(cl, 0),
            "n_features": len(used_cols),
            "cv_r2_mean": float(np.mean(cv_r2)),
            "cv_r2_std": float(np.std(cv_r2)),
            "spearman_rho": float(rho),
        })

    if not results:
        print("  WARNING: No valid cluster results, skipping.")
        return

    res_df = pd.DataFrame(results)
    res_df.to_csv(run_dir / "marginal_value_by_cluster.csv", index=False)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = res_df["n_clusters"].values

    color1 = "#2196F3"
    ax1.errorbar(x, res_df["cv_r2_mean"], yerr=res_df["cv_r2_std"],
                 fmt="o-", color=color1, capsize=4, label="5-fold CV R²")
    ax1.set_xlabel("Number of Benchmark Clusters", fontsize=12)
    ax1.set_ylabel("CV R²", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(x)

    ax2 = ax1.twinx()
    color2 = "#FF9800"
    ax2.plot(x, res_df["spearman_rho"], "s--", color=color2, label="Spearman ρ")
    ax2.set_ylabel("Spearman ρ", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=10)

    ax1.set_title("Marginal Benchmark Value: Incremental Cluster Contribution", fontsize=14)
    ax1.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(run_dir / "marginal_benchmark_value.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved marginal_benchmark_value.png ({len(results)} clusters)")


# ──────────────────────────────────────────────────────────────────
# Chart 11: Capability Archetype Map
# ──────────────────────────────────────────────────────────────────

def plot_capability_archetype_map(df_clean, df_imputed, df_combined, run_dir):
    print("\n--- Chart 11: Capability Archetype Map ---")

    fa_df, fa_cols = prepare_fa_matrix(df_clean, df_imputed)
    if len(fa_cols) < 2 or len(fa_df) < 5:
        print("  WARNING: Not enough data for PCA, skipping.")
        return

    X = fa_df[fa_cols].values
    imp = SimpleImputer(strategy="mean")
    X = imp.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_

    result_df = fa_df[["model_name"]].copy()
    result_df["PC1"] = scores[:, 0]
    result_df["PC2"] = scores[:, 1]

    # Merge source and reasoning
    df_source = build_source_map(df_combined)
    result_df = result_df.merge(df_source, on="model_name", how="left")
    result_df["openbench_Source"] = result_df["openbench_Source"].fillna("Other")

    if "openbench_Reasoning" in df_clean.columns:
        reason_map = df_clean[["model_name", "openbench_Reasoning"]].drop_duplicates("model_name")
        result_df = result_df.merge(reason_map, on="model_name", how="left")
        result_df["is_reasoning"] = result_df["openbench_Reasoning"].fillna(0).astype(bool)
    else:
        result_df["is_reasoning"] = False

    result_df.to_csv(run_dir / "archetype_pca_scores.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    sources = sorted(result_df["openbench_Source"].unique())
    s2c, s2m = source_styles(sources)

    texts = []
    for source in sources:
        mask = result_df["openbench_Source"] == source
        sub = result_df[mask]
        color = s2c.get(source, "#90A4AE")
        marker = s2m.get(source, "o")

        # Filled = reasoning, hollow = non-reasoning
        reasoning = sub[sub["is_reasoning"]]
        non_reasoning = sub[~sub["is_reasoning"]]

        if len(non_reasoning) > 0:
            ax.scatter(non_reasoning["PC1"], non_reasoning["PC2"],
                       c=color, marker=marker, s=60, edgecolors=color,
                       facecolors="none", linewidths=1.5, zorder=3)
        if len(reasoning) > 0:
            ax.scatter(reasoning["PC1"], reasoning["PC2"],
                       c=color, marker=marker, s=60, edgecolors="white",
                       linewidths=0.5, zorder=4)

        # Convex hull per vendor (if ≥ 3 models)
        if len(sub) >= 3:
            points = sub[["PC1", "PC2"]].values
            try:
                hull = ConvexHull(points)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                ax.fill(points[hull_pts, 0], points[hull_pts, 1],
                        alpha=0.08, color=color)
                ax.plot(points[hull_pts, 0], points[hull_pts, 1],
                        "-", color=color, alpha=0.3, linewidth=1)
            except Exception:
                pass

        # Labels
        for _, row in sub.iterrows():
            texts.append(ax.text(row["PC1"], row["PC2"], row["model_name"],
                                fontsize=6, alpha=0.8))

    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.4, lw=0.5))

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)", fontsize=12)
    ax.set_title("Capability Archetype Map (PCA on benchmarks)", fontsize=16)
    ax.grid(True, linestyle=":", alpha=0.3)

    # Legend
    handles = []
    for s in sources:
        handles.append(Line2D([0], [0], marker=s2m.get(s, "o"), linestyle="",
                              color=s2c.get(s, "#90A4AE"), label=s, markersize=8))
    handles.append(Line2D([0], [0], marker="o", linestyle="", color="gray",
                          markerfacecolor="none", markersize=8, label="Non-Reasoning"))
    handles.append(Line2D([0], [0], marker="o", linestyle="", color="gray",
                          markerfacecolor="gray", markersize=8, label="Reasoning"))
    ax.legend(handles=handles, frameon=False, loc="best", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(run_dir / "capability_archetype_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved capability_archetype_map.png ({len(result_df)} models, "
          f"PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%})")


# ──────────────────────────────────────────────────────────────────
# Chart 12: Token Productivity
# ──────────────────────────────────────────────────────────────────

def plot_token_productivity(df_scores, df_clean, df_predictions, run_dir):
    print("\n--- Chart 12: Token Productivity ---")

    token_col = "logic_avg_reasoning_tokens"
    if token_col not in df_clean.columns:
        print(f"  WARNING: {token_col} not found, skipping.")
        return

    # Build base data with token counts and reasoning flag
    df = df_clean[["model_name", token_col]].dropna(subset=[token_col]).copy()
    df[token_col] = pd.to_numeric(df[token_col], errors="coerce")
    df = df.dropna(subset=[token_col])
    df = df[df[token_col] > 0]

    if len(df) < 5:
        print("  WARNING: Too few models with token data, skipping.")
        return

    # Merge reasoning flag
    if "openbench_Reasoning" in df_clean.columns:
        reason_map = df_clean[["model_name", "openbench_Reasoning"]].drop_duplicates("model_name")
        df = df.merge(reason_map, on="model_name", how="left")
        df["reasoning_label"] = df["openbench_Reasoning"].map(
            {1.0: "Reasoning", 0.0: "Non-Reasoning"}
        ).fillna("Unknown")
    else:
        df["reasoning_label"] = "Unknown"

    # Merge factor1 score
    if df_scores is not None and "factor1_score" in df_scores.columns:
        df = df.merge(df_scores[["model_name", "factor1_score"]], on="model_name", how="left")
    elif df_scores is not None:
        score_col = [c for c in df_scores.columns if c != "model_name"][0] if len(df_scores.columns) > 1 else None
        if score_col:
            df = df.merge(df_scores[["model_name", score_col]].rename(columns={score_col: "factor1_score"}),
                          on="model_name", how="left")
        else:
            df["factor1_score"] = np.nan
    else:
        df["factor1_score"] = np.nan

    # Merge actual arena score
    if "actual_score" in df_predictions.columns:
        actual_map = df_predictions[["model_name", "actual_score"]].dropna(subset=["actual_score"])
        df = df.merge(actual_map, on="model_name", how="left")
    else:
        df["actual_score"] = np.nan

    df.to_csv(run_dir / "token_productivity.csv", index=False)

    has_f1 = df["factor1_score"].notna().sum() > 5
    has_arena = df["actual_score"].notna().sum() > 5
    n_panels = int(has_f1) + int(has_arena)
    if n_panels == 0:
        print("  WARNING: No valid y-axis data, skipping.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    palette = {"Reasoning": "#2196F3", "Non-Reasoning": "#FF9800", "Unknown": "#90A4AE"}
    panel_idx = 0

    def _plot_panel(ax, x_col, y_col, y_label, data):
        for label in ["Reasoning", "Non-Reasoning"]:
            sub = data[data["reasoning_label"] == label].dropna(subset=[y_col])
            if len(sub) < 2:
                continue
            color = palette.get(label, "#90A4AE")
            ax.scatter(sub[x_col], sub[y_col], c=color, label=label,
                       alpha=0.6, s=40, edgecolors="white", linewidths=0.5, zorder=3)
            # LOWESS
            if len(sub) >= 5:
                x_vals = np.log10(sub[x_col].values)
                y_vals = sub[y_col].values
                sorted_idx = np.argsort(x_vals)
                x_sorted, y_sorted = x_vals[sorted_idx], y_vals[sorted_idx]
                smooth = lowess(y_sorted, x_sorted, frac=0.6, return_sorted=True)
                ax.plot(10**smooth[:, 0], smooth[:, 1], "-", color=color, linewidth=2, alpha=0.7)

        ax.set_xscale("log")
        ax.set_xlabel("Avg Reasoning Tokens (log scale)", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.4)

    if has_f1:
        ax = axes[panel_idx]
        _plot_panel(ax, token_col, "factor1_score", "Overall Performance (Factor 1)", df)
        ax.set_title("Tokens vs Benchmark Performance", fontsize=13)
        panel_idx += 1

    if has_arena:
        ax = axes[panel_idx]
        df_arena = df.dropna(subset=["actual_score"])
        _plot_panel(ax, token_col, "actual_score", "Arena Score", df_arena)
        ax.set_title("Tokens vs Arena Score", fontsize=13)
        panel_idx += 1

    fig.suptitle("Token Productivity", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(run_dir / "token_productivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved token_productivity.png ({len(df)} models)")


# ──────────────────────────────────────────────────────────────────
# Chart 13: Capability Over Time
# ──────────────────────────────────────────────────────────────────

RELEASE_DATES_FILE = "../benchmark_combiner/benchmarks/openbench_release_dates.csv"


def plot_capability_over_time(df_scores, df_clean, df_predictions, df_combined, run_dir):
    print("\n--- Chart 13: Capability Over Time ---")

    # Load release dates
    if not os.path.exists(RELEASE_DATES_FILE):
        print(f"  WARNING: {RELEASE_DATES_FILE} not found, skipping.")
        return

    df_rd = pd.read_csv(RELEASE_DATES_FILE)
    df_rd["Release_Date"] = pd.to_datetime(df_rd["Release_Date"], errors="coerce")
    df_rd = df_rd.dropna(subset=["Release_Date"])
    df_rd = df_rd.rename(columns={"Model": "model_name"})
    df_rd["model_name"] = df_rd["model_name"].str.strip()

    # Build plot data
    df = df_rd[["model_name", "Release_Date"]].copy()

    # Merge factor1 score
    if df_scores is not None and "factor1_score" in df_scores.columns:
        df = df.merge(df_scores[["model_name", "factor1_score"]], on="model_name", how="left")
    else:
        df["factor1_score"] = np.nan

    # Merge actual arena score
    if "actual_score" in df_predictions.columns:
        actual_map = df_predictions[["model_name", "actual_score"]].dropna(subset=["actual_score"])
        df = df.merge(actual_map, on="model_name", how="left")
    else:
        df["actual_score"] = np.nan

    # Merge source
    df_source = build_source_map(df_combined)
    df = df.merge(df_source, on="model_name", how="left")
    df["openbench_Source"] = df["openbench_Source"].fillna("Unknown")

    # Merge reasoning flag
    if "openbench_Reasoning" in df_clean.columns:
        reason_map = df_clean[["model_name", "openbench_Reasoning"]].drop_duplicates("model_name")
        df = df.merge(reason_map, on="model_name", how="left")
        df["reasoning_label"] = df["openbench_Reasoning"].map(
            {1.0: "Reasoning", 0.0: "Non-Reasoning"}
        ).fillna("Unknown")
    else:
        df["reasoning_label"] = "Unknown"

    df.to_csv(run_dir / "capability_over_time.csv", index=False)

    has_f1 = df["factor1_score"].notna().sum() > 5
    has_arena = df["actual_score"].notna().sum() > 5
    n_panels = int(has_f1) + int(has_arena)
    if n_panels == 0:
        print("  WARNING: No valid y-axis data, skipping.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(14 * n_panels, 9))
    if n_panels == 1:
        axes = [axes]

    sources = sorted(df["openbench_Source"].unique().tolist())
    s2c, s2m = source_styles(sources)

    def _plot_time_panel(ax, y_col, y_label, data):
        data = data.dropna(subset=[y_col]).copy()
        if len(data) < 3:
            return

        texts = []
        for src in sources:
            sub = data[data["openbench_Source"] == src]
            if sub.empty:
                continue
            ax.scatter(sub["Release_Date"], sub[y_col],
                       marker=s2m[src], c=s2c[src], label=src,
                       alpha=0.7, s=60, edgecolors="white", linewidths=0.5, zorder=3)

        # LOWESS trendline (all data)
        x_num = (data["Release_Date"] - data["Release_Date"].min()).dt.days.values.astype(float)
        y_vals = data[y_col].values.astype(float)
        if len(data) >= 8:
            sorted_idx = np.argsort(x_num)
            smooth = lowess(y_vals[sorted_idx], x_num[sorted_idx], frac=0.4, return_sorted=True)
            smooth_dates = data["Release_Date"].min() + pd.to_timedelta(smooth[:, 0], unit="D")
            ax.plot(smooth_dates, smooth[:, 1], "-", color="#333333",
                    linewidth=2.5, alpha=0.7, zorder=2, label="LOWESS trend")

        # Label top-5 and bottom-3 by y-value for context
        top = data.nlargest(5, y_col)
        bottom = data.nsmallest(3, y_col)
        for _, r in pd.concat([top, bottom]).drop_duplicates("model_name").iterrows():
            ann = ax.annotate(
                r["model_name"], xy=(r["Release_Date"], r[y_col]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=7, alpha=0.7,
                bbox={**LABEL_BBOX, "pad": 0.15},
                arrowprops=dict(arrowstyle="-", lw=0.4, color="gray", alpha=0.4),
            )
            texts.append(ann)

        if texts:
            adjust_text(texts, only_move={"text": "xy"}, expand_text=(1.2, 1.2),
                        force_text=0.5, force_static=0.3)

        ax.set_xlabel("Release Date", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.tick_params(axis="x", rotation=30)
        handles = [Line2D([0], [0], marker=s2m[s], linestyle="", color=s2c[s],
                          label=s, markersize=8) for s in sources if s in data["openbench_Source"].values]
        handles.append(Line2D([0], [0], color="#333333", linewidth=2.5, alpha=0.7, label="LOWESS trend"))
        ax.legend(handles=handles, fontsize=8, ncol=2, loc="lower right", frameon=False)
        ax.grid(True, linestyle=":", alpha=0.4)

    panel_idx = 0
    if has_f1:
        ax = axes[panel_idx]
        _plot_time_panel(ax, "factor1_score", "Overall Performance (Factor 1)", df)
        ax.set_title("Benchmark Performance Over Time", fontsize=14)
        panel_idx += 1

    if has_arena:
        ax = axes[panel_idx]
        _plot_time_panel(ax, "actual_score", "Arena ELO Score", df)
        ax.set_title("Arena ELO Over Time", fontsize=14)
        panel_idx += 1

    fig.suptitle("Model Capability Over Time (by Release Date)", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(run_dir / "capability_over_time.png", dpi=300, bbox_inches="tight")
    fig.savefig(run_dir / "capability_over_time.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved capability_over_time.png + .svg ({df['factor1_score'].notna().sum()} benchmark, "
          f"{df['actual_score'].notna().sum()} arena)")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
# Chart 14: Capability Profiles (absolute z-scored category averages)
# ──────────────────────────────────────────────────────────────────

# Hand-picked benchmark categories for interpretable capability profiles.
# Each benchmark appears in exactly one category. Z-scores are averaged
# within category, so categories with fewer benchmarks are noisier but
# not over-counted.
CAPABILITY_CATEGORIES = {
    "Reasoning": [
        "livebench_zebra_puzzle", "livebench_consecutive_events", "livebench_integrals_with_game",
        "livebench_connections", "livebench_logic_with_navigation", "livebench_spatial",
        "livebench_theory_of_mind",
        "aa_eval_hle", "arc_ARC-AGI-1", "arc_ARC-AGI-2",
        "logic_trick_acc", "logic_physics_acc",
        "simplebench_Score (AVG@5)",
    ],
    "STEM\nKnowledge": [
        "aa_eval_aime_25", "aa_eval_gpqa", "aa_eval_mmlu_pro",
        "livebench_olympiad",
        "aaomniscience_OmniscienceAccuracy",
    ],
    "Coding": [
        "aa_eval_livecodebench", "aa_eval_scicode", "aa_eval_terminalbench_hard",
        "livebench_code_generation", "livebench_code_completion",
        "livebench_javascript", "livebench_typescript", "livebench_python",
        "aiderbench_Percent correct",
    ],
    "Creative\nWriting": [
        "eqbench_creative_elo", "ugileaderboard_Writing",
        "writing_Grok 4 Fast TrueSkill",
        "yupp_Text_Score", "livebench_typos",
    ],
    "Instruction\nFollowing": [
        "aa_eval_ifbench", "aa_eval_tau2", "livebench_IF Average",
    ],
    "Emotional\nIntelligence": [
        "eqbench_eq_elo",
        "eq_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill",
        "eq_Grok 4 Fast TrueSkill",
    ],
}


def compute_capability_profiles(df_imputed: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model capability scores via PCA PC1 within each category.

    For each category, z-scores the constituent benchmarks, runs PCA, and
    uses the first principal component as the category score. This lets
    benchmarks contribute proportionally to their shared variance rather
    than equally (which over-counts redundant benchmarks).
    """
    print("  Computing capability profiles (PCA within category)...")
    df = df_imputed.copy()

    result = pd.DataFrame({"model_name": df["model_name"]})

    for cat, cols in CAPABILITY_CATEGORIES.items():
        available = [c for c in cols if c in df.columns]
        if len(available) < 2:
            # Need at least 2 columns for PCA; fall back to z-score
            if available:
                vals = pd.to_numeric(df[available[0]], errors="coerce")
                mu, std = vals.mean(), vals.std()
                result[cat] = (vals - mu) / std if std > 0 else vals * 0
            else:
                result[cat] = np.nan
            continue

        # Z-score each benchmark
        X = df[available].apply(lambda c: pd.to_numeric(c, errors="coerce"))
        X_z = (X - X.mean()) / X.std()
        X_z = X_z.fillna(0)  # fill remaining NaNs for PCA

        # PCA: extract PC1
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X_z).ravel()

        # Flip sign so positive = better (check correlation with column means)
        avg_z = X_z.mean(axis=1)
        if np.corrcoef(pc1, avg_z)[0, 1] < 0:
            pc1 = -pc1

        result[cat] = pc1
        var_expl = pca.explained_variance_ratio_[0]
        print(f"    {cat.replace(chr(10), ' '):25s} {len(available):2d} cols, PC1 explains {var_expl:.1%}")

    # Scale each category to 0-100 across all models
    cats = list(CAPABILITY_CATEGORIES.keys())
    for cat in cats:
        if cat in result.columns:
            vals = result[cat]
            lo, hi = vals.min(), vals.max()
            if hi - lo > 1e-12:
                result[cat] = (vals - lo) / (hi - lo) * 100
            else:
                result[cat] = 50.0

    print(f"  {len(cats)} categories, {len(result)} models (scaled 0-100)")
    return result


# Colors for the 3 major labs + best-of-rest
HIGHLIGHT_COLORS_BY_GROUP = {
    "OpenAI": "#9C27B0",      # purple
    "Google": "#E53935",      # red
    "Anthropic": "#F5A623",   # orange/gold
    "Other": "#2196F3",       # blue
}


def _select_highlight_models(cap_df: pd.DataFrame) -> list:
    """Auto-select: top model from each major lab + top model outside them."""
    cats = [c for c in cap_df.columns if c != "model_name"]
    df = cap_df.copy()
    df["_overall"] = df[cats].mean(axis=1)

    def _brand(name):
        n = name.lower()
        if "claude" in n:
            return "Anthropic"
        if any(k in n for k in ("gpt", "chatgpt", "o3 ", "o3-", "o4 ", "o4-")):
            return "OpenAI"
        if "gemini" in n or "gemma" in n:
            return "Google"
        return "Other"

    df["_brand"] = df["model_name"].apply(_brand)
    picks = []
    for group in ["OpenAI", "Google", "Anthropic", "Other"]:
        grp = df[df["_brand"] == group]
        if not grp.empty:
            picks.append(grp.nlargest(1, "_overall").iloc[0]["model_name"])
    return picks


def plot_capability_profiles(cap_df: pd.DataFrame, df_predictions: pd.DataFrame,
                             df_combined: pd.DataFrame, run_dir: Path,
                             highlight: list = None):
    """Radar + bar chart of capability profiles for highlighted models.

    By default auto-selects: top model from OpenAI, Google, Anthropic,
    and the top model outside those three (by mean PC1 across categories).
    """
    print("  Plotting capability profiles...")

    cats = [c for c in cap_df.columns if c != "model_name"]
    highlight = highlight or _select_highlight_models(cap_df)
    print(f"    Highlighted: {highlight}")

    # Select highlighted models
    selected = cap_df[cap_df["model_name"].isin(highlight)].copy()
    missing = set(highlight) - set(selected["model_name"])
    if missing:
        print(f"  WARNING: models not found: {missing}")

    # Preserve requested order
    found_names = selected["model_name"].tolist()
    ordered = [m for m in highlight if m in found_names]
    selected = selected.set_index("model_name").loc[ordered].reset_index()

    n_models = len(selected)
    if n_models == 0:
        print("  WARNING: no highlight models found, skipping capability profiles plot")
        return

    # Assign colors by lab group
    def _brand(name):
        n = name.lower()
        if "claude" in n:
            return "Anthropic"
        if any(k in n for k in ("gpt", "chatgpt", "o3 ", "o3-", "o4 ", "o4-")):
            return "OpenAI"
        if "gemini" in n or "gemma" in n:
            return "Google"
        return "Other"

    colors_dict = {}
    for _, row in selected.iterrows():
        name = row["model_name"]
        colors_dict[name] = HIGHLIGHT_COLORS_BY_GROUP.get(_brand(name), "#888888")

    # Short display names
    def _short(name):
        return (name.replace(" Preview", "").replace(" Thinking", " T")
                .replace("(2025-", "(").replace("(2026-", "("))

    # --- Radar chart ---
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    vals_dict = {}
    for _, row in selected.iterrows():
        name = row["model_name"]
        vals_dict[name] = row[cats].values.astype(float)

    _radar_chart(ax, cats, vals_dict, colors_dict,
                 title="Capability Profile Comparison")
    fig.tight_layout()
    fig.savefig(run_dir / "capability_profiles_radar.png", dpi=300)
    plt.close(fig)

    # --- Bar chart ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cats))
    width = 0.8 / n_models
    for i, (_, row) in enumerate(selected.iterrows()):
        name = row["model_name"]
        vals = row[cats].values.astype(float)
        ax2.bar(x + i * width - 0.4 + width / 2, vals, width,
                label=_short(name), color=colors_dict[name],
                alpha=0.9, edgecolor="white", linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, fontsize=11)
    ax2.set_ylabel("PC1 score (higher = better)", fontsize=11)
    ax2.set_title("Capability Profile Comparison", fontsize=14)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(run_dir / "capability_profiles_bars.png", dpi=300)
    plt.close(fig2)

    print(f"  Saved capability_profiles_radar.png and capability_profiles_bars.png ({n_models} models)")


# ──────────────────────────────────────────────────────────────────
# Chart 15: Predicted vs Actual (Top 50)
# ──────────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(df_predictions, df_oof, df_combined, run_dir, top_n=50):
    """Chart 15: Scatter of predicted vs actual Arena ELO for the top N models."""
    print(f"\n--- Chart 15: Predicted vs Actual (Top {top_n}) ---")

    # Prefer OOF predictions (honest); fall back to full-fit
    if df_oof is not None and not df_oof.empty and len(df_oof) >= 10:
        df = df_oof.dropna(subset=["actual_score", "oof_predicted_score"]).copy()
        df = df.rename(columns={"oof_predicted_score": "predicted"})
        pred_label = "Out-of-Fold Predicted"
    else:
        df = df_predictions.dropna(subset=["actual_score"]).copy()
        df = df.rename(columns={"predicted_score": "predicted"})
        pred_label = "Predicted"

    # Top N by actual score
    df = df.nlargest(top_n, "actual_score")

    # Add vendor
    df_source = build_source_map(df_combined)
    df = df.merge(df_source, on="model_name", how="left")
    df["openbench_Source"] = df["openbench_Source"].fillna("Other")

    sources = sorted(df["openbench_Source"].unique().tolist())
    s2c, s2m = source_styles(sources)

    # Stats
    residuals = df["actual_score"] - df["predicted"]
    rmse = np.sqrt((residuals ** 2).mean())
    ss_res = (residuals ** 2).sum()
    ss_tot = ((df["actual_score"] - df["actual_score"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    corr = np.corrcoef(df["actual_score"], df["predicted"])[0, 1]

    df.to_csv(run_dir / f"predicted_vs_actual_top{top_n}.csv", index=False)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(10, 10))

    # Perfect-prediction line
    lo = min(df["actual_score"].min(), df["predicted"].min()) - 15
    hi = max(df["actual_score"].max(), df["predicted"].max()) + 15
    ax.plot([lo, hi], [lo, hi], ls="--", color="#888888", lw=1.5, zorder=1, label="Perfect prediction")

    # +/- RMSE band around the diagonal
    ax.fill_between([lo, hi], [lo - rmse, hi - rmse], [lo + rmse, hi + rmse],
                    color="#888888", alpha=0.08, zorder=0, label=f"\u00b1 1 RMSE ({rmse:.1f})")

    # Scatter by vendor
    texts = []
    for src in sources:
        mask = df["openbench_Source"] == src
        sub = df[mask]
        ax.scatter(sub["actual_score"], sub["predicted"],
                   marker=s2m[src], color=s2c[src], s=70, edgecolors="white",
                   linewidths=0.5, zorder=3, label=src)
        for _, row in sub.iterrows():
            # Shorten name for legibility
            name = row["model_name"]
            texts.append(ax.text(row["actual_score"], row["predicted"], name,
                                 fontsize=5.5, alpha=0.85, zorder=4))

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4))

    # Stats box
    stats_text = (f"Top {len(df)} models\n"
                  f"R\u00b2 = {r2:.3f}\n"
                  f"r  = {corr:.3f}\n"
                  f"RMSE = {rmse:.1f}")
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9))

    ax.set_xlabel("Actual Arena ELO", fontsize=13)
    ax.set_ylabel(f"{pred_label} Arena ELO", fontsize=13)
    ax.set_title(f"Predicted vs Actual Arena ELO — Top {len(df)} Models", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fname = f"predicted_vs_actual_top{top_n}"
    fig.savefig(run_dir / f"{fname}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}.png ({len(df)} models, RMSE={rmse:.1f}, R\u00b2={r2:.3f})")


# ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Post-Hoc Analysis Suite")
    print("=" * 60)

    df_clean, df_combined, df_imputed, df_predictions, df_oof, df_feature_ranking = load_extended_data()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutputs → {run_dir}\n")

    # Chart 1
    df_scores, fa_cols, X_scaled = plot_cost_vs_performance(df_clean, df_combined, df_imputed, run_dir)

    # Chart 2
    plot_arena_residuals(df_predictions, df_combined, run_dir)

    # Chart 2b
    plot_arena_residuals_oof(df_oof, df_combined, run_dir)

    # Chart 3
    scores_df, loadings_df, factor_names = extract_factors(df_clean, df_imputed, n_factors=4)
    loadings_df.to_csv(run_dir / "multi_factor_loadings.csv")
    scores_df.to_csv(run_dir / "multi_factor_scores.csv", index=False)
    plot_radar_top_models(scores_df, factor_names, df_combined, run_dir, df_f1_scores=df_scores)
    plot_radar_by_vendor(scores_df, factor_names, df_combined, run_dir)

    # Chart 4
    plot_benchmark_dendrogram(df_imputed, run_dir)

    # Chart 5
    plot_reasoning_comparison(scores_df, factor_names, df_clean, run_dir)

    # Chart 6
    plot_prediction_calibration(df_predictions, df_oof, run_dir)

    # Chart 7
    plot_residual_bias(df_predictions, df_combined, df_clean, run_dir, df_oof=df_oof)

    # Chart 8
    plot_rank_stability(df_predictions, df_combined, run_dir)

    # Chart 9
    plot_benchmark_redundancy(df_imputed, run_dir)

    # Chart 10
    plot_marginal_benchmark_value(df_imputed, df_predictions, df_feature_ranking, run_dir)

    # Chart 11
    plot_capability_archetype_map(df_clean, df_imputed, df_combined, run_dir)

    # Chart 12
    plot_token_productivity(df_scores, df_clean, df_predictions, run_dir)

    # Chart 13
    plot_capability_over_time(df_scores, df_clean, df_predictions, df_combined, run_dir)

    # Chart 14
    cap_df = compute_capability_profiles(df_imputed)
    cap_df.to_csv(run_dir / "capability_profiles.csv", index=False)
    plot_capability_profiles(cap_df, df_predictions, df_combined, run_dir)

    # Chart 15
    plot_predicted_vs_actual(df_predictions, df_oof, df_combined, run_dir)

    # Summary
    outputs = sorted(run_dir.glob("*"))
    print("\n" + "=" * 60)
    print(f"Done! {len(outputs)} files in {run_dir}")
    for f in outputs:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s} {size_kb:8.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
