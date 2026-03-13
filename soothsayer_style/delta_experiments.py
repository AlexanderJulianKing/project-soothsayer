"""Comprehensive experiments to improve style_predicted_delta OOF quality.

Reads responses.csv, judge_results.csv, combined benchmarks.
Tries many feature engineering and modeling approaches.
Prints a ranked leaderboard at the end.
"""

import glob
import os
import re
import sys
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    Ridge,
    SGDRegressor,
)
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)
from sklearn.svm import SVR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KF = KFold(n_splits=5, shuffle=True, random_state=42)
RESULTS: List[Dict[str, Any]] = []


def oof_metrics(y, oof):
    r = np.corrcoef(y, oof)[0, 1] if len(y) > 2 else float("nan")
    rmse = float(np.sqrt(np.mean((y - oof) ** 2)))
    return r, rmse


def record(name, y, oof, n_feats=None):
    r, rmse = oof_metrics(y, oof)
    RESULTS.append({"name": name, "r": r, "rmse": rmse, "n_feats": n_feats})
    print(f"  {name:60s} r={r:.3f}  RMSE={rmse:.2f}  (n_feats={n_feats})")


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_data():
    style_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "outputs", "style_2*.csv")))
    style = pd.read_csv(style_files[-1])

    feat_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "outputs", "style_features_v4_*.csv")))
    feat_df = pd.read_csv(feat_files[-1])

    tone_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "outputs", "tone_*.csv")))
    tone = pd.read_csv(tone_files[-1]) if tone_files else pd.DataFrame()

    resp = pd.read_csv(os.path.join(SCRIPT_DIR, "responses.csv"))
    ok = resp[resp.status == "ok"].copy()

    # Load benchmark scores for logic features
    bench_patterns = [
        os.path.join(SCRIPT_DIR, "..", "benchmark_combiner", "benchmarks", "combined_all_benches.csv"),
    ]
    bench_df = pd.DataFrame()
    for p in bench_patterns:
        if os.path.exists(p):
            bench_df = pd.read_csv(p)
            break

    return style, feat_df, tone, ok, bench_df


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
STYLE_METRICS = ["length", "header_count", "bold_count", "list_count"]


def extra_text_features(text):
    text = str(text) if text else ""
    return {
        "code_blocks": len(re.findall(r"```", text)) // 2,
        "inline_code": len(re.findall(r"`[^`]+`", text)),
        "blockquotes": len(re.findall(r"^\s*>", text, re.MULTILINE)),
        "paragraphs": len(re.split(r"\n\s*\n", text.strip())),
        "sentences": len(re.split(r"[.!?]+", text)),
        "emoji_count": len(
            re.findall(
                r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
                r"\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF"
                r"\U00002702-\U000027B0]",
                text,
            )
        ),
        "table_rows": len(re.findall(r"^\|.*\|\s*$", text, re.MULTILINE)),
        "exclamation_marks": text.count("!"),
        "question_marks": text.count("?"),
        "newlines": text.count("\n"),
        "words": len(text.split()),
        "avg_word_length": np.mean([len(w) for w in text.split()]) if text.split() else 0,
        "unique_words_ratio": len(set(text.lower().split())) / max(len(text.split()), 1),
        "max_line_length": max((len(line) for line in text.split("\n")), default=0),
        "short_lines_pct": sum(1 for line in text.split("\n") if 0 < len(line.strip()) < 40)
        / max(sum(1 for line in text.split("\n") if line.strip()), 1),
    }


def build_all_features(ok: pd.DataFrame, feat_df: pd.DataFrame, tone: pd.DataFrame, bench_df: pd.DataFrame, style: pd.DataFrame):
    """Build the full feature matrix with all engineered features."""
    new_metrics = list(extra_text_features("").keys())

    # Compute original style metrics + extra features for each response
    from score import calculate_markdown_stats
    orig_stats = ok["response"].apply(lambda x: pd.Series(calculate_markdown_stats(str(x) if x else "")))
    extras = ok["response"].apply(lambda x: pd.Series(extra_text_features(x)))
    ok_ext = pd.concat([ok, orig_stats, extras], axis=1)

    # --- Per-question new features (pivot to wide) ---
    all_metrics = STYLE_METRICS + new_metrics
    per_mq = ok_ext.groupby(["model_name", "question_id"])[all_metrics].mean().reset_index()
    q_ids = sorted(per_mq.question_id.unique())

    rows = []
    for model in per_mq.model_name.unique():
        md = per_mq[per_mq.model_name == model]
        row = {"model": model}
        for q in q_ids:
            qd = md[md.question_id == q]
            for m in all_metrics:
                row[f"Q{q}_{m}"] = qd[m].values[0] if len(qd) > 0 else 0
        rows.append(row)
    per_q_new = pd.DataFrame(rows)

    # --- Model-level summary stats ---
    per_model_avg = ok_ext.groupby("model_name")[all_metrics].mean().reset_index()
    per_model_avg.columns = ["model"] + [f"avg_{m}" for m in all_metrics]

    per_model_std = ok_ext.groupby("model_name")[all_metrics].std().reset_index()
    per_model_std.columns = ["model"] + [f"std_{m}" for m in all_metrics]

    per_model_max = ok_ext.groupby("model_name")[all_metrics].max().reset_index()
    per_model_max.columns = ["model"] + [f"max_{m}" for m in all_metrics]

    per_model_min = ok_ext.groupby("model_name")[all_metrics].min().reset_index()
    per_model_min.columns = ["model"] + [f"min_{m}" for m in all_metrics]

    # --- Engineered ratios ---
    ratios = per_model_avg[["model"]].copy()
    avg = per_model_avg
    ratios["bold_per_1k_chars"] = avg["avg_bold_count"] / (avg["avg_length"] + 1) * 1000
    ratios["headers_per_1k_chars"] = avg["avg_header_count"] / (avg["avg_length"] + 1) * 1000
    ratios["lists_per_1k_chars"] = avg["avg_list_count"] / (avg["avg_length"] + 1) * 1000
    ratios["code_per_1k_chars"] = avg["avg_code_blocks"] / (avg["avg_length"] + 1) * 1000
    ratios["inline_code_per_1k_chars"] = avg["avg_inline_code"] / (avg["avg_length"] + 1) * 1000
    ratios["emoji_per_1k_chars"] = avg["avg_emoji_count"] / (avg["avg_length"] + 1) * 1000
    ratios["formatting_density"] = (
        avg["avg_bold_count"] + avg["avg_header_count"] + avg["avg_list_count"] + avg["avg_code_blocks"] + avg["avg_table_rows"]
    ) / (avg["avg_length"] + 1) * 1000
    ratios["sentences_per_paragraph"] = avg["avg_sentences"] / (avg["avg_paragraphs"] + 1)
    ratios["words_per_sentence"] = avg["avg_words"] / (avg["avg_sentences"] + 1)
    ratios["chars_per_word"] = avg["avg_length"] / (avg["avg_words"] + 1)
    ratios["log_avg_length"] = np.log1p(avg["avg_length"])
    ratios["log_avg_sentences"] = np.log1p(avg["avg_sentences"])
    ratios["log_avg_words"] = np.log1p(avg["avg_words"])

    # --- Cross-question variance features ---
    variance = pd.DataFrame({"model": per_model_avg["model"]})
    for m in all_metrics:
        cols = [f"Q{q}_{m}" for q in q_ids]
        existing = [c for c in cols if c in per_q_new.columns]
        if existing:
            sub = per_q_new[["model"] + existing]
            variance[f"cv_{m}"] = sub[existing].std(axis=1) / (sub[existing].mean(axis=1).abs() + 1)
            variance[f"range_{m}"] = sub[existing].max(axis=1) - sub[existing].min(axis=1)

    # --- Merge everything ---
    result = per_q_new.copy()
    for df in [per_model_avg, per_model_std, per_model_max, per_model_min, ratios, variance]:
        result = result.merge(df, on="model", how="left")

    # Old features from feat_df
    old_only_cols = [c for c in feat_df.columns if c not in result.columns and c not in ["model", "code"]]
    if old_only_cols:
        result = result.merge(feat_df[["model"] + old_only_cols], on="model", how="left")

    # Tone features
    if not tone.empty and "judged_model" in tone.columns:
        tone_feats = [c for c in tone.columns if "TrueSkill" in c and "Sigma" not in c]
        if tone_feats:
            result = result.merge(tone[["judged_model"] + tone_feats], left_on="model", right_on="judged_model", how="left")
            if "judged_model" in result.columns:
                result.drop(columns=["judged_model"], inplace=True)

    # Logic / benchmark features
    if not bench_df.empty and "openbench_Model" in bench_df.columns:
        logic_cols = [c for c in ["logic_weighted_accuracy", "logic_PC1"] if c in bench_df.columns]
        if logic_cols:
            result = result.merge(
                bench_df[["openbench_Model"] + logic_cols].drop_duplicates(subset=["openbench_Model"]),
                left_on="model",
                right_on="openbench_Model",
                how="left",
            )
            if "openbench_Model" in result.columns:
                result.drop(columns=["openbench_Model"], inplace=True)

    # Merge target
    both = style.dropna(subset=["delta_score"])[["model", "delta_score"]]
    result = result.merge(both, on="model", how="inner")

    return result


def get_feature_cols(df):
    exclude = {"model", "code", "delta_score", "judged_model", "openbench_Model"}
    return [c for c in df.columns if c not in exclude]


# ==============================================================================
# FEATURE SUBSETS
# ==============================================================================
def make_feature_subsets(df, feature_cols):
    """Create various feature subsets for ablation."""
    subsets = {}

    # All features
    subsets["all"] = feature_cols

    # Old features only (from score.py)
    subsets["old_only"] = [c for c in feature_cols if not any(
        c.startswith(p) for p in ["avg_", "std_", "max_", "min_", "cv_", "range_"]
    ) and not any(
        m in c for m in ["code_blocks", "inline_code", "blockquotes", "paragraphs",
                         "sentences", "emoji_count", "table_rows", "exclamation_marks",
                         "question_marks", "newlines", "words", "avg_word_length",
                         "unique_words_ratio", "max_line_length", "short_lines_pct"]
    ) and not any(
        c.startswith(p) for p in ["bold_per_", "headers_per_", "lists_per_", "code_per_",
                                   "inline_code_per_", "emoji_per_", "formatting_density",
                                   "sentences_per_", "words_per_", "chars_per_", "log_avg_"]
    ) and "TrueSkill" not in c and "logic_" not in c]

    # Summary only (no per-question)
    subsets["summary_only"] = [c for c in feature_cols if not c.startswith("Q")]

    # Per-question only
    subsets["per_q_only"] = [c for c in feature_cols if c.startswith("Q")]

    # Old + new model-level summaries (no per-Q new)
    subsets["old_plus_summaries"] = [c for c in feature_cols if c.startswith("Q") and not any(
        m in c for m in ["code_blocks", "inline_code", "blockquotes", "paragraphs",
                         "sentences", "emoji_count", "table_rows", "exclamation_marks",
                         "question_marks", "newlines", "words", "avg_word_length",
                         "unique_words_ratio", "max_line_length", "short_lines_pct"]
    )] + [c for c in feature_cols if not c.startswith("Q")]

    # New features only
    new_metrics = ["code_blocks", "inline_code", "blockquotes", "paragraphs", "sentences",
                   "emoji_count", "table_rows", "exclamation_marks", "question_marks",
                   "newlines", "words", "avg_word_length", "unique_words_ratio",
                   "max_line_length", "short_lines_pct"]
    subsets["new_only"] = [c for c in feature_cols if any(m in c for m in new_metrics)
                           or any(c.startswith(p) for p in ["bold_per_", "headers_per_",
                                                             "lists_per_", "code_per_",
                                                             "inline_code_per_", "emoji_per_",
                                                             "formatting_density",
                                                             "sentences_per_", "words_per_",
                                                             "chars_per_", "log_avg_"])]

    return subsets


# ==============================================================================
# EXPERIMENTS
# ==============================================================================
def run_experiments():
    print("Loading data...")
    style, feat_df, tone, ok, bench_df = load_data()
    print("Building features...")
    df = build_all_features(ok, feat_df, tone, bench_df, style)

    feature_cols = get_feature_cols(df)
    y = df["delta_score"].values
    n = len(y)
    print(f"\nDataset: {n} samples, {len(feature_cols)} features, delta std={y.std():.2f}")
    print(f"{'='*80}")

    subsets = make_feature_subsets(df, feature_cols)
    for name, cols in subsets.items():
        print(f"  Subset '{name}': {len(cols)} features")

    X_all = df[feature_cols].fillna(0).values
    X_old = df[subsets["old_only"]].fillna(0).values
    X_summary = df[subsets["summary_only"]].fillna(0).values
    X_per_q = df[subsets["per_q_only"]].fillna(0).values
    X_new = df[subsets["new_only"]].fillna(0).values
    X_old_plus_sum = df[subsets["old_plus_summaries"]].fillna(0).values

    # ==== SECTION 1: BASELINES ====
    print(f"\n{'='*80}")
    print("SECTION 1: BASELINES")
    print(f"{'='*80}")

    record("Dummy (mean)", y, cross_val_predict(DummyRegressor(), X_all, y, cv=KF), 0)

    # ==== SECTION 2: FEATURE SUBSET ABLATION (ExtraTrees) ====
    print(f"\n{'='*80}")
    print("SECTION 2: FEATURE SUBSETS (ExtraTrees-500)")
    print(f"{'='*80}")

    for subset_name, cols in subsets.items():
        X = df[cols].fillna(0).values
        et = ExtraTreesRegressor(n_estimators=500, random_state=42)
        oof = cross_val_predict(et, X, y, cv=KF)
        record(f"ET500 [{subset_name}]", y, oof, len(cols))

    # ==== SECTION 3: MODEL ZOO ON ALL FEATURES ====
    print(f"\n{'='*80}")
    print("SECTION 3: MODEL ZOO (all features)")
    print(f"{'='*80}")

    models = {
        "ExtraTrees-500": ExtraTreesRegressor(n_estimators=500, random_state=42),
        "ExtraTrees-1000": ExtraTreesRegressor(n_estimators=1000, random_state=42),
        "ExtraTrees-500-d15": ExtraTreesRegressor(n_estimators=500, max_depth=15, random_state=42),
        "ExtraTrees-500-d10": ExtraTreesRegressor(n_estimators=500, max_depth=10, random_state=42),
        "ExtraTrees-500-d20": ExtraTreesRegressor(n_estimators=500, max_depth=20, random_state=42),
        "ExtraTrees-500-leaf3": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, random_state=42),
        "ExtraTrees-500-leaf5": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=5, random_state=42),
        "ExtraTrees-500-leaf10": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=10, random_state=42),
        "RandomForest-500": RandomForestRegressor(n_estimators=500, random_state=42),
        "RandomForest-500-d15": RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42),
        "GradientBoosting-200": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
        "GradientBoosting-500": GradientBoostingRegressor(n_estimators=500, max_depth=3, learning_rate=0.02, random_state=42),
        "GradientBoosting-200-d5": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
        "HistGB-default": HistGradientBoostingRegressor(random_state=42),
        "HistGB-200-d5": HistGradientBoostingRegressor(max_iter=200, max_depth=5, learning_rate=0.05, random_state=42),
        "HistGB-500-d3": HistGradientBoostingRegressor(max_iter=500, max_depth=3, learning_rate=0.02, random_state=42),
        "AdaBoost-200": AdaBoostRegressor(n_estimators=200, random_state=42),
        "AdaBoost-400": AdaBoostRegressor(n_estimators=400, random_state=42),
        "Ridge-1": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Ridge-10": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "Ridge-100": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=100.0))]),
        "BayesianRidge": Pipeline([("scaler", StandardScaler()), ("model", BayesianRidge())]),
        "ARDRegression": Pipeline([("scaler", StandardScaler()), ("model", ARDRegression(max_iter=500))]),
        "Lasso-0.1": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.1, max_iter=10000))]),
        "Lasso-1": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=1.0, max_iter=10000))]),
        "ElasticNet-0.01": Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000))]),
        "Huber-1.35": Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor(epsilon=1.35, max_iter=500))]),
        "Huber-1.0": Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor(epsilon=1.0, max_iter=500))]),
        "SVR-rbf-10": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=10.0))]),
        "SVR-rbf-100": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=100.0))]),
        "SVR-linear": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="linear", C=1.0))]),
        "KNN-5": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5))]),
        "KNN-10": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=10))]),
        "KNN-20": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=20))]),
        "MLP-64-32": Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=5000, random_state=42))]),
        "MLP-128-64": Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=5000, random_state=42))]),
        "MLP-256-128-64": Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=5000, random_state=42))]),
        "Bagging-ET": BaggingRegressor(
            estimator=ExtraTreesRegressor(n_estimators=100, random_state=42),
            n_estimators=10, random_state=42,
        ),
    }

    for name, model in models.items():
        try:
            oof = cross_val_predict(model, X_all, y, cv=KF)
            record(f"[all] {name}", y, oof, len(feature_cols))
        except Exception as e:
            print(f"  FAILED: {name}: {e}")

    # ==== SECTION 4: MODEL ZOO ON OLD FEATURES ====
    print(f"\n{'='*80}")
    print("SECTION 4: MODEL ZOO (old features only)")
    print(f"{'='*80}")

    key_models = {
        "ExtraTrees-500": ExtraTreesRegressor(n_estimators=500, random_state=42),
        "ExtraTrees-500-leaf3": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, random_state=42),
        "GradientBoosting-200": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
        "HistGB-default": HistGradientBoostingRegressor(random_state=42),
        "Ridge-10": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "BayesianRidge": Pipeline([("scaler", StandardScaler()), ("model", BayesianRidge())]),
    }

    for name, model in key_models.items():
        try:
            oof = cross_val_predict(model, X_old, y, cv=KF)
            record(f"[old] {name}", y, oof, len(subsets["old_only"]))
        except Exception as e:
            print(f"  FAILED: {name}: {e}")

    # ==== SECTION 5: PCA DIMENSIONALITY REDUCTION ====
    print(f"\n{'='*80}")
    print("SECTION 5: PCA DIMENSIONALITY REDUCTION")
    print(f"{'='*80}")

    for n_components in [5, 10, 15, 20, 30, 50]:
        if n_components > min(n, len(feature_cols)):
            continue
        pca_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("model", ExtraTreesRegressor(n_estimators=500, random_state=42)),
        ])
        oof = cross_val_predict(pca_pipe, X_all, y, cv=KF)
        record(f"PCA({n_components})->ET500", y, oof, n_components)

    for n_components in [10, 20, 30]:
        if n_components > min(n, len(feature_cols)):
            continue
        pca_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("model", Ridge(alpha=10.0)),
        ])
        oof = cross_val_predict(pca_pipe, X_all, y, cv=KF)
        record(f"PCA({n_components})->Ridge10", y, oof, n_components)

    for n_components in [10, 20, 30]:
        if n_components > min(n, len(feature_cols)):
            continue
        pca_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("model", BayesianRidge()),
        ])
        oof = cross_val_predict(pca_pipe, X_all, y, cv=KF)
        record(f"PCA({n_components})->BayesianRidge", y, oof, n_components)

    # ==== SECTION 6: FEATURE SELECTION ====
    print(f"\n{'='*80}")
    print("SECTION 6: FEATURE SELECTION")
    print(f"{'='*80}")

    for k in [10, 20, 30, 50]:
        if k > len(feature_cols):
            continue
        sel_pipe = Pipeline([
            ("selector", SelectKBest(mutual_info_regression, k=k)),
            ("model", ExtraTreesRegressor(n_estimators=500, random_state=42)),
        ])
        oof = cross_val_predict(sel_pipe, X_all, y, cv=KF)
        record(f"MI-Select(k={k})->ET500", y, oof, k)

    # Importance-based selection (pre-compute)
    et_imp = ExtraTreesRegressor(n_estimators=500, random_state=42)
    et_imp.fit(X_all, y)
    imp_order = np.argsort(et_imp.feature_importances_)[::-1]

    for k in [10, 20, 30, 50, 80]:
        if k > len(feature_cols):
            continue
        top_idx = imp_order[:k]
        X_top = X_all[:, top_idx]
        oof = cross_val_predict(ExtraTreesRegressor(n_estimators=500, random_state=42), X_top, y, cv=KF)
        record(f"ImpTop{k}->ET500", y, oof, k)

    # ==== SECTION 7: ENSEMBLES & STACKING ====
    print(f"\n{'='*80}")
    print("SECTION 7: ENSEMBLES & STACKING")
    print(f"{'='*80}")

    # Voting ensembles
    voting_configs = {
        "Vote(ET+GB)": VotingRegressor([
            ("et", ExtraTreesRegressor(n_estimators=500, random_state=42)),
            ("gb", GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
        ]),
        "Vote(ET+RF)": VotingRegressor([
            ("et", ExtraTreesRegressor(n_estimators=500, random_state=42)),
            ("rf", RandomForestRegressor(n_estimators=500, random_state=42)),
        ]),
        "Vote(ET+GB+RF)": VotingRegressor([
            ("et", ExtraTreesRegressor(n_estimators=500, random_state=42)),
            ("gb", GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
            ("rf", RandomForestRegressor(n_estimators=500, random_state=42)),
        ]),
        "Vote(ET+GB+Ada)": VotingRegressor([
            ("et", ExtraTreesRegressor(n_estimators=500, random_state=42)),
            ("gb", GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
            ("ada", AdaBoostRegressor(n_estimators=200, random_state=42)),
        ]),
        "Vote(ET+HistGB)": VotingRegressor([
            ("et", ExtraTreesRegressor(n_estimators=500, random_state=42)),
            ("hgb", HistGradientBoostingRegressor(random_state=42)),
        ]),
        "Vote(ET+GB+Ridge)": VotingRegressor([
            ("et", ExtraTreesRegressor(n_estimators=500, random_state=42)),
            ("gb", GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
            ("ridge", Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])),
        ]),
    }

    for name, model in voting_configs.items():
        try:
            oof = cross_val_predict(model, X_all, y, cv=KF)
            record(f"[all] {name}", y, oof, len(feature_cols))
        except Exception as e:
            print(f"  FAILED: {name}: {e}")

    # Stacking
    stacking_configs = {
        "Stack(ET+GB)->Ridge": StackingRegressor(
            estimators=[
                ("et", ExtraTreesRegressor(n_estimators=200, random_state=42)),
                ("gb", GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)),
            ],
            final_estimator=Ridge(alpha=10.0),
            cv=3,
        ),
        "Stack(ET+GB+RF)->Ridge": StackingRegressor(
            estimators=[
                ("et", ExtraTreesRegressor(n_estimators=200, random_state=42)),
                ("gb", GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)),
                ("rf", RandomForestRegressor(n_estimators=200, random_state=42)),
            ],
            final_estimator=Ridge(alpha=10.0),
            cv=3,
        ),
        "Stack(ET+GB)->BR": StackingRegressor(
            estimators=[
                ("et", ExtraTreesRegressor(n_estimators=200, random_state=42)),
                ("gb", GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)),
            ],
            final_estimator=BayesianRidge(),
            cv=3,
        ),
    }

    for name, model in stacking_configs.items():
        try:
            oof = cross_val_predict(model, X_all, y, cv=KF)
            record(f"[all] {name}", y, oof, len(feature_cols))
        except Exception as e:
            print(f"  FAILED: {name}: {e}")

    # ==== SECTION 8: TARGET TRANSFORMS ====
    print(f"\n{'='*80}")
    print("SECTION 8: TARGET TRANSFORMS")
    print(f"{'='*80}")

    # Log-transform (shift to make positive)
    y_shift = y - y.min() + 1
    y_log = np.log(y_shift)
    et = ExtraTreesRegressor(n_estimators=500, random_state=42)
    oof_log = cross_val_predict(et, X_all, y_log, cv=KF)
    oof_back = np.exp(oof_log) + y.min() - 1
    record("ET500 [log target]", y, oof_back, len(feature_cols))

    # Rank transform
    from scipy.stats import rankdata
    y_rank = rankdata(y) / len(y)
    oof_rank = cross_val_predict(ExtraTreesRegressor(n_estimators=500, random_state=42), X_all, y_rank, cv=KF)
    # Map back via linear interp
    from scipy.interpolate import interp1d
    sort_idx = np.argsort(y_rank)
    interp = interp1d(y_rank[sort_idx], y[sort_idx], kind="linear", fill_value="extrapolate")
    oof_rank_back = interp(np.clip(oof_rank, y_rank.min(), y_rank.max()))
    record("ET500 [rank target]", y, oof_rank_back, len(feature_cols))

    # ==== SECTION 9: MULTI-OUTPUT / AUXILIARY TARGETS ====
    print(f"\n{'='*80}")
    print("SECTION 9: AUXILIARY TARGETS")
    print(f"{'='*80}")

    # Predict lmarena and lmsys separately, take difference
    lmarena = df.merge(style.dropna(subset=["delta_score"])[["model", "lmarena_Score"]], on="model")["lmarena_Score"].values
    lmsys = df.merge(style.dropna(subset=["delta_score"])[["model", "lmsys_Score"]], on="model")["lmsys_Score"].values

    if len(lmarena) == len(y):
        oof_lmarena = cross_val_predict(ExtraTreesRegressor(n_estimators=500, random_state=42), X_all, lmarena, cv=KF)
        oof_lmsys = cross_val_predict(ExtraTreesRegressor(n_estimators=500, random_state=42), X_all, lmsys, cv=KF)
        oof_diff = oof_lmarena - oof_lmsys
        record("ET500 [predict lmarena-lmsys separately]", y, oof_diff, len(feature_cols))

    # ==== SECTION 10: MANUAL BLENDING OF OOF PREDICTIONS ====
    print(f"\n{'='*80}")
    print("SECTION 10: MANUAL BLENDING")
    print(f"{'='*80}")

    # Collect OOF predictions from top models
    blend_pool = {}
    blend_models = {
        "ET500": ExtraTreesRegressor(n_estimators=500, random_state=42),
        "ET1000": ExtraTreesRegressor(n_estimators=1000, random_state=42),
        "GB200": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
        "HistGB": HistGradientBoostingRegressor(random_state=42),
        "RF500": RandomForestRegressor(n_estimators=500, random_state=42),
        "Ada400": AdaBoostRegressor(n_estimators=400, random_state=42),
    }

    for name, model in blend_models.items():
        blend_pool[name] = cross_val_predict(model, X_all, y, cv=KF)

    # Try all pairs
    names = list(blend_pool.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            avg = (blend_pool[names[i]] + blend_pool[names[j]]) / 2
            record(f"Blend({names[i]}+{names[j]})", y, avg, len(feature_cols))

    # Try all triples
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for k in range(j + 1, len(names)):
                avg = (blend_pool[names[i]] + blend_pool[names[j]] + blend_pool[names[k]]) / 3
                record(f"Blend({names[i]}+{names[j]}+{names[k]})", y, avg, len(feature_cols))

    # Weighted blend: optimize weights via simple grid
    print(f"\n  Searching weighted blends (ET500+GB200)...")
    for w_et in np.arange(0.1, 1.0, 0.1):
        w_gb = 1.0 - w_et
        avg = w_et * blend_pool["ET500"] + w_gb * blend_pool["GB200"]
        r, rmse = oof_metrics(y, avg)
        if rmse < 12.5:  # only record good ones
            record(f"WBlend(ET500={w_et:.1f},GB200={w_gb:.1f})", y, avg, len(feature_cols))

    for w_et in np.arange(0.1, 0.9, 0.1):
        for w_gb in np.arange(0.1, 0.9 - w_et + 0.05, 0.1):
            w_hgb = 1.0 - w_et - w_gb
            if w_hgb < 0.05:
                continue
            avg = w_et * blend_pool["ET500"] + w_gb * blend_pool["GB200"] + w_hgb * blend_pool["HistGB"]
            r, rmse = oof_metrics(y, avg)
            if rmse < 12.5:
                record(f"WBlend3(ET={w_et:.1f},GB={w_gb:.1f},HGB={w_hgb:.1f})", y, avg, len(feature_cols))

    # ==== SECTION 11: REPEATED CV FOR STABILITY ====
    print(f"\n{'='*80}")
    print("SECTION 11: REPEATED CV (3 seeds)")
    print(f"{'='*80}")

    for name_base, model_fn in [
        ("ET500", lambda: ExtraTreesRegressor(n_estimators=500, random_state=42)),
        ("GB200", lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
    ]:
        oofs = []
        for seed in [42, 123, 456]:
            kf_s = KFold(n_splits=5, shuffle=True, random_state=seed)
            oofs.append(cross_val_predict(model_fn(), X_all, y, cv=kf_s))
        oof_avg = np.mean(oofs, axis=0)
        record(f"AvgSeeds(3) {name_base}", y, oof_avg, len(feature_cols))

    # ==== SECTION 12: BEST ON EACH FEATURE SUBSET ====
    print(f"\n{'='*80}")
    print("SECTION 12: TOP MODELS x FEATURE SUBSETS")
    print(f"{'='*80}")

    top_model_configs = {
        "ET500": ExtraTreesRegressor(n_estimators=500, random_state=42),
        "GB200": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
        "Vote(ET+GB)": VotingRegressor([
            ("et", ExtraTreesRegressor(n_estimators=500, random_state=42)),
            ("gb", GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
        ]),
    }

    for subset_name in ["old_plus_summaries", "new_only"]:
        cols = subsets[subset_name]
        X = df[cols].fillna(0).values
        for model_name, model in top_model_configs.items():
            try:
                oof = cross_val_predict(model, X, y, cv=KF)
                record(f"[{subset_name}] {model_name}", y, oof, len(cols))
            except Exception as e:
                print(f"  FAILED: [{subset_name}] {model_name}: {e}")

    # ==== SECTION 13: FEATURE SELECTION ON BEST SUBSETS ====
    print(f"\n{'='*80}")
    print("SECTION 13: FEATURE SELECTION ON SUBSETS")
    print(f"{'='*80}")

    # ImpTop on old_plus_summaries
    X_ops = df[subsets["old_plus_summaries"]].fillna(0).values
    et_imp2 = ExtraTreesRegressor(n_estimators=500, random_state=42)
    et_imp2.fit(X_ops, y)
    imp_order2 = np.argsort(et_imp2.feature_importances_)[::-1]

    for k in [20, 30, 50, 80, 100]:
        if k > len(subsets["old_plus_summaries"]):
            continue
        top_idx = imp_order2[:k]
        X_top = X_ops[:, top_idx]
        oof = cross_val_predict(ExtraTreesRegressor(n_estimators=500, random_state=42), X_top, y, cv=KF)
        record(f"[old+sum] ImpTop{k}->ET500", y, oof, k)

    # ImpTop on all features with GB
    for k in [30, 50, 80]:
        if k > len(feature_cols):
            continue
        top_idx = imp_order[:k]
        X_top = X_all[:, top_idx]
        oof = cross_val_predict(
            GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
            X_top, y, cv=KF
        )
        record(f"ImpTop{k}->GB200", y, oof, k)

    # ImpTop on all with Vote(ET+GB)
    for k in [50, 80]:
        if k > len(feature_cols):
            continue
        top_idx = imp_order[:k]
        X_top = X_all[:, top_idx]
        vote = VotingRegressor([
            ("et", ExtraTreesRegressor(n_estimators=500, random_state=42)),
            ("gb", GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
        ])
        oof = cross_val_predict(vote, X_top, y, cv=KF)
        record(f"ImpTop{k}->Vote(ET+GB)", y, oof, k)

    # MI-Select on all with different models
    for k in [50, 80]:
        if k > len(feature_cols):
            continue
        sel_pipe = Pipeline([
            ("selector", SelectKBest(mutual_info_regression, k=k)),
            ("model", GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
        ])
        oof = cross_val_predict(sel_pipe, X_all, y, cv=KF)
        record(f"MI-Select(k={k})->GB200", y, oof, k)

    # ImpTop with seed averaging
    for k in [50, 80]:
        top_idx = imp_order[:k]
        X_top = X_all[:, top_idx]
        oofs = []
        for seed in [42, 123, 456]:
            kf_s = KFold(n_splits=5, shuffle=True, random_state=seed)
            oofs.append(cross_val_predict(ExtraTreesRegressor(n_estimators=500, random_state=42), X_top, y, cv=kf_s))
        oof_avg = np.mean(oofs, axis=0)
        record(f"ImpTop{k}->ET500 AvgSeeds(3)", y, oof_avg, k)

    # Blend of ImpTop predictions
    oof_it50_et = cross_val_predict(ExtraTreesRegressor(n_estimators=500, random_state=42), X_all[:, imp_order[:50]], y, cv=KF)
    oof_it80_et = cross_val_predict(ExtraTreesRegressor(n_estimators=500, random_state=42), X_all[:, imp_order[:80]], y, cv=KF)
    oof_it50_gb = cross_val_predict(
        GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
        X_all[:, imp_order[:50]], y, cv=KF
    )
    record("Blend(ImpTop50-ET + ImpTop80-ET)", y, (oof_it50_et + oof_it80_et) / 2, "50+80")
    record("Blend(ImpTop50-ET + ImpTop50-GB)", y, (oof_it50_et + oof_it50_gb) / 2, 50)
    record("Blend(ImpTop80-ET + ImpTop50-GB)", y, (oof_it80_et + oof_it50_gb) / 2, "80+50")
    record("Blend3(IT50ET+IT80ET+IT50GB)", y, (oof_it50_et + oof_it80_et + oof_it50_gb) / 3, "50+80+50")

    # ==== LEADERBOARD ====
    print(f"\n{'='*80}")
    print("LEADERBOARD (sorted by RMSE)")
    print(f"{'='*80}")

    results_df = pd.DataFrame(RESULTS).sort_values("rmse")
    results_df["rank"] = range(1, len(results_df) + 1)
    results_df = results_df[["rank", "name", "r", "rmse", "n_feats"]]

    print(results_df.head(30).to_string(index=False))

    print(f"\n--- Bottom 10 ---")
    print(results_df.tail(10).to_string(index=False))

    # Save full results
    out_path = os.path.join(SCRIPT_DIR, "outputs", "delta_experiment_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    # Summary stats
    best = results_df.iloc[0]
    print(f"\n{'='*80}")
    print(f"BEST: {best['name']}  r={best['r']:.3f}  RMSE={best['rmse']:.2f}")
    print(f"Current production (ET500 old_only): see above")
    print(f"Delta std: {y.std():.2f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_experiments()
