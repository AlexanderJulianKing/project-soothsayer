"""Stage 3: Aggregate judge scores, compute markdown features, predict Arena ELO delta.

Reads:
  - responses.csv   (long-format model responses from collect.py)
  - judge_results.csv (per-response judge scores from style_analysis.py)
  - combined_all_benches.csv (external benchmarks for ML target)

Outputs (backward-compatible with combine.py df8/df10):
  - outputs/style_YYYYMMDD.csv
  - outputs/tone_YYYYMMDD.csv
  - outputs/style_features_v4_YYYYMMDD.csv   (diagnostic)
  - outputs/style_models_v4_YYYYMMDD.csv     (diagnostic)
  - outputs/style_feature_importance_v4_YYYYMMDD.csv (diagnostic)
"""

import datetime
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.utils import get_latest_file
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.model_selection import KFold, cross_val_predict

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESPONSES_CSV = os.path.join(SCRIPT_DIR, 'responses.csv')
JUDGE_RESULTS_CSV = os.path.join(SCRIPT_DIR, 'judge_results.csv')

_BENCHMARK_PATTERNS = [
    os.path.join(SCRIPT_DIR, '..', 'benchmark_combiner', 'benchmarks', 'combined_all_benches.csv'),
    os.path.join(SCRIPT_DIR, 'combined_all_benches.csv'),
]
BENCHMARK_SCORES_FILE = next((p for p in _BENCHMARK_PATTERNS if os.path.exists(p)), _BENCHMARK_PATTERNS[-1])

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d")
STYLE_OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'outputs', f'style_{TIMESTAMP}.csv')
TONE_OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'outputs', f'tone_{TIMESTAMP}.csv')
PER_QUESTION_FEATURE_FILE = os.path.join(SCRIPT_DIR, 'outputs', f'style_features_v4_{TIMESTAMP}.csv')
MODEL_PERFORMANCE_FILE = os.path.join(SCRIPT_DIR, 'outputs', f'style_models_v4_{TIMESTAMP}.csv')
MODEL_IMPORTANCE_FILE = os.path.join(SCRIPT_DIR, 'outputs', f'style_feature_importance_v4_{TIMESTAMP}.csv')

SCORING_WEIGHTS = {
    'length': 0.249,
    'header_count': 0.031,
    'bold_count': 0.024,
    'list_count': 0.019,
}
STYLE_METRICS = list(SCORING_WEIGHTS.keys())

# Extra text metrics computed per response alongside the original 4
EXTRA_TEXT_METRICS = [
    'sentences', 'words', 'paragraphs', 'code_blocks', 'inline_code',
    'blockquotes', 'table_rows', 'emoji_count', 'exclamation_marks',
    'question_marks', 'newlines',
]

FEATURE_SELECT_K = 80


# ==============================================================================
# --- JUDGE SCORE AGGREGATION (replaces process_analysis.py) ---
# ==============================================================================
def aggregate_judge_scores(judge_csv: str) -> pd.DataFrame:
    """Read judge_results.csv, normalize each axis per judge, aggregate to one row per model.

    Handles 2-axis scores (score_density, score_confidence).
    Returns a DataFrame with columns: judged_model, scaled_avg_density_by_<judge>, scaled_avg_confidence_by_<judge>, ...
    """
    if not os.path.exists(judge_csv):
        print(f"Warning: Judge results file '{judge_csv}' not found. Skipping tone output.")
        return pd.DataFrame()

    df = pd.read_csv(judge_csv)
    if df.empty:
        return pd.DataFrame()

    score_cols = ['score_density', 'score_confidence']
    for col in score_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in judge results.")
            return pd.DataFrame()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=score_cols)

    if df.empty:
        print("Warning: No valid numeric scores found in judge results.")
        return pd.DataFrame()

    judge_models = df['judge_model_id'].unique()
    aggregation_rules = {}

    for judge_id in judge_models:
        mask = df['judge_model_id'] == judge_id
        sanitized = judge_id.replace('/', '_').replace('-', '_').replace('.', '_').replace(':', '_')

        for axis in ('density', 'confidence'):
            src_col = f'score_{axis}'
            valid_scores = df.loc[mask, src_col]

            if valid_scores.empty:
                continue

            min_val = valid_scores.min()
            max_val = valid_scores.max()

            norm_col = f'normalized_{axis}_{sanitized}'

            if max_val == min_val:
                df.loc[mask, norm_col] = 100.0
            else:
                df.loc[mask, norm_col] = 100.0 * (df.loc[mask, src_col] - min_val) / (max_val - min_val)

            final_col = f'scaled_avg_{axis}_by_{sanitized}'
            aggregation_rules[final_col] = (norm_col, 'mean')

    if not aggregation_rules:
        print("Warning: No aggregation rules generated from judge scores.")
        return pd.DataFrame()

    grouped = df.groupby('model_name').agg(**aggregation_rules).reset_index()
    grouped.rename(columns={'model_name': 'judged_model'}, inplace=True)
    return grouped


def load_tone_trueskill_scores() -> pd.DataFrame:
    """Read the latest results/tone_*.csv produced by super_bench.py.

    The TrueSkill mu values are already normalized — no additional scaling needed.
    """
    results_dir = os.path.join(SCRIPT_DIR, 'results')
    tone_path = get_latest_file(os.path.join(results_dir, 'tone_*.csv'))
    if not tone_path:
        print("Warning: No tone TrueSkill results found in results/. Run super_bench.py first.")
        return pd.DataFrame()
    df = pd.read_csv(tone_path)
    print(f"Loaded tone TrueSkill scores from {tone_path}")
    return df


# ==============================================================================
# --- TEXT FEATURE ENGINEERING ---
# ==============================================================================
def calculate_markdown_stats(text: str) -> Dict[str, int]:
    text = text or ""
    stats = {}
    stats['length'] = len(text)
    stats['header_count'] = len(re.findall(r'^\s*#{1,6}\s+', text, re.MULTILINE))
    bold_asterisk = len(re.findall(r'\*\*(?!\s)(.+?)(?<!\s)\*\*', text))
    bold_underscore = len(re.findall(r'__(?!\s)(.+?)(?<!\s)__', text))
    stats['bold_count'] = bold_asterisk + bold_underscore
    unordered = len(re.findall(r'^\s*[-*+]\s+', text, re.MULTILINE))
    ordered = len(re.findall(r'^\s*\d+[.)]\s+', text, re.MULTILINE))
    stats['list_count'] = unordered + ordered
    return stats


def calculate_extra_text_stats(text: str) -> Dict[str, float]:
    """Additional text features: sentence/word counts, code blocks, emoji, etc."""
    text = str(text) if text else ""
    words = text.split()
    lines = text.split("\n")
    non_empty_lines = [l for l in lines if l.strip()]
    return {
        'sentences': len(re.split(r'[.!?]+', text)),
        'words': len(words),
        'paragraphs': len(re.split(r'\n\s*\n', text.strip())),
        'code_blocks': len(re.findall(r'```', text)) // 2,
        'inline_code': len(re.findall(r'`[^`]+`', text)),
        'blockquotes': len(re.findall(r'^\s*>', text, re.MULTILINE)),
        'table_rows': len(re.findall(r'^\|.*\|\s*$', text, re.MULTILINE)),
        'emoji_count': len(re.findall(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
            r'\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF'
            r'\U00002702-\U000027B0]', text)),
        'exclamation_marks': text.count('!'),
        'question_marks': text.count('?'),
        'newlines': text.count('\n'),
    }


def min_max_scale(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(0.0, index=series.index)
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_style_features(responses_csv: str) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """Compute text features from long-format responses.csv.

    Computes per-response stats (original markdown + extra text), then averages
    across runs per (model, question). Returns wide-format features + summaries.
    """
    if not os.path.exists(responses_csv):
        print(f"Error: Responses file '{responses_csv}' not found.")
        return pd.DataFrame(), [], [], []

    df = pd.read_csv(responses_csv)
    if df.empty:
        print("Error: responses.csv is empty.")
        return pd.DataFrame(), [], [], []

    # Only compute features on successful responses
    df = df[df['status'] == 'ok'].copy()
    if df.empty:
        print("Error: No 'ok' responses found in responses.csv.")
        return pd.DataFrame(), [], [], []

    # Compute per-response stats: original 4 + extra 11
    all_metrics = STYLE_METRICS + EXTRA_TEXT_METRICS
    for metric in all_metrics:
        df[metric] = 0
    for idx, row in df.iterrows():
        text = str(row.get('response', '') or '')
        md_stats = calculate_markdown_stats(text)
        extra_stats = calculate_extra_text_stats(text)
        for metric in STYLE_METRICS:
            df.at[idx, metric] = md_stats[metric]
        for metric in EXTRA_TEXT_METRICS:
            df.at[idx, metric] = extra_stats[metric]

    # Average raw stats across runs per (model_name, question_id)
    per_model_question = df.groupby(['model_name', 'question_id'])[all_metrics].mean().reset_index()

    # Get model-level code mapping (first occurrence)
    model_codes = df.drop_duplicates(subset=['model_name'])[['model_name', 'model_id']].copy()

    # Pivot to wide format: one row per model, columns like Q1_length, Q1_sentences, etc.
    q_ids = sorted(per_model_question['question_id'].unique())
    feature_rows = []
    for model_name in per_model_question['model_name'].unique():
        model_data = per_model_question[per_model_question['model_name'] == model_name]
        code = model_codes[model_codes['model_name'] == model_name]['model_id'].iloc[0]
        row_dict: Dict[str, Any] = {'model': model_name, 'code': code}

        for q_id in q_ids:
            q_row = model_data[model_data['question_id'] == q_id]
            for metric in all_metrics:
                col_name = f'Q{q_id}_{metric}'
                row_dict[col_name] = q_row[metric].values[0] if len(q_row) > 0 else 0

        feature_rows.append(row_dict)

    features_df = pd.DataFrame(feature_rows)

    # Compute combined (full-text) stats: average across all questions per model
    combined = per_model_question.groupby('model_name')[all_metrics].mean().reset_index()
    n_questions = len(q_ids)
    for metric in all_metrics:
        combined[f'combined_{metric}'] = combined[metric] * n_questions
    combined = combined.drop(columns=all_metrics)
    features_df = features_df.merge(combined, left_on='model', right_on='model_name', how='left')
    if 'model_name' in features_df.columns:
        features_df.drop(columns=['model_name'], inplace=True)

    # Model-level summary stats (mean, std) from per-question raw values
    for metric in all_metrics:
        q_cols = [f'Q{q}_{metric}' for q in q_ids if f'Q{q}_{metric}' in features_df.columns]
        if q_cols:
            features_df[f'avg_{metric}'] = features_df[q_cols].mean(axis=1)
            features_df[f'std_{metric}'] = features_df[q_cols].std(axis=1)

    # Shape/variance features: CV, min, frac_used for primary style metrics
    for metric in STYLE_METRICS:
        q_cols = [f'Q{q}_{metric}' for q in q_ids if f'Q{q}_{metric}' in features_df.columns]
        if not q_cols:
            continue
        q_vals = features_df[q_cols].values.astype(float)
        avg = features_df.get(f'avg_{metric}', pd.Series(0, index=features_df.index)).values
        std = features_df.get(f'std_{metric}', pd.Series(0, index=features_df.index)).values
        features_df[f'cv_{metric}'] = np.where(avg > 1e-6, std / avg, 0.0)
        features_df[f'min_{metric}'] = np.nanmin(q_vals, axis=1)
        features_df[f'frac_used_{metric}'] = np.nanmean(q_vals > 0, axis=1)

    # Q7 (creative programming) per-question features — strongest single-question
    # predictors of Arena ELO (r=0.57 for headers, r=0.47 for length)
    for metric in STYLE_METRICS:
        q7_col = f'Q7_{metric}'
        if q7_col in features_df.columns:
            features_df[f'q7_{metric}'] = features_df[q7_col]

    # Normalized columns (original metrics only, for backward compat with combine.py)
    q_col_labels = [f'Q{q_id}' for q_id in q_ids]
    per_question_raw_cols: List[str] = []
    per_question_norm_cols: List[str] = []

    for q_label in q_col_labels:
        for metric in STYLE_METRICS:
            col_name = f'{q_label}_{metric}'
            if col_name in features_df.columns:
                per_question_raw_cols.append(col_name)
                norm_col = f'{q_label}_normalized_{metric}'
                features_df[norm_col] = min_max_scale(features_df[col_name]) * 100
                per_question_norm_cols.append(norm_col)

    for metric in STYLE_METRICS:
        combined_col = f'combined_{metric}'
        norm_col = f'normalized_{metric}'
        if combined_col in features_df.columns:
            features_df[norm_col] = min_max_scale(features_df[combined_col]) * 100

    # Log transforms on skewed metrics
    log_metrics = ['length', 'bold_count', 'list_count']
    log_norm_cols: List[str] = []
    for metric in log_metrics:
        norm_col = f'normalized_{metric}'
        combined_col = f'combined_{metric}'
        if norm_col in features_df.columns:
            features_df[f'log_normalized_{metric}'] = np.log1p(features_df[norm_col])
            log_norm_cols.append(f'log_normalized_{metric}')
        if combined_col in features_df.columns:
            features_df[f'log_combined_{metric}'] = np.log1p(features_df[combined_col])
            log_norm_cols.append(f'log_combined_{metric}')
        for c in per_question_norm_cols:
            if f'normalized_{metric}' in c:
                features_df[f'log_{c}'] = np.log1p(features_df[c])
                log_norm_cols.append(f'log_{c}')

    # Engineered ratios
    avg_len = features_df.get('avg_length', features_df.get('combined_length', pd.Series(0, index=features_df.index)))
    for metric in ['bold_count', 'header_count', 'list_count', 'code_blocks', 'inline_code', 'emoji_count']:
        avg_col = f'avg_{metric}'
        if avg_col in features_df.columns:
            features_df[f'{metric}_per_1k_chars'] = features_df[avg_col] / (avg_len + 1) * 1000
    if 'avg_sentences' in features_df.columns and 'avg_paragraphs' in features_df.columns:
        features_df['sentences_per_paragraph'] = features_df['avg_sentences'] / (features_df['avg_paragraphs'] + 1)
    if 'avg_words' in features_df.columns and 'avg_sentences' in features_df.columns:
        features_df['words_per_sentence'] = features_df['avg_words'] / (features_df['avg_sentences'] + 1)
    if 'avg_length' in features_df.columns and 'avg_words' in features_df.columns:
        features_df['chars_per_word'] = features_df['avg_length'] / (features_df['avg_words'] + 1)

    # Composite style score (backward compat)
    total_weight = sum(SCORING_WEIGHTS.values())
    features_df['style_score'] = 0.0
    for metric, weight in SCORING_WEIGHTS.items():
        norm_col = f'normalized_{metric}'
        if norm_col in features_df.columns:
            features_df['style_score'] += features_df[norm_col] * (weight / total_weight)

    return features_df, per_question_raw_cols, per_question_norm_cols, log_norm_cols


# ==============================================================================
# --- MACHINE LEARNING ---
# ==============================================================================
def load_benchmark_scores(score_file: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(score_file)
    except FileNotFoundError:
        print(f"Error: Combined benchmark file not found at '{score_file}'.")
        return pd.DataFrame()

    if 'openbench_Model' in df.columns:
        pass
    elif 'model_name' in df.columns:
        df = df.rename(columns={'model_name': 'openbench_Model'})
    else:
        raise ValueError("Combined benchmark file missing model column (openbench_Model or model_name)")

    needed_cols = {'openbench_Model', 'lmarena_Score', 'lmsys_Score'}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"Combined benchmark file missing columns: {missing}")

    keep_cols = ['openbench_Model', 'lmarena_Score', 'lmsys_Score']
    sbf_extras = ['logic_weighted_accuracy', 'logic_PC1']
    keep_cols += [c for c in sbf_extras if c in df.columns]

    df = df[keep_cols]
    df['lmarena_Score'] = pd.to_numeric(df['lmarena_Score'], errors='coerce')
    df['lmsys_Score'] = pd.to_numeric(df['lmsys_Score'], errors='coerce')
    for c in sbf_extras:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.drop_duplicates(subset=['openbench_Model'], keep='last')
    return df


def _build_delta_model():
    """Vote(ExtraTrees + GradientBoosting) — best from 182-experiment sweep."""
    return VotingRegressor([
        ('et', ExtraTreesRegressor(n_estimators=500, random_state=42)),
        ('gb', GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
    ])


def _select_top_features(X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
    """Importance-based feature selection using ExtraTrees."""
    selector = ExtraTreesRegressor(n_estimators=500, random_state=42)
    selector.fit(X, y)
    importances = pd.Series(selector.feature_importances_, index=X.columns)
    top_cols = importances.nlargest(k).index.tolist()
    return top_cols


def train_and_predict_delta(
    features_df: pd.DataFrame,
    per_question_raw_cols: List[str],
    per_question_norm_cols: List[str],
    log_norm_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    benchmark_df = load_benchmark_scores(BENCHMARK_SCORES_FILE)
    if benchmark_df.empty:
        print("Benchmark file is empty or missing required columns. Cannot train ML models.")
        features_df['predicted_delta'] = pd.NA
        features_df['best_model'] = pd.NA
        features_df['delta_score'] = pd.NA
        return features_df, {}, []

    features_with_scores = features_df.merge(benchmark_df, how='left', left_on='model', right_on='openbench_Model')
    if features_with_scores.empty:
        print("No style features available after merging benchmark scores.")
        return pd.DataFrame(), {}, []

    features_with_scores['delta_score'] = features_with_scores['lmarena_Score'] - features_with_scores['lmsys_Score']

    train_df = features_with_scores.dropna(subset=['lmarena_Score', 'lmsys_Score'])
    if train_df.empty:
        print("No overlapping models with both lmarena and lmsys scores; cannot train predictor.")
        features_with_scores['predicted_delta'] = pd.NA
        features_with_scores['best_model'] = pd.NA
        return features_with_scores, {}, []

    # Build feature column list (all numeric features available)
    exclude_cols = {'model', 'code', 'openbench_Model', 'lmarena_Score', 'lmsys_Score', 'delta_score',
                    'predicted_delta', 'best_model'}
    all_feature_cols = [c for c in features_with_scores.columns
                        if c not in exclude_cols and features_with_scores[c].dtype in ('float64', 'int64', 'float32')]

    sbf_cols = [c for c in ['logic_weighted_accuracy', 'logic_PC1']
                if c in features_with_scores.columns]
    base_feature_cols = [c for c in all_feature_cols if c not in sbf_cols]

    feature_cols = all_feature_cols
    importance_rows: List[Dict[str, Any]] = []

    if sbf_cols:
        train_with_sbf = train_df.dropna(subset=sbf_cols)
        n_with = len(train_with_sbf)
        n_without = len(train_df) - n_with
        print(f"Delta training: {n_with} models with logic, {n_without} without.")
    else:
        train_with_sbf = train_df

    X_train_full = train_with_sbf[feature_cols].fillna(0.0)
    y_train = train_with_sbf['delta_score']

    # Feature selection: pick top-k by ExtraTrees importance
    k = min(FEATURE_SELECT_K, len(feature_cols))
    selected_cols = _select_top_features(X_train_full, y_train, k)
    print(f"Selected {len(selected_cols)} features from {len(feature_cols)} total.")

    X_train = train_with_sbf[selected_cols].fillna(0.0)
    cv_splits = min(5, len(X_train))

    # Train production model on all training data
    best_model_name = 'vote_et_gb'
    best_model = _build_delta_model()
    best_model.fit(X_train, y_train)

    # OOF predictions for training models to prevent leakage into downstream pipeline
    kf_oof = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    oof_model = _build_delta_model()
    oof_preds = cross_val_predict(oof_model, X_train, y_train, cv=kf_oof)
    oof_r = np.corrcoef(y_train, oof_preds)[0, 1] if len(y_train) > 2 else float('nan')
    oof_rmse = float(np.sqrt(np.mean((y_train.values - oof_preds) ** 2)))
    print(f"Delta OOF: r={oof_r:.3f}, RMSE={oof_rmse:.2f} (delta std={y_train.std():.2f})")

    has_sbf = features_with_scores[sbf_cols].notna().all(axis=1) if sbf_cols else pd.Series(True, index=features_with_scores.index)
    features_with_scores['predicted_delta'] = pd.NA
    features_with_scores['best_model'] = best_model_name

    # Assign OOF predictions to training rows, full-model predictions to non-training rows
    train_sbf_idx = train_with_sbf.index
    features_with_scores.loc[train_sbf_idx, 'predicted_delta'] = oof_preds

    non_train_sbf = has_sbf & ~features_with_scores.index.isin(train_sbf_idx)
    if non_train_sbf.any():
        X_non_train = features_with_scores.loc[non_train_sbf, selected_cols].fillna(0.0)
        features_with_scores.loc[non_train_sbf, 'predicted_delta'] = best_model.predict(X_non_train)

    # Fallback for models without logic features
    if sbf_cols and (~has_sbf).any():
        n_fallback = (~has_sbf).sum()
        fallback_base_cols = [c for c in selected_cols if c not in sbf_cols]
        if not fallback_base_cols:
            fallback_base_cols = base_feature_cols[:k]

        X_fb_train = train_df[fallback_base_cols].fillna(0.0)
        y_fb_train = train_df['delta_score']
        fallback_model = _build_delta_model()
        fallback_model.fit(X_fb_train, y_fb_train)

        # OOF for fallback training rows too
        fallback_train_idx = train_df[train_df[sbf_cols].isna().any(axis=1)].index
        if len(fallback_train_idx) >= 2:
            kf_fb = KFold(n_splits=min(cv_splits, len(fallback_train_idx)), shuffle=True, random_state=42)
            fb_oof_model = _build_delta_model()
            fb_oof = cross_val_predict(
                fb_oof_model,
                X_fb_train.loc[fallback_train_idx],
                y_fb_train.loc[fallback_train_idx],
                cv=kf_fb,
            )
            features_with_scores.loc[fallback_train_idx, 'predicted_delta'] = fb_oof

        non_train_no_sbf = (~has_sbf) & ~features_with_scores.index.isin(train_df.index)
        if non_train_no_sbf.any():
            X_no_sbf = features_with_scores.loc[non_train_no_sbf, fallback_base_cols].fillna(0.0)
            features_with_scores.loc[non_train_no_sbf, 'predicted_delta'] = fallback_model.predict(X_no_sbf)
        print(f"Fallback used for {n_fallback} models without logic.")

    # Feature importances (from the ET component of the voting model)
    et_model = best_model.named_estimators_['et']
    importances = et_model.feature_importances_
    importance_rows = [
        {
            'feature': feature,
            'importance': float(value),
            'best_model': best_model_name,
            'type': 'feature_importance',
        }
        for feature, value in zip(selected_cols, importances)
    ]

    performance = {
        best_model_name: {
            'mean_rmse': oof_rmse,
            'std_rmse': float('nan'),
            'splits': cv_splits,
        }
    }

    return features_with_scores, performance, importance_rows


# ==============================================================================
# --- OUTPUT ---
# ==============================================================================
def save_outputs(
    feature_df: pd.DataFrame,
    per_question_raw_cols: List[str],
    per_question_norm_cols: List[str],
    prediction_df: pd.DataFrame,
    performance: Dict[str, Dict[str, float]],
    importance_rows: List[Dict[str, Any]],
):
    os.makedirs(os.path.join(SCRIPT_DIR, 'outputs'), exist_ok=True)

    # Per-question features (diagnostic)
    per_question_export_cols = ['code', 'model'] + per_question_raw_cols + per_question_norm_cols
    per_question_export_cols += [f'combined_{metric}' for metric in STYLE_METRICS]
    per_question_export_cols += [f'normalized_{metric}' for metric in STYLE_METRICS]
    available_per_question_cols = [col for col in per_question_export_cols if col in feature_df.columns]
    feature_df[available_per_question_cols].to_csv(PER_QUESTION_FEATURE_FILE, index=False)

    # Style summary (consumed by combine.py df8)
    summary_cols = ['code', 'model', 'style_score']
    summary_cols += [f'combined_{metric}' for metric in STYLE_METRICS]
    summary_cols += [f'normalized_{metric}' for metric in STYLE_METRICS]
    summary_cols += ['log_normalized_length']
    # Shape/variance features
    for metric in STYLE_METRICS:
        summary_cols += [f'cv_{metric}', f'min_{metric}', f'frac_used_{metric}']
    # Q7 (creative programming) per-question features
    for metric in STYLE_METRICS:
        summary_cols.append(f'q7_{metric}')
    for col in ['lmarena_Score', 'lmsys_Score', 'delta_score', 'predicted_delta']:
        if col in prediction_df.columns:
            summary_cols.append(col)

    available_summary_cols = [col for col in summary_cols if col in prediction_df.columns]
    summary_df = prediction_df[available_summary_cols]
    summary_df.to_csv(STYLE_OUTPUT_FILE, index=False)

    # Model performance (diagnostic)
    performance_rows = []
    for name, stats in performance.items():
        performance_rows.append({
            'model_name': name,
            'mean_rmse': stats.get('mean_rmse'),
            'std_rmse': stats.get('std_rmse'),
            'splits': stats.get('splits'),
        })
    pd.DataFrame(performance_rows).to_csv(MODEL_PERFORMANCE_FILE, index=False)

    # Feature importances (diagnostic)
    importance_df = pd.DataFrame(importance_rows)
    if importance_df.empty:
        importance_df = pd.DataFrame(columns=['feature', 'importance', 'best_model', 'type'])
    importance_df.to_csv(MODEL_IMPORTANCE_FILE, index=False)

    print(f"\nSaved style summary to {STYLE_OUTPUT_FILE}")
    print(f"Saved per-question features to {PER_QUESTION_FEATURE_FILE}")
    print(f"Saved model RMSE comparisons to {MODEL_PERFORMANCE_FILE}")
    print(f"Saved feature importances to {MODEL_IMPORTANCE_FILE}")


# ==============================================================================
# --- MAIN ---
# ==============================================================================
def main():
    # 1. Load tone TrueSkill scores from super_bench.py output -> tone CSV
    tone_df = load_tone_trueskill_scores()
    if not tone_df.empty:
        os.makedirs(os.path.join(SCRIPT_DIR, 'outputs'), exist_ok=True)
        tone_df.to_csv(TONE_OUTPUT_FILE, index=False, float_format='%.4f')
        print(f"Saved tone scores to {TONE_OUTPUT_FILE}")
    else:
        print("Warning: No tone scores to save.")

    # 2. Compute text style features from responses
    features_df, per_question_raw_cols, per_question_norm_cols, log_norm_cols = compute_style_features(RESPONSES_CSV)
    if features_df.empty:
        print("Unable to compute style features. Exiting.")
        return

    # 3. Train ML model and predict Arena ELO delta
    prediction_df, performance, importance_rows = train_and_predict_delta(
        features_df, per_question_raw_cols, per_question_norm_cols, log_norm_cols
    )
    if prediction_df.empty:
        print("Unable to generate style predictions.")
        return

    # 4. Save all outputs
    save_outputs(features_df, per_question_raw_cols, per_question_norm_cols, prediction_df, performance, importance_rows)


if __name__ == "__main__":
    main()
