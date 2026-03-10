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

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.utils import get_latest_file
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

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
LINEAR_MODELS = {'ridge', 'lasso', 'elastic_net', 'bayesian_ridge'}


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
# --- MARKDOWN FEATURE ENGINEERING (from old collect.py) ---
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


def min_max_scale(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(0.0, index=series.index)
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_style_features(responses_csv: str) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """Compute markdown style features from long-format responses.csv.

    Computes per-response stats, then averages across runs per (model, question).
    Normalization and log-transforms applied after averaging.
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

    # Compute per-response markdown stats
    for metric in STYLE_METRICS:
        df[metric] = 0
    for idx, row in df.iterrows():
        stats = calculate_markdown_stats(str(row.get('response', '') or ''))
        for metric in STYLE_METRICS:
            df.at[idx, metric] = stats[metric]

    # Average raw stats across runs per (model_name, question_id)
    per_model_question = df.groupby(['model_name', 'question_id'])[STYLE_METRICS].mean().reset_index()

    # Get model-level code mapping (first occurrence)
    model_codes = df.drop_duplicates(subset=['model_name'])[['model_name', 'model_id']].copy()

    # Pivot to wide format: one row per model, columns like Q1_length, Q1_header_count, etc.
    q_ids = sorted(per_model_question['question_id'].unique())
    feature_rows = []
    for model_name in per_model_question['model_name'].unique():
        model_data = per_model_question[per_model_question['model_name'] == model_name]
        code = model_codes[model_codes['model_name'] == model_name]['model_id'].iloc[0]
        row_dict: Dict[str, Any] = {'model': model_name, 'code': code}

        for q_id in q_ids:
            q_row = model_data[model_data['question_id'] == q_id]
            for metric in STYLE_METRICS:
                col_name = f'Q{q_id}_{metric}'
                row_dict[col_name] = q_row[metric].values[0] if len(q_row) > 0 else 0

        feature_rows.append(row_dict)

    features_df = pd.DataFrame(feature_rows)

    # Compute combined (full-text) stats: average across all questions per model
    combined = per_model_question.groupby('model_name')[STYLE_METRICS].mean().reset_index()
    # Scale combined by number of questions to approximate full-text stats
    n_questions = len(q_ids)
    for metric in STYLE_METRICS:
        combined[f'combined_{metric}'] = combined[metric] * n_questions
    combined = combined.drop(columns=STYLE_METRICS)
    features_df = features_df.merge(combined, left_on='model', right_on='model_name', how='left')
    if 'model_name' in features_df.columns:
        features_df.drop(columns=['model_name'], inplace=True)

    # Build column name lists
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

    # Composite style score
    total_weight = sum(SCORING_WEIGHTS.values())
    features_df['style_score'] = 0.0
    for metric, weight in SCORING_WEIGHTS.items():
        norm_col = f'normalized_{metric}'
        if norm_col in features_df.columns:
            features_df['style_score'] += features_df[norm_col] * (weight / total_weight)

    return features_df, per_question_raw_cols, per_question_norm_cols, log_norm_cols


# ==============================================================================
# --- MACHINE LEARNING (from old collect.py) ---
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


def build_model_zoo() -> Dict[str, Any]:
    ridge = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))])
    lasso = Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.0005, max_iter=10000, random_state=42))])
    elastic_net = Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=42))])
    bayesian_ridge = Pipeline([('scaler', StandardScaler()), ('model', BayesianRidge())])
    svr_rbf = Pipeline([('scaler', StandardScaler()), ('model', SVR(kernel='rbf', C=10.0, epsilon=0.1))])
    knn = Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor(n_neighbors=5))])
    mlp = Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=5000, random_state=42))])
    decision_tree = DecisionTreeRegressor(random_state=42)
    random_forest = RandomForestRegressor(n_estimators=500, random_state=42)
    extra_trees = ExtraTreesRegressor(n_estimators=500, random_state=42)
    gradient_boosting = GradientBoostingRegressor(random_state=42)
    hist_gradient_boosting = HistGradientBoostingRegressor(random_state=42)
    ada_boost = AdaBoostRegressor(random_state=42, n_estimators=400)
    base_models: Dict[str, Any] = {
        'ridge': ridge,
        'lasso': lasso,
        'elastic_net': elastic_net,
        'bayesian_ridge': bayesian_ridge,
        'svr_rbf': svr_rbf,
        'knn': knn,
        'mlp': mlp,
        'decision_tree': decision_tree,
        'random_forest': random_forest,
        'extra_trees': extra_trees,
        'gradient_boosting': gradient_boosting,
        'hist_gradient_boosting': hist_gradient_boosting,
        'ada_boost': ada_boost,
        'dummy_mean': DummyRegressor(strategy='mean'),
    }

    top_for_ensemble = [
        ('bayesian_ridge', base_models['bayesian_ridge']),
        ('hist_gradient_boosting', base_models['hist_gradient_boosting']),
        ('extra_trees', base_models['extra_trees']),
    ]
    base_models['voting_ensemble'] = VotingRegressor(estimators=top_for_ensemble)

    return base_models


def evaluate_models(X: pd.DataFrame, y: pd.Series, cv_splits: int) -> Tuple[str, Dict[str, Any]]:
    models = build_model_zoo()

    if cv_splits < 2:
        return 'ridge', {'ridge': {'mean_rmse': float('nan'), 'std_rmse': float('nan'), 'splits': cv_splits}}

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    performance: Dict[str, Dict[str, float]] = {}
    best_model_name = None
    best_score = float('inf')

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
        rmse_scores = -scores
        performance[name] = {
            'mean_rmse': float(rmse_scores.mean()),
            'std_rmse': float(rmse_scores.std()),
            'splits': cv_splits,
        }
        if performance[name]['mean_rmse'] < best_score:
            best_score = performance[name]['mean_rmse']
            best_model_name = name

    assert best_model_name is not None
    return best_model_name, performance


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

    combined_cols = [f'combined_{metric}' for metric in STYLE_METRICS]
    normalized_cols = [f'normalized_{metric}' for metric in STYLE_METRICS]
    base_feature_cols = per_question_norm_cols + combined_cols + normalized_cols + ['style_score']
    if log_norm_cols:
        base_feature_cols += log_norm_cols
    base_feature_cols = [col for col in base_feature_cols if col in features_with_scores.columns]

    if not base_feature_cols:
        print("No feature columns available for model training.")
        features_with_scores['predicted_delta'] = pd.NA
        features_with_scores['best_model'] = pd.NA
        return features_with_scores, {}, []

    sbf_cols = [c for c in ['logic_weighted_accuracy', 'logic_PC1']
                if c in features_with_scores.columns]
    feature_cols = base_feature_cols + sbf_cols

    importance_rows: List[Dict[str, Any]] = []

    if sbf_cols:
        train_with_sbf = train_df.dropna(subset=sbf_cols)
        train_without_sbf = train_df[train_df[sbf_cols].isna().any(axis=1)]
        n_with = len(train_with_sbf)
        n_without = len(train_without_sbf)
        print(f"Delta training: {n_with} models with logic, {n_without} without.")
    else:
        train_with_sbf = train_df
        train_without_sbf = pd.DataFrame()

    X_train = train_with_sbf[feature_cols].fillna(0.0)
    y_train = train_with_sbf['delta_score']

    cv_splits = min(5, len(X_train))
    best_model_name, performance = evaluate_models(X_train, y_train, cv_splits)

    if best_model_name not in performance:
        print("Unable to evaluate models; skipping prediction.")
        features_with_scores['predicted_delta'] = pd.NA
        features_with_scores['best_model'] = pd.NA
        return features_with_scores, performance, []

    models = build_model_zoo()
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

    # OOF predictions for training models to prevent leakage into downstream pipeline
    kf_oof = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    oof_model = build_model_zoo()[best_model_name]
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
        X_non_train = features_with_scores.loc[non_train_sbf, feature_cols].fillna(0.0)
        features_with_scores.loc[non_train_sbf, 'predicted_delta'] = best_model.predict(X_non_train)

    if sbf_cols and (~has_sbf).any():
        n_fallback = (~has_sbf).sum()
        X_fallback_train = train_df[base_feature_cols].fillna(0.0)
        y_fallback_train = train_df['delta_score']
        fallback_name, _ = evaluate_models(X_fallback_train, y_fallback_train, min(5, len(X_fallback_train)))
        fallback_model = build_model_zoo()[fallback_name]
        fallback_model.fit(X_fallback_train, y_fallback_train)
        # OOF for fallback training rows too
        fallback_train_idx = train_df[train_df[sbf_cols].isna().any(axis=1)].index
        if len(fallback_train_idx) >= 2:
            kf_fb = KFold(n_splits=min(cv_splits, len(fallback_train_idx)), shuffle=True, random_state=42)
            fb_oof_model = build_model_zoo()[fallback_name]
            fb_oof = cross_val_predict(fb_oof_model, X_fallback_train.loc[fallback_train_idx], y_fallback_train.loc[fallback_train_idx], cv=kf_fb)
            features_with_scores.loc[fallback_train_idx, 'predicted_delta'] = fb_oof
        non_train_no_sbf = (~has_sbf) & ~features_with_scores.index.isin(train_df.index)
        if non_train_no_sbf.any():
            X_no_sbf = features_with_scores.loc[non_train_no_sbf, base_feature_cols].fillna(0.0)
            features_with_scores.loc[non_train_no_sbf, 'predicted_delta'] = fallback_model.predict(X_no_sbf)
        print(f"Fallback ({fallback_name}) used for {n_fallback} models without logic.")

    if best_model_name in LINEAR_MODELS:
        model = best_model.named_steps['model']
        coef_series = pd.Series(model.coef_, index=feature_cols)
        importance_rows = [
            {
                'feature': feature,
                'importance': float(value),
                'best_model': best_model_name,
                'type': 'coefficient',
            }
            for feature, value in coef_series.items()
        ]
        importance_rows.append({
            'feature': '__intercept__',
            'importance': float(model.intercept_),
            'best_model': best_model_name,
            'type': 'coefficient_intercept',
        })
    elif hasattr(best_model, 'feature_importances_'):
        importances = getattr(best_model, 'feature_importances_')
        importance_rows = [
            {
                'feature': feature,
                'importance': float(value),
                'best_model': best_model_name,
                'type': 'feature_importance',
            }
            for feature, value in zip(feature_cols, importances)
        ]

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

    # 2. Compute markdown style features from responses
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
