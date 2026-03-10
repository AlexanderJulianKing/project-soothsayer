
import csv
import os
import sys
import warnings

csv.field_size_limit(sys.maxsize)
from collections import defaultdict
import datetime

import numpy as np # type: ignore
import pandas as pd # type: ignore

# Suppress sklearn convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
warnings.filterwarnings("ignore", message=".*Objective did not converge.*")
warnings.filterwarnings("ignore", message=".*optimal value found.*close to.*bound.*")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.linear_model import (
    BayesianRidge, LogisticRegression, Ridge, HuberRegressor, QuantileRegressor,
    Lasso, ElasticNet, ARDRegression, TheilSenRegressor, RANSACRegressor,
    LinearRegression, SGDRegressor,
)
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import (
    HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, RationalQuadratic
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.metrics import make_scorer

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# --- Configuration ---
INPUT_CSV_FILE = 'benchmark_results_multi_run.csv'
QUESTION_COEFFICIENTS_FILE = 'question_coefficients.csv'
DEFAULT_QUESTION_COEFFICIENT = 1.0
TARGETS_FILE = 'combined_all_benches.csv'

current_date = datetime.datetime.now().strftime("%Y%m%d")

OUTPUT_CSV_FILE = 'output/logic_{}.csv'.format(current_date)
TOKEN_MODEL_EXCLUSIONS = {'Nova 2 Lite'}

MODEL_SPECS = {
    'simplebench': {
        'target_col': 'simplebench_Score (AVG@5)',
        'label': 'SimpleBench AVG@5',
        'clip_bounds': (0.0, 100.0),
        'excluded_candidates': ['HistGradientBoosting', 'Ensemble'],
    },
}

N_PRINCIPAL_COMPONENTS = 4

MODEL_CACHE = {}

# Define how judge responses map to categories (case-insensitive check)
# You can adjust this if your judge uses different terms
JUDGE_CATEGORIES = {
    'correct': 'correct',
    'partially correct': 'partially_correct',
    'partially_correct': 'partially_correct',
    'partial': 'partially_correct',
    'incorrect': 'incorrect',
    'wrong': 'incorrect',
    'no answer': 'no_answer',
    'no_answer': 'no_answer',
    'refusal': 'refusal',
    'refused': 'refusal',
    'unsafe': 'unsafe'
}

def normalize_judge_label(label: str) -> str:
    """Map a free-form judge response string into a canonical category."""
    if not label:
        return 'unknown'
    s = str(label).strip().lower()
    # direct hit
    if s in JUDGE_CATEGORIES:
        return JUDGE_CATEGORIES[s]
    # substring/heuristics
    if 'partial' in s:
        return 'partially_correct'
    if 'correct' in s:
        return 'correct'
    if 'wrong' in s or 'incorrect' in s:
        return 'incorrect'
    if 'no' in s and 'answer' in s:
        return 'no_answer'
    if 'refus' in s:
        return 'refusal'
    if 'unsafe' in s:
        return 'unsafe'
    return s  # fall back to raw (so we can see unexpected labels)

def safe_int(x):
    try:
        return int(x)
    except Exception:
        try:
            # some CSVs might contain floats
            return int(float(x))
        except Exception:
            return 0

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def _parse_numeric_value(value):
    """Best-effort conversion of strings like '45.2%' into floats."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return np.nan
        cleaned = cleaned.replace('%', '').replace(',', '')
        try:
            return float(cleaned)
        except ValueError:
            return np.nan
    # Fallback for any other type
    try:
        return float(value)
    except Exception:
        return np.nan


def _extract_linear_coefficients(pipeline, feature_columns):
    if not hasattr(pipeline, 'steps'):
        estimator = pipeline
        scaler = None
    else:
        estimator = pipeline.steps[-1][1]
        scaler = pipeline.named_steps.get('standardscaler') if hasattr(pipeline, 'named_steps') else None

    if not hasattr(estimator, 'coef_'):
        return None

    coef_scaled = np.asarray(estimator.coef_, dtype=float)
    if coef_scaled.ndim > 1:
        if 1 in coef_scaled.shape:
            coef_scaled = coef_scaled.reshape(-1)
        else:
            return None

    intercept = getattr(estimator, 'intercept_', 0.0)
    if isinstance(intercept, (list, tuple, np.ndarray)):
        intercept_arr = np.asarray(intercept, dtype=float)
        if intercept_arr.size != 1:
            return None
        intercept = float(intercept_arr.reshape(-1)[0])
    else:
        intercept = float(intercept)

    if scaler is not None and hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_'):
        scale = np.asarray(scaler.scale_, dtype=float)
        mean = np.asarray(scaler.mean_, dtype=float)
        safe_scale = np.where(scale == 0, 1.0, scale)
        coef_original = coef_scaled / safe_scale
        intercept = intercept - float(np.dot(coef_original, mean))
    else:
        coef_original = coef_scaled

    return {
        'coefficients': {name: float(val) for name, val in zip(feature_columns, coef_original)},
        'intercept': intercept
    }


def _summarize_feature_importance(pipeline, feature_columns):
    if not hasattr(pipeline, 'steps'):
        estimator = pipeline
    else:
        estimator = pipeline.steps[-1][1]

    if hasattr(estimator, 'feature_importances_'):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
        return {
            name: float(val)
            for name, val in zip(feature_columns, importances)
        }
    return None


def _build_column_table(agg):
    """Build per-question score DataFrame from aggregated benchmark results.

    Replaces the standalone generate_column_table.py script by constructing
    the same data in-memory from the already-aggregated results.
    """
    rows = []
    for model_name, a in agg.items():
        row = {'openbench_Model': model_name}
        for qid, score_sum in a['question_score_sum'].items():
            row[str(qid)] = score_sum

        # Token-derived features
        n_tok = max(a['count_with_token_info'], 1)
        avg_reasoning = a['sum_reasoning_tokens'] / n_tok
        avg_output = a['sum_output_tokens'] / n_tok

        row['is_reasoner'] = 1.0 if avg_reasoning > 0 else 0.0
        row['log_reasoning'] = float(np.log1p(avg_reasoning))

        # std of output tokens: sqrt(E[X^2] - E[X]^2)
        mean_sq = a['sum_sq_output_tokens'] / n_tok
        variance = max(mean_sq - avg_output * avg_output, 0.0)
        row['std_output_tokens'] = float(np.sqrt(variance))

        # Targeted interactions
        q10_score = a['question_score_sum'].get('10', 0.0)
        row['q10_x_logr'] = q10_score * float(np.log1p(avg_reasoning))
        total_score = sum(a['question_score_sum'].values())
        row['total_score_sq'] = total_score * total_score

        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def train_benchmark_model(column_table_df, target_col: str, label: str, excluded_candidates: list = None):
    """Train a regression model that predicts a benchmark score from question-level stats.

    Features: per-question columns (1-12) from the in-memory column table.
    Targets: benchmark scores from combined_all_benches.csv, joined on model name.
    """
    info = {
        'pipeline': None,
        'feature_columns': [],
        'best_model_name': None,
        'cv_summary': {},
        'label': label
    }

    if column_table_df is None or column_table_df.empty:
        print(f"Warning: No column table data available. Predictions disabled for {label}.")
        return info

    feature_df_raw = column_table_df.copy()

    # Load target values from combined_all_benches.csv
    if not os.path.exists(TARGETS_FILE):
        print(
            f"Warning: Targets file '{TARGETS_FILE}' not found. "
            f"Predictions disabled for {label}."
        )
        return info

    try:
        targets_df = pd.read_csv(TARGETS_FILE)
    except Exception as exc:
        print(f"Warning: Failed to read targets file '{TARGETS_FILE}': {exc}")
        return info

    if targets_df.empty:
        print(f"Warning: Targets file '{TARGETS_FILE}' is empty. Predictions disabled for {label}.")
        return info

    if target_col not in targets_df.columns:
        print(
            f"Warning: Column '{target_col}' not found in '{TARGETS_FILE}'. "
            f"Predictions disabled for {label}."
        )
        return info

    # Prepare feature data with model names
    if 'openbench_Model' not in feature_df_raw.columns:
        print(f"Warning: 'openbench_Model' column not found in column table.")
        return info

    feature_df_raw['model_name'] = feature_df_raw['openbench_Model'].astype(str).str.strip()
    feature_df_raw = feature_df_raw.dropna(subset=['model_name'])
    feature_df_raw = feature_df_raw.drop_duplicates(subset=['model_name'])
    feature_df_raw = feature_df_raw.set_index('model_name')

    # Prepare target data with model names
    name_col = 'openbench_Model' if 'openbench_Model' in targets_df.columns else 'model_name'
    if name_col not in targets_df.columns:
        print(f"Warning: No model name column found in '{TARGETS_FILE}'.")
        return info

    targets_df['model_name'] = targets_df[name_col].astype(str).str.strip()
    targets_df = targets_df.dropna(subset=['model_name'])
    targets_df = targets_df.drop_duplicates(subset=['model_name'])
    targets_df = targets_df.set_index('model_name')

    # Per-question columns 1-12 plus token-derived features as predictors.
    feature_cols = [str(i) for i in range(1, 13) if str(i) in feature_df_raw.columns]
    for extra in ['is_reasoner', 'log_reasoning', 'std_output_tokens', 'q10_x_logr', 'total_score_sq']:
        if extra in feature_df_raw.columns:
            feature_cols.append(extra)

    if not feature_cols:
        print(
            f"Warning: No feature columns available in column table. "
            f"Predictions disabled for {label}."
        )
        return info

    # Find shared models between features and targets
    shared_models = feature_df_raw.index.intersection(targets_df.index)
    if shared_models.empty:
        print(f"Warning: No overlapping models between features and targets for {label}.")
        return info

    # Extract features and targets for shared models
    feature_df = feature_df_raw.loc[shared_models, feature_cols].apply(
        lambda col: col.apply(_parse_numeric_value)
    )
    target_series = targets_df.loc[shared_models, target_col].apply(_parse_numeric_value)

    # Keep only rows with a target value.
    valid_mask = ~target_series.isna()
    feature_df = feature_df.loc[valid_mask]
    target_series = target_series.loc[valid_mask]

    # Drop rows with any missing feature values
    complete_mask = ~feature_df.isna().any(axis=1)
    feature_df = feature_df.loc[complete_mask]
    target_series = target_series.loc[complete_mask]

    if feature_df.empty:
        print(
            f"Warning: Column table has no usable target values. "
            f"Predictions disabled for {label}."
        )
        return info

    n_samples = len(feature_df)
    n_splits = min(5, n_samples)

    # Build candidate estimators - FAST models first
    candidate_pipelines = {
        'BayesianRidge': make_pipeline(StandardScaler(), BayesianRidge()),
        'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'Ridge_a10': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'Lasso': make_pipeline(StandardScaler(), Lasso(alpha=0.1, random_state=42, max_iter=2000)),
        'ElasticNet': make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)),
        'LinearRegression': make_pipeline(StandardScaler(), LinearRegression()),
        'PolyRidge_d2': make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0)),
        'PolyBayesian_d2': make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), BayesianRidge()),
        'HistGradientBoosting': make_pipeline(HistGradientBoostingRegressor(random_state=42, max_depth=4, max_iter=100)),
    }
    # Add XGBoost if available
    if HAS_XGBOOST:
        candidate_pipelines['XGBoost'] = make_pipeline(XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0))

    # SLOW models - additional options
    slow_pipelines = {
        'HuberRegressor': make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, max_iter=500)),
        'ARDRegression': make_pipeline(StandardScaler(), ARDRegression()),
        'PolyElasticNet_d2': make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)),
        'PolyLasso_d2': make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), Lasso(alpha=0.1, random_state=42, max_iter=2000)),
        'GP_Linear': make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(), random_state=42, n_restarts_optimizer=2)),
        'GP_Poly': make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=DotProduct(sigma_0=1.0) ** 2 + WhiteKernel(), random_state=42, n_restarts_optimizer=2)),
        'MLP_small': make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,), activation='relu', solver='adam', max_iter=500, early_stopping=True, random_state=42)),
        'MLP_medium': make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, early_stopping=True, random_state=42)),
        'LinearSVR': make_pipeline(StandardScaler(), LinearSVR(C=1.0, random_state=42, max_iter=2000)),
        'SVR_Linear': make_pipeline(StandardScaler(), SVR(kernel="linear", C=10.0)),
        'SVR_RBF': make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0, gamma="scale")),
        'ExtraTrees': make_pipeline(ExtraTreesRegressor(n_estimators=200, random_state=42)),
        'LinearEnsemble': make_pipeline(VotingRegressor([
            ('bayesian', make_pipeline(StandardScaler(), BayesianRidge())),
            ('ridge', make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ])),
        'TheilSen': make_pipeline(StandardScaler(), TheilSenRegressor(random_state=42)),
        'RANSAC': make_pipeline(StandardScaler(), RANSACRegressor(estimator=LinearRegression(), min_samples=0.5, random_state=42)),
        'SGD': make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)),
        'RandomForest': make_pipeline(RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)),
        'AdaBoost': make_pipeline(AdaBoostRegressor(n_estimators=100, random_state=42)),
        'Bagging': make_pipeline(BaggingRegressor(n_estimators=50, random_state=42)),
        'KNeighbors': make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        'GP_RBF': make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), random_state=42, n_restarts_optimizer=2)),
        'GP_Matern': make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=Matern() + WhiteKernel(), random_state=42, n_restarts_optimizer=2)),
        'GP_RationalQuadratic': make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=RationalQuadratic() + WhiteKernel(), random_state=42, n_restarts_optimizer=2)),
        'QuantileRegressor': make_pipeline(StandardScaler(), QuantileRegressor(quantile=0.5, solver="highs")),
    }
    # Add XGBoost-based slow models
    if HAS_XGBOOST:
        slow_pipelines['XGBoost_deep'] = make_pipeline(XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0))
        slow_pipelines['Ensemble'] = make_pipeline(VotingRegressor([
            ('linear', make_pipeline(StandardScaler(), BayesianRidge())),
            ('xgb', XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)),
        ]))
    else:
        slow_pipelines['Ensemble'] = make_pipeline(VotingRegressor([
            ('linear', make_pipeline(StandardScaler(), BayesianRidge())),
            ('hgb', HistGradientBoostingRegressor(random_state=42, max_depth=4, max_iter=100)),
        ]))

    baseline_pipeline = make_pipeline(DummyRegressor())

    if n_splits < 2:
        # Not enough data to cross-validate; fall back to baseline fit.
        baseline_pipeline.fit(feature_df, target_series)
        info.update({
            'pipeline': baseline_pipeline,
            'feature_columns': feature_cols,
            'best_model_name': 'DummyRegressor',
            'cv_summary': {
                'n_samples': n_samples,
                'mean_rmse': None,
                'std_rmse': None
            }
        })
        return info

    candidate_pipelines['DummyRegressor'] = baseline_pipeline

    if n_samples >= 30:
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    def weighted_rmse(y_true, y_pred):
        # Slightly upweight higher-score samples so tail fit matters more.
        weights = 1.0 + (np.asarray(y_true, dtype=float) / 50.0)
        return np.sqrt(np.average((y_pred - y_true) ** 2, weights=weights))

    scorer = make_scorer(weighted_rmse, greater_is_better=False)
    best_name = None
    best_score = -np.inf
    best_scores = None
    best_pipeline = None

    # Run fast models first
    for name, pipeline in candidate_pipelines.items():
        if excluded_candidates and name in excluded_candidates:
            continue
        try:
            scores = cross_val_score(
                pipeline,
                feature_df,
                target_series,
                cv=cv,
                scoring=scorer
            )
        except Exception as exc:
            print(f"Warning: Model '{name}' failed during cross-validation: {exc}")
            continue

        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_scores = scores
            best_name = name
            best_pipeline = pipeline

    # Run slow models too (they may perform better)
    for name, pipeline in slow_pipelines.items():
        if excluded_candidates and name in excluded_candidates:
            continue
        try:
            scores = cross_val_score(
                pipeline,
                feature_df,
                target_series,
                cv=cv,
                scoring=scorer
            )
        except Exception as exc:
            # Silently skip slow model failures
            continue

        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_scores = scores
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None:
        # All candidates failed; use baseline.
        baseline_pipeline.fit(feature_df, target_series)
        info.update({
            'pipeline': baseline_pipeline,
            'feature_columns': feature_cols,
            'best_model_name': 'DummyRegressor',
            'cv_summary': {
                'n_samples': n_samples,
                'mean_rmse': None,
                'std_rmse': None
            }
        })
        return info

    # Compute honest out-of-sample predictions via LOO before final refit
    loo_predictions = {}
    try:
        loo_cv = LeaveOneOut()
        loo_preds = cross_val_predict(best_pipeline, feature_df, target_series, cv=loo_cv)
        for model_name, pred in zip(feature_df.index, loo_preds):
            loo_predictions[model_name] = float(pred)
    except Exception as exc:
        print(f"Warning: LOO cross_val_predict failed: {exc}")

    # Fit the best candidate on the full dataset
    best_pipeline.fit(feature_df, target_series)

    rmse_scores = -best_scores  # convert to positive RMSE values
    train_r_value = None
    coef_report = None
    importance_report = None

    try:
        train_preds = best_pipeline.predict(feature_df)
        corr_mat = np.corrcoef(train_preds, target_series)
        if corr_mat.shape == (2, 2):
            train_r_value = float(corr_mat[0, 1])
    except Exception as exc:
        print(f"Warning: Could not compute training correlation: {exc}")

    try:
        coef_report = _extract_linear_coefficients(best_pipeline, feature_cols)
    except Exception:
        coef_report = None
    if coef_report is None:
        importance_report = _summarize_feature_importance(best_pipeline, feature_cols)

    info.update({
        'pipeline': best_pipeline,
        'feature_columns': feature_cols,
        'best_model_name': best_name,
        'cv_summary': {
            'n_samples': n_samples,
            'mean_rmse': float(rmse_scores.mean()),
            'std_rmse': float(rmse_scores.std(ddof=1)) if len(rmse_scores) > 1 else 0.0
        },
        'train_r_value': train_r_value,
        'coefficients': coef_report,
        'feature_importances': importance_report,
        'label': label,
        'loo_predictions': loo_predictions
    })

    print(
        f"Selected '{best_name}' for {label} prediction "
        f"(CV RMSE ~ {info['cv_summary']['mean_rmse']:.3f}, n={n_samples})."
    )

    if train_r_value is not None:
        print(f"{label}: Training R (Pearson) ~ {train_r_value:.3f}")

    if coef_report is not None:
        print(f"{label}: Model coefficients (original feature scale):")
        for name, value in coef_report['coefficients'].items():
            print(f"  {name}: {value:+.4f}")
        print(f"  intercept: {coef_report['intercept']:+.4f}")
    elif importance_report is not None:
        print(f"{label}: Feature importances:")
        for name, value in sorted(importance_report.items(), key=lambda kv: -abs(kv[1])):
            print(f"  {name}: {value:+.4f}")

    return info


def get_benchmark_model(model_key: str, column_table_df):
    """Load or train the cached prediction model for the specified benchmark."""
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model key '{model_key}'.")

    if model_key not in MODEL_CACHE:
        spec = MODEL_SPECS[model_key]
        MODEL_CACHE[model_key] = train_benchmark_model(
            column_table_df,
            spec['target_col'],
            spec['label'],
            excluded_candidates=spec.get('excluded_candidates')
        )
        MODEL_CACHE[model_key]['clip_bounds'] = spec.get('clip_bounds')

    return MODEL_CACHE[model_key]


def build_feature_vector(agg_entry, overall_accuracy):
    """Derive per-question feature vector for a model's aggregated responses."""
    features = {'logic_accuracy': overall_accuracy}

    score_sums = agg_entry.get('question_score_sum', {})
    attempts = agg_entry.get('question_attempts', {})
    for question_id, summed_score in score_sums.items():
        n = attempts.get(question_id, 1) or 1
        # Normalize to the 0-4 scale used in training data (N_RUNS=4)
        features[str(question_id)] = float(summed_score) * 4.0 / n

    # Token-derived features (match _build_column_table)
    n_tok = max(agg_entry.get('count_with_token_info', 1), 1)
    avg_reasoning = agg_entry.get('sum_reasoning_tokens', 0) / n_tok
    avg_output = agg_entry.get('sum_output_tokens', 0) / n_tok

    features['is_reasoner'] = 1.0 if avg_reasoning > 0 else 0.0
    features['log_reasoning'] = float(np.log1p(avg_reasoning))

    mean_sq = agg_entry.get('sum_sq_output_tokens', 0) / n_tok
    variance = max(mean_sq - avg_output * avg_output, 0.0)
    features['std_output_tokens'] = float(np.sqrt(variance))

    # Targeted interactions (match _build_column_table)
    score_sums_raw = agg_entry.get('question_score_sum', {})
    q10_score = score_sums_raw.get('10', 0.0)
    features['q10_x_logr'] = q10_score * float(np.log1p(avg_reasoning))
    total_score = sum(score_sums_raw.values())
    features['total_score_sq'] = total_score * total_score

    return features


def compute_principal_components(feature_rows, n_components=N_PRINCIPAL_COMPONENTS):
    """Extract principal components from per-question feature vectors.

    Returns a list of dicts with keys PC1, PC2, ..., PCn (one dict per model),
    plus prints the explained variance ratio.
    """
    feature_df = pd.DataFrame(feature_rows)
    # Use per-question columns only (numeric IDs like '1', '2', ... '12')
    question_cols = sorted(
        [c for c in feature_df.columns if c.isdigit()],
        key=lambda x: int(x),
    )
    if not question_cols:
        print("Warning: No question columns found for PCA.")
        return [{}] * len(feature_rows)

    X = feature_df[question_cols].fillna(0.0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=k, random_state=42)
    components = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    print(f"PCA on {len(question_cols)} question features → {k} components:")
    for i, ev in enumerate(explained):
        print(f"  PC{i+1}: {ev:.1%} variance explained")
    print(f"  Total: {sum(explained):.1%}")

    # Save PC loadings to CSV
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=question_cols,
        columns=[f'PC{i+1}' for i in range(k)],
    )
    loadings_df.index.name = 'question'
    # Append explained variance as a summary row
    var_row = pd.DataFrame(
        [explained],
        index=['explained_variance_ratio'],
        columns=loadings_df.columns,
    )
    loadings_out = pd.concat([loadings_df, var_row])
    loadings_path = f'output/pc_loadings_{current_date}.csv'
    ensure_output_dir(loadings_path)
    loadings_out.to_csv(loadings_path)
    print(f"  PC loadings written to '{loadings_path}'")

    results = []
    for row_idx in range(len(feature_rows)):
        d = {}
        for pc_idx in range(k):
            d[f'PC{pc_idx+1}'] = round(float(components[row_idx, pc_idx]), 6)
        results.append(d)
    return results


def predict_benchmark_scores(model_key: str, feature_rows, column_table_df):
    """Predict benchmark scores using the trained regression model."""
    model_info = get_benchmark_model(model_key, column_table_df)
    pipeline = model_info.get('pipeline')
    feature_columns = model_info.get('feature_columns', [])
    clip_bounds = model_info.get('clip_bounds')

    if pipeline is None or not feature_columns:
        return [None] * len(feature_rows)

    try:
        feature_df = pd.DataFrame(feature_rows)
    except Exception as exc:
        print(f"Warning: Failed to assemble prediction features for '{model_key}': {exc}")
        return [None] * len(feature_rows)

    for col in feature_columns:
        if col not in feature_df.columns:
            # Treat unanswered questions as zero contribution rather than missing
            feature_df[col] = 0.0

    feature_df = feature_df[feature_columns]

    missing_mask = feature_df.isna().any(axis=1)
    predictions = [None] * len(feature_df)

    if (~missing_mask).any():
        try:
            valid_predictions = pipeline.predict(feature_df.loc[~missing_mask])
        except Exception as exc:
            print(f"Warning: Failed to generate '{model_key}' predictions: {exc}")
            return [None] * len(feature_rows)

        for idx, value in zip(feature_df.index[~missing_mask], valid_predictions):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            clipped = float(value)
            if clip_bounds:
                lower, upper = clip_bounds
                if lower is not None and clipped < lower:
                    clipped = lower
                if upper is not None and clipped > upper:
                    clipped = upper
            predictions[idx] = clipped

    return predictions


def load_question_coefficients(path):
    """Load per-question weights from a CSV file keyed by question_id."""
    coeffs = {}
    if not path or not os.path.exists(path):
        return coeffs

    try:
        with open(path, 'r', newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return coeffs

            # Normalize headers and allow a couple common aliases
            normalized = {name.strip().lower(): name for name in reader.fieldnames}
            q_col = normalized.get('question_id') or normalized.get('questionid')
            c_col = normalized.get('coefficient') or normalized.get('weight')
            if not q_col or not c_col:
                print(f"Warning: Coefficient file '{path}' is missing required headers.")
                return coeffs

            for row in reader:
                raw_qid = (row.get(q_col) or '').strip()
                raw_coeff = (row.get(c_col) or '').strip()
                if not raw_qid or not raw_coeff:
                    continue
                try:
                    coeffs[str(raw_qid)] = float(raw_coeff)
                except ValueError:
                    continue
    except (IOError, csv.Error) as exc:
        print(f"Warning: Could not read question coefficients from '{path}': {exc}")

    return coeffs


def load_actual_simplebench_scores(path):
    """Return a mapping of model name to actual SimpleBench AVG@5 scores."""
    scores = {}
    if not path or not os.path.exists(path):
        return scores

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: Failed to read actual SimpleBench scores from '{path}': {exc}")
        return scores

    if df.empty:
        return scores

    normalized = {col.strip().lower(): col for col in df.columns}
    name_col = normalized.get('openbench_model') or normalized.get('model')
    score_col = normalized.get('simplebench_score (avg@5)')

    if not name_col or not score_col:
        return scores

    for _, row in df[[name_col, score_col]].dropna(how='all').iterrows():
        name = str(row[name_col]).strip()
        if not name:
            continue
        value = _parse_numeric_value(row[score_col])
        if value is None or np.isnan(value):
            continue
        scores[name] = float(value)

    return scores


def report_holdout_metrics(rows):
    baseline_pairs = []
    predicted_pairs = []

    for row in rows:
        actual_val = row.get('simplebench_actual_score')
        if actual_val is None:
            continue
        act = float(actual_val)

        baseline_val = row.get('run_weighted_accuracy')
        if baseline_val is not None:
            baseline_pairs.append((float(baseline_val), act))

        predicted_val = row.get('weighted_accuracy')
        if predicted_val is not None:
            predicted_pairs.append((float(predicted_val), act))

    def _summarize(label, pairs):
        if not pairs:
            return None
        est = np.array([p[0] for p in pairs], dtype=float)
        act = np.array([p[1] for p in pairs], dtype=float)
        if est.size == 0:
            return None
        diff = est - act
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        bias = float(np.mean(diff))
        corr = float(np.corrcoef(est, act)[0, 1]) if est.size > 1 else float('nan')
        return label, est.size, mae, rmse, bias, corr

    summaries = [
        _summarize('Run-weighted', baseline_pairs),
        _summarize('Model prediction', predicted_pairs),
    ]

    summaries = [s for s in summaries if s]
    if not summaries:
        return

    n_overlap = max(s[1] for s in summaries)
    print(f"Hold-out evaluation on {n_overlap} overlapping models:")
    for label, _, mae, rmse, bias, corr in summaries:
        print(
            f"  {label}: MAE={mae:.3f}, RMSE={rmse:.3f}, Bias={bias:+.3f}, PearsonR={corr:.3f}"
        )


def calculate_scores(input_filename):
    """
    Read multi-run benchmark results and aggregate scores & token stats per (model_name, model_id).
    
    Expected CSV columns (latest as of 2025-08-13):
    question_id,prompt,reference_answer,model_name,model_id,run_number,model_response,
    judge_model_id,judge_response,model_output_tokens,model_reasoning_tokens,judge_output_tokens,judge_reasoning_tokens
    """
    if not os.path.exists(input_filename):
        print(f"Input file '{input_filename}' not found.")
        return []

    question_coefficients = load_question_coefficients(QUESTION_COEFFICIENTS_FILE)
    if not question_coefficients:
        print(
            f"Note: No question coefficients loaded from '{QUESTION_COEFFICIENTS_FILE}'. "
            "Weighted accuracy will default to standard accuracy."
        )
    else:
        print(f"Loaded coefficients for {len(question_coefficients)} questions.")

    # Per model aggregates
    agg = defaultdict(lambda: {
        'model_name': None,
        'model_id': None,
        'total': 0,
        'correct': 0,
        'partially_correct': 0,
        'incorrect': 0,
        'no_answer': 0,
        'refusal': 0,
        'unsafe': 0,
        'other_labels': defaultdict(int),
        # token sums (we will convert to averages at the end)
        'sum_output_tokens': 0,
        'sum_reasoning_tokens': 0,
        'sum_answer_tokens': 0,
        'sum_sq_output_tokens': 0,
        'count_with_token_info': 0,
        # weighted accuracy tracking
        'weighted_score_sum': 0.0,
        'weighted_abs_sum': 0.0,
        # per-question statistics
        'question_score_sum': defaultdict(float),
        'question_attempts': defaultdict(int),
    })

    with open(input_filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Normalize expected fieldnames
        fieldnames = [c.strip() for c in reader.fieldnames] if reader.fieldnames else []
        # Token column names (robustly handle variants just in case)
        col_output = next((c for c in fieldnames if c.lower() in {'model_output_tokens','output_tokens','tokens_out'}), None)
        col_reason = next((c for c in fieldnames if c.lower() in {'model_reasoning_tokens','reasoning_tokens'}), None)
        for row in reader:
            model_name = (row.get('model_name') or row.get('model') or 'UNKNOWN_MODEL').strip()
            model_id = row.get('model_id') or ''
            key = model_name  # collapse any changing model_ids under the same name

            a = agg[key]
            a['model_name'] = model_name
            # keep the first non-empty id we see for reference
            if not a['model_id'] and model_id:
                a['model_id'] = model_id
            a['total'] += 1

            # Judge label
            raw_label = row.get('judge_response', '')
            norm = normalize_judge_label(raw_label)
            if norm in a:
                a[norm] += 1
            else:
                a['other_labels'][norm] += 1

            # Weighted accuracy contribution
            question_id = str(row.get('question_id') or '').strip()
            weight = question_coefficients.get(question_id, DEFAULT_QUESTION_COEFFICIENT)
            # Map normalized label to a numeric score (same semantics as accuracy calc)
            if norm == 'correct':
                score_value = 1.0
            elif norm == 'partially_correct':
                score_value = 0.5
            else:
                score_value = 0.0

            a['weighted_score_sum'] += weight * score_value
            a['weighted_abs_sum'] += abs(weight)

            if question_id:
                a['question_score_sum'][question_id] += score_value
                a['question_attempts'][question_id] += 1

            # Token accounting
            output_tokens = safe_int(row.get(col_output, 0)) if col_output else 0
            reasoning_tokens = safe_int(row.get(col_reason, 0)) if col_reason else 0
            # Answer tokens are output - reasoning (guaranteed non-negative)
            answer_tokens = max(output_tokens - reasoning_tokens, 0)

            if col_output:
                a['sum_output_tokens'] += output_tokens
                a['sum_sq_output_tokens'] += output_tokens * output_tokens
                a['count_with_token_info'] += 1
            if col_reason:
                a['sum_reasoning_tokens'] += reasoning_tokens
            # Always accumulate answer estimate if we had output (even if reasoning was 0/missing)
            if col_output:
                a['sum_answer_tokens'] += answer_tokens

    # Build rows
    rows = []
    feature_rows = []
    MIN_QUESTIONS = 36  # ignore models with too few answered questions
    for model_name, a in agg.items():
        total = max(a['total'], 1)
        if total < MIN_QUESTIONS:
            continue

        display_name = model_name
        if model_name == 'Gemini 3.0 Pro Preview (2025-11-18)':
            display_name = model_name + ' '
        correct = a['correct']
        partial = a['partially_correct']

        # Simple accuracy with partial credit = 0.5
        acc = (correct + 0.5 * partial) / total

        weight_denom = a['weighted_abs_sum']
        weighted_acc = (a['weighted_score_sum'] / weight_denom) if weight_denom else 0.0
        baseline_weighted_percent = round(weighted_acc * 100.0, 4)

        # Compute averages (guard divide-by-zero)
        n_tok = max(a['count_with_token_info'], 1)
        avg_output = a['sum_output_tokens'] / n_tok
        avg_reason = a['sum_reasoning_tokens'] / n_tok
        avg_answer = a['sum_answer_tokens'] / n_tok

        row = {
            'model_name': display_name,
            'model_id': model_id,
            'total': total,
            'correct': correct,
            'partially_correct': partial,
            'incorrect': a['incorrect'],
            'no_answer': a['no_answer'],
            'refusal': a['refusal'],
            'unsafe': a['unsafe'],
            'accuracy': round(acc, 4),
            'weighted_accuracy': None,
            'avg_output_tokens': round(avg_output, 2),
            'avg_reasoning_tokens': round(avg_reason, 2),
            'avg_answer_tokens': round(avg_answer, 2),
        }
        # PC columns will be filled in after PCA is computed
        for pc_i in range(1, N_PRINCIPAL_COMPONENTS + 1):
            row[f'PC{pc_i}'] = None
        if display_name in TOKEN_MODEL_EXCLUSIONS:
            row['avg_output_tokens'] = 'NA'
            row['avg_reasoning_tokens'] = 'NA'
            row['avg_answer_tokens'] = 'NA'
        row['_baseline_weighted_accuracy'] = baseline_weighted_percent

        feature_rows.append(build_feature_vector(a, acc))
        # Include unexpected labels as separate columns if any (helps debugging)
        if a['other_labels']:
            for k, v in a['other_labels'].items():
                row[f'label_{k}'] = v

        rows.append(row)

    # Build column table in-memory for regression training
    column_table_df = _build_column_table(agg)

    # Compute weighted_accuracy (predicted simplebench score)
    def _compute_predictions(model_key):
        try:
            preds = predict_benchmark_scores(model_key, feature_rows, column_table_df)
        except Exception as exc:
            print(f"Warning: Could not compute {model_key} predictions: {exc}")
            preds = [None] * len(rows)
        if len(preds) < len(rows):
            preds = list(preds) + [None] * (len(rows) - len(preds))
        return preds

    simple_predictions = _compute_predictions('simplebench')

    # Get LOO predictions and actual scores for holdout evaluation
    model_info = MODEL_CACHE.get('simplebench', {})
    loo_preds = model_info.get('loo_predictions', {})
    actual_scores = load_actual_simplebench_scores(TARGETS_FILE)

    for idx, row in enumerate(rows):
        simple_pred = simple_predictions[idx] if idx < len(simple_predictions) else None
        model_name = row['model_name']

        # Use LOO prediction for training-set models (honest OOS estimate)
        loo_pred = loo_preds.get(model_name)
        if loo_pred is not None:
            clip_bounds = model_info.get('clip_bounds')
            clipped = float(loo_pred)
            if clip_bounds:
                clipped = max(clip_bounds[0], min(clip_bounds[1], clipped))
            raw_pred = clipped
        elif simple_pred is not None:
            raw_pred = float(simple_pred)
        else:
            raw_pred = None

        # Conditional blend: for high-accuracy reasoners, blend model prediction
        # with raw accuracy (the accuracy→SimpleBench relationship is tighter
        # for strong reasoners, so trust the raw signal more).
        acc_pct = row['accuracy'] * 100.0
        feat = feature_rows[idx] if idx < len(feature_rows) else {}
        is_reasoner = feat.get('is_reasoner', 0.0)
        if raw_pred is not None and is_reasoner and acc_pct > 60.0:
            raw_pred = 0.4 * raw_pred + 0.6 * acc_pct

        if raw_pred is not None:
            row['weighted_accuracy'] = round(raw_pred, 4)
        else:
            row['weighted_accuracy'] = row['_baseline_weighted_accuracy']

        # Wire up holdout metrics keys
        actual = actual_scores.get(model_name)
        if actual is not None:
            row['simplebench_actual_score'] = actual
            row['run_weighted_accuracy'] = row['_baseline_weighted_accuracy']

    # Compute principal components from per-question features
    pc_results = compute_principal_components(feature_rows)
    for idx, row in enumerate(rows):
        pc_dict = pc_results[idx] if idx < len(pc_results) else {}
        for key, val in pc_dict.items():
            row[key] = val

    report_holdout_metrics(rows)

    # Sort by accuracy desc, then total desc
    rows.sort(key=lambda r: (-r['accuracy'], -r['total'], r['model_name']))

    return rows

def ensure_output_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def write_scores_csv(output_filename, score_data):
    """Write the aggregated scores to a CSV file with stable column order."""
    if not score_data:
        print("No score data to write.")
        return
    ensure_output_dir(output_filename)

    # Collect all dynamic headers (e.g., label_foo)
    dynamic_keys = set()
    for r in score_data:
        dynamic_keys.update(k for k in r.keys() if k.startswith('label_'))
    dynamic_keys = sorted(dynamic_keys)

    # Stable header order
    headers = [
        'model_name',
        'model_id',
        'total',
        'correct',
        'partially_correct',
        'incorrect',
        'no_answer',
        'refusal',
        'unsafe',
        'accuracy',
        'weighted_accuracy',
    ] + [f'PC{i}' for i in range(1, N_PRINCIPAL_COMPONENTS + 1)] + [
        'avg_output_tokens',
        'avg_reasoning_tokens',
        'avg_answer_tokens',
    ] + dynamic_keys

    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as out:
            writer = csv.DictWriter(out, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(score_data)
        print(f"Successfully wrote scores for {len(score_data)} models to '{output_filename}'")
    except IOError as e:
        print(f"Error writing output file '{output_filename}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing the output CSV: {e}")

def main():
    """Main function to orchestrate reading and writing."""
    calculated_scores = calculate_scores(INPUT_CSV_FILE)
    if calculated_scores:
        write_scores_csv(OUTPUT_CSV_FILE, calculated_scores)

if __name__ == "__main__":
    main()
