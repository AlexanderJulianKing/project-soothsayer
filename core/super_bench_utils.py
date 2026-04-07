"""Shared utilities for TrueSkill super-bench scripts (EQ & Writing).

Deduplicates: judge resolution, TrueSkill column augmentation,
results CSV saving, and judge-name normalization.
"""

import datetime as dt
import os
from typing import Dict, List, Optional

import pandas as pd

from core.utils import discover_openbench_csv, load_models, normalize_reasoning_flag


def resolve_judge_model(script_dir: str, judge_name: str) -> Dict[str, str]:
    """Look up a judge model by display name in the OpenBench CSV.

    Returns dict with keys: name, id, Reasoning.
    """
    model_csv = discover_openbench_csv(script_dir)
    for model in load_models(model_csv):
        if model["name"].strip() == judge_name.strip():
            if not model.get("id"):
                raise ValueError(f"Judge '{judge_name}' is missing an openbench_id.")
            return {
                "name": model["name"],
                "id": model["id"],
                "Reasoning": normalize_reasoning_flag(model.get("Reasoning", False)),
            }
    raise ValueError(f"Judge '{judge_name}' not found in {model_csv}.")


def augment_trueskill_columns(
    df: pd.DataFrame,
    ratings: dict,
    judge_name: str,
    model_col: str = "model",
) -> None:
    """Add TrueSkill rating + sigma columns for *judge_name* to *df* (in-place).

    *model_col* is the column containing model names (e.g. "model" or
    "writer_model").
    """
    rating_col = f"{judge_name} TrueSkill"
    sigma_col = f"{judge_name} Sigma"

    def _stats(model: str) -> pd.Series:
        rating = ratings.get(model)
        if rating is None:
            return pd.Series({rating_col: None, sigma_col: None})
        return pd.Series({rating_col: rating.mu, sigma_col: rating.sigma})

    df[[rating_col, sigma_col]] = df[model_col].apply(_stats)


def save_trueskill_csv(
    df: pd.DataFrame,
    results_dir: str,
    prefix: str,
    sort_by_col: Optional[str] = None,
) -> str:
    """Sort (optionally) and save TrueSkill results to a dated CSV.

    Returns the output path.
    """
    os.makedirs(results_dir, exist_ok=True)
    if sort_by_col and sort_by_col in df.columns:
        df.sort_values(by=sort_by_col, ascending=False, inplace=True, na_position="last")
    out_path = os.path.join(results_dir, f"{prefix}{dt.datetime.now().strftime('%Y%m%d')}.csv")
    df.to_csv(out_path, index=False, float_format="%.4f")
    return out_path


def normalize_judge_names(history_df: pd.DataFrame) -> List[str]:
    """Normalize whitespace in judge_model column, return sorted unique names.

    Returns an empty list when there is no judge_model column or no judges.
    """
    if "judge_model" not in history_df.columns:
        return []
    history_df["judge_model"] = (
        history_df["judge_model"].str.strip().str.replace(r"\s+", " ", regex=True)
    )
    judges = sorted(history_df["judge_model"].dropna().unique())
    return judges
