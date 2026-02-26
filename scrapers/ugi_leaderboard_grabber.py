#!/usr/bin/env python3
"""Download UGI leaderboard writing scores."""
import datetime
import io
import os

import pandas as pd
import requests


SOURCE_URL = "https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard/resolve/main/ugi-leaderboard-data.csv"
AUTHOR_COL = "author/model_name"
WRITING_COL = "Writing"
WRITING_SOURCE_CANDIDATES = ["Writing \u270d\ufe0f", "Writing"]
OUT_DIR = "benchmarks"


def today_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fetch_leaderboard() -> pd.DataFrame:
    response = requests.get(SOURCE_URL, timeout=30)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    writing_col = next((c for c in WRITING_SOURCE_CANDIDATES if c in df.columns), None)
    if writing_col is None or AUTHOR_COL not in df.columns:
        missing_cols = [col for col in [AUTHOR_COL, WRITING_SOURCE_CANDIDATES[0]] if col not in df.columns]
        raise RuntimeError(f"Missing expected columns: {', '.join(missing_cols)}")

    filtered = df
    if "Is Finetuned" in filtered.columns:
        filtered = filtered[filtered["Is Finetuned"].fillna(False) == False]
    if writing_col != WRITING_COL:
        filtered = filtered.rename(columns={writing_col: WRITING_COL})
    return filtered[[AUTHOR_COL, WRITING_COL]].copy()


def save_csv(df: pd.DataFrame) -> str:
    ensure_dir(OUT_DIR)
    out_path = os.path.join(OUT_DIR, f"UGI_Leaderboard_{today_stamp()}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    return out_path


def main() -> None:
    df = fetch_leaderboard()
    trimmed = select_columns(df)
    save_csv(trimmed)


if __name__ == "__main__":
    main()
