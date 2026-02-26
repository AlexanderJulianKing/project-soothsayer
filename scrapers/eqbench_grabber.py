# eqbench3_to_csv requests version
import io
import os
import re
from datetime import datetime

import pandas as pd
import requests

EQBENCH_DATA_JS = "https://eqbench.com/eqbench3.js?v=latest"
CREATIVE_DATA_JS = "https://eqbench.com/creative_writing.js?v=latest"

# Manual aliases for models that are spelled differently between leaderboards
RAW_MODEL_ALIASES = {
    "gemma 3 12b": "google/gemma-3-12b-it",
    "claude 3.5 haiku": "anthropic/claude-3.5-haiku-20241022",
    "gpt-4.5 preview": "gpt-4.5-preview-2025-02-27",
    "gpt 4.5 preview": "gpt-4.5-preview-2025-02-27",
    "gpt-4.5-preview": "gpt-4.5-preview-2025-02-27",
    "gemini 2.5 pro preview (2025-03-25)": "gemini-2.5-pro-preview-03-25",
    "gemini 2.5 pro preview 2025-03-25": "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-preview-2025-03-25": "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-exp-03-25": "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-preview-03-25": "gemini-2.5-pro-preview-03-25",
    "llama 3.1 405b instruct turbo": "meta-llama/Llama-3.1-405B-Instruct",
    "llama 3.1 405b instruct": "meta-llama/Llama-3.1-405B-Instruct",
    "mistral large": "mistralai/Mistral-Large-Instruct-2411",
    "mistralai/mistral-large-2411": "mistralai/Mistral-Large-Instruct-2411",
}


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


MODEL_ALIAS_LOOKUP = {
    _normalize_key(variant): canonical.strip()
    for variant, canonical in RAW_MODEL_ALIASES.items()
    if variant and canonical
}

# Remember canonical choices per normalized key so both tables collapse to the same label.
_CANONICAL_BY_KEY = {}


def canonicalize_model_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    cleaned = name.strip()
    if not cleaned:
        return cleaned
    key = _normalize_key(cleaned)
    canonical = MODEL_ALIAS_LOOKUP.get(key)
    if canonical is None:
        canonical = _CANONICAL_BY_KEY.get(key, cleaned)
    _CANONICAL_BY_KEY.setdefault(key, canonical)
    return canonical


def today_stamp() -> str:
    return datetime.now().strftime("%Y%m%d")  # adjust to your timezone if needed


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def extract_embedded_csv(js_text: str, variable_name: str) -> str:
    """Return the CSV payload between backticks for the given JS variable."""
    pattern = rf"{re.escape(variable_name)}\s*=\s*`(?P<csv>[^`]*)`"
    match = re.search(pattern, js_text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find CSV payload for {variable_name}.")
    return match.group("csv").strip()


def fetch_leaderboard_dataframe(js_url: str, variable_name: str) -> pd.DataFrame:
    resp = requests.get(js_url, timeout=30)
    resp.raise_for_status()
    csv_text = extract_embedded_csv(resp.text, variable_name)
    return pd.read_csv(io.StringIO(csv_text))


def load_eqbench() -> pd.DataFrame:
    df = fetch_leaderboard_dataframe(EQBENCH_DATA_JS, "leaderboardDataEQBench3")
    df = df.rename(
        columns={
            "model_name": "model",
            "elo_norm": "eq_elo",
            "rubric_0_100": "eq_rubric_score",
            "humanlike": "eq_humanlike",
            "safe": "eq_safe",
            "assertive": "eq_assertive",
            "social_iq": "eq_social_iq",
            "warm": "eq_warm",
            "analytical": "eq_analytical",
            "insightful": "eq_insightful",
            "empathy": "eq_empathy",
            "compliant": "eq_compliant",
            "moral": "eq_moral",
            "pragmatic": "eq_pragmatic",
            "ci_low_norm": "eq_elo_ci_low",
            "ci_high_norm": "eq_elo_ci_high",
        }
    )
    df["model"] = df["model"].apply(canonicalize_model_name)
    return df


def load_creative_writing() -> pd.DataFrame:
    resp = requests.get(CREATIVE_DATA_JS, timeout=30)
    resp.raise_for_status()
    csv_text = extract_embedded_csv(resp.text, "leaderboardDataCreativeWritingV3")
    df = pd.read_csv(
        io.StringIO(csv_text),
        header=None,
        names=[
            "model",
            "creative_elo",
            "creative_rubric_score",
            "creative_length",
            "creative_vocab_complexity",
            "creative_slop_score",
            "creative_repetition_score",
        ],
    )
    df["model"] = df["model"].str.lstrip("*")
    df["model"] = df["model"].apply(canonicalize_model_name)
    return df


def merge_benchmarks(eq_df: pd.DataFrame, creative_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.merge(eq_df, creative_df, on="model", how="outer")
    desired_order = [
        "model",
        "eq_elo",
        "eq_elo_ci_low",
        "eq_elo_ci_high",
        "eq_rubric_score",
        "eq_humanlike",
        "eq_safe",
        "eq_assertive",
        "eq_social_iq",
        "eq_warm",
        "eq_analytical",
        "eq_insightful",
        "eq_empathy",
        "eq_compliant",
        "eq_moral",
        "eq_pragmatic",
        "creative_elo",
        "creative_rubric_score",
        "creative_length",
        "creative_slop_score",
        "creative_repetition_score",
        "creative_vocab_complexity",
    ]
    existing_cols = [c for c in desired_order if c in combined.columns]
    remaining_cols = [c for c in combined.columns if c not in existing_cols]
    return combined[existing_cols + remaining_cols]


def save_csv(df: pd.DataFrame, outdir="benchmarks"):
    ensure_dir(outdir)
    outpath = os.path.join(outdir, f"EQ-Bench_combined_{today_stamp()}.csv")
    df.to_csv(outpath, index=False)
    print(f"Saved {len(df)} rows to {outpath}")


def main():
    eq_df = load_eqbench()
    creative_df = load_creative_writing()
    combined_df = merge_benchmarks(eq_df, creative_df)
    save_csv(combined_df)


if __name__ == "__main__":
    main()
