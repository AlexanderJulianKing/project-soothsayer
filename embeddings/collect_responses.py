"""Walk the 4 custom-benchmark response sources and emit one long-format parquet.

Schema: (model, benchmark, prompt_id, run_id, response_text, text_len)

Strips common reasoning-block wrappers (<think>, <thinking>, <reasoning>) so the
embedded text reflects what a judge would see. Skips empty and errored responses.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EQ_DIR = PROJECT_ROOT / "soothsayer_eq" / "generated_responses"
WRITING_DIR = PROJECT_ROOT / "soothsayer_writing" / "generated_stories"
LOGIC_CSV = PROJECT_ROOT / "soothsayer_logic" / "benchmark_results_multi_run.csv"
STYLE_CSV = PROJECT_ROOT / "soothsayer_style" / "responses.csv"

OUT_DIR = PROJECT_ROOT / "embeddings" / "cache"
OUT_FILE = OUT_DIR / "all_responses.parquet"

REASONING_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
]


def strip_reasoning(text) -> str:
    if not isinstance(text, str):
        return ""
    out = text
    for pat in REASONING_PATTERNS:
        out = pat.sub("", out)
    return out.strip()


def collect_eq() -> list[dict]:
    rows = []
    for model_dir in sorted(EQ_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for f in sorted(model_dir.glob("scenario_*.json")):
            try:
                with open(f) as fp:
                    d = json.load(fp)
            except (json.JSONDecodeError, OSError):
                continue
            model = d.get("model", model_dir.name)
            scenario_id = d.get("scenario_id", f.stem.replace("scenario_", ""))
            for t in d.get("turns", []):
                resp = strip_reasoning(t.get("response", ""))
                if not resp:
                    continue
                rows.append({
                    "model": model,
                    "benchmark": "eq",
                    "prompt_id": f"s{scenario_id}_t{t.get('turn', 0)}",
                    "run_id": 0,
                    "response_text": resp,
                })
    return rows


def collect_writing() -> list[dict]:
    rows = []
    for model_dir in sorted(WRITING_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for f in sorted(model_dir.glob("prompt_*.txt")):
            try:
                text = f.read_text()
            except OSError:
                continue
            resp = strip_reasoning(text)
            if not resp:
                continue
            prompt_id = f.stem.replace("prompt_", "")
            rows.append({
                "model": model_dir.name,
                "benchmark": "writing",
                "prompt_id": prompt_id,
                "run_id": 0,
                "response_text": resp,
            })
    return rows


def collect_logic() -> list[dict]:
    df = pd.read_csv(LOGIC_CSV, low_memory=False)
    df["run_id"] = pd.to_numeric(df["run_number"], errors="coerce").fillna(0).astype(int)
    rows = []
    for _, r in df.iterrows():
        resp = strip_reasoning(r.get("model_response", ""))
        if not resp:
            continue
        rows.append({
            "model": r["model_name"],
            "benchmark": "logic",
            "prompt_id": str(r["question_id"]),
            "run_id": int(r["run_id"]),
            "response_text": resp,
        })
    return rows


def collect_style() -> list[dict]:
    df = pd.read_csv(STYLE_CSV, low_memory=False)
    df = df[df["status"] == "ok"].copy()
    df["run_id"] = pd.to_numeric(df["run_number"], errors="coerce").fillna(0).astype(int)
    rows = []
    for _, r in df.iterrows():
        resp = strip_reasoning(r.get("response", ""))
        if not resp:
            continue
        rows.append({
            "model": r["model_name"],
            "benchmark": "style",
            "prompt_id": str(r["question_id"]),
            "run_id": int(r["run_id"]),
            "response_text": resp,
        })
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("collecting EQ...", flush=True)
    eq_rows = collect_eq()
    print(f"  {len(eq_rows)} rows")

    print("collecting writing...", flush=True)
    w_rows = collect_writing()
    print(f"  {len(w_rows)} rows")

    print("collecting logic...", flush=True)
    l_rows = collect_logic()
    print(f"  {len(l_rows)} rows")

    print("collecting style...", flush=True)
    s_rows = collect_style()
    print(f"  {len(s_rows)} rows")

    df = pd.DataFrame(eq_rows + w_rows + l_rows + s_rows)
    df["text_len"] = df["response_text"].str.len()

    print(f"\ntotal: {len(df)} rows, {df['model'].nunique()} unique models")
    print("per-benchmark counts:")
    print(df.groupby("benchmark").size().to_string())
    print(f"\nmean text length: {df['text_len'].mean():.0f} chars, "
          f"median: {df['text_len'].median():.0f}, "
          f"max: {df['text_len'].max():.0f}")

    df.to_parquet(OUT_FILE, index=False)
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
