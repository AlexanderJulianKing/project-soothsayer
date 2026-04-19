"""
Extract production turn-3 first-person features from EQ benchmark raw responses.

These are the 2 features that meaningfully improve KNN Arena prediction:
LOO RMSE -0.78 mean across 5 seeds, robust across knn_power_alpha/c hyperparams.

Earlier sweeps tested 14 other aggregates (draft continuity, back-reference rate,
length stability, hedge rate, persona growth, criterion winrates, etc.) — none
survived ablation, and ingesting them all through combine.py inflated the
correlations.py row-filter denominator enough to silently admit ~22 sparse
models, blowing up OOF RMSE from 15.22 to 35.20. Production now emits only the
2 surviving features. See git history pre-2026-04-08 for the search code.

Output: benchmark_combiner/benchmarks/eq_multiturn_YYYYMMDD.csv with 2 features
per model, using canonical openbench display names.
"""

import glob
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESPONSE_ROOT = os.path.join(HERE, "generated_responses")
OUT_DIR = os.path.join(HERE, "..", "benchmark_combiner", "benchmarks")

WORD_RE = re.compile(r"[A-Za-z']+")
FIRST_PERSON_TOKENS = {"i", "i'm", "im", "i'd", "i've", "i'll", "my", "mine", "me", "myself"}


def _tokenize(text: str):
    return [t.lower() for t in WORD_RE.findall(text or "")]


def _first_person_rate(text: str) -> float:
    toks = _tokenize(text)
    if not toks:
        return np.nan
    return sum(1 for t in toks if t in FIRST_PERSON_TOKENS) / len(toks)


def extract_model_features(model_dir: str) -> dict:
    """Compute turn-3 first-person rate + turn-3-vs-turn-1 delta for one model.

    Per-turn raw fallback so a single unparseable turn doesn't sink the scenario.
    """
    scenario_files = sorted(glob.glob(os.path.join(model_dir, "scenario_*.json")))
    if not scenario_files:
        return {}

    scenario_rows = []
    for sf in scenario_files:
        try:
            with open(sf) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        turns = data.get("turns") or []
        if len(turns) < 3:
            continue

        drafts = [t.get("parsed", {}).get("draft") or "" for t in turns]
        raws = [t.get("parsed", {}).get("raw") or t.get("response") or "" for t in turns]

        t1_text = drafts[0] or raws[0]
        t3_text = drafts[2] or raws[2]
        fp_t1 = _first_person_rate(t1_text)
        fp_t3 = _first_person_rate(t3_text)

        scenario_rows.append({
            "first_person_rate_t3": fp_t3,
            "first_person_delta_t3": (
                fp_t3 - fp_t1 if not (np.isnan(fp_t1) or np.isnan(fp_t3)) else np.nan
            ),
        })

    if not scenario_rows:
        return {}

    sdf = pd.DataFrame(scenario_rows)
    return {
        "first_person_rate_t3": float(sdf["first_person_rate_t3"].mean(skipna=True)),
        "first_person_delta_t3": float(sdf["first_person_delta_t3"].mean(skipna=True)),
    }


def main():
    print(f"Scanning {RESPONSE_ROOT}...")
    model_dirs = sorted(
        d for d in glob.glob(os.path.join(RESPONSE_ROOT, "*")) if os.path.isdir(d)
    )
    print(f"Found {len(model_dirs)} model directories")

    rows = []
    for md in model_dirs:
        model = os.path.basename(md)
        feats = extract_model_features(md)
        if feats:
            feats["model_name"] = model
            rows.append(feats)
    df = pd.DataFrame(rows)
    print(f"Extracted features for {len(df)} models")

    df = df[["model_name", "first_person_rate_t3", "first_person_delta_t3"]]

    stamp = datetime.now().strftime("%Y%m%d")
    out_path = os.path.join(OUT_DIR, f"eq_multiturn_{stamp}.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({df.shape})")

    print("\nFeature summary:")
    for col in df.columns:
        if col == "model_name":
            continue
        s = df[col]
        print(f"  {col}: n={s.notna().sum()} mean={s.mean():.4f} std={s.std():.4f} min={s.min():.4f} max={s.max():.4f}")


if __name__ == "__main__":
    main()
