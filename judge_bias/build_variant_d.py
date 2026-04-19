"""Variant D — user's preferred final configuration.

- EQ: adjusted only
  - Keep `eq_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill` with CONTROLLED values.
  - DROP `eq_Grok 4 Fast TrueSkill` (we didn't fit Grok-EQ).

- Writing: adjusted only
  - ADD `writing_Grok 4 Fast TrueSkill` with CONTROLLED values (net new column —
    no raw writing TS existed in combined CSV due to combine.py glob tie-break).
  - Leave the existing `writing_*_score` reference-comparison columns alone
    (different feature family — not raw TrueSkill).

- Tonebench: both adjusted AND raw
  - RENAME existing `tone_Grok 4.1 Fast density/confidence TrueSkill` → `..._raw`
  - ADD new `tone_Grok 4.1 Fast density/confidence TrueSkill_controlled`
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "judge_bias" / "output"
BENCH_DIR = ROOT / "benchmark_combiner" / "benchmarks"


def load_controlled(tag: str) -> pd.Series:
    df = pd.read_csv(OUTDIR / f"style_controlled_ratings_{tag}.csv")
    return df.set_index("model")["skill_controlled"]


def apply_variant_d(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()

    # ---- EQ ----
    # eq_Gemini: replace with controlled values
    col_eq = "eq_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill"
    out[col_eq] = out["model_name"].map(load_controlled("eq_gemini"))
    # eq_Grok: drop
    col_grok_eq = "eq_Grok 4 Fast TrueSkill"
    if col_grok_eq in out.columns:
        out = out.drop(columns=[col_grok_eq])
        print(f"  EQ:       kept {col_eq!r} (controlled), dropped {col_grok_eq!r}")

    # ---- Writing ----
    # Add new column with controlled Grok values
    out["writing_Grok 4 Fast TrueSkill"] = out["model_name"].map(load_controlled("writing_grok"))
    print(f"  Writing:  added 'writing_Grok 4 Fast TrueSkill' (controlled; net-new column)")

    # ---- Tonebench ----
    # Rename existing → _raw, add new _controlled
    renames = {
        "tone_Grok 4.1 Fast density TrueSkill": "tone_Grok 4.1 Fast density TrueSkill_raw",
        "tone_Grok 4.1 Fast confidence TrueSkill": "tone_Grok 4.1 Fast confidence TrueSkill_raw",
    }
    out = out.rename(columns=renames)
    out["tone_Grok 4.1 Fast density TrueSkill_controlled"] = (
        out["model_name"].map(load_controlled("tone_signal_density"))
    )
    out["tone_Grok 4.1 Fast confidence TrueSkill_controlled"] = (
        out["model_name"].map(load_controlled("tone_conv_confidence"))
    )
    print(f"  Tone:     renamed 2 raw TS cols to _raw, added 2 _controlled variants")

    return out


print("=== Variant D (baseline): adjusted-only EQ/Writing + both for Tone ===")
base = pd.read_csv(BENCH_DIR / "clean_combined_all_benches.csv")
out_d = apply_variant_d(base)
path_d = BENCH_DIR / "clean_combined_all_benches_variantD.csv"
out_d.to_csv(path_d, index=False)
print(f"  wrote {path_d.relative_to(ROOT)}  shape={out_d.shape}")

print("\n=== Variant D+sem: same rules on sem v4 d32 champion CSV ===")
champ = pd.read_csv(BENCH_DIR / "clean_combined_all_benches_with_sem_v4_d32.csv")
out_d_sem = apply_variant_d(champ)
path_d_sem = BENCH_DIR / "clean_combined_all_benches_with_sem_v4_d32_variantD.csv"
out_d_sem.to_csv(path_d_sem, index=False)
print(f"  wrote {path_d_sem.relative_to(ROOT)}  shape={out_d_sem.shape}")
