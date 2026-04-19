"""Variant E — keep raw AND add controlled as parallel columns everywhere.

Instead of replacing raw with controlled (which hurts because arena voters
share the same shape biases), give the predictor BOTH per-judge variants
and let it decide.

- EQ:
  - Rename `eq_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill` → `..._raw`
  - Add `eq_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill_controlled`
  - Keep `eq_Grok 4 Fast TrueSkill` unchanged (no controlled version fit for Grok)

- Writing (rescued from combine.py's glob tie-break):
  - Add `writing_Grok 4 Fast TrueSkill_raw` (from writing_20260407.csv)
  - Add `writing_Grok 4 Fast TrueSkill_controlled`
  - Add `writing_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill_raw` (bonus)
  - Leave existing `writing_*_score` columns alone

- Tonebench:
  - Rename existing `tone_Grok 4.1 Fast density/confidence TrueSkill` → `..._raw`
  - Add `tone_Grok 4.1 Fast density/confidence TrueSkill_controlled`

Outputs two variants:
  E_baseline: clean_combined_all_benches + all of above
  E_sem:      sem v4 d32 champion + all of above
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "judge_bias" / "output"
BENCH_DIR = ROOT / "benchmark_combiner" / "benchmarks"
WRITING_RAW_CSV = ROOT / "benchmark_combiner" / "benchmarks" / "writing_20260407.csv"


def load_controlled(tag: str) -> pd.Series:
    df = pd.read_csv(OUTDIR / f"style_controlled_ratings_{tag}.csv")
    return df.set_index("model")["skill_controlled"]


def apply_variant_e(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()

    # ---- EQ ----
    eq_col = "eq_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill"
    out = out.rename(columns={eq_col: f"{eq_col}_raw"})
    out[f"{eq_col}_controlled"] = out["model_name"].map(load_controlled("eq_gemini"))
    print(f"  EQ:       renamed 'eq_Gemini...TrueSkill' → _raw, added _controlled; kept eq_Grok")

    # ---- Writing ----
    writing_raw = pd.read_csv(WRITING_RAW_CSV).set_index("writer_model")
    out["writing_Grok 4 Fast TrueSkill_raw"] = (
        out["model_name"].map(writing_raw["Grok 4 Fast TrueSkill"])
    )
    out["writing_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill_raw"] = (
        out["model_name"].map(writing_raw["Gemini 3.0 Flash Preview (2025-12-17) TrueSkill"])
    )
    out["writing_Grok 4 Fast TrueSkill_controlled"] = (
        out["model_name"].map(load_controlled("writing_grok"))
    )
    n_raw = out["writing_Grok 4 Fast TrueSkill_raw"].notna().sum()
    n_gem = out["writing_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill_raw"].notna().sum()
    n_ctl = out["writing_Grok 4 Fast TrueSkill_controlled"].notna().sum()
    print(f"  Writing:  added _raw Grok ({n_raw}), _raw Gemini ({n_gem}), _controlled Grok ({n_ctl})")

    # ---- Tonebench ----
    tone_d_raw = "tone_Grok 4.1 Fast density TrueSkill"
    tone_c_raw = "tone_Grok 4.1 Fast confidence TrueSkill"
    out = out.rename(columns={
        tone_d_raw: f"{tone_d_raw}_raw",
        tone_c_raw: f"{tone_c_raw}_raw",
    })
    out[f"{tone_d_raw}_controlled"] = out["model_name"].map(load_controlled("tone_signal_density"))
    out[f"{tone_c_raw}_controlled"] = out["model_name"].map(load_controlled("tone_conv_confidence"))
    print(f"  Tone:     renamed 2 raw TS cols to _raw, added 2 _controlled variants")

    return out


print("=== Variant E (baseline): raw + controlled parallel for all ===")
base = pd.read_csv(BENCH_DIR / "clean_combined_all_benches.csv")
out_e = apply_variant_e(base)
path_e = BENCH_DIR / "clean_combined_all_benches_variantE.csv"
out_e.to_csv(path_e, index=False)
print(f"  wrote {path_e.relative_to(ROOT)}  shape={out_e.shape}")

print("\n=== Variant E+sem: same on sem v4 d32 champion ===")
champ = pd.read_csv(BENCH_DIR / "clean_combined_all_benches_with_sem_v4_d32.csv")
out_e_sem = apply_variant_e(champ)
path_e_sem = BENCH_DIR / "clean_combined_all_benches_with_sem_v4_d32_variantE.csv"
out_e_sem.to_csv(path_e_sem, index=False)
print(f"  wrote {path_e_sem.relative_to(ROOT)}  shape={out_e_sem.shape}")
