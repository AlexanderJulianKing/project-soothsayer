"""Variant F — isolate EQ only. Leave writing and tone untouched.

F1: baseline CSV, eq_Gemini TrueSkill SWAPPED to controlled
F2: baseline CSV, eq_Gemini TrueSkill kept as raw + _controlled added PARALLEL
F3: sem champion CSV, same swap as F1
F4: sem champion CSV, same parallel as F2

All four keep eq_Grok 4 Fast TrueSkill unchanged, writing cols unchanged
(no bug-fix), and tone cols unchanged.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "judge_bias" / "output"
BENCH_DIR = ROOT / "benchmark_combiner" / "benchmarks"

EQ_COL = "eq_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill"


def load_controlled(tag: str) -> pd.Series:
    df = pd.read_csv(OUTDIR / f"style_controlled_ratings_{tag}.csv")
    return df.set_index("model")["skill_controlled"]


def apply_swap(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out[EQ_COL] = out["model_name"].map(load_controlled("eq_gemini"))
    return out


def apply_parallel(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out = out.rename(columns={EQ_COL: f"{EQ_COL}_raw"})
    out[f"{EQ_COL}_controlled"] = out["model_name"].map(load_controlled("eq_gemini"))
    return out


base = pd.read_csv(BENCH_DIR / "clean_combined_all_benches.csv")
champ = pd.read_csv(BENCH_DIR / "clean_combined_all_benches_with_sem_v4_d32.csv")

f1 = apply_swap(base)
f1.to_csv(BENCH_DIR / "clean_combined_all_benches_variantF1.csv", index=False)
print(f"F1 (baseline + eq swap): shape={f1.shape}")

f2 = apply_parallel(base)
f2.to_csv(BENCH_DIR / "clean_combined_all_benches_variantF2.csv", index=False)
print(f"F2 (baseline + eq parallel): shape={f2.shape}")

f3 = apply_swap(champ)
f3.to_csv(BENCH_DIR / "clean_combined_all_benches_with_sem_v4_d32_variantF3.csv", index=False)
print(f"F3 (sem + eq swap): shape={f3.shape}")

f4 = apply_parallel(champ)
f4.to_csv(BENCH_DIR / "clean_combined_all_benches_with_sem_v4_d32_variantF4.csv", index=False)
print(f"F4 (sem + eq parallel): shape={f4.shape}")
