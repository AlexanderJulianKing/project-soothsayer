"""Variant G — sem champion + writing TS raw ADDED (no style control).

Tests whether the combine.py bug-fix win (Δ −0.22 on baseline → variant C)
stacks on top of the sem champion. If it does, we have a new champion.

G_sem_writing_ts: clean_combined_all_benches_with_sem_v4_d32.csv
                  + writing_Grok 4 Fast TrueSkill_raw
                  + writing_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill_raw

No style-controlled modifications. No column renames. Just adds the 2
writing TS columns that combine.py's glob tie-break currently drops.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmark_combiner" / "benchmarks"
WRITING_RAW_CSV = BENCH_DIR / "writing_20260407.csv"

writing_raw = pd.read_csv(WRITING_RAW_CSV).set_index("writer_model")
grok_ts = writing_raw["Grok 4 Fast TrueSkill"]
gem_ts = writing_raw["Gemini 3.0 Flash Preview (2025-12-17) TrueSkill"]

champ = pd.read_csv(BENCH_DIR / "clean_combined_all_benches_with_sem_v4_d32.csv")
champ["writing_Grok 4 Fast TrueSkill_raw"] = champ["model_name"].map(grok_ts)
champ["writing_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill_raw"] = champ["model_name"].map(gem_ts)

out = BENCH_DIR / "clean_combined_all_benches_with_sem_v4_d32_plus_writing_ts.csv"
champ.to_csv(out, index=False)
print(f"wrote {out.relative_to(ROOT)}  shape={champ.shape}")
print(f"  writing_Grok TS_raw non-null:   {champ['writing_Grok 4 Fast TrueSkill_raw'].notna().sum()}")
print(f"  writing_Gemini TS_raw non-null: {champ['writing_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill_raw'].notna().sum()}")
