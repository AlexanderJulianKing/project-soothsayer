"""Swap raw TrueSkill columns in the combined CSV with style-controlled
equivalents, plus pull in the Writing TrueSkill columns that combine.py
currently loses to a glob-tie-break.

Known pre-existing bug in combine.py: `get_latest_file` on pattern
`benchmarks/writing_*.csv` ties between `writing_20260407.csv` (has
TrueSkill) and `writing_direct_20260407.csv` (score-vs-reference only) and
silently picks the `_direct` variant. As a result no Writing TrueSkill
columns appear in `clean_combined_all_benches.csv`. For this experiment we
pull the TrueSkill columns directly from `writing_20260407.csv`.

Swaps (direct column replacement — same name, new values):
  eq_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill ← eq_gemini skill_controlled
  tone_Grok 4.1 Fast density TrueSkill               ← tone_signal_density skill_controlled
  tone_Grok 4.1 Fast confidence TrueSkill            ← tone_conv_confidence skill_controlled

New additions (Writing TrueSkill cols currently dropped by combine.py):
  writing_Grok 4 Fast TrueSkill_raw        — raw from writing_*.csv
  writing_Grok 4 Fast TrueSkill_controlled ← writing_grok skill_controlled
  writing_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill_raw  — raw from writing_*.csv
    (no controlled version — we only fit Writing/Grok)

Outputs two variants:
  A. clean_combined_all_benches_style_controlled.csv (baseline + swap + writing adds)
  B. clean_combined_all_benches_with_sem_v4_d32_style_controlled.csv (champion + swap + writing adds)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "judge_bias" / "output"
BENCH_DIR = ROOT / "benchmark_combiner" / "benchmarks"
WRITING_RAW_CSV = ROOT / "benchmark_combiner" / "benchmarks" / "writing_20260407.csv"

SWAPS = [
    ("eq_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill", "eq_gemini"),
    ("tone_Grok 4.1 Fast density TrueSkill",               "tone_signal_density"),
    ("tone_Grok 4.1 Fast confidence TrueSkill",            "tone_conv_confidence"),
]


def load_controlled(tag: str) -> pd.Series:
    df = pd.read_csv(OUTDIR / f"style_controlled_ratings_{tag}.csv")
    return df.set_index("model")["skill_controlled"]


def apply_swaps(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()

    # 1) Direct swaps for cols that already exist in combined CSV
    for col, tag in SWAPS:
        if col not in out.columns:
            print(f"  [warn] column {col!r} not found — skipping")
            continue
        controlled = load_controlled(tag)
        mapped = out["model_name"].map(controlled)
        n_before = out[col].notna().sum()
        n_after = mapped.notna().sum()
        out[col] = mapped
        print(f"  swapped  {col!r}  ({n_before} → {n_after} non-null)")

    # 2) Pull Writing TrueSkill columns that combine.py's tie-break drops
    writing_raw = pd.read_csv(WRITING_RAW_CSV)
    # Index by model name (writer_model → model_name)
    writing_raw = writing_raw.set_index("writer_model")

    grok_ts = writing_raw["Grok 4 Fast TrueSkill"]
    gem_ts = writing_raw["Gemini 3.0 Flash Preview (2025-12-17) TrueSkill"]

    out["writing_Grok 4 Fast TrueSkill_raw"] = out["model_name"].map(grok_ts)
    out["writing_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill_raw"] = (
        out["model_name"].map(gem_ts)
    )
    # 3) Add Writing Grok style-controlled
    out["writing_Grok 4 Fast TrueSkill_controlled"] = (
        out["model_name"].map(load_controlled("writing_grok"))
    )

    n_raw = out["writing_Grok 4 Fast TrueSkill_raw"].notna().sum()
    n_ctl = out["writing_Grok 4 Fast TrueSkill_controlled"].notna().sum()
    n_gem = out["writing_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill_raw"].notna().sum()
    print(f"  added    writing_Grok 4 Fast TrueSkill_raw        ({n_raw} non-null)")
    print(f"  added    writing_Grok 4 Fast TrueSkill_controlled ({n_ctl} non-null)")
    print(f"  added    writing_Gemini 3.0 Flash ... TrueSkill_raw ({n_gem} non-null)")

    return out


print("=== Variant A: baseline + style-controlled swap + writing adds ===")
base = pd.read_csv(BENCH_DIR / "clean_combined_all_benches.csv")
out_a = apply_swaps(base)
path_a = BENCH_DIR / "clean_combined_all_benches_style_controlled.csv"
out_a.to_csv(path_a, index=False)
print(f"  wrote {path_a.relative_to(ROOT)}  shape={out_a.shape}")

print("\n=== Variant B: champion (sem_v4_d32) + style-controlled swap + writing adds ===")
champ = pd.read_csv(BENCH_DIR / "clean_combined_all_benches_with_sem_v4_d32.csv")
out_b = apply_swaps(champ)
path_b = BENCH_DIR / "clean_combined_all_benches_with_sem_v4_d32_style_controlled.csv"
out_b.to_csv(path_b, index=False)
print(f"  wrote {path_b.relative_to(ROOT)}  shape={out_b.shape}")

# Also emit "raw writing added, no style control" variant — isolates the bug-fix effect
print("\n=== Variant C: baseline + writing TrueSkill raw ONLY (no style control) ===")
out_c = base.copy()
writing_raw = pd.read_csv(WRITING_RAW_CSV).set_index("writer_model")
out_c["writing_Grok 4 Fast TrueSkill_raw"] = out_c["model_name"].map(writing_raw["Grok 4 Fast TrueSkill"])
out_c["writing_Gemini 3.0 Flash Preview (2025-12-17) TrueSkill_raw"] = (
    out_c["model_name"].map(writing_raw["Gemini 3.0 Flash Preview (2025-12-17) TrueSkill"])
)
path_c = BENCH_DIR / "clean_combined_all_benches_with_writing_ts.csv"
out_c.to_csv(path_c, index=False)
print(f"  wrote {path_c.relative_to(ROOT)}  shape={out_c.shape}")
