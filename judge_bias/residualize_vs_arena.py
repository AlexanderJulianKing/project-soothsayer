"""Residualize judge preferences against Arena ELO, then interpret via style_*.

For each (judge, benchmark) slice we have:
  direction_j,b = mean over battles of [ embed(winner) - embed(loser) ]        (384-dim)
  fingerprint_m,b = mean embedding of model m's responses on benchmark b       (384-dim)
  pref_m,j,b = fingerprint_m,b · direction_j,b                                 (scalar)

Three analyses:

  1. Pool judges: z-score pref within each (judge, bench) slice, then average
     across the 5 slices to produce one "universal LLM-judge preferences model
     m" score per model. This is the shape-alignment quantity.

  2. How well does this universal pref predict Arena ELO? r(pref, lmarena_Score).
     High r = judges reward skill. Low r = judges reward style unrelated to skill.

  3. Residualize pref against Arena ELO (pref_residual = pref - OLS(pref|ELO)).
     Regress residual on the 21 style_* structural features. Coefficients tell
     us what FORMATTING FEATURES the judges reward BEYOND what Arena skill predicts.

Style_* columns describe structural response shape (length, headers, bold, lists)
independent of judge ratings, so regressing on them isolates bias from skill.
Tone_* columns are the judge ratings themselves — excluded to avoid tautology.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from judge_bias.shape_feature_probe import load_battles

EMB_PATH = ROOT / "embeddings/cache/response_embeddings.parquet"
COMBINED_PATH = ROOT / "benchmark_combiner/benchmarks/clean_combined_all_benches.csv"

SLICES = [
    ("eq", "Grok 4 Fast"),
    ("eq", "Gemini 3.0 Flash Preview (2025-12-17)"),
    ("writing", "Grok 4 Fast"),
    ("writing", "Gemini 3.0 Flash Preview (2025-12-17)"),
    ("style", "Grok 4.1 Fast"),
]


def unit_norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# --- Load bge-small embeddings ---
emb = pd.read_parquet(EMB_PATH)
emb_cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]

# For each benchmark, build (model, prompt-key) embedding and model fingerprint.
fp_per_bench: dict[str, pd.DataFrame] = {}
key_per_bench: dict[str, pd.DataFrame] = {}

for bench in ["eq", "writing", "style"]:
    sub = emb[emb.benchmark == bench].copy()
    if bench == "eq":
        sub["key"] = sub["prompt_id"].str.extract(r"s(\d+)_t\d+").astype(int)
    else:
        sub["key"] = sub["prompt_id"].astype(int)
    # Per-(model, key) — average turns & runs
    by_key = sub.groupby(["model", "key"])[emb_cols].mean()
    by_key_norm = by_key.values / np.linalg.norm(by_key.values, axis=1, keepdims=True).clip(min=1e-12)
    key_per_bench[bench] = pd.DataFrame(by_key_norm, index=by_key.index, columns=emb_cols)
    # Per-model fingerprint — average all responses on this benchmark
    fp = sub.groupby("model")[emb_cols].mean()
    fp_norm = fp.values / np.linalg.norm(fp.values, axis=1, keepdims=True).clip(min=1e-12)
    fp_per_bench[bench] = pd.DataFrame(fp_norm, index=fp.index, columns=emb_cols)

print(f"Embedding fingerprints ready for {list(fp_per_bench.keys())}")


# --- For each slice, compute the direction vector from winner-loser deltas ---
directions: dict[tuple[str, str], np.ndarray] = {}
for bench, judge in SLICES:
    battles = load_battles(bench)
    sub = battles[battles["judge_model"] == judge]
    lookup = key_per_bench[bench]
    deltas = []
    for _, b in sub.iterrows():
        w = (b["winner_model"], b["key"])
        l = (b["loser_model"], b["key"])
        if w not in lookup.index or l not in lookup.index:
            continue
        deltas.append(lookup.loc[w].values - lookup.loc[l].values)
    d = np.vstack(deltas).mean(axis=0)
    directions[(bench, judge)] = unit_norm(d)
    print(f"{bench:<8} / {judge:<50} → direction built from {len(deltas)} battles, raw norm {np.linalg.norm(d):.4f}")


# --- Per-model preference score for each slice ---
pref = {}
for (bench, judge), dir_vec in directions.items():
    fp = fp_per_bench[bench]
    scores = pd.Series(fp.values @ dir_vec, index=fp.index, name=f"pref_{bench}_{judge[:12]}")
    pref[(bench, judge)] = scores


# --- Z-score within each slice, then average to get universal preference index ---
pref_df = pd.concat(pref.values(), axis=1)
pref_z = (pref_df - pref_df.mean()) / pref_df.std()
universal_pref = pref_z.mean(axis=1)
universal_pref.name = "judge_pref_universal"
print(f"\nuniversal judge-preference index available for {len(universal_pref)} models")


# --- Join with Arena ELO and style_* features ---
combined = pd.read_csv(COMBINED_PATH)
style_cols = [c for c in combined.columns if c.startswith("style_")]
print(f"style_* cols: {len(style_cols)}")

df = combined[["model_name", "lmarena_Score"] + style_cols].copy()
df = df.set_index("model_name")
df["pref"] = universal_pref
df_full = df.dropna(subset=["lmarena_Score", "pref"]).copy()
print(f"\n{len(df_full)} models have both lmarena_Score and judge-preference")


# --- Analysis 1: how well does judge preference track Arena ELO? ---
r = df_full[["pref", "lmarena_Score"]].corr().iloc[0, 1]
print(f"\n=== A1: judge preference vs Arena ELO ===")
print(f"Pearson r = {r:+.3f}   (high → judges reward skill; low → judges reward bias)")


# --- Analysis 2: residualize preference against ELO, then regress on style_* ---
print(f"\n=== A2: residualize preference against Arena ELO, regress onto style_* ===")
# Fit pref ~ lmarena_Score
X0 = df_full[["lmarena_Score"]].values
y = df_full["pref"].values
lr = LinearRegression().fit(X0, y)
residual = y - lr.predict(X0)
print(f"OLS pref = {lr.coef_[0]:+.5f} * lmarena_Score + {lr.intercept_:+.5f}")
print(f"R² of ELO alone on preference: {lr.score(X0, y):.3f}")
print(f"residual std: {residual.std():.3f}")

# --- Top-10 models by residual ---
df_full["pref_residual"] = residual
pos = df_full.nlargest(10, "pref_residual")[["lmarena_Score", "pref", "pref_residual"]]
neg = df_full.nsmallest(10, "pref_residual")[["lmarena_Score", "pref", "pref_residual"]]
print(f"\n--- Top 10: judges like MORE than Arena ELO would predict (over-rewarded by judge) ---")
for m, row in pos.iterrows():
    print(f"  +{row['pref_residual']:+.3f}  pref={row['pref']:+.3f}  arena={row['lmarena_Score']:.1f}  {m}")

print(f"\n--- Bottom 10: judges like LESS than Arena ELO would predict (under-rewarded by judge) ---")
for m, row in neg.iterrows():
    print(f"  {row['pref_residual']:+.3f}  pref={row['pref']:+.3f}  arena={row['lmarena_Score']:.1f}  {m}")


# --- Analysis 3: Regress residuals on style_* features ---
# Drop style cols that have too many NaN or that are constant
style_usable = []
for c in style_cols:
    col = df_full[c]
    if col.notna().sum() >= 50 and col.nunique() > 1:
        style_usable.append(c)

df_reg = df_full.dropna(subset=style_usable)
print(f"\n=== A3: regress residual on {len(style_usable)} style_* features ({len(df_reg)} models after dropna) ===")

X = df_reg[style_usable].values
# z-score style features for comparable coefficients
X_z = (X - X.mean(axis=0)) / X.std(axis=0)
y = df_reg["pref_residual"].values

# Simple per-feature Pearson r (independent of confounding other features)
print(f"\n--- per-feature Pearson r (residual vs one feature at a time) ---")
rows = []
for c, col_z in zip(style_usable, X_z.T):
    r = np.corrcoef(col_z, y)[0, 1]
    rows.append((c, r))
rows.sort(key=lambda x: -abs(x[1]))
for c, r in rows:
    sig = "★" if abs(r) > 0.2 else " "
    print(f"  {sig} {r:+.3f}  {c}")

# Ridge regression to balance multicollinearity
print(f"\n--- Ridge regression (alpha=1.0, standardized X) ---")
ridge = Ridge(alpha=1.0).fit(X_z, y)
coefs = pd.Series(ridge.coef_, index=style_usable).sort_values(key=lambda s: -s.abs())
print(f"R² (in-sample): {ridge.score(X_z, y):.3f}")
for c, co in coefs.items():
    sig = "★" if abs(co) > 0.15 else " "
    print(f"  {sig} {co:+.4f}  {c}")

# --- Also compare: judge preference vs style features (not just residual) ---
# For contrast — how much of this is just that judges reward length, which correlates with ELO?
print(f"\n=== A4: for comparison — raw preference (not residualized) vs style_* ===")
y_raw = df_reg["pref"].values
rows = []
for c, col_z in zip(style_usable, X_z.T):
    r = np.corrcoef(col_z, y_raw)[0, 1]
    rows.append((c, r))
rows.sort(key=lambda x: -abs(x[1]))
for c, r in rows[:10]:
    print(f"  {r:+.3f}  {c}")
