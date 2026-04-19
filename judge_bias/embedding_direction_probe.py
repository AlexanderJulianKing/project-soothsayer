"""Quick judge-bias probe: does Gemini 3.0 Flash reward different things than Grok 4 Fast?

For each battle: delta = embed(winner) - embed(loser).
Average per (judge, benchmark) gives the direction in embedding space that judge rewards.
We then:
  1. Compare judges to each other (cosine similarity of their directions).
  2. Score every model's fingerprint against each judge direction — which models benefit
     most / least from Gemini vs Grok.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/alexanderking/Desktop/random_stuff/project_soothsayer")
EMB_PATH = ROOT / "embeddings/cache/response_embeddings.parquet"  # bge-small v4
EQ_BATTLES = ROOT / "soothsayer_eq/results/battle_history.csv"
WR_BATTLES = ROOT / "soothsayer_writing/results/battle_history.csv"

print("loading embeddings...")
emb = pd.read_parquet(EMB_PATH)
emb_cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
dim = len(emb_cols)
print(f"  {len(emb)} rows, {dim} dims")

# Build per-(model, scenario/prompt) embeddings ready for lookup
# EQ: average across turns + runs for each (model, scenario)
eq = emb[emb.benchmark == "eq"].copy()
eq["scenario"] = eq["prompt_id"].str.extract(r"s(\d+)_t\d+").astype(int)
eq_key = eq.groupby(["model", "scenario"])[emb_cols].mean()
eq_key_norm = eq_key.values / np.linalg.norm(eq_key.values, axis=1, keepdims=True).clip(min=1e-12)
eq_key = pd.DataFrame(eq_key_norm, index=eq_key.index, columns=emb_cols)

# Writing: average across runs for each (model, prompt_id)
wr = emb[emb.benchmark == "writing"].copy()
wr["prompt_id_int"] = wr["prompt_id"].astype(int)
wr_key = wr.groupby(["model", "prompt_id_int"])[emb_cols].mean()
wr_key_norm = wr_key.values / np.linalg.norm(wr_key.values, axis=1, keepdims=True).clip(min=1e-12)
wr_key = pd.DataFrame(wr_key_norm, index=wr_key.index, columns=emb_cols)


def compute_deltas(battles: pd.DataFrame, emb_lookup: pd.DataFrame, key_col: str):
    """For each battle row, look up winner/loser embeddings and return the delta matrix."""
    deltas = []
    used_judges = []
    missing = 0
    for _, row in battles.iterrows():
        w_key = (row["winner_model"], row[key_col])
        l_key = (row["loser_model"], row[key_col])
        if w_key not in emb_lookup.index or l_key not in emb_lookup.index:
            missing += 1
            continue
        w = emb_lookup.loc[w_key].values
        l = emb_lookup.loc[l_key].values
        deltas.append(w - l)
        used_judges.append(row["judge_model"].strip())
    print(f"  {missing} battles skipped (missing embeddings)")
    return np.vstack(deltas), np.array(used_judges)


print("\n=== EQ battles ===")
eq_b = pd.read_csv(EQ_BATTLES, usecols=["judge_model", "scenario_id", "winner_model", "loser_model"])
eq_b.rename(columns={"scenario_id": "scenario"}, inplace=True)
eq_deltas, eq_judges = compute_deltas(eq_b, eq_key, "scenario")
print(f"  {len(eq_deltas)} usable battles")
print(f"  judge counts: {pd.Series(eq_judges).value_counts().to_dict()}")

print("\n=== Writing battles ===")
wr_b = pd.read_csv(WR_BATTLES, usecols=["judge_model", "prompt_id", "winner_model", "loser_model"])
wr_b.rename(columns={"prompt_id": "prompt_id_int"}, inplace=True)
wr_deltas, wr_judges = compute_deltas(wr_b, wr_key, "prompt_id_int")
print(f"  {len(wr_deltas)} usable battles")
print(f"  judge counts: {pd.Series(wr_judges).value_counts().to_dict()}")


def direction(deltas: np.ndarray) -> np.ndarray:
    """Average the delta vectors and unit-normalize to get the preference direction."""
    mean = deltas.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


# Compute per-(judge, benchmark) directions for the two judges of interest.
JUDGES = ["Grok 4 Fast", "Gemini 3.0 Flash Preview (2025-12-17)"]

dirs = {}
for judge in JUDGES:
    eq_mask = eq_judges == judge
    wr_mask = wr_judges == judge
    eq_d = direction(eq_deltas[eq_mask]) if eq_mask.sum() > 50 else None
    wr_d = direction(wr_deltas[wr_mask]) if wr_mask.sum() > 50 else None
    dirs[(judge, "eq")] = eq_d
    dirs[(judge, "writing")] = wr_d
    print(f"\n{judge}:")
    print(f"  EQ      battles: {eq_mask.sum()}, direction-norm (pre-unit): {np.linalg.norm(eq_deltas[eq_mask].mean(axis=0)):.4f}" if eq_mask.sum() else f"  EQ: no data")
    print(f"  Writing battles: {wr_mask.sum()}, direction-norm (pre-unit): {np.linalg.norm(wr_deltas[wr_mask].mean(axis=0)):.4f}" if wr_mask.sum() else f"  Writing: no data")


def cos(a, b):
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


print("\n=== Cross-judge direction similarities ===")
print("(cosine between unit-normalized preference directions)")
print()
g_eq = dirs[("Grok 4 Fast", "eq")]
g_wr = dirs[("Grok 4 Fast", "writing")]
m_eq = dirs[("Gemini 3.0 Flash Preview (2025-12-17)", "eq")]
m_wr = dirs[("Gemini 3.0 Flash Preview (2025-12-17)", "writing")]

print(f"Grok EQ ↔ Grok Writing:       {cos(g_eq, g_wr):+.3f}   (is Grok's bias consistent across tasks?)")
print(f"Gemini EQ ↔ Gemini Writing:   {cos(m_eq, m_wr):+.3f}   (is Gemini's bias consistent across tasks?)")
print(f"Grok EQ ↔ Gemini EQ:          {cos(g_eq, m_eq):+.3f}   (do they agree on EQ?)")
print(f"Grok Writing ↔ Gemini Writing: {cos(g_wr, m_wr):+.3f}  (do they agree on Writing?)")


# For the rest of the analysis, use per-benchmark model fingerprints
# (mean embedding per model per benchmark, unit-normalized).
print("\n=== Models that each judge's direction favors ===\n")

def model_fingerprints(cache_df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    sub = cache_df[cache_df.benchmark == benchmark].copy()
    fp = sub.groupby("model")[emb_cols].mean()
    fp_norm = fp.values / np.linalg.norm(fp.values, axis=1, keepdims=True).clip(min=1e-12)
    return pd.DataFrame(fp_norm, index=fp.index, columns=emb_cols)


fp_eq = model_fingerprints(emb, "eq")
fp_wr = model_fingerprints(emb, "writing")


def favored(fp: pd.DataFrame, direction_vec: np.ndarray, k: int = 8) -> pd.Series:
    scores = pd.Series(fp.values @ direction_vec, index=fp.index).sort_values(ascending=False)
    return scores


def report(label: str, fp: pd.DataFrame, direction_vec: np.ndarray):
    if direction_vec is None:
        return
    scores = favored(fp, direction_vec)
    print(f"--- {label} — top 10 favored / bottom 10 disfavored ---")
    top = scores.head(10)
    bot = scores.tail(10).iloc[::-1]
    max_len = max(len(m) for m in list(top.index) + list(bot.index))
    for m, s in top.items():
        print(f"  + {s:+.3f}  {m}")
    print("  ...")
    for m, s in bot.items():
        print(f"  - {s:+.3f}  {m}")
    print()


report("Grok 4 Fast on EQ", fp_eq, g_eq)
report("Gemini 3.0 Flash on EQ", fp_eq, m_eq)
report("Grok 4 Fast on Writing", fp_wr, g_wr)
report("Gemini 3.0 Flash on Writing", fp_wr, m_wr)
