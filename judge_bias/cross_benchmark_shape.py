"""Cross-benchmark comparison of Grok's (and Gemini's) shape preferences.

We have:
  Style (tonebench) — only Grok 4.1 Fast as judge
  EQ + Writing      — Grok 4 Fast and Gemini 3.0 Flash Preview as judges

For each (judge, benchmark) slice with enough battles, compute the 17-feature
winner-loser mean delta. Then compute pairwise cosine similarity across all
slices to map how the "winning shape" varies with task vs with judge.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from judge_bias.shape_feature_probe import (
    build_response_store, extract_features, load_battles, compute_deltas,
)

SLICES = [
    ("eq", "Grok 4 Fast"),
    ("eq", "Gemini 3.0 Flash Preview (2025-12-17)"),
    ("writing", "Grok 4 Fast"),
    ("writing", "Gemini 3.0 Flash Preview (2025-12-17)"),
    ("style", "Grok 4.1 Fast"),
]

means = {}
for bench, judge in SLICES:
    texts = build_response_store(bench)
    feats = texts.apply(extract_features).apply(pd.Series)
    battles = load_battles(bench)
    sub = battles[battles["judge_model"] == judge]
    if len(sub) < 50:
        print(f"skip {bench}/{judge}: only {len(sub)} battles")
        continue
    deltas = compute_deltas(sub, feats)
    m = deltas.mean(axis=0)
    means[(bench, judge)] = m
    print(f"{bench:<10} {judge:<45} n={len(sub):>5}  feature cols={len(feats.columns)}")

print(f"\nCollected {len(means)} slices.")

# Pairwise cosine similarity
print("\n=== pairwise cosine similarities (mean winner-loser direction, 17 features) ===\n")
names = list(means.keys())
header = "slice".ljust(55) + " " + " ".join(f"{n[0][:5]}/{n[1][:12]:<12}" for n in names)
print(header)
for a in names:
    row = f"{a[0][:10]+'/'+a[1][:45]:<55}"
    for b in names:
        ma, mb = means[a], means[b]
        cos = (ma @ mb) / (np.linalg.norm(ma) * np.linalg.norm(mb))
        row += f" {cos:>+18.3f}"
    print(row)

print("\n\n=== Pearson r (same thing, but rank-sensitive) ===\n")
print(header)
for a in names:
    row = f"{a[0][:10]+'/'+a[1][:45]:<55}"
    for b in names:
        ma, mb = means[a], means[b]
        r = np.corrcoef(ma, mb)[0, 1]
        row += f" {r:>+18.3f}"
    print(row)

# Feature-level universality: which features have same sign in ALL 5 slices?
print("\n=== Feature-level universality across all 5 slices ===\n")
feats_order = list(pd.Series(extract_features("test text.")).index)
counts = []
for i, f in enumerate(feats_order):
    signs = [np.sign(means[n][i]) for n in names]
    agree = len(set(signs))
    pos_count = sum(1 for s in signs if s > 0)
    neg_count = sum(1 for s in signs if s < 0)
    direction = "rewards" if pos_count > neg_count else ("penalizes" if neg_count > pos_count else "mixed")
    consistency = f"{max(pos_count, neg_count)}/{len(signs)}"
    counts.append((f, direction, consistency, signs))

counts.sort(key=lambda x: -max(sum(1 for s in x[3] if s > 0), sum(1 for s in x[3] if s < 0)))
print(f"{'feature':<30} {'direction':>10} {'consistency':>12}  signs [eq-gr, eq-ge, wr-gr, wr-ge, st-gr4.1]")
for f, direction, consistency, signs in counts:
    sign_str = " ".join("+" if s > 0 else ("-" if s < 0 else "0") for s in signs)
    print(f"{f:<30} {direction:>10} {consistency:>12}  [{sign_str}]")
