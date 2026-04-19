"""Ablation + interpretation of the judge-preference probe.

Ablation (per-battle):
  A. 17 interpretable shape features (length, pronouns, em-dashes, etc.)
  B. 384-dim bge-small embedding
  C. concat of both (A + B = 401-dim)

Compare held-out accuracy across all 5 (judge, benchmark) slices under the
same leave-pair-of-models-out CV as `predict_from_embeddings.py`.

Interpretation:
  For each slice, train one logistic regression on the full embedding (B) to
  recover the preference direction (coef vector). For each of the 17 shape
  features, compute correlation between the feature's per-response value and
  the response's projection onto the preference direction. This tells us
  which interpretable features are encoded by the embedding direction the
  judge rewards.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from judge_bias.shape_feature_probe import (
    build_response_store, extract_features, load_battles,
)

EMB_PATH = ROOT / "embeddings/cache/response_embeddings.parquet"

SLICES = [
    ("eq", "Grok 4 Fast"),
    ("eq", "Gemini 3.0 Flash Preview (2025-12-17)"),
    ("writing", "Grok 4 Fast"),
    ("writing", "Gemini 3.0 Flash Preview (2025-12-17)"),
    ("style", "Grok 4.1 Fast"),
]


def build_embeddings(bench: str) -> pd.DataFrame:
    emb = pd.read_parquet(EMB_PATH)
    sub = emb[emb.benchmark == bench].copy()
    if bench == "eq":
        sub["key"] = sub["prompt_id"].str.extract(r"s(\d+)_t\d+").astype(int)
    else:
        sub["key"] = sub["prompt_id"].astype(int)
    emb_cols = [c for c in sub.columns if c.startswith("e") and c[1:].isdigit()]
    by_key = sub.groupby(["model", "key"])[emb_cols].mean()
    v = by_key.values / np.linalg.norm(by_key.values, axis=1, keepdims=True).clip(min=1e-12)
    return pd.DataFrame(v, index=by_key.index, columns=emb_cols)


def build_shape_features(bench: str) -> pd.DataFrame:
    texts = build_response_store(bench)
    feats = texts.apply(extract_features).apply(pd.Series)
    # z-score so dimensions are comparable to embedding dims
    feats_z = (feats - feats.mean()) / feats.std().replace(0, 1)
    return feats_z


def pair_dataset(bench, judge, shape, emb):
    battles = load_battles(bench)
    sub = battles[battles["judge_model"] == judge]
    rng = np.random.RandomState(0)

    rows_shape = []; rows_emb = []; y = []; groups = []
    for _, b in sub.iterrows():
        w_key = (b["winner_model"], b["key"])
        l_key = (b["loser_model"], b["key"])
        if (w_key not in shape.index or l_key not in shape.index or
            w_key not in emb.index or l_key not in emb.index):
            continue
        # Randomize A/B orientation so label isn't a function of order
        if rng.rand() < 0.5:
            rows_shape.append(shape.loc[w_key].values - shape.loc[l_key].values)
            rows_emb.append(emb.loc[w_key].values - emb.loc[l_key].values)
            y.append(1)
        else:
            rows_shape.append(shape.loc[l_key].values - shape.loc[w_key].values)
            rows_emb.append(emb.loc[l_key].values - emb.loc[w_key].values)
            y.append(0)
        groups.append(frozenset([b["winner_model"], b["loser_model"]]))

    X_shape = np.vstack(rows_shape)
    X_emb = np.vstack(rows_emb)
    y = np.array(y)
    g2id = {g: i for i, g in enumerate(set(groups))}
    group_ids = np.array([g2id[g] for g in groups])
    return X_shape, X_emb, y, group_ids


def cv_accuracy(X, y, group_ids, C=1.0):
    n_splits = min(5, len(set(group_ids)))
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    for tr, te in gkf.split(X, y, group_ids):
        accs.append(LogisticRegression(max_iter=1000, C=C).fit(X[tr], y[tr]).score(X[te], y[te]))
    return float(np.mean(accs)), float(np.std(accs))


# ------------------- Run ablation -------------------
print("=" * 92)
print("Ablation: how much of the per-battle predictability is in 17 interpretable features")
print("vs 384-dim bge-small embedding vs both concatenated?")
print("(leave-pair-of-models out, 5-fold)")
print("=" * 92)

shapes = {b: build_shape_features(b) for b in ["eq", "writing", "style"]}
embeds = {b: build_embeddings(b) for b in ["eq", "writing", "style"]}

print(f"\n{'slice':<55} {'shape (17d)':>13} {'emb (384d)':>13} {'both':>13}")
print("-" * 94)
results_tbl = []
for bench, judge in SLICES:
    Xs, Xe, y, g = pair_dataset(bench, judge, shapes[bench], embeds[bench])
    a_s, _ = cv_accuracy(Xs, y, g)
    a_e, _ = cv_accuracy(Xe, y, g)
    a_b, _ = cv_accuracy(np.hstack([Xs, Xe]), y, g)
    label = f"{bench}/{judge[:40]}"
    print(f"{label:<55} {a_s:>11.3f}   {a_e:>11.3f}   {a_b:>11.3f}")
    results_tbl.append((bench, judge, len(y), a_s, a_e, a_b))

print()
print("Key question per slice: how much does going from 17d → 384d BUY?")
for bench, judge, n, a_s, a_e, a_b in results_tbl:
    gap = a_e - a_s
    label = f"{bench}/{judge[:40]}"
    print(f"  {label:<55} Δ(emb - shape) = {gap:+.3f}  (both ≈ {a_b:.3f})")


# ------------------- Interpretation: project embedding direction onto shape features -------------------
print()
print("=" * 92)
print("Interpretation: train full-embedding probe, then see which shape features")
print("correlate with each response's projection onto the probe's preference direction.")
print("=" * 92)

for bench, judge in SLICES:
    shape = shapes[bench]
    emb = embeds[bench]
    feat_cols = list(shape.columns)
    # Get all (key-level) response rows that appear in either winner or loser for any battle
    # Simpler: just use all keys present in both indexes
    shared = shape.index.intersection(emb.index)
    Xs = shape.loc[shared].values
    Xe = emb.loc[shared].values

    # Train the embedding probe on ALL this slice's battles (no CV, we want the best direction)
    battles = load_battles(bench)
    sub = battles[battles["judge_model"] == judge]
    rng = np.random.RandomState(1)
    Xd, yd, groups = [], [], []
    for _, b in sub.iterrows():
        wk, lk = (b["winner_model"], b["key"]), (b["loser_model"], b["key"])
        if wk not in emb.index or lk not in emb.index:
            continue
        if rng.rand() < 0.5:
            Xd.append(emb.loc[wk].values - emb.loc[lk].values); yd.append(1)
        else:
            Xd.append(emb.loc[lk].values - emb.loc[wk].values); yd.append(0)
    Xd = np.vstack(Xd); yd = np.array(yd)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xd, yd)
    direction = clf.coef_.ravel()
    direction = direction / np.linalg.norm(direction)

    # For each response, project onto the direction
    projections = Xe @ direction  # scalar per response
    # Correlate with each shape feature
    print(f"\n--- {bench} / {judge[:40]} ---")
    rows = []
    for i, col in enumerate(feat_cols):
        r = np.corrcoef(Xs[:, i], projections)[0, 1]
        rows.append((col, r))
    rows.sort(key=lambda x: -abs(x[1]))
    for col, r in rows:
        star = "★" if abs(r) > 0.4 else (" " if abs(r) > 0.2 else ".")
        print(f"  {star} {r:+.3f}  {col}")
