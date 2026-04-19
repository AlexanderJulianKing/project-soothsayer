"""Can we predict judge preferences from embedding space alone?

Two linear-probe evaluations per (judge, benchmark) slice:

  A. Per-battle: given embed(A) - embed(B), predict which the judge picked.
     Chance = 50%. Leave-pair-of-models out: every held-out battle involves at
     least one model never seen in training, so the probe must generalize via
     response shape, not memorize per-model voice.

  B. Per-model: given a model's fingerprint (mean embedding on the benchmark,
     unit-normalized), predict the model's win-rate with this judge on this
     benchmark. Leave-one-model-out. Compare held-out R² vs chance (r=0).

Both tests are honest: no training data touches the test model's battles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, LeaveOneOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from judge_bias.shape_feature_probe import load_battles

EMB_PATH = ROOT / "embeddings/cache/response_embeddings.parquet"

SLICES = [
    ("eq", "Grok 4 Fast"),
    ("eq", "Gemini 3.0 Flash Preview (2025-12-17)"),
    ("writing", "Grok 4 Fast"),
    ("writing", "Gemini 3.0 Flash Preview (2025-12-17)"),
    ("style", "Grok 4.1 Fast"),
]


def build_key_embeddings(bench: str) -> pd.DataFrame:
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


def build_fingerprints(bench: str) -> pd.DataFrame:
    emb = pd.read_parquet(EMB_PATH)
    sub = emb[emb.benchmark == bench].copy()
    emb_cols = [c for c in sub.columns if c.startswith("e") and c[1:].isdigit()]
    fp = sub.groupby("model")[emb_cols].mean()
    v = fp.values / np.linalg.norm(fp.values, axis=1, keepdims=True).clip(min=1e-12)
    return pd.DataFrame(v, index=fp.index, columns=emb_cols)


def test_a_per_battle(bench, judge, key_emb):
    battles = load_battles(bench)
    sub = battles[battles["judge_model"] == judge]
    rows = []
    for _, b in sub.iterrows():
        w_key = (b["winner_model"], b["key"])
        l_key = (b["loser_model"], b["key"])
        if w_key not in key_emb.index or l_key not in key_emb.index:
            continue
        rows.append((b["winner_model"], b["loser_model"], b["key"],
                     key_emb.loc[w_key].values, key_emb.loc[l_key].values))
    if not rows:
        return None

    # Randomize A/B orientation per battle so the classifier can't cheat on order.
    rng = np.random.RandomState(0)
    X, y, groups = [], [], []
    for wmod, lmod, key, w_vec, l_vec in rows:
        if rng.rand() < 0.5:
            X.append(w_vec - l_vec); y.append(1)
        else:
            X.append(l_vec - w_vec); y.append(0)
        # group = pair-of-models identifier so GroupKFold splits on model pairs
        groups.append(frozenset([wmod, lmod]))
    X = np.vstack(X); y = np.array(y)
    # Replace groups with an integer encoding for GroupKFold
    group_to_id = {g: i for i, g in enumerate(set(groups))}
    group_ids = np.array([group_to_id[g] for g in groups])

    n_splits = min(5, len(set(group_ids)))
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    for tr, te in gkf.split(X, y, group_ids):
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return dict(n=len(X), mean_acc=float(np.mean(accs)), std_acc=float(np.std(accs)),
                n_pairs=len(set(group_ids)))


def test_b_per_model(bench, judge, fp):
    battles = load_battles(bench)
    sub = battles[battles["judge_model"] == judge]
    # Per-model win rate: wins / (wins + losses) across this judge's battles.
    wins = sub["winner_model"].value_counts()
    losses = sub["loser_model"].value_counts()
    models = sorted(set(wins.index) | set(losses.index))
    n_models = []; wr = []
    for m in models:
        w = wins.get(m, 0); l = losses.get(m, 0)
        if (w + l) < 5:  # need a few battles for a stable win-rate
            continue
        if m not in fp.index:
            continue
        n_models.append(m); wr.append(w / (w + l))
    X = fp.loc[n_models].values
    y = np.array(wr)

    # Leave-one-model-out Ridge regression
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        m = Ridge(alpha=1.0).fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    r = np.corrcoef(y, preds)[0, 1]
    return dict(n=len(y), r=float(r), r2=float(r2))


# ---------- Run both probes for every slice ----------

print("=" * 80)
print("A. Per-battle probe: predict which response the judge picked")
print("   Chance = 50%. Leave-pair-of-models out (train and test share no model pair)")
print("=" * 80)

key_embs = {b: build_key_embeddings(b) for b in ["eq", "writing", "style"]}
fps = {b: build_fingerprints(b) for b in ["eq", "writing", "style"]}

a_rows = []
for bench, judge in SLICES:
    r = test_a_per_battle(bench, judge, key_embs[bench])
    if r:
        a_rows.append((bench, judge, r))
        print(f"  {bench:<8} {judge[:40]:<40} n={r['n']:>5}  pairs={r['n_pairs']:>4}  "
              f"accuracy = {r['mean_acc']:.3f} ± {r['std_acc']:.3f}")

print()
print("=" * 80)
print("B. Per-model probe: predict a model's win-rate from its fingerprint")
print("   Leave-one-model-out Ridge. Chance = r=0. 'Good' = r>0.3 (modest), r>0.5 (strong).")
print("=" * 80)

b_rows = []
for bench, judge in SLICES:
    r = test_b_per_model(bench, judge, fps[bench])
    b_rows.append((bench, judge, r))
    print(f"  {bench:<8} {judge[:40]:<40} n_models={r['n']:>3}  "
          f"r = {r['r']:+.3f}   LOO R² = {r['r2']:+.3f}")
