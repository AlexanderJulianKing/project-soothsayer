"""Arena-style style-controlled ratings for each of our pairwise benchmarks.

Fits a Bradley-Terry-with-covariates logistic regression per benchmark:

    log P(A beats B) / P(B beats A)
        = skill_A - skill_B + β · (style_A - style_B)

Coefficients on model indicators are the style-controlled skill ratings.
Coefficients on shape-feature deltas are the universal style preferences
the judge rewards NET of model skill.

Compare to the skill-only baseline (same fit without style deltas) to get
rating shifts: who gains / loses from style-controlling.

Runs on EQ (Grok 4 Fast + Gemini 3.0 Flash pooled), Writing (Grok + Gemini
pooled), and Style tonebench (Grok 4.1 Fast).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, lil_matrix
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from judge_bias.shape_feature_probe import (
    build_response_store, extract_features, load_battles,
)


# Style feature subset — pick the ones most directly analogous to Arena's
# (length + structural formatting + the universal voice markers).
STYLE_FEATURES = [
    "char_count", "word_count", "newlines", "sentence_count",
    "avg_sentence_words",
    "first_person_rate", "second_person_rate", "hedge_rate",
    "em_dashes_per_100w", "ellipses_per_100w",
    "bullet_lines_per_1kw", "numbered_lines_per_1kw",
    "bold_markers_per_1kw", "italic_markers_per_1kw",
    "all_caps_words_per_1kw",
    "exclamation_rate", "question_rate",
]


def build_dataset(bench: str, judges: list[str] | None = None,
                  criterion: str | None = None):
    """Return (X_model_sparse, X_style_dense, y, model_names, battle_df_used).

    - X_model: sparse (n_battles × n_models) signed indicator — +1 for A's model,
      -1 for B's. The logistic regression on this predicts log-odds as
      β_A - β_B.
    - X_style: dense (n_battles × n_style_features) standardized style delta.

    If `criterion` is set, the battle winner is the per-criterion winner from
    criteria_json instead of the top-level winner.
    """
    import json as _json

    texts = build_response_store(bench)
    feats = texts.apply(extract_features).apply(pd.Series)
    feats = feats[STYLE_FEATURES].copy()
    # Z-score style features for numerical stability + interpretable coefficients
    mu = feats.mean(); sd = feats.std().replace(0, 1)
    feats_z = (feats - mu) / sd

    battles = load_battles(bench, include_criteria=criterion is not None)
    if judges:
        battles = battles[battles["judge_model"].isin(judges)]

    # Randomize A/B orientation per row so label isn't a function of order.
    rng = np.random.RandomState(0)
    rows = []
    for _, b in battles.iterrows():
        if criterion is None:
            winner, loser = b["winner_model"], b["loser_model"]
        else:
            # Use per-criterion winner from criteria_json
            try:
                crits = _json.loads(b["criteria_json"])
            except Exception:
                continue
            match = [c for c in crits if c.get("name") == criterion]
            if not match or not match[0].get("winner"):
                continue
            crit_winner = match[0]["winner"]
            a_m, b_m = b["winner_model"], b["loser_model"]
            if crit_winner == a_m:
                winner, loser = a_m, b_m
            elif crit_winner == b_m:
                winner, loser = b_m, a_m
            else:
                continue

        w = (winner, b["key"]); l = (loser, b["key"])
        if w not in feats_z.index or l not in feats_z.index:
            continue
        if rng.rand() < 0.5:
            rows.append((winner, loser, b["key"],
                         feats_z.loc[w].values - feats_z.loc[l].values, 1))
        else:
            rows.append((loser, winner, b["key"],
                         feats_z.loc[l].values - feats_z.loc[w].values, 0))

    model_names = sorted(set(r[0] for r in rows) | set(r[1] for r in rows))
    idx = {m: i for i, m in enumerate(model_names)}

    n = len(rows); m = len(model_names); d = len(STYLE_FEATURES)
    Xmod = lil_matrix((n, m), dtype=np.float32)
    Xsty = np.empty((n, d), dtype=np.float32)
    y = np.empty(n, dtype=np.int32)
    for i, (a, b_, _, sdelta, label) in enumerate(rows):
        Xmod[i, idx[a]] = 1.0
        Xmod[i, idx[b_]] = -1.0
        Xsty[i, :] = sdelta
        y[i] = label
    return Xmod.tocsr(), Xsty, y, model_names


def fit_bradley_terry(Xmod, Xsty, y, use_style=True, C=1.0):
    if use_style:
        X = hstack([Xmod, csr_matrix(Xsty)]).tocsr()
    else:
        X = Xmod
    # fit_intercept=False because model identifiers absorb any global bias
    lr = LogisticRegression(penalty="l2", C=C, fit_intercept=False,
                             solver="lbfgs", max_iter=2000).fit(X, y)
    return lr


def run(bench: str, judges: list[str] | None, criterion: str | None = None, tag: str | None = None):
    label = tag or (f"{bench}" + (f" / {criterion}" if criterion else "") + (f" / {judges[0][:30]}" if judges and len(judges)==1 else ""))
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {label}   judges: {judges or 'all'}   criterion: {criterion or '(battle winner)'}")
    print(f"{'='*80}")
    Xmod, Xsty, y, model_names = build_dataset(bench, judges, criterion)
    n_bat, n_mod = Xmod.shape
    print(f"  {n_bat} battles, {n_mod} models, {len(STYLE_FEATURES)} style features")
    if n_bat < 200 or n_mod < 20:
        print("  skipping — too few battles or models")
        return None

    # Baseline: no style controls.
    lr0 = fit_bradley_terry(Xmod, Xsty, y, use_style=False)
    skill0 = lr0.coef_[0, :n_mod]

    # Style-controlled.
    lr1 = fit_bradley_terry(Xmod, Xsty, y, use_style=True)
    skill1 = lr1.coef_[0, :n_mod]
    style_coef = lr1.coef_[0, n_mod:]

    # In-sample accuracy
    from sklearn.metrics import log_loss
    acc0 = lr0.score(Xmod, y)
    acc1 = lr1.score(hstack([Xmod, csr_matrix(Xsty)]).tocsr(), y)

    print(f"\n  in-sample accuracy  (model-only): {acc0:.3f}")
    print(f"  in-sample accuracy  (+ style):    {acc1:.3f}   Δ = {acc1 - acc0:+.3f}")

    # Style coefficients (z-scored deltas → "1 sd increase in feature delta →
    # exp(coef) odds multiplier on winning")
    print(f"\n  Style coefficients (on standardized deltas, bench={bench}):")
    order = np.argsort(-np.abs(style_coef))
    for i in order:
        feat = STYLE_FEATURES[i]
        coef = style_coef[i]
        or_ = np.exp(coef)
        star = "★" if abs(coef) > 0.2 else " "
        print(f"    {star} {coef:+.3f}   (odds × {or_:.2f})   {feat}")

    # Rating shifts: who gained / lost from style-controlling?
    shift = skill1 - skill0
    order = np.argsort(-np.abs(shift))
    up = np.argsort(-shift)[:10]
    down = np.argsort(shift)[:10]

    print(f"\n  Biggest WINNERS from style control (ratings went up when style was")
    print(f"  controlled → their wins weren't due to style privilege):")
    for i in up:
        print(f"    {shift[i]:+.3f}   raw={skill0[i]:+.3f} → controlled={skill1[i]:+.3f}   {model_names[i]}")

    print(f"\n  Biggest LOSERS from style control (ratings went down when style was")
    print(f"  controlled → their wins leaned on style advantages):")
    for i in down:
        print(f"    {shift[i]:+.3f}   raw={skill0[i]:+.3f} → controlled={skill1[i]:+.3f}   {model_names[i]}")

    return dict(
        bench=bench, tag=tag or bench, models=model_names, skill_raw=skill0, skill_controlled=skill1,
        style_coef=style_coef, style_features=STYLE_FEATURES,
    )


if __name__ == "__main__":
    GEMINI = "Gemini 3.0 Flash Preview (2025-12-17)"
    results = []
    # EQ: Gemini only
    results.append(run("eq", [GEMINI], tag="eq_gemini"))
    # Writing: Grok only
    results.append(run("writing", ["Grok 4 Fast"], tag="writing_grok"))
    # Tonebench: two criteria treated as two separate pairwise benchmarks
    results.append(run("style", ["Grok 4.1 Fast"], criterion="signal_density",
                       tag="tone_signal_density"))
    results.append(run("style", ["Grok 4.1 Fast"], criterion="conversational_confidence",
                       tag="tone_conv_confidence"))

    # Compare style coefficients across the 4 targeted fits
    print(f"\n\n{'='*95}")
    print("Style coefficients across the 4 target fits (same standardized features)")
    print(f"{'='*95}\n")
    header = f"{'feature':<30}"
    for r in results:
        header += f" {r['tag'][:15]:>16}"
    print(header)
    print("-" * len(header))
    for i, feat in enumerate(STYLE_FEATURES):
        line = f"{feat:<30}"
        for r in results:
            if r is None:
                line += f" {'--':>16}"
            else:
                line += f" {r['style_coef'][i]:>+16.3f}"
        print(line)

    # Save style-controlled ratings
    out_dir = ROOT / "judge_bias" / "output"
    out_dir.mkdir(exist_ok=True)
    for r in results:
        if r is None:
            continue
        df = pd.DataFrame({
            "model": r["models"],
            "skill_raw": r["skill_raw"],
            "skill_controlled": r["skill_controlled"],
            "shift": r["skill_controlled"] - r["skill_raw"],
        }).sort_values("shift", ascending=False)
        path = out_dir / f"style_controlled_ratings_{r['tag']}.csv"
        df.to_csv(path, index=False)
        print(f"\nwrote {path.relative_to(ROOT)}")
