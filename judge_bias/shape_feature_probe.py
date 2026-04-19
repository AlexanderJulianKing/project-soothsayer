"""Judge-bias shape probe: what textual features predict winning?

For each pairwise battle (winner, loser) on a given scenario/prompt, compute 17
interpretable textual features for both responses, then compute
(winner_features - loser_features). Averaging across battles isolates the
feature direction the judge systematically rewards, while fully controlling for
prompt content (both responses answered the same prompt).

Paired-within-battle design means any residual signal is judge preference over
response *shape*, not scenario difficulty or model skill.

Supports both EQ (multi-turn conversations joined into one string per scenario)
and Writing (single-turn stories).

Usage:
    python3 judge_bias/shape_feature_probe.py --benchmark eq
    python3 judge_bias/shape_feature_probe.py --benchmark writing --judge "Grok 4 Fast"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESP_CACHE = ROOT / "embeddings/cache/all_responses.parquet"
EQ_BATTLES = ROOT / "soothsayer_eq/results/battle_history.csv"
WR_BATTLES = ROOT / "soothsayer_writing/results/battle_history.csv"
ST_BATTLES = ROOT / "soothsayer_style/results/battle_history.csv"


WORD_RE = re.compile(r"\b\w+\b")
SENT_END = re.compile(r"[.!?]+\s")


def extract_features(text: str) -> dict:
    words = WORD_RE.findall(text)
    word_count = len(words)
    char_count = len(text)
    newlines = text.count("\n")
    sentence_count = max(1, len(SENT_END.findall(text)) + 1)
    first_person = sum(1 for w in words if w.lower() in {"i", "me", "my", "mine", "myself"})
    second_person = sum(1 for w in words if w.lower() in {"you", "your", "yours", "yourself"})
    hedges = sum(1 for w in words if w.lower() in {"perhaps", "maybe", "might", "could", "possibly", "seems", "somewhat"})
    questions = text.count("?")
    exclamations = text.count("!")
    em_dashes = text.count("—") + text.count("--")
    ellipses = text.count("…") + text.count("...")
    bullet_lines = len(re.findall(r"^\s*[-*•]\s", text, re.MULTILINE))
    numbered_lines = len(re.findall(r"^\s*\d+[\.\)]\s", text, re.MULTILINE))
    bold_markers = text.count("**") // 2
    italic_markers = max(0, text.count("*") - 2 * (text.count("**") // 2)) // 2
    all_caps_words = sum(1 for w in words if len(w) > 2 and w.isupper())
    return {
        "char_count": char_count,
        "word_count": word_count,
        "newlines": newlines,
        "sentence_count": sentence_count,
        "avg_sentence_words": word_count / sentence_count,
        "first_person_rate": first_person / max(1, word_count),
        "second_person_rate": second_person / max(1, word_count),
        "hedge_rate": hedges / max(1, word_count),
        "question_rate": questions / max(1, sentence_count),
        "exclamation_rate": exclamations / max(1, sentence_count),
        "em_dashes_per_100w": 100 * em_dashes / max(1, word_count),
        "ellipses_per_100w": 100 * ellipses / max(1, word_count),
        "bullet_lines_per_1kw": 1000 * bullet_lines / max(1, word_count),
        "numbered_lines_per_1kw": 1000 * numbered_lines / max(1, word_count),
        "bold_markers_per_1kw": 1000 * bold_markers / max(1, word_count),
        "italic_markers_per_1kw": 1000 * italic_markers / max(1, word_count),
        "all_caps_words_per_1kw": 1000 * all_caps_words / max(1, word_count),
    }


def build_response_store(bench: str) -> pd.Series:
    """Return one string per (model, scenario_or_prompt) key."""
    resp = pd.read_parquet(RESP_CACHE)
    if bench == "eq":
        sub = resp[resp.benchmark == "eq"].copy()
        sub["key"] = sub["prompt_id"].str.extract(r"s(\d+)_t\d+").astype(int)
        sub["turn"] = sub["prompt_id"].str.extract(r"s\d+_t(\d+)").astype(int)
        sub = sub.sort_values(["model", "key", "run_id", "turn"])
        # Concatenate all 3 turns for each (model, scenario, run_id)
        per_run = (sub.groupby(["model", "key", "run_id"])["response_text"]
                   .agg(lambda s: "\n\n".join(s)))
        # Take first run per (model, scenario); the judge evaluated one specific conversation.
        out = per_run.reset_index().groupby(["model", "key"]).first()["response_text"]
    elif bench == "writing":
        sub = resp[resp.benchmark == "writing"].copy()
        sub["key"] = sub["prompt_id"].astype(int)
        sub = sub.sort_values(["model", "key", "run_id"])
        out = sub.groupby(["model", "key"]).first()["response_text"]
    elif bench == "style":
        # Style has multiple runs per (model, question). The judge saw one
        # specific response per battle (run_id tracked in model_outputs.csv,
        # but not in the battle key). Take first run as the canonical text —
        # consistent with EQ/Writing handling.
        sub = resp[resp.benchmark == "style"].copy()
        sub["key"] = sub["prompt_id"].astype(int)
        sub = sub.sort_values(["model", "key", "run_id"])
        out = sub.groupby(["model", "key"]).first()["response_text"]
    else:
        raise ValueError(f"unknown benchmark: {bench}")
    return out


def load_battles(bench: str, include_criteria: bool = False) -> pd.DataFrame:
    cols_base = ["judge_model", "winner_model", "loser_model"]
    if bench == "eq":
        cols = cols_base + ["scenario_id"] + (["criteria_json"] if include_criteria else [])
        df = pd.read_csv(EQ_BATTLES, usecols=cols)
        df.rename(columns={"scenario_id": "key"}, inplace=True)
    elif bench == "writing":
        cols = cols_base + ["prompt_id"] + (["criteria_json"] if include_criteria else [])
        df = pd.read_csv(WR_BATTLES, usecols=cols)
        df.rename(columns={"prompt_id": "key"}, inplace=True)
    elif bench == "style":
        cols = cols_base + ["question_id"] + (["criteria_json"] if include_criteria else [])
        df = pd.read_csv(ST_BATTLES, usecols=cols)
        df.rename(columns={"question_id": "key"}, inplace=True)
        df = df.dropna(subset=["judge_model"]).reset_index(drop=True)
    else:
        raise ValueError(f"unknown benchmark: {bench}")
    # Normalize trailing whitespace that occurs on some judge labels.
    df["judge_model"] = df["judge_model"].str.strip()
    return df


def compute_deltas(battles: pd.DataFrame, feats: pd.DataFrame) -> np.ndarray:
    deltas = []
    for _, b in battles.iterrows():
        w = (b["winner_model"], b["key"])
        l = (b["loser_model"], b["key"])
        if w not in feats.index or l not in feats.index:
            continue
        deltas.append((feats.loc[w] - feats.loc[l]).values)
    return np.vstack(deltas) if deltas else np.empty((0, feats.shape[1]))


def report(label: str, deltas: np.ndarray, feat_cols: list[str], loser_levels: pd.Series | None = None):
    n = len(deltas)
    if n == 0:
        print(f"[{label}] no battles")
        return None
    mean = deltas.mean(axis=0)
    std = deltas.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(std > 0, mean / (std / np.sqrt(n)), 0.0)
    win_rate = (deltas > 0).mean(axis=0)
    df = pd.DataFrame({
        "mean_delta": mean,
        "t_stat": t,
        "winner>loser %": win_rate * 100,
    }, index=feat_cols)
    if loser_levels is not None:
        df.insert(0, "loser_mean", loser_levels.values)
    df["abs_t"] = df["t_stat"].abs()
    df = df.sort_values("abs_t", ascending=False).drop(columns=["abs_t"])

    print(f"\n=== {label} (n={n} paired battles) ===\n")
    header = f"{'feature':<30}"
    if loser_levels is not None:
        header += f" {'loser mean':>12}"
    header += f" {'mean Δ':>12} {'t-stat':>8} {'w>l %':>8}"
    print(header)
    print("-" * len(header))
    for feat, row in df.iterrows():
        sign = "+" if row["mean_delta"] >= 0 else ""
        line = f"{feat:<30}"
        if loser_levels is not None:
            line += f" {row['loser_mean']:>12.3f}"
        line += f" {sign}{row['mean_delta']:>11.3f} {row['t_stat']:>+8.2f} {row['winner>loser %']:>7.1f}%"
        print(line)
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["eq", "writing", "style"])
    ap.add_argument("--judges", nargs="*", default=None,
                    help="subset of judges to report (default: all with >=50 battles)")
    ap.add_argument("--per_criterion", action="store_true",
                    help="split battles by per-criterion winner (from criteria_json) and report per criterion")
    args = ap.parse_args()

    print(f"loading {args.benchmark} responses...")
    texts = build_response_store(args.benchmark)
    print(f"  {len(texts)} (model, {'scenario' if args.benchmark=='eq' else 'prompt'}) pairs")

    print("extracting features...")
    feats = texts.apply(extract_features).apply(pd.Series)
    feat_cols = list(feats.columns)

    battles = load_battles(args.benchmark, include_criteria=args.per_criterion)
    print(f"\njudge counts: {battles['judge_model'].value_counts().to_dict()}")

    judges = args.judges
    if not judges:
        counts = battles["judge_model"].value_counts()
        judges = counts[counts >= 50].index.tolist()

    # Per-criterion mode: split each battle into one synthetic sub-battle
    # per criterion (using that criterion's winner/loser) so we can measure
    # per-criterion shape preferences.
    if args.per_criterion:
        import json as _json
        print("\nper-criterion mode: parsing criteria_json ...")
        rows = []
        for _, b in battles.iterrows():
            try:
                crits = _json.loads(b["criteria_json"])
            except Exception:
                continue
            for c in crits:
                name = c.get("name")
                win = c.get("winner")
                if not name or not win:
                    continue
                a_m = b["winner_model"]
                b_m = b["loser_model"]
                # criterion winner may be either the battle winner or loser
                if win == a_m:
                    sub_w, sub_l = a_m, b_m
                elif win == b_m:
                    sub_w, sub_l = b_m, a_m
                else:
                    continue
                rows.append({
                    "judge_model": b["judge_model"], "criterion": name,
                    "winner_model": sub_w, "loser_model": sub_l, "key": b["key"],
                })
        per_crit = pd.DataFrame(rows)
        print(f"  expanded {len(battles)} battles → {len(per_crit)} per-criterion sub-battles")
        print(f"  criteria: {per_crit['criterion'].value_counts().to_dict()}")

        crit_means = {}
        for (j, crit), grp in per_crit.groupby(["judge_model", "criterion"]):
            if len(grp) < 50:
                continue
            deltas = compute_deltas(grp, feats)
            m = report(f"{j} on {args.benchmark} — criterion '{crit}'", deltas, feat_cols, None)
            crit_means[(j, crit)] = m

        if len(crit_means) >= 2:
            print(f"\n=== per-criterion feature-preference agreement ({args.benchmark}) ===")
            print("(Pearson r across the 17 feature deltas between criteria)")
            keys = list(crit_means.keys())
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    ma, mb = crit_means[a], crit_means[b]
                    r = np.corrcoef(ma, mb)[0, 1]
                    cos = (ma @ mb) / (np.linalg.norm(ma) * np.linalg.norm(mb))
                    a_lbl = f"{a[0]} | {a[1]}"
                    b_lbl = f"{b[0]} | {b[1]}"
                    print(f"  {a_lbl:<55} ↔ {b_lbl:<55}  r={r:+.3f}  cos={cos:+.3f}")
        return

    # Per-judge feature deltas (default mode)
    judge_means = {}
    for j in judges:
        sub = battles[battles["judge_model"] == j]
        deltas = compute_deltas(sub, feats)
        # loser level relative only for the first judge (same denominator for all)
        if j == judges[0]:
            ll = feats.loc[[(b["loser_model"], b["key"]) for _, b in sub.iterrows()
                            if (b["loser_model"], b["key"]) in feats.index]].mean()
        else:
            ll = None
        m = report(f"{j} on {args.benchmark}", deltas, feat_cols, ll)
        judge_means[j] = m

    # Pairwise agreement between judges (Pearson r across features)
    if len(judges) >= 2:
        print(f"\n=== judge-to-judge feature-preference agreement ({args.benchmark}) ===")
        print("(Pearson r across the 17 feature deltas; +1 = reward exactly the same features)")
        names = [j for j in judges if judge_means[j] is not None]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ma, mb = judge_means[a], judge_means[b]
                r = np.corrcoef(ma, mb)[0, 1]
                cos = (ma @ mb) / (np.linalg.norm(ma) * np.linalg.norm(mb))
                print(f"  {a:<50} ↔ {b:<50}  r={r:+.3f}  cos={cos:+.3f}")


if __name__ == "__main__":
    main()
