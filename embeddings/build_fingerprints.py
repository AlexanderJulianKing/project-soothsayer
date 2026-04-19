"""Pool response embeddings into per-model fingerprints and reduce via PCA.

Produces `model_fingerprints.csv` with schema (model_name, sem_f01 ... sem_fK)
ready to merge into `benchmark_combiner/benchmarks/clean_combined_all_benches.csv`
on the `model_name` column.

Five pooling modes:

--mode cross_bench
    Average every response from a model across all 4 benchmarks into one
    unit-normalized vector. Simplest baseline; discards per-task variation.

--mode per_bench (default)
    Average per (model, benchmark). Concatenate the 4 per-benchmark vectors
    per model (in fixed alphabetical benchmark order). Models missing a
    benchmark get the per-benchmark centroid (mean across models that have
    it) — light imputation to keep feature vectors aligned.

--mode per_bench_eq_split
    Like per_bench, but splits EQ into its first turn (t1) and last turn (t3)
    as two separate slots, dropping the middle turn. Produces a 5-slot concat
    per model: eq_t1, eq_t3, logic, style, writing. Tests whether multi-turn
    escalation carries semantic signal that mean-pooling across turns discards.

--mode per_bench_eq_and_style_split
    Extends per_bench_eq_split by also splitting Style into two register
    slots: style_tech (prompts 1-5: STEM/technical — ice-cube math, CPR
    ethics, genetics, chemistry, astronomy) and style_casual (prompts 6-9:
    React opinion, Python-in-anime-voice, games, sourdough). Produces a
    6-slot concat: eq_t1, eq_t3, logic, style_tech, style_casual, writing.
    Tests whether register-switching between technical and casual prompts
    carries semantic signal that pooling across registers discards.

--mode per_bench_eq_split_style_per_prompt
    Extends per_bench_eq_split by giving every individual Style prompt its
    own 384-dim slot (9 slots for the 9 Style prompts). Each style slot =
    run-averaged embedding for that prompt. Produces a 13-slot concat:
    eq_t1, eq_t3, logic, style_q1 ... style_q9, writing. Tests the
    hypothesis that per-prompt structure in Style carries signal that
    pooling across prompts averages away, while keeping run-averaging as
    within-slot denoising.

PCA is fit on the full cross-benchmark matrix for cross_bench mode, or the
full concatenated matrix for the per_bench* modes. Unsupervised, no target
leakage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "embeddings" / "cache"
IN_FILE = CACHE_DIR / "response_embeddings.parquet"
OUT_FILE = CACHE_DIR / "model_fingerprints.csv"
COMBINED_CSV = PROJECT_ROOT / "benchmark_combiner" / "benchmarks" / "clean_combined_all_benches.csv"

BENCHMARKS = ["eq", "logic", "style", "writing"]  # fixed order for concatenation
# For per_bench_eq_split mode: EQ is split into first/last turn, middle turn dropped.
# Slot order matches the rest of the pipeline (eq first, then the other 3 alphabetical).
EQ_SPLIT_SLOTS = ["eq_t1", "eq_t3", "logic", "style", "writing"]

# For per_bench_eq_and_style_split: also split Style by register.
# Prompt ids in cache/response_embeddings.parquet for style are strings "1".."9".
# See soothsayer_style/questions.txt for the source prompts.
STYLE_TECH_PROMPTS = {"1", "2", "3", "4", "5"}  # STEM / technical reasoning
STYLE_CASUAL_PROMPTS = {"6", "7", "8", "9"}     # casual / lifestyle / creative
EQ_STYLE_SPLIT_SLOTS = ["eq_t1", "eq_t3", "logic", "style_tech", "style_casual", "writing"]

# For per_bench_eq_split_style_per_prompt: every Style prompt is its own slot.
STYLE_PROMPTS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
EQ_STYLE_PER_PROMPT_SLOTS = (
    ["eq_t1", "eq_t3", "logic"] + [f"style_q{p}" for p in STYLE_PROMPTS] + ["writing"]
)


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms == 0, 1.0, norms)


def build_cross_bench(df: pd.DataFrame, emb_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby("model")[emb_cols].mean()
    grouped_values = _normalize(grouped.values)
    return pd.DataFrame(grouped_values, index=grouped.index, columns=emb_cols)


def build_per_bench(df: pd.DataFrame, emb_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Return (matrix, column_names) where each row is a concatenated per-benchmark
    fingerprint for one model.
    """
    per = df.groupby(["model", "benchmark"])[emb_cols].mean()
    # Re-normalize each (model, benchmark) vector
    per_vals = _normalize(per.values)
    per = pd.DataFrame(per_vals, index=per.index, columns=emb_cols)

    # Per-benchmark centroid for imputation
    centroids = {}
    for b in BENCHMARKS:
        if b in per.index.get_level_values("benchmark"):
            centroids[b] = _normalize(per.xs(b, level="benchmark").mean(axis=0).values.reshape(1, -1)).ravel()
        else:
            centroids[b] = np.zeros(len(emb_cols), dtype=np.float32)

    models = sorted(df["model"].unique())
    wide_cols = [f"{b}_{c}" for b in BENCHMARKS for c in emb_cols]
    out = np.empty((len(models), len(wide_cols)), dtype=np.float32)

    n_imputed = {b: 0 for b in BENCHMARKS}
    for i, m in enumerate(models):
        chunks = []
        for b in BENCHMARKS:
            key = (m, b)
            if key in per.index:
                chunks.append(per.loc[key].values)
            else:
                chunks.append(centroids[b])
                n_imputed[b] += 1
        out[i] = np.concatenate(chunks)

    print(f"per-bench imputation counts (missing benchmark → centroid):")
    for b, n in n_imputed.items():
        print(f"  {b}: {n}/{len(models)} models")

    return pd.DataFrame(out, index=pd.Index(models, name="model"), columns=wide_cols), wide_cols


def _eq_turn_slot(prompt_id: str) -> str | None:
    """Map an EQ prompt_id like 's101_t1' to the slot label 'eq_t1'/'eq_t3', or
    None if this row should be dropped (middle turn t2 or malformed id).
    """
    if "_t" not in prompt_id:
        return None
    turn = prompt_id.rsplit("_t", 1)[-1]
    if turn == "1":
        return "eq_t1"
    if turn == "3":
        return "eq_t3"
    return None  # drop t2 and anything else


def build_per_bench_eq_split(df: pd.DataFrame, emb_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Per-benchmark fingerprint with EQ split into t1/t3 slots (middle turn dropped).

    Produces a 5-slot concat per model: eq_t1, eq_t3, logic, style, writing.
    """
    df = df.copy()
    # Assign a "slot" label per row: EQ rows map to eq_t1 / eq_t3 / (drop), other
    # benchmarks map to their benchmark name directly.
    eq_mask = df["benchmark"] == "eq"
    df.loc[~eq_mask, "slot"] = df.loc[~eq_mask, "benchmark"]
    df.loc[eq_mask, "slot"] = df.loc[eq_mask, "prompt_id"].map(_eq_turn_slot)
    dropped = eq_mask.sum() - df.loc[eq_mask, "slot"].notna().sum()
    print(f"EQ rows: {int(eq_mask.sum())} total, {int(dropped)} dropped (middle turn t2)")
    df = df[df["slot"].notna()].reset_index(drop=True)

    per = df.groupby(["model", "slot"])[emb_cols].mean()
    per_vals = _normalize(per.values)
    per = pd.DataFrame(per_vals, index=per.index, columns=emb_cols)

    # Per-slot centroid for imputation
    centroids = {}
    for s in EQ_SPLIT_SLOTS:
        if s in per.index.get_level_values("slot"):
            centroids[s] = _normalize(per.xs(s, level="slot").mean(axis=0).values.reshape(1, -1)).ravel()
        else:
            centroids[s] = np.zeros(len(emb_cols), dtype=np.float32)

    models = sorted(df["model"].unique())
    wide_cols = [f"{s}_{c}" for s in EQ_SPLIT_SLOTS for c in emb_cols]
    out = np.empty((len(models), len(wide_cols)), dtype=np.float32)

    n_imputed = {s: 0 for s in EQ_SPLIT_SLOTS}
    for i, m in enumerate(models):
        chunks = []
        for s in EQ_SPLIT_SLOTS:
            key = (m, s)
            if key in per.index:
                chunks.append(per.loc[key].values)
            else:
                chunks.append(centroids[s])
                n_imputed[s] += 1
        out[i] = np.concatenate(chunks)

    print("per-slot imputation counts (missing slot → centroid):")
    for s, n in n_imputed.items():
        print(f"  {s}: {n}/{len(models)} models")

    return pd.DataFrame(out, index=pd.Index(models, name="model"), columns=wide_cols), wide_cols


def _style_register_slot(prompt_id: str) -> str | None:
    """Map a Style prompt_id ('1'..'9') to its register slot, or None if unknown."""
    if prompt_id in STYLE_TECH_PROMPTS:
        return "style_tech"
    if prompt_id in STYLE_CASUAL_PROMPTS:
        return "style_casual"
    return None


def build_per_bench_eq_and_style_split(df: pd.DataFrame, emb_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """6-slot fingerprint: EQ split by turn (t1/t3) + Style split by register.

    Slots: eq_t1, eq_t3, logic, style_tech, style_casual, writing.
    """
    df = df.copy()
    eq_mask = df["benchmark"] == "eq"
    style_mask = df["benchmark"] == "style"
    other_mask = ~(eq_mask | style_mask)

    df.loc[other_mask, "slot"] = df.loc[other_mask, "benchmark"]
    df.loc[eq_mask, "slot"] = df.loc[eq_mask, "prompt_id"].map(_eq_turn_slot)
    df.loc[style_mask, "slot"] = df.loc[style_mask, "prompt_id"].astype(str).map(_style_register_slot)

    eq_dropped = eq_mask.sum() - df.loc[eq_mask, "slot"].notna().sum()
    style_dropped = style_mask.sum() - df.loc[style_mask, "slot"].notna().sum()
    print(f"EQ rows: {int(eq_mask.sum())} total, {int(eq_dropped)} dropped (middle turn t2)")
    print(f"Style rows: {int(style_mask.sum())} total, {int(style_dropped)} dropped (unmapped prompt_id)")

    df = df[df["slot"].notna()].reset_index(drop=True)

    per = df.groupby(["model", "slot"])[emb_cols].mean()
    per_vals = _normalize(per.values)
    per = pd.DataFrame(per_vals, index=per.index, columns=emb_cols)

    centroids = {}
    for s in EQ_STYLE_SPLIT_SLOTS:
        if s in per.index.get_level_values("slot"):
            centroids[s] = _normalize(per.xs(s, level="slot").mean(axis=0).values.reshape(1, -1)).ravel()
        else:
            centroids[s] = np.zeros(len(emb_cols), dtype=np.float32)

    models = sorted(df["model"].unique())
    wide_cols = [f"{s}_{c}" for s in EQ_STYLE_SPLIT_SLOTS for c in emb_cols]
    out = np.empty((len(models), len(wide_cols)), dtype=np.float32)

    n_imputed = {s: 0 for s in EQ_STYLE_SPLIT_SLOTS}
    for i, m in enumerate(models):
        chunks = []
        for s in EQ_STYLE_SPLIT_SLOTS:
            key = (m, s)
            if key in per.index:
                chunks.append(per.loc[key].values)
            else:
                chunks.append(centroids[s])
                n_imputed[s] += 1
        out[i] = np.concatenate(chunks)

    print("per-slot imputation counts (missing slot → centroid):")
    for s, n in n_imputed.items():
        print(f"  {s}: {n}/{len(models)} models")

    return pd.DataFrame(out, index=pd.Index(models, name="model"), columns=wide_cols), wide_cols


def build_per_bench_eq_split_style_per_prompt(df: pd.DataFrame, emb_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """13-slot fingerprint: EQ t1/t3 + logic + 9 per-prompt Style slots + writing.

    Each Style slot = run-averaged embedding for that single prompt.
    """
    df = df.copy()
    eq_mask = df["benchmark"] == "eq"
    style_mask = df["benchmark"] == "style"
    other_mask = ~(eq_mask | style_mask)

    df.loc[other_mask, "slot"] = df.loc[other_mask, "benchmark"]
    df.loc[eq_mask, "slot"] = df.loc[eq_mask, "prompt_id"].map(_eq_turn_slot)
    # Per-prompt Style slots: map prompt_id '1'..'9' → 'style_q1'..'style_q9'
    df.loc[style_mask, "slot"] = df.loc[style_mask, "prompt_id"].astype(str).map(
        lambda p: f"style_q{p}" if p in STYLE_PROMPTS else None
    )

    eq_dropped = eq_mask.sum() - df.loc[eq_mask, "slot"].notna().sum()
    style_dropped = style_mask.sum() - df.loc[style_mask, "slot"].notna().sum()
    print(f"EQ rows: {int(eq_mask.sum())} total, {int(eq_dropped)} dropped (middle turn t2)")
    print(f"Style rows: {int(style_mask.sum())} total, {int(style_dropped)} dropped (unmapped prompt_id)")

    df = df[df["slot"].notna()].reset_index(drop=True)

    per = df.groupby(["model", "slot"])[emb_cols].mean()
    per_vals = _normalize(per.values)
    per = pd.DataFrame(per_vals, index=per.index, columns=emb_cols)

    centroids = {}
    for s in EQ_STYLE_PER_PROMPT_SLOTS:
        if s in per.index.get_level_values("slot"):
            centroids[s] = _normalize(per.xs(s, level="slot").mean(axis=0).values.reshape(1, -1)).ravel()
        else:
            centroids[s] = np.zeros(len(emb_cols), dtype=np.float32)

    models = sorted(df["model"].unique())
    wide_cols = [f"{s}_{c}" for s in EQ_STYLE_PER_PROMPT_SLOTS for c in emb_cols]
    out = np.empty((len(models), len(wide_cols)), dtype=np.float32)

    n_imputed = {s: 0 for s in EQ_STYLE_PER_PROMPT_SLOTS}
    for i, m in enumerate(models):
        chunks = []
        for s in EQ_STYLE_PER_PROMPT_SLOTS:
            key = (m, s)
            if key in per.index:
                chunks.append(per.loc[key].values)
            else:
                chunks.append(centroids[s])
                n_imputed[s] += 1
        out[i] = np.concatenate(chunks)

    print("per-slot imputation counts (missing slot → centroid):")
    for s, n in n_imputed.items():
        print(f"  {s}: {n}/{len(models)} models")

    return pd.DataFrame(out, index=pd.Index(models, name="model"), columns=wide_cols), wide_cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cross_bench", "per_bench", "per_bench_eq_split", "per_bench_eq_and_style_split", "per_bench_eq_split_style_per_prompt"], default="per_bench")
    ap.add_argument("--n_components", type=int, default=24)
    ap.add_argument("--in", dest="in_file", type=str, default=str(IN_FILE),
                    help="input parquet from embed_responses.py (default: cache/response_embeddings.parquet)")
    ap.add_argument("--out", type=str, default=str(OUT_FILE))
    args = ap.parse_args()

    in_path = Path(args.in_file)
    if not in_path.exists():
        raise SystemExit(f"input missing: {in_path} — run embed_responses.py first")

    df = pd.read_parquet(in_path)
    emb_cols = [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]
    dim = len(emb_cols)
    print(f"loaded {len(df)} rows, embedding dim={dim}, mode={args.mode}")

    if args.mode == "cross_bench":
        fingerprints = build_cross_bench(df, emb_cols)
    elif args.mode == "per_bench":
        fingerprints, _ = build_per_bench(df, emb_cols)
    elif args.mode == "per_bench_eq_split":
        fingerprints, _ = build_per_bench_eq_split(df, emb_cols)
    elif args.mode == "per_bench_eq_and_style_split":
        fingerprints, _ = build_per_bench_eq_and_style_split(df, emb_cols)
    elif args.mode == "per_bench_eq_split_style_per_prompt":
        fingerprints, _ = build_per_bench_eq_split_style_per_prompt(df, emb_cols)
    else:
        raise ValueError(f"unknown mode: {args.mode}")
    X = fingerprints.values

    print(f"fingerprint matrix: {X.shape}")

    pca = PCA(n_components=args.n_components, random_state=0)
    reduced = pca.fit_transform(X)
    cumvar = pca.explained_variance_ratio_.cumsum()
    print(f"PCA explained variance: first {args.n_components} components = {cumvar[-1]:.3f}")
    top10 = np.round(pca.explained_variance_ratio_[:10], 3).tolist()
    print(f"  top 10 component ratios: {top10}")

    out_cols = [f"sem_f{i+1:02d}" for i in range(args.n_components)]
    out = pd.DataFrame(reduced, index=fingerprints.index, columns=out_cols)
    out.index.name = "model_name"
    out = out.reset_index()
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(out)} models × {args.n_components} components)")

    if COMBINED_CSV.exists():
        combined = pd.read_csv(COMBINED_CSV, usecols=["model_name"])
        both = set(out["model_name"]) & set(combined["model_name"])
        only_comb = set(combined["model_name"]) - set(out["model_name"])
        print(f"\noverlap with clean_combined_all_benches.csv:")
        print(f"  models in both: {len(both)}")
        print(f"  only in combined: {len(only_comb)}")
        if only_comb:
            print(f"  sample: {sorted(only_comb)[:5]}")


if __name__ == "__main__":
    main()
