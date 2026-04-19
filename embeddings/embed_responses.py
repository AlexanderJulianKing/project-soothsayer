"""Encode every row of all_responses.parquet into a dense vector.

Defaults to BAAI/bge-small-en-v1.5 (384-dim, 512-token context) on Metal.
--model swaps in any HF embedder; --out writes to a custom parquet so
multiple embedders can be cached side-by-side.

For responses longer than the model's context window, the text is split into
overlapping chunks, each is embedded, and the chunk embeddings are mean-pooled
into a single vector per response.

Resumable: if the output parquet already exists, already-embedded rows are
skipped and the new ones are appended.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "embeddings" / "cache"
IN_FILE = CACHE_DIR / "all_responses.parquet"
DEFAULT_OUT_FILE = CACHE_DIR / "response_embeddings.parquet"

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# Default chunk length stays under bge-small's 512 ceiling. Overridden by
# --chunk_tokens when using a long-context model (nomic, gte-large-v1.5).
DEFAULT_CHUNK_TOKENS = 480
# Overlap between consecutive chunks — preserves some local context at chunk
# boundaries so sentence fragments aren't orphaned.
CHUNK_STRIDE = 48
# Below this, don't bother with a trailing chunk.
MIN_TRAILING_TOKENS = 32
# Char-level cap before tokenization (belt-and-suspenders for giant outliers).
MAX_CHARS = 120_000  # ~30k tokens, still far under the worst case
DEFAULT_BATCH_SIZE = 64
# Default checkpoint cadence: flush pooled responses to disk every N emitted.
# Long-context models on MPS can thrash memory and tank mid-run — flushing lets
# a resumed run pick up near where it left off rather than restarting from zero.
DEFAULT_CHECKPOINT_EVERY = 500
KEY_COLS = ["model", "benchmark", "prompt_id", "run_id"]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def row_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["model"].astype(str)
        + "|" + df["benchmark"].astype(str)
        + "|" + df["prompt_id"].astype(str)
        + "|" + df["run_id"].astype(str)
    )


def chunk_text(text: str, tokenizer, chunk_tokens: int) -> list[str]:
    """Split `text` into overlapping chunks each ≤ chunk_tokens tokens.

    Returns the original text as a single-element list if it already fits.
    """
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(ids) <= chunk_tokens:
        return [text]
    chunks = []
    step = chunk_tokens - CHUNK_STRIDE
    for start in range(0, len(ids), step):
        piece = ids[start:start + chunk_tokens]
        if start > 0 and len(piece) < MIN_TRAILING_TOKENS:
            break
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True))
        if start + chunk_tokens >= len(ids):
            break
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model id for sentence-transformers (default bge-small)")
    ap.add_argument("--out", default=str(DEFAULT_OUT_FILE),
                    help="output parquet path")
    ap.add_argument("--chunk_tokens", type=int, default=DEFAULT_CHUNK_TOKENS,
                    help="max tokens per embedding chunk (stay under model's max_seq_length)")
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                    help="encoder batch size (lower for larger models / longer contexts)")
    ap.add_argument("--max_seq_length", type=int, default=None,
                    help="override model.max_seq_length (e.g. 8192 for nomic/gte-large)")
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="required for nomic and gte-large-v1.5")
    ap.add_argument("--passage_prefix", default="",
                    help="string prepended to each text (e.g. 'search_document: ' for nomic)")
    ap.add_argument("--checkpoint_every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
                    help="flush pooled responses to disk every N responses; pass 0 to disable "
                         "and only write at the end (legacy behavior)")
    args = ap.parse_args()

    out_file = Path(args.out)
    if not IN_FILE.exists():
        raise SystemExit(f"input missing: {IN_FILE} — run collect_responses.py first")

    df = pd.read_parquet(IN_FILE)
    df["_key"] = row_key(df)

    if out_file.exists():
        existing = pd.read_parquet(out_file)
        existing_keys = set(existing["_key"]) if "_key" in existing.columns else set()
        todo = df[~df["_key"].isin(existing_keys)].reset_index(drop=True)
        print(f"resuming: {len(existing)} already embedded, {len(todo)} remaining")
    else:
        existing = None
        todo = df.reset_index(drop=True)
        print(f"fresh run: {len(todo)} rows to embed")

    if len(todo) == 0:
        print("nothing to do")
        return

    device = pick_device()
    print(f"loading {args.model} on {device}...")
    st_kwargs = {"device": device}
    if args.trust_remote_code:
        st_kwargs["trust_remote_code"] = True
    model = SentenceTransformer(args.model, **st_kwargs)
    if args.max_seq_length is not None:
        model.max_seq_length = args.max_seq_length
    dim = model.get_sentence_embedding_dimension()
    print(f"max_seq_length={model.max_seq_length}, embedding_dim={dim}, "
          f"chunk_tokens={args.chunk_tokens}, batch_size={args.batch_size}")
    if args.passage_prefix:
        print(f"prefixing passages with: {args.passage_prefix!r}")

    tokenizer = model.tokenizer
    raw_texts = [t[:MAX_CHARS] for t in todo["response_text"].tolist()]

    # 1) Build a flat list of (response_idx, chunk_text) covering every chunk.
    print("chunking...", flush=True)
    flat_chunks: list[str] = []
    response_to_chunk_count: list[int] = []
    for text in tqdm(raw_texts, desc="chunk"):
        chunks = chunk_text(text, tokenizer, args.chunk_tokens)
        response_to_chunk_count.append(len(chunks))
        flat_chunks.extend(chunks)
    total_chunks = len(flat_chunks)
    multi = sum(1 for c in response_to_chunk_count if c > 1)
    print(f"  {len(raw_texts)} responses → {total_chunks} chunks; "
          f"{multi} responses required >1 chunk ({multi / len(raw_texts) * 100:.1f}%)")

    # 2) Set up per-response chunk boundaries so we can emit responses as their
    #    chunks finish encoding (needed for incremental checkpointing).
    response_chunk_starts = np.zeros(len(raw_texts) + 1, dtype=np.int64)
    np.cumsum(response_to_chunk_count, out=response_chunk_starts[1:])
    emb_cols = [f"e{i:03d}" for i in range(dim)]
    checkpoint_every = args.checkpoint_every if args.checkpoint_every > 0 else float("inf")

    def pool_response(i: int, chunk_embs: np.ndarray) -> np.ndarray:
        """Mean-pool + re-normalize chunks for response i into one vector."""
        s = response_chunk_starts[i]
        e = response_chunk_starts[i + 1]
        if e - s == 1:
            return chunk_embs[s]
        pooled = chunk_embs[s:e].mean(axis=0)
        norm = np.linalg.norm(pooled)
        return pooled / norm if norm > 0 else pooled

    def flush(staged_idx: list[int], staged_vecs: list[np.ndarray]) -> None:
        """Append staged response embeddings to out_file atomically."""
        if not staged_idx:
            return
        mat = np.stack(staged_vecs)
        emb_df = pd.DataFrame(mat, columns=emb_cols)
        key_df = todo.iloc[staged_idx][KEY_COLS + ["_key"]].reset_index(drop=True)
        new_rows = pd.concat([key_df, emb_df], axis=1)
        if out_file.exists():
            prev = pd.read_parquet(out_file)
            combined = pd.concat([prev, new_rows], ignore_index=True)
        else:
            combined = new_rows
        tmp = Path(str(out_file) + ".tmp")
        combined.to_parquet(tmp, index=False)
        tmp.replace(out_file)
        staged_idx.clear()
        staged_vecs.clear()

    # 3) Encode chunk-by-chunk; emit + checkpoint responses as they complete.
    prefix = args.passage_prefix
    chunk_embeddings = np.empty((total_chunks, dim), dtype=np.float32)
    next_response = 0
    staged_idx: list[int] = []
    staged_vecs: list[np.ndarray] = []
    emitted = 0
    last_flush_at = 0

    pbar = tqdm(range(0, total_chunks, args.batch_size), desc="encode")
    for start in pbar:
        batch = flat_chunks[start:start + args.batch_size]
        if prefix:
            batch = [prefix + t for t in batch]
        vecs = model.encode(
            batch,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        chunk_embeddings[start:start + len(batch)] = vecs

        # Emit every response whose chunks have all been processed.
        chunks_done = start + len(batch)
        while (next_response < len(response_to_chunk_count)
               and response_chunk_starts[next_response + 1] <= chunks_done):
            staged_idx.append(next_response)
            staged_vecs.append(pool_response(next_response, chunk_embeddings))
            next_response += 1
            emitted += 1

        # Checkpoint when we've accumulated enough new responses.
        if emitted - last_flush_at >= checkpoint_every:
            flush(staged_idx, staged_vecs)
            last_flush_at = emitted
            pbar.set_postfix({"flushed": emitted})

    # Final flush for any stragglers past the last checkpoint boundary.
    flush(staged_idx, staged_vecs)

    # Report
    print(f"wrote {out_file} "
          f"(total rows now: {pd.read_parquet(out_file).shape[0]}, {dim} dims, "
          f"{emitted} new this run)")


if __name__ == "__main__":
    main()
