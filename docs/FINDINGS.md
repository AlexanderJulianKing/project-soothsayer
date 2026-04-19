# Findings: Alt-Target Prediction Pipeline

## Reference numbers (current as of 2026-04-18, post-CritPT rerun, n=127)

| Config | OOF RMSE | R² | ρ |
|---|---:|---:|---:|
| Baseline (no sem) | 15.39 | — | — |
| Sem v4 @ 32 dims | 14.12 | — | — |
| Sem + PLS hybrid + drop_style_tone (2026-04-18 ship, pre-CritPT n=126) | 13.48 | 0.943 | 0.971 |
| **Sem + PLS hybrid + drop_style_tone (SHIPPED, post-CritPT n=127)** | **13.61** | **0.941** | **0.971** |

**Honest walk-forward (2026-04-18, n=23 newest):** RMSE 14.69, R² 0.900, Spearman ρ 0.940. Re-fits ModelBankImputer + PCA-32 + PLS-3 at every step — see `docs/WALKFORWARD_ANALYSIS.md`.

**Shipped 7-flag config** (from `predict.sh`, lift verbatim for any new sweep):

```bash
python3 predict.py \
    --csv_path ../benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_v4_d32.csv \
    --imputer_type model_bank \
    --coherence_lambda 8.0 --coherence_shape exp \
    --predictor_selection loo_forward \
    --drop_style_tone \
    --pls_hybrid_k 3 \
    --cv_repeats_outer 10
```

**Note on pre-/post-fix numbers:** The 15.39 / 14.12 reference points were 15.48 / 14.19 before fixing a combine.py glob tie-break that had been dropping Writing TrueSkill columns for about a week (see "Combine.py writing glob fix" section below). All ablations written before 2026-04-15 used the pre-fix reference; where we contrast against a baseline, the Δ values remain comparable since the bug affected both arms equally.

## Response-Embedding Features (2026-04-14 / 2026-04-15)

Added `sem_f*` PCA-compressed response-embedding fingerprints as a derived feature set alongside `style_` and `tone_`. Best config cuts OOF RMSE from 15.48 → **14.19** (Δ −1.29) on n=124 arena-known rows, 10-repeat outer CV, same predict.py flags as **2026-04-15 production at the time** (`--imputer_type model_bank --coherence_lambda 1.0 --coherence_shape exp --eb_parent`). Post-fix reference: 15.39 → **14.12** (Δ −1.27 preserves the gain). *Historical context: `--eb_parent` and `coherence_lambda 1.0` were retired in the 2026-04-18 PLS cleanup; current shipped config is in the 2026-04-18 section at the end of this doc.*

### Ablation ladder

| Version | Pooling | Raw dims | PCA | OOF RMSE | Δ vs baseline (15.48) |
|---|---|---|---|---|---|
| v1 (2026-04-14) | cross-bench mean | 384 | 16 | 14.96 | −0.52 |
| v2 | per-bench concat (eq/logic/style/writing) | 1536 | 24 | 14.74 | −0.75 |
| v3 | per-bench + chunk-and-pool | 1536 | 24 | 15.01 | −0.47 |
| v4 @ 24 | per-bench + EQ split into t1/t3 | 1920 | 24 | 14.52 | −0.97 |
| v5 @ 24 | v4 + Style split into tech/casual | 2304 | 24 | 14.84 | −0.64 |
| v6 @ 24 | v4 + 9 per-prompt Style slots | 4992 | 24 | 14.64 | −0.84 |
| **v4 @ 32** | **v4 with wider PCA budget (champion)** | **1920** | **32** | **14.19** | **−1.29** |
| v4 @ 48 | v4 with oversized PCA | 1920 | 48 | 14.73 | −0.75 |
| v6 @ 32 | v6 with wider PCA | 4992 | 32 | 14.78 | −0.70 |
| v6 @ 48 | v6 with widest PCA | 4992 | 48 | 14.55 | −0.93 |

### Key findings

- **Truncation was an accidental signal filter (v3 < v2).** Chunk-and-pool preserves the full response but dilutes the fingerprint; the first 512 tokens carry the model-identity signal, and later chunks are prompt-driven content that averages toward a topic centroid. Top PCA components stayed aligned between truncated and chunked (|r|>0.7); later components diverged, confirming the early tokens carry the load.
- **EQ per-turn split helps (v4 > v2).** Splitting EQ into t1 (first-turn) and t3 (last-turn) slots and dropping t2 exposed multi-turn escalation signal that the cross-turn pool was averaging away. Consistent with the prior validated feature `first_person_delta_t3`.
- **Style splits HURT (v5, v6).** Splitting Style's 9 prompts into tech/casual (v5) or per-prompt (v6) sub-slots regressed below v4. Diagnostic Mantel r on between-model distance matrices: EQ t1/t3 = **0.69** (correlated views of shared signal), Style tech/casual = **0.39** (decorrelated noisy views). PCA sweep confirmed v6 cannot be rescued with more PCA dims — the problem isn't compression but that per-prompt slots contain prompt-topic variance that becomes nuisance variance in PCA's top components.
- **Pooling does pre-PCA denoising that PCA cannot replicate.** Averaging 9 per-prompt slots cancels prompt-topic (same topic mix across models → identical constant → removed by centering) while preserving model-identity (same signature across slots → accumulates coherently). Splitting keeps topic visible in each slot, PCA sees topic as "high variance" and spends top components on it, and the predictor distance function is contaminated with topic-axes that don't correlate with ELO.
- **PCA budget sweet spot at 32 dims (d/n ≈ 0.26).** v4 @ 24 undersizes — real signal axes exist past component 24. v4 @ 48 oversizes — components 33-48 are noise-dominated and hurt KNN distance. 32 dims is the bias-variance elbow. Curse of dimensionality starts to bite around d/n ≈ 0.4 (n=124 training rows).

### Design rule

> Before splitting a benchmark into sub-slots, ask whether the split-axis carries **per-model variance** or **per-prompt content variance**. Split on per-model axes (EQ turn escalation — shared topic, different behavior). Pool across per-prompt axes (Style's 9 topics — different topic, shared behavior). Embedding models encode topic; PCA can't separate topic variance from model variance when both are present.

### Artifacts

- Champion augmented CSV: `benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_v4_d32.csv`
- Fingerprint source: `embeddings/cache/model_fingerprints_v4_d32.csv`
- Reproducers: `embeddings/run_ablation_v{2,4,5,6}.bash`, `embeddings/run_pca_sweep.bash`

### Embedder capacity sweep (2026-04-15)

Tested whether a larger-capacity embedder beats bge-small on the same v4 (per_bench_eq_split) fingerprint pipeline. Downloaded candidates: bge-base (109M, 768d, 512 ctx), bge-large (335M, 1024d, 512 ctx), nomic-embed-v1.5 (137M, 768d, 8192 ctx), gte-large-v1.5 (434M, 1024d, 8192 ctx). Only bge-base run so far; others queued.

**bge-base PCA sweep (same embeddings, v4 pooling, varying only n_components):**

| dims | OOF RMSE | Δ vs baseline (15.48) | Δ vs bge-small champion (14.19) |
|------|----------|-----------------------|----------------------------------:|
| 16   | 14.83    | −0.65 | +0.64 |
| **24** | **14.40** | **−1.08** | **+0.21** |
| 32   | 14.59    | −0.89 | +0.40 |
| 48   | 15.48    | 0.00  | +1.29 |
| 64   | 16.20    | +0.72 | +2.01 |

**Key findings:**

- **bge-base at its own optimum loses to bge-small at its own optimum by only 0.21 RMSE** (14.40 vs 14.19) — within CV noise territory. Fixed-d=32 comparison overstated the gap (0.40). Each embedder must be swept at its own PCA budget before comparison.
- **Compression ratio — not absolute PCA dim count — determines the optimum.** bge-small champion: 32/1920 ≈ **1.7%** retention. bge-base optimum: 24/3840 ≈ **0.6%** retention. The sweet spot drops in absolute dims as raw dim goes up, not up. Components beyond a small fraction of the raw space are noise-dominated regardless of embedder choice.
- **Bigger embedder ≠ better fingerprint.** bge-base's extra 384 dims relative to bge-small are spent on finer-grained **content distinctions** (MTEB training objective), not finer-grained **voice distinctions**. The averaging trick we use to extract model identity benefits from an embedder that aggressively compresses into the top statistical axes (where voice lives) rather than one that expressively encodes content across many axes. This predicts bge-large will lose even at its own optimum, and reframes the nomic comparison — if nomic wins, the gain will come from 8192-ctx fixing the 55% truncation rate, not from extra capacity.
- **Monotonic curve beyond optimum.** Past d=24, bge-base's RMSE rises monotonically (14.40 → 14.59 → 15.48 → 16.20). Past d=32, bge-small's RMSE rose too (14.19 → 14.73). Both embedders sit in the "less is more" regime for this task at the current n=124 training rows.

**Artifacts:**

- Per-dim augmented CSVs: `benchmark_combiner/benchmarks/clean_combined_all_benches_with_sem_bge_base_d{16,24,32,48,64}.csv`
- Per-dim fingerprints: `embeddings/cache/model_fingerprints_bge_base_d{16,24,32,48,64}.csv` (d=32 saved as `model_fingerprints_bge_base.csv`)
- Per-dim predict.py outputs: `arena_predictor/analysis_output/_sem_embedder_sweep_bge_base{,_d16,_d24,_d48,_d64}/`
- Reproducer: `embeddings/run_embedder_sweep.bash bge_base` (fresh embedding + fingerprint + predict) and `embeddings/run_bge_base_pca_sweep.bash` (reuses cached embeddings for dim sweep)

### Embedder + PCA + fingerprint-mode sweep (2026-04-16)

Completed the embedder capacity test queued on 2026-04-15 (bge-large, nomic @ 2048 ctx,
gte-large @ 2048 ctx) and extended it with a PCA sweep on bge-large and a fingerprint-mode
sweep on bge-small. All runs on post-fix baseline (n=160, 15.39 OOF RMSE); champion is the
bge-small d=32 per_bench_eq_split model at 14.12.

**Note on nomic/gte-large context:** original plan was 8192 ctx. On MPS this blows up: the
attention matrix is O(L²), so at L=8192 a single forward pass needs ~20-40 GB of peak memory
(model-dependent), which on Apple Silicon Unified Memory quickly hits swap and the encoder
rate collapses from ~2 s/it → 240 s/it around batch 88 of a 23k-response run. Capping
max_seq_length=2048 drops peak attention memory by 16× and still captures ~99% of responses
in a single chunk (EQ p95 is ~2200 tokens; no benchmark's p90 exceeds 2048). We also added
incremental checkpointing to `embed_responses.py` (flush pooled responses to the output
parquet every N, atomic-rename write) so any thrash never costs more than 500 responses.

**Embedder sweep at d=32 (fixed champion PCA budget):**

| Embedder | Raw dims | Context | OOF RMSE | vs champion |
|---|---:|---:|---:|---:|
| **bge-small** (champion) | 384 | 512 | **14.12** | — |
| bge-large | 1024 | 512 | 14.30 | +0.18 |
| gte-large @ 2048 | 1024 | 2048 | 14.35 | +0.23 |
| nomic @ 2048 | 768 | 2048 | 14.45 | +0.33 |

**bge-large PCA sweep** (per_bench_eq_split mode, 5120 raw-dim concat):

| d | OOF RMSE | compression | note |
|---:|---:|---:|---|
| 8 | 14.86 | 0.16% | underfit |
| 16 | 14.56 | 0.31% | |
| **24** | **14.23** | **0.47%** | bge-large optimum |
| 32 | 14.30 | 0.63% | |
| 48 | 14.30 | 0.94% | plateau |
| 64 | 14.87 | 1.25% | overfit cliff |
| 96 | 15.03 | 1.88% | |
| 128 | 16.84 | 2.50% | severe overfit |

**bge-small PCA sweep** (post-fix baseline — extends d=32/48 from 2026-04-15):

| d | OOF RMSE |
|---:|---:|
| 16 | 14.61 |
| 24 | 14.40 |
| **32** | **14.12** (champion) |
| 48 | 14.82 |

**bge-small fingerprint-mode sweep** (all at d=32):

| Mode | Slots | OOF RMSE |
|---|---:|---:|
| `per_bench_eq_split` (champion) | 5 | **14.12** |
| `per_bench_eq_and_style_split` | 6 | 14.41 |
| `per_bench_eq_split_style_per_prompt` | 13 | 14.60 |

**Key findings:**

- **bge-small d=32 + 5-slot is a narrow global optimum** across all three axes (embedder choice,
  PCA dim, fingerprint-mode granularity). Every variant we tried loses by ≥Δ+0.11.
- **Bigger embedder ≠ better fingerprint, even at its own PCA optimum.** bge-large's U-curve
  minimum (d=24, 14.23) is still Δ+0.11 behind bge-small's d=32 minimum (14.12). Pushing
  bge-large toward bge-small's 1.67% compression ratio — i.e. d≈85 — hits the overfit cliff
  hard (d=96 → 15.03; d=128 → 16.84). The 2026-04-15 compression-ratio hypothesis is only
  *partly* right: both embedders have a narrow sweet spot in compression-ratio space, but
  bge-small's sweet spot produces strictly better fingerprints than bge-large's.
- **Long context is neutral-to-harmful.** nomic @ 2048 — where ~99% of responses fit
  unchunked — regressed Δ+0.33 vs bge-small's 512-with-chunking. gte-large @ 2048 regressed
  Δ+0.23. Interpretation: the arena-predictive signal lives in roughly the first 512 tokens
  of a response (the voice/identity signature), so extending context adds content variance
  without adding predictive signal. Consistent with the earlier "chunk-and-pool hurts" (v3)
  result — the opening is what matters.
- **Adding pooling slots hurts.** 6-slot (split Style by register) and 13-slot (per-prompt
  Style) both regressed vs 5-slot. Reconfirms the 2026-04-15 Mantel diagnostic: Style prompts
  contribute per-prompt content variance, not per-model variance. Splitting injects nuisance
  variance that PCA's top components can't filter away.

**Design rule (upgraded to global)**: bge-small at 512 ctx, 5-slot `per_bench_eq_split`, d=32
PCA is the global optimum for this task at n=160. No laptop hours should be spent on bigger
embedders, longer context, or finer-grained pooling without first changing the task structure
(more models, different benchmarks, different prediction target, etc.).

**Artifacts:**

- Embedder queue driver: `embeddings/_run_embedder_queue.bash` (with 2048-ctx caps and
  `--checkpoint_every 500` flag on `embed_responses.py`)
- PCA + mode sweep driver: `embeddings/_run_pca_and_mode_sweep.bash`
- Per-run predict.py outputs: `arena_predictor/analysis_output/_sem_embedder_sweep_{bge_large,nomic_2k,gte_large_2k}/`
  and `_pca_mode_sweep_{bgeL_pb5_d*,bgeS_pb5_d*,bgeS_pb6_d32,bgeS_pb13_d32}/`
- Response-embedding parquets: `embeddings/cache/response_embeddings_{bge_large,nomic_2k,gte_large_2k}.parquet`

---

### Combine.py writing glob fix (2026-04-15)

Late in the judge-bias work we noticed `clean_combined_all_benches.csv` had
no Writing TrueSkill columns. Root cause: `benchmark_combiner/combine.py`
line 591 used `pattern="benchmarks/writing_*.csv"`, which matched both
`writing_20260407.csv` (has TrueSkill) and `writing_direct_20260407.csv`
(score-vs-reference only). `get_latest_file` picks by date; both files
tied on `20260407`, and the glob tie-break silently picked the `_direct`
variant. As a result Writing TrueSkill hadn't been in the pipeline since
those two files started coexisting (roughly a week).

**Fix**: narrow the pattern to `benchmarks/writing_[0-9]*.csv`, mirroring
the EQ pattern on line 593. Excludes `_direct` variants entirely.

**Impact on reference numbers** (via full `combine.bash` regeneration, not
hand-edit):

| Config | Pre-fix | Post-fix | Δ |
|---|---:|---:|---:|
| Baseline (no sem) | 15.48 | **15.39** | **−0.09** |
| Sem v4 d32 champion | 14.19 | **14.12** | **−0.07** |

The post-fix n changed too (159 → 160) because one model was in
`writing_20260407.csv` but missing from `_direct_20260407.csv`, so the
comparison isn't strictly apples-to-apples. Directionally: the fix is
a permanent free gain, and sem-champion stays ahead by ~Δ −1.27 over
post-fix baseline (vs −1.29 pre-fix; same story).

Not fixed here but worth noting for later: `correlations.py` around line
1387 has a drop-list with double-space typos in several writing column
names (e.g. `writing_Gemini 3.0 Flash Preview (2025-12-17)  TrueSkill`
with two spaces). The real column has one space, so the drop rule fails
to match. Happens to work out (we want TS kept) but the Sigma drops
should be audited next time someone touches that file.

---

## Verified Results (2026-03-09, post-leakage fix)

All numbers below use honest OOF predictions for `style_predicted_delta` (cross_val_predict, OOF r=0.870). Prior results used in-sample predictions (r=1.000) due to a leakage bug in `soothsayer_style/score.py`.

### Imputer & Configuration Comparison

| # | Config | OOF RMSE | Best Model | Notes |
|---|--------|----------|------------|-------|
| 1 | Specialized imputer | 22.74 | ARDRegression | Baseline |
| 2 | ModelBank, no coherence | 22.89 | ARDRegression | Worse than Specialized without coherence |
| 3 | ModelBank + coherence (λ=1.0, exp) | 22.17 | ARDRegression | Coherence is what makes ModelBank win |
| 4 | ModelBank + coherence + trajectory in target | **21.69** | ARDRegression | Best verified config |
| 5 | ModelBank + coherence + trajectory in both | **21.69** | ARDRegression | ALT trajectory irrelevant |

All runs (**historical, pre-KNN era**): `--poly_interactions --poly_limit 7 --no_residual_head --cv_repeats_outer 10 --cv_repeats_inner 5 --feature_cv_repeats 1 --alt_cv_repeats 1`. These flags were retired in the 2026-04-18 cleanup — see the section at the end of this doc for the current shipped config.

### Verified Improvement Deltas

| Change | Delta | From → To |
|--------|-------|-----------|
| Coherence projection (λ=1.0, exp shape) | **-0.72** | 22.89 → 22.17 |
| Trajectory features in target model | **-0.48** | 22.17 → 21.69 |
| ModelBank+coherence vs Specialized | **-0.57** | 22.74 → 22.17 |
| ALT trajectory features | **0.00** | No effect on final RMSE |

### Best Production Config (shipped 2026-04-18)

**Target:** `lmarena_Score` (style-controlled). `lmsys_Score` is leakage (derived from the same Arena voting process).

See the reference block at the top of this doc for the canonical 7-flag invocation. The 2026-03-29 config (`--coherence_lambda 1.0 --eb_parent`) was superseded on 2026-04-18 when the PLS hybrid ablation showed EB parent was over-shrinking once PLS was on. Historical intermediate results below are preserved for lineage.

**Intermediate reference (2026-03-29 config, pre-sem, pre-PLS):** RMSE 15.63 (LOO), 16.03 (10×5-fold), R²=0.924, Spearman=0.962. Walk-forward RMSE: 14.86. Historical — not the current shipped config.

<details>
<summary>Legacy ALT pipeline config (deprecated)</summary>

```bash
python3 predict.py \
    --csv_path ../benchmark_combiner/benchmarks/clean_combined_all_benches.csv \
    --imputer_type model_bank \
    --coherence_lambda 1.0 --coherence_shape exp \
    --poly_interactions --poly_limit 7 \
    --no_residual_head --no_traj_in_alt \
    --eb_parent \
    --top_tier_boost 2 --top_tier_threshold 1400
```
This predicted lmsys_Score via lmarena imputation. Best RMSE was 18.84.
</details>

**OOF RMSE: 18.84** | Top-20 RMSE: **14.82** | Top-10 RMSE: **11.40** | >1400 RMSE: **13.95** | >1400 bias: **-3.4**

n=123 training models, 157 total. Key changes from 21.48 baseline:
- Top-tier boost (2x row duplication for ELO ≥ 1400): -1.3 RMSE
- lmarena style-restricted interactions (hardcoded in `_LMARENA_STYLE_ONLY_PARTNERS`): -0.95 RMSE
- Piecewise top-style feature (`_pw_top_style = max(lmarena-1400,0) * style_delta`): -0.12 RMSE
- LOBO column drop (`eqbench_creative_elo`): -0.20 RMSE
- Grouped conformal CIs: ~92% honest coverage (was 100% fake)

---

## Architectural Findings (qualitatively robust)

These findings were established across 70+ experiments. While the absolute RMSE numbers from the leaked era are invalid, the qualitative patterns are confirmed by the re-verification (relative ordering preserved, same features win/lose).

### The prediction is dominated by the ALT feature

The lmarena_Score (ALT target) is by far the strongest predictor. For models that already have an Arena score, prediction is near-trivial. For models that don't, everything depends on ALT imputation quality. This is why coherence projection helps — it improves imputed ALT values.

### Trajectory features encode imputation quality

The `_traj_mean_delta`, `_traj_max_delta`, `_traj_n_imputed` features (per-row statistics from the imputer's iterative passes) are the single most valuable meta-features. They encode:
1. **Coverage** — how many benchmarks the model was tested on (correlated with model prominence and ELO stability)
2. **Imputation difficulty** — how much imputed values shifted across passes (epistemic uncertainty)
3. **Surviving imputation smoothing** — binary facts from raw data that pass through untouched

The 2×2 ablation confirms the gain comes entirely from the target model pathway, not from trajectory improving ALT imputation. ALT trajectory features improve the ALT model's own cross-validation diagnostic (18.11 → 17.33) but this doesn't propagate to final predictions.

### Small n dominates everything

~90 training rows, 75+ features, ~40% missing. This explains most experiment outcomes:
- Nonlinear models catastrophically overfit (KernelRidge: 220, PLS: 20.5)
- Stacking and meta-learning overfit (learning blend weights from OOF with n=90)
- More than 2 blend members consistently hurts
- Aggressive feature selection is essential (~10 features from 78)
- The sparsest regularizer (ARD) wins

### Coherence projection is the only imputer lever that worked

23 imputer experiments were run. Only coherence projection (SVD low-rank projection of imputed matrix, weighted by completeness) improved results. Everything else was neutral or worse: adaptive gates, learned gates, per-column coherence, multiple imputation, SVD anchors, iterative coherence, prediction-level stacking.

---

## Model Coverage

### Current state

- ~140 models in combined CSV, ~100 with lmarena_Score
- 303 models on LMArena total
- 344 models on OpenRouter

### Custom benchmark coverage gaps

| Benchmark | Models missing (of training set) |
|-----------|----------------------------------|
| logic | 0 |
| style | 0 |
| writing | 0 |
| eq | 1 |
| weirdml | ~44 |

WeirdML is the largest coverage gap.

---

## Imputation Improvement Experiments

### Cycle 1 (in progress)

Baseline: **21.41** OOF RMSE (ModelBank + coherence + trajectory, improved style delta)

| # | Idea | Source | Description |
|---|------|--------|-------------|
| 1 | Spearman-augmented ranking | Claude | Use max(|pearson|, |spearman|) for candidate predictor ranking |
| 2 | KNN fallback (k=5) | Claude | Replace column-median fallback with k-NN weighted average |
| 3 | Per-column adaptive lambda | Claude | Scale coherence λ by column's SVD reconstruction quality |
| 4 | EB parent model | Codex | Shrink cell predictions toward empirical-Bayes parent model |
| 5 | Masked-cell calibration | Codex | Calibrate σ² via synthetic masking of observed cells |
| 6 | Graph-Laplacian coherence | Codex | Correlation-graph per-row smoothing instead of global SVD |

| # | Experiment | RMSE | Delta | Verdict |
|---|-----------|------|-------|---------|
| 0 | Baseline | 21.69 | — | — |
| 1 | Spearman ranking | 21.87 | +0.18 | Worse — non-linear ranking reorders predictors unhelpfully |
| 2 | KNN fallback (k=5) | 21.69 | 0.00 | Neutral — median fallback cells are too few to matter |
| 3 | Adaptive col lambda | 21.72 | +0.02 | Neutral — per-column λ scaling doesn't help |
| 4 | **EB parent** | **21.48** | **-0.22** | **Winner — shrinking uncertain cells toward parent model helps** |
| 5 | Masked calibration | 22.12 | +0.42 | Worse — recalibrated σ² disrupts coherence weighting |
| 6 | Graph Laplacian | exploded | — | Broken — numerical instability in per-row linear solve |

### Cycle 2

Baseline: **21.48** OOF RMSE (EB parent ON from Cycle 1)

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 0 | EB parent baseline | — | 21.48 | — | — |
| 1 | EB residual (blend in pass1) | Codex | 22.02 | +0.55 | Worse — early blending disrupts per-cell models |
| 2 | Exact joint support + EB | Codex | 21.49 | +0.01 | Neutral — approx support heuristic was fine |
| 3 | SPD graph smoother + EB | Codex | 22.82 | +1.35 | Much worse — local graph corrections hurt SVD coherence |
| 4 | EB after coherence | Claude | 22.25 | +0.77 | Worse — EB must come before coherence |
| 5 | Double EB | Claude | 22.25 | +0.77 | Same as Exp 4 — second EB after coherence dominates |
| 6 | EB Bayesian sigma | Claude | 21.68 | +0.20 | Slightly worse — per-prediction uncertainty adds noise |

**Key insight**: EB parent works best as a simple post-pass1, pre-coherence step with fixed parent σ². Any modification to its position or σ² formula hurts.

### Cycle 3

Baseline: **21.48** OOF RMSE (EB parent ON)

Testing pipeline structural choices: poly interaction settings, residual head, ALT trajectory.

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 0 | Baseline (poly_limit=7) | — | 21.48 | — | — |
| 1 | No poly interactions | Claude | 22.35 | +0.88 | Much worse — poly interactions are essential |
| 2 | Poly limit=10 | Claude | 21.88 | +0.40 | Worse — more interactions overfit |
| 3 | Poly limit=5 | Claude | 21.73 | +0.26 | Worse — fewer interactions lose signal |
| 4 | With residual head | Codex | 21.63 | +0.15 | Slightly worse — residual head adds noise |
| 5 | ALT trajectory ON | Codex | 21.48 | 0.00 | Neutral — ALT trajectory still irrelevant |
| 6 | Outer CV=20 repeats | Codex | — | — | Killed (100-fold eval too slow, no expected gain) |

**Key insight**: poly_limit=7 is the sweet spot. Fewer loses signal, more overfits. The residual head and ALT trajectory confirm prior findings — neither helps.

### Cycle 4

Baseline: **21.48** OOF RMSE (EB parent ON)

Testing post-imputation pipeline ideas: ALT bagging, delta head, reliability gating, imputer ensembles.

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 0 | Baseline | — | 21.48 | — | — |
| 1 | ALT bagged (5 seeds) | Codex | 21.48 | 0.00 | Neutral — interaction pair variance is already handled by consensus search |
| **2** | **Delta head** | **Codex** | **20.29** | **-1.19** | **LEAKED — delta head trained on all TARGET rows before CV; must re-verify** |
| 3 | ALT reliability gate | Codex | 21.42 | -0.06 | Neutral |
| 4 | Imputer ensemble (3 seeds) | Claude | 21.53 | +0.05 | Neutral |
| 5 | Obs-weighted SVD | Claude | 21.57 | +0.09 | Neutral |
| 6 | ALT LOO calibrate | Claude | 21.48 | 0.00 | Neutral |

**Key insight**: Delta head result is LEAKED — the -1.19 improvement is invalid. The delta head model was trained on ALL known-TARGET rows before CV, so validation rows see predictions from a model trained on their own labels. The signal (lmsys-lmarena gap predictable from style features, train r=0.937) may be genuine, but the honest lift must be measured with fold-internal delta head computation. See leakage analysis below.

### Cycle 5

Baseline: **21.48** OOF RMSE (EB parent ON)

Testing problem framing and regularization: completeness weighting, sparse column dropping, winsorization, feature noise.

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 0 | Baseline | — | 21.48 | — | — |
| 1 | No completeness weighting | Codex | 21.53 | +0.05 | Neutral |
| 2 | Weight power=4 | Codex | 22.81 | +1.33 | Much worse — over-weighting complete rows hurts |
| 3 | Pairwise rank feature | Codex | 15.57 | -5.91 | **CONFIRMED LEAKAGE** — see leakage analysis below |
| 4 | Drop sparse cols >70% | Claude | 21.70 | +0.22 | Worse — sparse columns still contribute through imputation |
| 5 | Winsorize 2% | Claude | 21.57 | +0.09 | Neutral |
| 6 | Feature noise 5% | Claude | 21.48 | 0.00 | Neutral |

**Key insight**: The default completeness weighting (power=2) is well-tuned. Dropping sparse columns hurts because even highly incomplete columns contribute through the imputation graph. Pairwise rank feature is confirmed leakage (see below).

### Cycle 6

Baseline: **21.48** OOF RMSE (EB parent ON, PCA-10 ALT feature mode)

Testing whether PCA in the ALT pathway compresses away useful capability-specific signals: raw columns, Factor Analysis, hybrid approaches.

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 0 | PCA-10 baseline | — | 21.48 | — | — |
| 1 | Raw columns (no PCA) | Claude | 21.79 | +0.31 | Worse |
| 2 | Factor Analysis (10 factors) | Claude | 21.89 | +0.41 | Worse |
| 3 | Hybrid (top-15 raw + PCA-5) | Claude | 21.98 | +0.50 | Worse |
| 4 | Raw, no interactions | Claude | 22.11 | +0.63 | Worse |
| 5 | FA, no interactions | Claude | 22.80 | +1.32 | Much worse |

**Key insight**: PCA(10) is optimal for the ALT pathway. At n≈140 rows with 76 features, PCA's regularization benefit outweighs any capability-specific signal that FA or raw columns might preserve. The interaction search on top of PCA is also confirmed valuable — removing it consistently hurts (+0.3 to +0.9 RMSE).

### Summary Across All Cycles

| Cycle | Experiments | Winners | Best Δ |
|-------|------------|---------|--------|
| 1 | 6 | EB parent (-0.22) | -0.22 |
| 2 | 6 | None | 0.00 |
| 3 | 5 | None | 0.00 |
| 4 | 6 | Delta head (-1.19, **LEAKED**) | ? |
| 5 | 6 | Pairwise rank (**LEAKED**) | ? |
| 6 | 5 | None | 0.00 |
| 7 | 7 | None (delta head confirmed leakage) | 0.00 |
| 8 | 3 | None | 0.00 |
| 9 | 3 | None | 0.00 |

**Total: 54 experiments, 1 confirmed winner (EB parent).** 53 experiments failed to beat 21.48. The prediction floor appears real at this sample size. Honest best: EB parent → **21.48 RMSE** (down from 21.69 baseline).

### Cycle 9

Baseline: **21.48** OOF RMSE (EB parent ON)

Moonshot experiments: semi-supervised self-training, model name features, prediction-level shrinkage.

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 1 | Self-training (pseudo-labels) | Claude | 22.05 | +0.57 | Worse — pseudo-labels add noise, don't regularize |
| 2 | Alias archaeology (name features) | Codex | 24.35 | +2.87 | Much worse — name tokens create spurious correlations at n=90 |
| 3 | Prediction shrinkage (blend w/ ALT-only) | Claude | 22.11 | +0.63 | Worse — simple model is too inaccurate, blending loses signal |

**Key insight**: Even fundamentally different approaches fail. Self-training adds ~50 pseudo-labeled rows but the pseudo-labels are too noisy (predicted from a 21.48-RMSE model) to help. Model name features (is_reasoning, is_mini, version) are spuriously correlated at n=90. Prediction shrinkage toward a 1-feature model sacrifices the real signal in the other 14 features.

### Cycle 8

Baseline: **21.48** OOF RMSE (EB parent ON)

Testing feature transforms: quantile normalization, reliability-weighted PCA, ALT orthogonalization.

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 1 | Quantile transform | Claude | 24.54 | +3.06 | Much worse — destroying natural scale hurts linear model |
| 2 | Reliability-weighted PCA | Codex | 21.48 | 0.00 | Neutral — obs_rate weighting doesn't change PCA enough |
| 3 | Orthogonalize to ALT | Codex | 22.58 | +1.10 | Worse — ALT-correlated signal in features is useful, not redundant |

**Key insight**: Raw benchmark scales carry important information. Quantile normalization destroys distances between scores that the linear model exploits. Features correlated with ALT are useful (not redundant) — the model needs redundant ALT signal across features to overcome imputed ALT noise. The existing StandardScaler → PCA(10) is already the right preprocessing.

### Cycle 7

Baseline: **21.48** OOF RMSE (EB parent ON)

Testing outside-the-box ideas + delta head leakage re-verification.

| # | Experiment | Source | RMSE | Delta | Verdict |
|---|-----------|--------|------|-------|---------|
| 0 | Baseline | — | 21.48 | — | — |
| 1 | Delta head (fold-internal fix) | Codex C4 | 23.38 | +1.90 | **Much worse — confirms signal was 100% leakage** |
| 2 | LOBO residuals | Claude | 21.48 | 0.00 | Neutral |
| 3 | Target-aware coherence | Claude | 21.48 | 0.00 | Neutral |
| 4 | Provider-family EB | Claude | 21.75 | +0.27 | Worse — family means too noisy with few models per family |
| 5 | Partial-linear ALT | Codex | 21.71 | +0.23 | Worse — isotonic overfits with ~80 fold-train rows |
| 6 | Pairwise anchor head | Codex | 43.01 | +21.53 | Catastrophic — pairwise differencing destroys signal at small n |

**Key insight**: Delta head is confirmed as pure leakage. With fold-internal computation, train r dropped from 0.937 (leaked, all rows) to 0.625 (honest, ~80 rows), and the feature actively hurts (+1.90 RMSE). The lmsys-lmarena gap is NOT reliably predictable from ~80 training rows. Provider-family and isotonic approaches also hurt — both add parameters that overfit at this sample size. Pairwise anchor head catastrophically fails — creating O(n²) difference examples from n=80 rows adds massive noise that overwhelms any relative signal.

---

## Leakage Analysis (2026-03-10)

Two features added in Cycles 4-5 were computed globally BEFORE the CV loop, bypassing the nested-CV safeguards. Both confirmed as leakage by code review + Codex (gpt-5.4) audit.

### Bug 1: Pairwise rank feature (Cycle 5 Exp 3, RMSE 15.57)

**Mechanism**: `_alt_rank_pctile = rank(global_ALT_prediction)` is computed at line 5546-5549 on the full-data ALT prediction (trained on all ~140 rows). Inside each CV fold, the ALT column is properly replaced with fold-honest OOF predictions, but `_alt_rank_pctile` is NOT updated — it retains the full-data ALT signal.

**Why it's severe**: The per-fold feature selector may keep `_alt_rank_pctile` because its r² with the noisy OOF-ALT falls below the 0.95 collinearity threshold. The model then uses `_alt_rank_pctile` as a superior proxy for ALT, completely circumventing OOF protection. Additionally, `_alt_rank_pctile` contaminates the ALT OOF model itself — `X_no_alt_df` only drops `ALT_TARGET`, not derived features.

**Fix**: Remove the feature entirely. A monotonic transform of ALT adds no information if computed honestly (fold-internally it would be perfectly collinear with ALT and dropped).

### Bug 2: Delta head (Cycle 4 Exp 2, RMSE 20.29)

**Mechanism**: `_delta_head_pred` is a BayesianRidge trained on `TARGET - ALT` using ALL known-target rows (line 5552-5573), then predictions are added as a feature for ALL rows. Inside CV folds, validation rows see predictions from a model trained on their own TARGET values — classic label leakage.

**Why the signal may be partially real**: The lmsys-lmarena gap IS predictable from style features (response length, formatting patterns). But the reported -1.19 RMSE improvement is contaminated and must be treated as entirely invalid until re-run.

**Fix** (implemented 2026-03-10): Delta head computation moved inside `_precompute_single_fold`. For each outer fold: delta model trained on fold's train rows only, predicts for both train and val rows. Also added to final model path in `fit_and_predict_all_with_alt`. Pairwise rank feature code removed entirely.

### Impact on production config

The "best production config" previously included `--delta_head`. The leakage fix has been implemented (delta head is now fold-internal), but needs re-verification. Until re-verified, the honest best config is:

```bash
python3 predict.py \
    --csv_path ../benchmark_combiner/benchmarks/clean_combined_all_benches.csv \
    --imputer_type model_bank \
    --coherence_lambda 1.0 --coherence_shape exp \
    --poly_interactions --poly_limit 7 \
    --no_residual_head --no_traj_in_alt \
    --eb_parent
```

Honest best RMSE: **21.48** (EB parent only, no delta head).

---

## Error Decomposition Analysis (2026-03-11)

### Where does the 21.48 RMSE come from?

The pipeline predicts target (lmsys_Score) in two stages: benchmarks → ALT, then ALT + benchmarks → target. Diagnostic experiments decompose the error:

| Scenario | ALT RMSE | Target RMSE | Notes |
|----------|----------|-------------|-------|
| Perfect ALT + all benchmarks → target | — | 12.93 | Floor: if ALT were oracle |
| Real ALT + style features → target | — | ~10-13 | Cheating: real ALT never available at inference |
| 73 benchmarks (median-fill) → ALT | 23.84 | — | Stage 1 bottleneck |
| 33 low-miss benchmarks → ALT | 22.99 | — | Slightly better, fewer imputed cells |
| Benchmarks → ALT → target (naive 2-stage) | 23.84 | 29.69 | No feature selection, no poly, no OOF stacking |
| Full pipeline | ~23 | **21.48** | Pipeline machinery recovers ~8 pts over naive |

**Key insight**: The bottleneck is stage 1 (benchmarks → ALT). Benchmarks only predict debiased Arena score to ~24 RMSE. Everything downstream is damage control. With perfect ALT the problem is easy (12.93), but predicting ALT from benchmarks is hard.

### The style correction is working as designed

The ALT target (lmarena_Score) is a style-debiased transform of the target (lmsys_Score). Style features (length, formatting, bold/list counts) capture exactly what makes these scores diverge. This is the entire point of the style benchmark — the r=-0.87 correlation between `style_predicted_delta` and the `lmsys - lmarena` gap is the design working, not leakage.

### IRT latent factors beat raw benchmarks for ALT prediction

A continuous 2-parameter IRT model (sigmoid link per benchmark, gradient-descent fit on observed cells only) extracts latent ability θ from the partially-observed benchmark matrix **without any imputation**.

| Approach | θ dims | ALT RMSE | Target RMSE (2-stage) |
|----------|--------|----------|-----------------------|
| IRT θ only | k=1 | 35.39 | 42.40 |
| IRT θ only | k=2 | 25.29 | 39.25 |
| IRT θ only | k=3 | **20.82** | 26.57 |
| IRT θ only | k=5 | 21.38 | 27.49 |
| IRT θ only | k=8 | 20.33 | 26.68 |
| Raw benchmarks (median-fill, 73 cols) | — | 23.84 | 29.69 |
| Full pipeline | — | ~23 | **21.48** |

**IRT θ (k=3) predicts ALT at 20.82 — 3 points better than 73 raw benchmarks (23.84).** Three nonlinear latent dimensions, fit on observed cells only, compress information more efficiently than 73 linearly-imputed columns. 68/73 benchmarks have genuinely nonlinear S-curves (not just linear regime), and 27/73 hit saturation bounds.

However, the IRT 2-stage target RMSE (26.57) is still behind the full pipeline (21.48) because the pipeline's OOF stacking, feature selection, and polynomial interactions recover signal in stage 2 that raw θ alone doesn't capture.

### IRT optimization: regularization and polynomial features matter

Grid sweep over rank × regularization, plus polynomial expansion of θ:

| Config | ALT RMSE | Notes |
|--------|----------|-------|
| k=3, λ=0.01 (original) | 20.82 | Initial prototype |
| k=3, λ=0.001 | 19.77 | Less regularization helps |
| k=4, λ=0.0001 | **18.29** | Best raw θ |
| k=4, λ=0.0001, poly(2) | **16.82** | Cross-terms between latent dims capture nonlinear ability combos |
| k=3, λ=0.0005, poly(2) | 17.50 | |
| k=6, λ=0.0005, poly(2) | 17.44 | Diminishing returns past k=4 |
| k=4, poly(3) | 27.37 | Degree 3 overfits |

**Poly(2) interactions of IRT θ are critical**: θ₁×θ₂, θ₁×θ₃, etc. capture nonlinear combinations of latent abilities. This is analogous to the pipeline's finding that poly_limit=7 is essential.

### IRT 2-stage: approaching the pipeline

Honest 2-stage evaluation (IRT θ_poly → ALT, then θ_poly + features → target):

| Stage 2 features | ALT RMSE | Target RMSE |
|------------------|----------|-------------|
| θ_poly only | 16.82 | 26.12 |
| θ_poly + style features | 16.82 | 22.66 |
| θ_poly + all 17 fully-observed features | 16.82 | **22.41** |
| Full pipeline (impute + ALT + feat sel + poly) | ~23 | **21.48** |

The IRT predicts ALT better (16.82 vs ~23) but the pipeline still wins on target (21.48 vs 22.41) due to its OOF stacking, feature selection, and polynomial interaction machinery for the ALT→target bridge. The ~1 point gap suggests feeding IRT θ into the existing pipeline could beat 21.48.

**Key insight**: Adding ANY features to θ for ALT prediction makes it worse (θ alone: 19.77, θ+features: 21.52+). The IRT factors are already the optimal compression — extra columns dilute signal at n=112. But for the target prediction stage, the style features are essential to model the Arena style bias.

**Next step**: Feed IRT θ into the existing pipeline as features, replacing or supplementing imputed benchmarks. The θ factors are clean, low-dimensional, require no imputation, and encode nonlinear benchmark relationships.

---

## Style Shape Features (2026-03-12)

### New features: per-question style variance

Added 15 style shape features to the pipeline, derived from per-question (Q1–Q9) response data in soothsayer_style. These capture *how* a model varies its style across different question types, not just aggregate formatting.

**New columns (prefixed `style_` in combined CSV):**
- `cv_{length,header_count,bold_count,list_count}` — coefficient of variation across 9 questions (adaptability)
- `min_{length,header_count,bold_count,list_count}` — minimum across questions (formatting floor)
- `frac_used_{header_count,bold_count,list_count}` — fraction of questions where formatting is used (consistency)
- `q7_{length,header_count,bold_count,list_count}` — Q7-specific metrics (creative programming task: Python slots in anime voice)

**Individual correlations with Arena ELO (n=110):**
- `style_q7_header_count`: r=+0.57 (best single feature — creative programming structure)
- `style_q7_length`: r=+0.47 (creative programming effort)
- `style_cv_length`: r=+0.41 (response length adaptability)
- `style_frac_used_bold_count`: r=+0.35 (consistent bold formatting)
- `style_min_bold_count`: r=+0.33 (formatting even on simple prompts)

**Key insight:** `style_cv_length` (adaptability) and `style_q7_header_count` (creative task structure) have NO overlap (r<0.5) with any existing combined column. They are genuinely new signals.

### Result: 17.89 RMSE (top-50 LOO, verified)

| Config | Top-50 LOO | 10×10-fold (all 112) | 10×5-fold (all 112) |
|--------|-----------|---------------------|---------------------|
| Previous best (EB parent, old features) | 21.48 | — | — |
| + style shape features, PCA-10 ALT, poly | **17.89** (CI: 14.67–20.92) | **19.76** (CI: 17.19–22.29) | **20.25** (CI: 17.57–22.90) |

Top-50 LOO evaluates prediction quality on the 50 strongest models (Arena score 1397–1502), which is the primary use case — predicting where new frontier models will land. The full 112-model K-fold includes weaker models that are harder to predict and less practically relevant. Both show clear improvement over the 21.48 baseline.

Selected features (7): `lmarena_Score`, `aa_eval_livecodebench`, `style_predicted_delta`, `eqbench_eq_elo`, **`style_q7_length`**, `arc_ARC-AGI-2`, `style_normalized_bold_count`

Only `style_q7_length` is new — but poly interactions expand it into 6 cross-terms (e.g., `style_q7_length × lmarena_Score`) that capture how creative coding effort modulates the benchmark-to-arena mapping.

**Previously underpredicted models improved:**

| Model | Old Error | New Error | Improvement |
|-------|-----------|-----------|-------------|
| Claude Sonnet 4.6 Thinking | -38.8 | -15.3 | +23.5 |
| Mistral Medium 3.1 | +46.4 | +24.9 | +21.5 |
| Claude Opus 4.6 | +50.9 | +31.8 | +19.1 |
| GLM-4.5 | +28.1 | +8.6 | +19.5 |
| Claude Opus 4.1 Thinking | +35.6 | +20.3 | +15.2 |

**Remaining hard cases:** Claude Opus 4.1 (+44.0), ChatGPT-4o (+35.1), DeepSeek R1 (+34.4). These likely need signals no current benchmark captures.

### Cross-domain ALT vs PCA-10 ALT

Tested a hybrid architecture: cross-domain interaction search for ALT prediction (greedy-selected pairwise interactions, RMSE ~17 vs PCA-10's ~28), then PCA components + greedy residual correlates for the target model.

**CD ALT consistently loses to PCA-10 ALT at the final target prediction despite much better stage-1 RMSE.** Pin-down experiments isolate why:

| Controls | PCA-10 | CD ALT | Gap |
|----------|--------|--------|-----|
| No selection, no poly | 23.39 | 24.39 | 1.00 |
| No selection, poly | 21.38 | 23.07 | 1.69 |
| Selection + poly (full pipeline) | 17.89 | 18.58 | 0.70 |
| ALT-centric poly | 22.70 | 24.67 | 1.97 |

**Diagnosis (confirmed by Codex review + ablation):**

The gap is ~60% fundamental, ~40% poly amplification.

1. **Fundamental (Wave 1, gap=1.0):** PCA-10 ALT is a better *raw feature* for predicting lmsys_Score despite being worse at predicting lmarena_Score. PCA-10 preserves a clean linear capability axis that the target model can use directly; CD ALT's accuracy at predicting lmarena doesn't translate to lmsys prediction utility.

2. **Poly amplification (Wave 2, gap grows to 1.7):** PCA-10's "structured errors" — systematic mispredictions for models with distinctive style — interact productively with raw features via poly terms. CD ALT's tighter fit leaves smaller, more random residuals that poly can't exploit. This is a division-of-labor effect: PCA-10 handles linear signal, poly handles nonlinear.

3. **Selector compensation (Wave 3, gap shrinks to 0.7):** Tree-based feature selection actually *helps* CD ALT more than PCA-10, partially compensating for the weaker poly interactions. The selector is not the villain.

4. **ALT-centric poly is counterproductive (Wave 4, gap=2.0):** Forcing explicit `ALT × feature` interactions hurts both ALT types and widens the gap. The model needs generic poly, not ALT-focused interactions.

**Literature context (via Codex):** This is a known phenomenon in stacked models — `argmin_h E[(ALT - h(X))²]` ≠ `argmin_h min_g E[(TARGET - g(X, h(X)))²]` when g is a restricted model class (Wolpert 1992, Breiman 1996). The optimal stage-1 model depends on stage-2's capacity.

**Practical conclusion:** PCA-10 ALT is the correct choice for this pipeline. Better ALT prediction is not a productive direction.

**Update (2026-03-15):** Re-tested with style-restricted pipeline. Wolpert phenomenon persists: cross-domain ALT gets 19.01 (vs bayes 21.16) but final RMSE goes from 19.16 → 21.46. Also tested lgbm (21.66) and stack (19.41) ALT regressors — neither beats bayes. The BayesianRidge ALT produces structured errors that the style bridge can exploit; better ALT removes that structure.

---

## lmarena Style Restriction (2026-03-14)

### Discovery
The lmsys↔lmarena difference is computed via Contextual Bradley-Terry with 4 style features (token count, headers, bold, lists). Source: `github.com/lmarena/arena-rank`. Therefore `lmarena_Score` should only interact with style features in our pipeline.

Before the fix, `eqbench_eq_elo × lmarena_Score` (importance=334) was the #1 feature. This created a near-circular "predict Arena from Arena × EQ" path that overfit for existing models (near-perfect predictions) and extrapolated badly for new ones (GPT-5.4 Thinking predicted at 1536 vs ~1475 actual for the high reasoning variant).

### Implementation
Added `_LMARENA_STYLE_ONLY_PARTNERS` set in `predict.py`. After `PolynomialFeatures` generation, any `lmarena_Score × non_style_feature` interaction is dropped. lmarena still enters as standalone and can interact with all `style_*` columns.

### Results
| Config | RMSE | Top-50 | GPT-5.4 |
|---|---|---|---|
| Old (unrestricted) | 20.11 | 15.10 | 1536 |
| **Style-restricted** | **19.16** | **15.25** | **1468** |

New top features: `lmarena × style_predicted_delta` (+660), `style_predicted_delta` (-591), `eqbench × style_predicted_delta` (-96).

### Why eqbench still matters
EQ correlates r=0.147 with ALT OOF residual — it compensates for lmarena imputation error, not the style adjustment. The BayesianRidge ALT uses PCA which smears EQ signal; the final model recovers leftover EQ information via `eqbench × style_delta`.

The lmsys-lmarena gap correlates r=-0.900 with `style_predicted_delta` and only r=0.082 with eqbench (n=43). A linear model with just `lmarena + style_delta + interaction` gets R²=0.981 on training data (RMSE=9.41). But OOF with fold-predicted lmarena gives RMSE=33.55 — capability features are essential to compensate for ALT imputation noise.

### Failed top-tier optimization experiments
All tested, none beat binary boost + style restriction:
- Continuous ELO weighting: 20.37
- ELO weight + boost: 20.65
- Top-50 residual correction: 21.88
- Pure style-only final: 43.11

### Grouped conformal CIs
Replaced 6-feature heteroscedastic scale model with coarse group-based conformal. 4 groups by predicted score × missingness, empirical quantile per group. Coverage: 92.5% (was fake 100%).

---

## Residual Analysis (2026-03-12)

In-sample residual analysis on the best model's final predictions (predictions_best_model.csv). These are NOT OOF predictions — they're trained on all data — but patterns should be directionally similar.

### By Provider

| Provider | Count | RMSE | Mean Error | Notes |
|----------|-------|------|------------|-------|
| Google | 19 | 3.0 | -0.2 | Best predicted — strong, consistent performers |
| OpenAI | 12 | 3.9 | -0.4 | Excellent — tight cluster around expected performance |
| Meta | 11 | 5.0 | -0.7 | Good |
| Anthropic | 8 | 7.2 | +3.6 | Systematically underpredicted |
| Mistral | 10 | 8.1 | +2.3 | Mixed — frontier models underpredicted |
| xAI | 4 | **13.2** | +5.5 | Worst — Grok 3 Beta has extreme formatting (40× length, 52× lists vs median) |

### By Model Type

| Group | RMSE | Notes |
|-------|------|-------|
| Reasoning models | 4.8 | Easier to predict — benchmarks capture their strengths well |
| Non-reasoning | 7.2 | Harder — Arena rewards conversational quality not captured by benchmarks |

### Key Findings

- **xAI/Grok is the worst outlier**: Grok 3 Beta's extreme formatting divergence (40× length, 52× lists vs median model) maps to huge Arena underprediction. Style features help but can't fully capture this.
- **Missingness does NOT explain errors**: Models with 20+ missing benchmarks can still be well-predicted (e.g., Google models). Imputation quality matters more than missingness count.
- **Style features fixed prior hard cases**: ChatGPT-4o went from +35 → -1.4 error, DeepSeek R1 from +34 → +12.8. But Claude Opus 4.1 (+44.0) and Grok models remain hard.
- **Provider effect is real**: Arena voters may have provider-specific preferences (conversation style, safety behavior) that benchmark scores can't capture.

---

## LOBO Column Analysis (2026-03-16)

Full leave-one-benchmark-out analysis on all 65 non-style, non-target columns. For each column, drop it, re-impute from scratch, re-run full pipeline, measure RMSE. Baseline: 18.84 (after dropping eqbench_creative_elo).

### Confirmed harmful (LOBO-dropped)
- `eqbench_creative_elo`: -0.21 all, -0.79 T20, -1.06 T10 — harmful across all 3 metrics

### Harmful overall but help top-tier (kept)
| Column | d_all | d_T20 | d_T10 | Notes |
|---|---|---|---|---|
| eqbench_eq_elo | -0.98 | +1.67 | +5.01 | Critical for T10 |
| aa_eval_lcr | -0.65 | +1.07 | +2.45 | |
| livebench_spatial | -0.52 | +0.05 | +1.09 | |
| livebench_tablejoin | -0.49 | +1.11 | +2.88 | |
| yupp_Coding_Score | -0.39 | -0.83 | -0.33 | Harmful across all 3 but dropping destabilizes pipeline |

### Most valuable columns
| Column | d_all | d_T20 | d_T10 | Notes |
|---|---|---|---|---|
| livebench_code_generation | +2.47 | +3.77 | +4.24 | |
| ugileaderboard_Writing | +2.36 | +4.42 | +6.59 | |
| livebench_typescript | +2.22 | +2.90 | +5.58 | |
| livebench_python | +2.11 | +2.83 | +4.55 | |
| livebench_code_completion | +2.08 | +3.80 | +5.51 | |
| aa_pricing_output_tokens | +1.87 | +3.72 | +5.39 | |
| livebench_javascript | +1.86 | +2.84 | +5.40 | |
| aa_eval_ifbench | +1.76 | +3.59 | +6.09 | Most valuable for T10 |
| logic_PC3 | +1.78 | +3.57 | +5.12 | |
| openbench_Reasoning | +1.56 | +2.70 | +5.69 | |
| writing_GPT-5 (low)_score | +1.02 | +4.82 | +6.86 | Most valuable single column for T10 |

### Key insight: tier-dependent value
Many columns hurt overall RMSE but help top-tier prediction. At the top, Arena voters differentiate on specific capabilities (coding, IF, writing). At mid-tier, those same benchmarks add noise because the models haven't crossed the threshold where those capabilities matter for preference.

### Partial correlations are unreliable for imputation pipelines
yupp_Coding_Score was the ONLY significant column in partial correlation analysis (r=-0.259, p=0.004) but LOBO showed it's harmful (-0.39 all). lechmazur_confab showed as harmful in partial correlation (r=+0.192, p=0.033) but LOBO showed it's valuable (+0.67 all). The imputation chain creates indirect effects that observational analysis cannot detect. **LOBO is the only reliable method for column value assessment.**

---

## Joint Prediction Experiments (2026-03-12)

Testing whether joint optimization of benchmark reconstruction + target prediction can break the 17.89/19.76 wall. Two approaches implemented in `arena_predictor/joint_predict.py`:

1. **SCMF (Supervised Collective Matrix Factorization)**: ALS-based, learns latent factors Z that jointly minimize reconstruction + target prediction loss. SVD warm-start.
2. **BHLT (Bayesian Hierarchical Latent-Trait)**: EM-based factor analysis with hierarchical family priors. Returns calibrated posterior uncertainty.

Both tested in standalone mode (pure latent model predicts directly), hybrid mode (inject Z factors into existing pipeline), and inductive mode (train-only factor fitting).

### Results (Waves 1-2, 10×10-fold on all 112)

Baseline: **19.76** (existing pipeline, 10×10-fold)

| Exp | Approach | Config | RMSE | Delta | Verdict |
|-----|----------|--------|------|-------|---------|
| 1 | SCMF standalone | rank=6, λ_target=5 | 29.24 | +9.5 | Much worse |
| 2 | **BHLT standalone** | k=6, family_prior=1.0 | **23.77** | +4.0 | Best joint model, still worse |
| 3 | SCMF hybrid | rank=6, λ_target=10 | 29.15 | +9.4 | Hybrid doesn't help SCMF |
| 4 | SCMF hybrid | rank=4, λ_target=5 | 27.10 | +7.3 | Lower rank slightly better |
| 5 | SCMF hybrid | rank=8, λ_target=5 | 34.33 | +14.6 | Higher rank overfits badly |
| 6 | SCMF hybrid | rank=6, λ_target=10 | 29.13 | +9.4 | Lambda insensitive |
| 7-12 | Various | — | — | — | Terminated (OOM), rerunning |

### Key Findings

- **SCMF is uniformly bad (27-34 RMSE)**: The ALS optimization finds latent factors that reconstruct benchmarks well but don't predict Arena. Joint supervision (λ_target) doesn't help — the reconstruction loss dominates.
- **BHLT is better (23.77) but still loses by 4 points**: The hierarchical structure provides better regularization than SCMF, but pure latent factors can't compete with the pipeline's direct ALT + trajectory + style features.
- **Hybrid injection doesn't rescue SCMF**: Injecting SCMF Z factors as additional features into the existing pipeline (exps 3-6) gives essentially the same RMSE as standalone. The Z factors are redundant with what the pipeline already computes.
- **Rank sensitivity**: rank=4 slightly better than 6, rank=8 much worse. At n=112, even 8 latent factors overfit.

**Conclusion**: Joint optimization of reconstruction + prediction is theoretically appealing but practically inferior to the staged pipeline. The existing pipeline's PCA-10 ALT + feature selection + poly interactions is a better inductive bias for this small-n problem than learned latent factors. The IRT experiments (Section above) showed the same pattern — better ALT prediction (16.82 vs ~23) but worse final target prediction.

---

## Cycle 10: Target Switch + KNN Pipeline (2026-03-28/29)

### Key Discovery: lmsys_Score is leakage

lmsys_Score (raw Arena ELO) and lmarena_Score (style-controlled Arena ELO) are both derived from the same Arena (arena.ai) voting process. Using lmarena as a feature to predict lmsys was circular — the high correlation (r=0.95) was because they measure the same thing, not because lmarena adds independent signal. The gap between them is driven by subjective style preference that benchmarks can't capture.

**Decision:** Switch target from lmsys_Score to lmarena_Score. Drop lmsys_Score from clean CSV entirely.

### lmarena is much more predictable

| Target | Global Ridge RMSE | KNN(50) RMSE | Spearman |
|--------|-------------------|-------------|----------|
| lmsys_Score | 22.29 | 21.02 | 0.928 |
| **lmarena_Score** | **18.95** | **17.20** | **0.955** |

12 of the top-30 lmsys models were preferred by humans for reasons benchmarks can't capture (Mistral Medium -57, ChatGPT-4o -46). When controlling for style (lmarena), most rank where benchmarks expect. Top-quartile identification: 25/30 for lmarena vs 18/30 for lmsys.

### Why KNN beats global Ridge

Different features predict lmarena at different score levels:
- **Top tier:** YUPP text/coding (+0.79), omniscience (+0.78), weirdml (+0.78), hardest benchmarks
- **Mid tier:** EQ (+0.49), writing (+0.45), conciseness (style length flips negative above ~1400)
- **Bottom tier:** Raw capability (MMLU-Pro, livebench coding), response length (+0.30)

KNN(50) + Ridge naturally adapts: each model's 50 nearest benchmark-space neighbors get their own Ridge fit with locally-relevant coefficients. No explicit tiers needed.

### Adaptive k is critical

Fixed k=50 forces bottom-tier models (sparse in feature space) to include distant, irrelevant neighbors spanning 170+ ELO. Distance-based cutoff at 2.0× nearest neighbor distance gives:

| Tier | Fixed k=50 | Adaptive k (avg) |
|------|-----------|-----------------|
| Top 30 | k=50 | k≈54 |
| Mid 30+30 | k=50 | k≈56 |
| Bottom 33 | k=50 | **k≈37** |

Bottom models get tighter neighborhoods → RMSE drops from 23.33 to 20.30 for that tier.

### Kernel weighting + jackknife variance inflation

Ridge regression shrinks predictions toward the neighborhood mean (centering bias):
- Top models underpredicted by +7.5 points
- Bottom models overpredicted by -5.7 points

**Gaussian kernel weighting** (bandwidth at 30th percentile neighbor) emphasizes closer neighbors.

**Jackknife variance inflation** measures how much Ridge compresses within each neighborhood (leave-each-neighbor-out, refit, compare jackknife predictions to actuals). Estimates compression slope b≈1.16. Corrects by stretching predictions away from neighborhood mean.

After correction: top bias +4.5, bottom bias -2.1.

### Final pipeline: Adaptive KNN + Kernel Ridge + Jackknife VI

| Step | Detail |
|------|--------|
| 1. Impute | ModelBankImputer fills missing benchmark features (unchanged) |
| 2. Standardize | StandardScaler on all 105 features |
| 3. Find neighbors | Adaptive k: all within 2.0× nearest distance, min 20, max 80 |
| 4. Kernel weight | Gaussian kernel, bandwidth at 30th percentile of neighbor distances |
| 5. Ridge fit | alpha = max(10, std(neighbor_scores)) |
| 6. Jackknife VI | Leave-each-neighbor-out, estimate compression slope b, clip [1.0, 1.5], inflate |

**Results:** RMSE 15.27 (LOO), R²=0.927, Spearman=0.958

### What was tried and didn't help (for lmarena prediction)

| Approach | RMSE | Notes |
|----------|------|-------|
| GP regressor for ALT | 31.82 | Overfits with n=123 |
| BayesianRidge within KNN | 17.70-19.18 | Overfits at small k, ok at k=80 |
| Elastic Net within KNN | 17.25-18.10 | L1 doesn't help locally |
| Local PLS | 17.28-22.00 | 3 components ok, more overfit |
| Local feature selection (corr) | 19.15-21.60 | Unstable at n=50 |
| Feature-pruned distance | 17.41-18.12 | Marginal |
| Multi-scale ensemble (k=30,50,80 avg) | 17.16 | Marginal improvement |
| Bagged KNN (random feature subsets) | 17.07 | Marginal |
| Residual KNN | 18.09-18.68 | Global model anchor hurts |
| Affine de-shrink (inner LOO) | 16.89 | Too noisy |
| Quantile mapping | 16.97-18.05 | Marginal to worse |
| Isotonic recalibration | 17.28-18.59 | Worse |
| Directional weighting | 17.35-24.35 | Hurts overall |
| Predict rank → map to score | 19.22 | Much worse |

### Other approaches tried for lmsys prediction (before target switch)

- **Two-stage lmarena pipeline** (old approach): 19.33 RMSE. lmarena was used as a feature. Now recognized as leakage.
- **PC-tier archetype split** (PC2/PC3): 21.02 honest RMSE for lmsys. PC2=style axis, PC3=reasoning axis. Helps bottom tier but hurts top.
- **Tiered target models**: All worse than global when honest (cross-validated tree splits destabilize).
- **Boosted classification for tier routing**: 82% LOO accuracy for top-30 vs mid-hi, but routing errors amplify prediction errors.
- **Direct prediction (no lmarena feature)**: 22.29 RMSE for lmsys. Ridge on 108 features.
- **Sliding window by predicted score**: 19.85 RMSE for lmsys (hybrid with PC-tier for bottom).

### SVD features matter

SVD factors from imputation matrix encode latent model archetype. Ablation on lmarena:

| Config | RMSE |
|--------|------|
| All 108 (raw + SVD raw + SVD sq + SVD cross) | 15.27 |
| No SVD cross interactions | 15.84 |
| No SVD squared | 15.89 |
| SVD raw only | 16.04 |
| No SVD at all | 16.48 |

Every SVD component helps. Total SVD contribution: 1.21 RMSE.

---

## Open Questions

1. ~~**Sliced error analysis**~~ — **Done.** Per-tier analysis complete. Bottom 33 models account for 45% of total error.
2. **Exogenous metadata features** — provider, release date, parameter count, open/closed source. Genuinely orthogonal to benchmark scores.
3. **More training rows** — still a major lever. n=123 → 150+ would help bottom-tier prediction where neighbors are sparse.
4. **Bottom-tier prediction** — RMSE 20.30 for bottom 33 models vs 11-15 for upper tiers. These models are spread in feature space with neighbors spanning 170+ ELO. Fundamentally harder — small/niche models whose Arena performance is driven by factors benchmarks don't capture.
5. **ChatGPT-4o / Mistral Medium underprediction** — still systematically underpredicted (+36, +37 for lmarena). These models are preferred by humans beyond what any benchmark measures. BullshitBench doesn't help (r=0.25 with Arena).
6. **New benchmarks for top-tier differentiation** — Top-30 vs Mid-Hi effect sizes are only 0.6-0.96 on existing benchmarks. Need benchmarks measuring: adaptive verbosity, conversational coherence, appropriate confidence calibration.
7. ~~**Full repeated K-fold verification**~~ — **Done.** KNN pipeline: 16.03 (10×5-fold), 15.63 (LOO) with power cutoff.
8. **Pipeline SVD factor count** — Pipeline produces 6 SVD factors (105 features) vs 8 in standalone experiments (108 features). The 3 extra features contribute ~0.6 RMSE. Could configure imputer for 8 factors.

---

## Cycle 11: Sublinear Power Cutoff + Walk-Forward Analysis (2026-03-30)

### Walk-Forward CV Discovery

Built temporal (walk-forward) CV: sort models by release date, train on older, predict each new one. This simulates the actual use case — predicting Arena scores before a model goes live.

Walk-forward RMSE was 16.64 (vs 16.06 random CV with old linear cutoff). The gap exposed problems with frontier model prediction — particularly Claude Opus 4.6 (error -26.6) and other top models.

### Feature Sign Flips

Within the top 15 models (≥1450), features flip sign vs the overall population:
- `style_q7_list_count`: ρ=-0.74 at top, ρ=+0.16 overall
- `livebench_code_completion`: ρ=-0.32 at top, ρ=+0.61 overall
- `style_normalized_list_count`: ρ=-0.62 at top, ρ=+0.02 overall

With k≈58 neighbors (linear cutoff), the local Ridge spans both sides of the flip and averages the coefficients to near zero. Features that predict the #1 model at the top (like `aa_eval_mmlu_pro` ρ=+0.95, `weirdml_avg_acc` ρ=+0.93) get diluted by mid-tier models where those features are irrelevant.

### Power Cutoff: `max_dist = d0^0.7 × 3.0`

Replaced the linear cutoff (`d0 × 2.0`) with a sublinear power function. This naturally gives tighter neighborhoods in dense regions:
- d=10 (top models): effective mult ≈1.5× → k≈25
- d=15 (sparse models): effective mult ≈1.35× → k≈20

Swept 2,700 configs across: power exponent (0.5-1.0), coefficient (2.0-4.0), bandwidth (0.10-0.40), Ridge alpha strategy (adaptive/fixed, floor 5-20), and local feature selection (0/15/30).

**Winner:** power_alpha=0.7, power_C=3.0, bw_pct=0.15, adaptive alpha (`max(10, std(neighbors))`), no feature selection. Ranked #1 on combined LOO + WF score out of 2,700 configurations.

### Results

| Metric | Old (dm=2.0, bw=0.3) | New (d^0.7×3, bw=0.15) |
|--------|----------------------|------------------------|
| LOO RMSE | 15.39 | 15.63 (+0.24) |
| LOO R² | 0.926 | 0.924 |
| LOO Spearman | 0.963 | 0.962 |
| LOO Top-Q | 25/30 | 25/30 |
| WF RMSE | 16.64 | 14.86 (-1.78) |
| WF R² | 0.873 | 0.899 |
| WF Pearson | 0.937 | 0.949 |
| Opus 4.6 WF err | -26.6 | -21.9 |

### Key Findings

1. **Interpolation vs extrapolation tradeoff**: tighter neighborhoods help walk-forward (extrapolation to novel frontier models) but slightly hurt LOO (interpolation among known models). The power cutoff gives the best balance.
2. **Feature selection within neighborhoods** (LocalCorr) helps on raw 85-feature data but hurts on the production 105-feature pipeline. The power cutoff provides implicit feature adaptation — explicit selection is redundant and costs degrees of freedom.
3. **Learned distance weighting** (feature importance × residual correlation) helps on raw features but is neutral on the production pipeline.
4. **Fixed alpha=20 vs adaptive alpha**: fixed helps walk-forward slightly but hurts LOO. Adaptive alpha (`max(10, std)`) naturally scales with neighborhood diversity and is the better default.
5. **Jackknife clip is not the bottleneck**: Opus 4.6's natural b=1.49 (not hitting the 1.5 clip). The error is because it's 26 points above its max neighbor — genuine extrapolation beyond training data.
6. **Remaining residuals correlate with hard reasoning benchmarks** (AIME ρ=-0.64, LiveCodeBench ρ=-0.55). Models that crush hard benchmarks get overpredicted — humans don't reward hard reasoning proportionally at the frontier.

---

## PLS Hybrid + drop_style_tone (2026-04-18, SHIPPED)

### Summary

Two flags added on the same day after an ablation cycle on the `_sweep_*` branch:

- `--drop_style_tone` — drop every `style_*` and `tone_*` column before the KNN feature matrix is built. These features still inform the `style_predicted_delta` feature, the judge-bias analysis, and the imputer's correlation-based predictor selection; they just hurt the KNN distance once PLS supervision is introduced.
- `--pls_hybrid_k 3` — inside each CV fold, fit `PLSRegression(n_components=3)` on `(Xtr, y[tr])` and append the 3 transformed components to both `Xtr` and `Xte` before the adaptive-KNN call. No leakage: PLS is fit only on training rows in each fold.

At the same time, `--eb_parent` (and its sidecars `--eb_parent_tier_n`, `--redundancy_threshold`, `--rank_by_corr_only`) was **removed** from the shipped invocation. The ablation showed EB parent over-shrunk once PLS was on.

### Result

| Config | 10×5 OOF RMSE | Top-15 RMSE | Opus 4.7 err |
|---|---:|---:|---:|
| Sem v4 @ 32, pre-PLS (earlier baseline) | 14.45 | 15.53 | −24.24 |
| **Sem + PLS hybrid + drop_style_tone (SHIPPED)** | **13.48** | **13.44** | **−19.38** |
| Δ | −0.97 | −2.09 | +4.86 ELO |

Biggest honest-4.7 gain of the 2026-04-18 session. Top-tier error dropped disproportionately (−2.09) — PLS supervises the distance toward "the direction of ELO," which is most helpful where neighborhood structure was weakest.

**Post-CritPT rerun (2026-04-18 PM, n=127):** Same SHIPPED config on the CritPT-augmented CSV lands at RMSE 13.61 / R² 0.941 / ρ 0.971 / top-15 RMSE 14.26 / Opus 4.7 Thinking err −17.78 (+1.60 ELO vs ship time). The ship-event delta above is preserved as historical.

### Why drop_style_tone now

The style/tone columns were added specifically to give the KNN distance a style-aware axis. Once PLS-3 is appended, the top PLS component captures style-relevant variance better than the raw style_ / tone_ columns, and the raw columns start diluting the distance with redundant dimensions. Ablation isolated the effect: dropping alone was worth −0.25 RMSE on top of the PLS gain, vs keeping them on was +0.25 RMSE lost.

### Why EB parent was removed

Pre-PLS, EB parent shrunk uncertain cells toward the column mean, and the `--rank_by_corr_only` / `--redundancy_threshold` flags governed which predictors survived the EB filter. Once PLS is appended to the KNN features, it already absorbs the coarse column-mean effect that EB parent was providing. Keeping EB parent on led to measurable over-shrinking at the frontier (Opus/Sonnet thinking variants).

### Flags removed from predict.py (post-cleanup sweep)

`--local_corr_k`, `--learned_dist`, `--resid_weight`, `--coherence_fade_pct`, `--coherence_disagreement_beta`, `--coherence_tau_cap_pct`, `--redundancy_threshold`, `--rank_by_corr_only`, `--relax_support_escalator`, `--loo_size_fair`, `--eb_parent`, `--eb_parent_sigma_mult`, `--eb_parent_tier_n`. Also: `_eb_parent_shrinkage` method (~100 lines) removed from `column_imputer.py`. Flag count 60 → 46.

### Artifacts

- Sweep scripts: `arena_predictor/run_*_experiments.bash`
- Honest walk-forward: `arena_predictor/_walkforward_honest.py` → RMSE 14.69 / R² 0.900 / ρ 0.940 on n=23 newest
- Per-step walk-forward results: `/tmp/walkforward_honest_80.csv`
