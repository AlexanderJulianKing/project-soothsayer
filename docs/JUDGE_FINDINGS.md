# Judge Findings

Characterization of **what LLM judges reward** when evaluating pairwise model
battles across the four project benchmarks. Distinct from the rest of the
project, which treats battle outcomes as ground truth and minimizes RMSE in
an Arena-ELO predictor. This document treats battle outcomes as the *object
of study*: what are the judges measuring, how consistent are they, and how
predictable are their preferences from response content alone?

Reframing caveat: a judge rewarding specific response-shape features is not
automatically "bias" — each benchmark asks the judge to measure something
specific (empathy, creative writing quality, conversational tone), and
rewarding appropriate surface features on those tasks is a correct behavior,
not a miscalibration. The findings below are **instrument-characterization**:
what is this measurement device measuring, how consistently, and how
predictably?

---

## Data sources

- **EQ battles**: `soothsayer_eq/results/battle_history.csv` (10,773 rows across 112 scenarios).
  - Judges: Grok 4 Fast (5,868), Gemini 3.0 Flash Preview 2025-12-17 (4,805), Claude Opus 4.6 Thinking (100).
- **Writing battles**: `soothsayer_writing/results/battle_history.csv` (13,207 rows across 8 prompts).
  - Judges: Grok 4 Fast (12,207), Gemini 3.0 Flash Preview 2025-12-17 (1,000).
- **Style battles (tonebench)**: `soothsayer_style/results/battle_history.csv` (3,240 judged rows across 9 questions).
  - Judges: Grok 4.1 Fast (3,240). Criteria: `signal_density`, `conversational_confidence`.
- **Response text + embeddings**: `embeddings/cache/all_responses.parquet`
  (23,166 rows, 4 benchmarks) and `embeddings/cache/response_embeddings.parquet`
  (bge-small-en-v1.5, 384-dim, unit-normalized).

All battles are paired A-vs-B comparisons on the **same** scenario/prompt, so
prompt-content variation is fully controlled within each row.

---

## Methodology

### Two complementary probes

**1. Embedding-direction probe** (`judge_bias/embedding_direction_probe.py`)

For each battle, compute:
- `w = embed(winner_model, scenario)` (mean-pooled across turns/runs, unit-normalized)
- `l = embed(loser_model, scenario)`
- `delta = w - l`

Average deltas per (judge, benchmark) to get the **preference direction** the
judge rewards in 384-dim embedding space. Compare directions across judges
and across tasks via cosine similarity.

Scoring every model's per-benchmark fingerprint against the direction
(`dot(fingerprint, direction)`) ranks models by "systematically favored by this
judge, all else equal."

**2. Interpretable-shape probe** (`judge_bias/shape_feature_probe.py`)

Extract 17 interpretable textual features per response: char count, word count,
newlines, sentence count, avg sentence length, first-person rate, second-person
rate, hedge rate, question rate, exclamation rate, em-dash density, ellipsis
density, bullet-line density, numbered-line density, bold-marker density,
italic-marker density, all-caps-word density.

For each battle, compute `(winner_features - loser_features)`. Average
per-judge across battles — paired-within-battle design controls for scenario
difficulty and prompt content, so residual signal is pure judge preference
over **response shape**, not over content.

Report the per-feature mean delta, a paired t-statistic (`mean / sem`), and
the fraction of battles where winner > loser on that feature.

### Why this design is defensible

- **Paired within battle**: prompt content is constant within each
  (winner, loser) comparison. No scenario-difficulty confound.
- **Large n per judge** (1,000-12,207 battles). Even weak systematic biases
  yield high t-statistics via volume.
- **Unsupervised**: no fitting to Arena ELO, so no risk of leakage into the
  downstream RMSE-minimization task. The direction vectors are observational.
- **Two-probe consistency**: if the interpretable-shape probe and the
  embedding-direction probe tell the same story (they do), both are likely
  tracking the same latent judge-preference axis.

---

## Results (2026-04-15, bge-small embeddings, EQ + Writing)

### 1. Judges agree with each other more than they agree with themselves across tasks

Embedding-direction cosine similarities:

| Comparison | Cosine |
|---|---:|
| **Grok EQ ↔ Gemini EQ** | **+0.905** |
| **Grok Writing ↔ Gemini Writing** | **+0.939** |
| Grok EQ ↔ Grok Writing (same judge, different task) | +0.287 |
| Gemini EQ ↔ Gemini Writing (same judge, different task) | +0.279 |

Two different judge models from two different labs (xAI and Google) reward
**nearly the same direction** in embedding space within a task. The same judge
rewards **substantially different directions** on different tasks. This
asymmetry is the core result: *the bias is shared more across providers than
it is across tasks*.

### 2. Feature-level agreement between judges is near-perfect

Pearson correlation of the 17-feature mean-delta vectors:

| Benchmark | Gemini vs Grok (r) | Gemini vs Grok (cosine) |
|---|---:|---:|
| EQ (n=4,805 / 5,868) | **+1.000** | +1.000 |
| Writing (n=1,000 / 12,207) | **+0.998** | +0.998 |

On EQ: **17/17 features** have the same sign between judges.
On Writing: 14/17 features agree in sign; the three disagreements
(`exclamation_rate`, `numbered_lines_per_1kw`, `bold_markers_per_1kw`) are all
at |t| < 1.3 — noise-level effects.

### 3. The winning shape (what the judges reward) — Gemini on EQ

n = 4,805 paired battles; winner-minus-loser feature deltas; t-stat is paired.
Sorted by |t|:

| Feature | Direction | t-stat | winner > loser |
|---|---|---:|---:|
| hedge_rate (perhaps/maybe/might/could/…) | **penalized** | **−17.07** | 40.2% |
| em_dashes_per_100w | rewarded | +10.42 | 55.5% |
| first_person_rate | rewarded | +9.65 | 58.7% |
| numbered_lines_per_1kw | penalized | −8.29 | 32.4% |
| bullet_lines_per_1kw | penalized | −6.73 | 25.3% |
| bold_markers_per_1kw | penalized | −6.45 | 33.8% |
| newlines | penalized | −6.23 | 48.7% |
| second_person_rate | rewarded | +4.35 | 52.5% |
| avg_sentence_words | penalized | −4.25 | 44.6% |
| char_count | penalized | −3.74 | 47.7% |

Condensed: **prose with personal voice beats structured/formatted/hedged
responses.** Shorter wins marginally.

### 4. The winning shape — Gemini on Writing

n = 1,000 paired battles.

| Feature | Direction | t-stat | winner > loser |
|---|---|---:|---:|
| em_dashes_per_100w | rewarded | +5.11 | 49.2% |
| italic_markers_per_1kw | rewarded | +4.34 | 12.7% |
| second_person_rate | rewarded | +3.95 | 24.0% |
| first_person_rate | rewarded | +2.89 | 28.7% |
| sentence_count | rewarded | +2.51 | 52.4% |
| newlines | rewarded | +2.06 | 47.6% |
| question_rate | rewarded | +2.04 | 19.9% |
| hedge_rate | penalized | −1.33 | 38.3% |

Note: bullet and numbered lists have near-zero base rate in Writing, so their
signal is unavailable.

### 5. Stable preferences across *both* tasks *and* both judges

Features that reward winning in EQ AND Writing, for Gemini AND Grok:

- **em-dashes** (reliably reward; t > 5 in all 4 slices)
- **first-person pronouns** (reliably reward)
- **second-person pronouns** (reliably reward)
- **hedging** (reliably penalize; t = −17 on EQ, weaker but still negative on Writing)
- **shorter average sentences** (reliably reward)

Features that flip sign between tasks (genre-dependent, but judges still agree
with each other within each task):

- **Length** (char / word count): EQ penalizes length, Writing marginally rewards it.
- **Newlines**: EQ penalizes (prose expected), Writing rewards (paragraph breaks expected).
- **Italic / bold markers**: EQ penalizes (structured formatting out of place in conversation), Writing rewards (emphasis in creative prose).

### 6. Tonebench (style benchmark) — Grok 4.1 Fast, 3,240 battles

Added 2026-04-15 by extending the probe to `soothsayer_style/results/battle_history.csv`.
One judge only (Grok 4.1 Fast), so no cross-judge comparison here — but we do
get cross-task and per-criterion decompositions.

**Per-criterion shape preferences (same judge, two criteria on same battles):**

The tonebench evaluates two criteria per battle: `signal_density` ("how
information-dense is the writing?") and `conversational_confidence` ("how
confident is the speaker?"). Treat each criterion as a synthetic sub-battle
and recompute deltas — result:

- Pearson r of mean-delta vectors: **+1.000**
- Sign-level agreement: **11/17 features**

The two criteria share most of their bias (hedging penalized, em-dashes rewarded,
shorter rewarded — 11 of 17 features point the same direction) but diverge in a
specific, interpretable way:

| Feature | conversational_confidence | signal_density |
|---|---|---|
| exclamation_rate | **rewarded (t = +6.9)** | penalized (t = −2.3) |
| bullet_lines / bold_markers | penalized | rewarded |
| ellipses | rewarded | penalized |

- `conversational_confidence` wants **emphatic punctuation** (exclamations, ellipses) and rejects structured formatting (bullets, bold).
- `signal_density` wants **scannable structure** (bullets, bold, italics) and rejects exclamations.

The near-unit Pearson correlation is inflated by the 11 features that *do*
agree and dominate variance (hedging, first-person, em-dashes, length). This
is a methodological warning: **the r = +1.000 numbers in sections 1-5 above
should be read "correlated on the high-magnitude axes", not "identical on
every axis."**

### 7. Cross-benchmark universality across 5 (judge, benchmark) slices

Pooling 5 independent (judge × benchmark) slices — Grok 4 Fast on EQ and
Writing, Gemini 3.0 Flash on EQ and Writing, Grok 4.1 Fast on Style — gives:

**Universal (5/5 slices agree on direction):**

| Feature | Direction |
|---|---|
| hedge_rate | penalize |
| first_person_rate | reward |
| second_person_rate | reward |
| em_dashes_per_100w | reward |
| ellipses_per_100w | reward |

**Strongly consistent (4/5 slices agree):**
- sentence_count (reward, style dissents)
- avg_sentence_words (penalize — shorter wins, style dissents)
- exclamation_rate (reward, Gemini-Writing dissents)
- italic_markers_per_1kw (reward, Gemini-EQ dissents)
- all_caps_words_per_1kw (reward, Gemini-EQ dissents)

**Task-dependent (judges agree within-task, flip across tasks):**

The length axis (char_count, word_count, newlines) is the main driver of
*inverse* cosine similarity (~ −0.99) between Writing and the other two
benchmarks. EQ and Style both reward short+concise; Writing rewards
long+developed — judge-invariantly. The shared-bias property holds within each
task but cosine similarity flips when length-axis direction flips.

Pairwise cosine similarity across the 5 slices:

| | EQ-Grok | EQ-Gemini | Wr-Grok | Wr-Gemini | Style-Grok4.1 |
|---|---:|---:|---:|---:|---:|
| EQ-Grok | +1.000 | +1.000 | −0.991 | −0.997 | +0.998 |
| EQ-Gemini | +1.000 | +1.000 | −0.991 | −0.996 | +0.999 |
| Wr-Grok | −0.991 | −0.991 | +1.000 | +0.998 | −0.989 |
| Wr-Gemini | −0.997 | −0.996 | +0.998 | +1.000 | −0.994 |
| Style-Grok4.1 | +0.998 | +0.999 | −0.989 | −0.994 | +1.000 |

**Interpretation:** the shape-preference axis has basically two modes — EQ/Style
(short+direct+personal) and Writing (long+developed+personal). Within each
mode, three independent judges from two labs agree at cos > 0.99.

---

## Interpretation

### The summary, reframed

Each of these benchmarks has a specific job: EQ judges empathy and emotional
intelligence, Writing judges creative-prose quality, Style judges
conversational tone. That a judge on any of these tasks rewards specific
response-shape features is not automatically "bias" — it's the thing the
benchmark is asking the judge to measure. Shape preferences that a reasonable
human would also hold on the same task are not miscalibration; they are the
signal.

What's interesting isn't "the judges are biased." What's interesting is:

1. **Preferences are highly learnable from embedding space alone.** 70-74%
   per-battle accuracy and r ≈ 0.6-0.8 per-model win-rate prediction via
   linear probes on bge-small embeddings, under honest leave-out CV.
2. **Preferences are shared across judges within a task.** Gemini and Grok
   agree at r ≈ +1.0 (by Pearson on feature deltas) and cosine ≈ +1.0 (by
   embedding-space direction). Grok 4.1 Fast on Style points in near-identical
   direction to Grok 4 Fast on EQ (cos ≈ +0.999).
3. **Preferences are interpretable.** 59-69% of per-battle accuracy is captured
   by 17 hand-coded shape features. The embedding's extra 5-13pp is
   semantic/stylistic signal that doesn't reduce to counting em-dashes.
4. **A compact universal preference axis exists**: across 5 independent
   (judge × benchmark) slices, 5 shape features agree on direction (penalize
   hedging; reward em-dashes, ellipses, first-person, second-person pronouns).

This is an **instrument-characterization result**. LLM-as-judge is a
measurement device; these findings describe what that device is measuring,
how consistent it is across different device instances, and how much of its
output is predictable from inputs alone.

### Where "bias" framing would be appropriate

If one later wanted to audit these judges for bias in a human-facing sense
(demographic, political, correctness-calibration, etc.) the ingredients are
here:

- The 384-dim direction vector in embedding space for each (judge, benchmark)
  slice — an artifact the judge's preference is encoded in.
- The interpretable shape-feature regression — which surface features the
  direction correlates with.
- Per-model alignment scores — which specific models are most/least
  judge-favored.

The current writeup stops at characterization. Judgment about whether any of
these preferences constitute harmful bias requires an external yardstick
(human ratings on shared-task data, demographic-coded probe set, etc.) not
collected here.

---

## Predictability: can we predict judge preferences from embeddings alone?

If judge preferences are a real, learnable structure in embedding space rather
than noise around per-pair idiosyncrasy, simple probes should predict them
from embeddings alone — without any knowledge of model identity.

### Per-battle probe — leave-pair-of-models-out CV

Given a battle, take `embed(A) - embed(B)` and ask a logistic classifier to
predict which response the judge picked. Randomize A/B orientation per row
so the probe can't cheat on position. `GroupKFold` splits are by model-pair,
so the probe is never tested on model pairs it trained on.

| Slice | n battles | pairs | Accuracy | ± |
|---|---:|---:|---:|---:|
| EQ / Grok 4 Fast | 5,868 | 2,402 | **73.9%** | ±0.6% |
| EQ / Gemini 3.0 Flash | 4,805 | 2,127 | **73.4%** | ±1.3% |
| Writing / Grok 4 Fast | 12,207 | 3,680 | **74.2%** | ±0.8% |
| Writing / Gemini 3.0 Flash | 1,000 | 193 | **69.7%** | ±3.4% |
| Style / Grok 4.1 Fast | 3,202 | 1,192 | **70.1%** | ±1.7% |

**70-74% accuracy vs 50% chance.** ~3 out of 4 judge decisions are predictable
before any judge ever sees the responses. The remaining ~26% is the noise
floor — content correctness, reasoning quality, and idiosyncrasy that the
384-dim bge-small embedding can't distinguish.

### Per-model probe — leave-one-model-out Ridge

Given a model's fingerprint (average normalized embedding on a benchmark),
predict its win-rate with this judge via Ridge regression. Held-out target
model is excluded from training.

| Slice | n models | Pearson r (LOO) | R² (LOO) |
|---|---:|---:|---:|
| EQ / Grok 4 Fast | 151 | **+0.826** | +0.450 |
| EQ / Gemini 3.0 Flash | 197 | +0.675 | +0.338 |
| Writing / Grok 4 Fast | 211 | **+0.758** | +0.450 |
| Writing / Gemini 3.0 Flash | 151 | +0.586 | +0.258 |
| Style / Grok 4.1 Fast | 230 | +0.670 | +0.240 |

**r = +0.59 to +0.83 between predicted and actual win-rate.** 25-45% of
win-rate variance is captured by a linear function of the 384-dim fingerprint,
without ever training on that model.

### Ablation — how much of the probe signal is in simple features?

For the same per-battle setup, compare three feature sets:

  A. 17 interpretable shape features (length, pronouns, em-dashes, hedging, etc.)
  B. 384-dim bge-small embedding delta
  C. Concatenation

| Slice | Shape (17d) | Embed (384d) | Both | Δ (emb − shape) |
|---|---:|---:|---:|---:|
| EQ / Grok 4 Fast | 0.691 | 0.740 | 0.742 | **+5.0pp** |
| EQ / Gemini | 0.670 | 0.740 | 0.750 | **+7.0pp** |
| Writing / Grok 4 Fast | 0.612 | 0.743 | 0.741 | **+13.2pp** |
| Writing / Gemini | 0.591 | 0.713 | 0.685 | **+12.2pp** |
| Style / Grok 4.1 Fast | 0.615 | 0.702 | 0.710 | **+8.7pp** |

**Takeaways:**

- **59-69% of per-battle accuracy is captured by 17 hand-coded features alone.** Judge preferences are mostly reducible to surface-level formatting choices (hedging, em-dashes, first/second-person, list usage).
- **The embedding adds +5-13pp on top.** The extra signal is semantic/stylistic and doesn't reduce to any single countable feature.
- **Writing benefits most from the embedding (+13pp)**: creative-prose voice has more subtle distinctions than EQ's "hedging vs not."
- **EQ benefits least (+5-7pp)**: the "ineffable EQ quality" judges claim to evaluate is mostly the universal 5 shape features.
- **Concat rarely beats embedding alone** — shape features are redundant with embedding. On Writing/Gemini concat *hurts* (small n + more dims → overfitting).

### Interpretation — what is the embedding probe's direction "about"?

For each slice, train a full-embedding probe on all battles, then correlate
each shape feature with the response's projection onto the probe's preference
direction. Top |correlation| features per slice:

**EQ / Gemini 3.0 Flash** (cleanest signal):
- +0.53 first_person_rate
- −0.31 hedge_rate
- +0.28 em_dashes_per_100w
- −0.28 numbered_lines_per_1kw

**EQ / Grok 4 Fast**:
- +0.41 first_person_rate
- +0.37 em_dashes_per_100w
- −0.33 hedge_rate

**Style / Grok 4.1 Fast**:
- +0.45 exclamation_rate
- +0.32 ellipses_per_100w
- +0.30 first_person_rate
- −0.27 hedge_rate

**Writing / Grok 4 Fast** (weaker — confirms shape features miss more here):
- +0.34 em_dashes_per_100w
- +0.25 second_person_rate
- Everything else |r| < 0.2

**Writing / Gemini 3.0 Flash** (weakest):
- +0.23 second_person_rate
- +0.22 italic_markers_per_1kw
- +0.20 em_dashes_per_100w

The embedding direction *is* roughly "more first-person, more em-dashes, less
hedging" on EQ and Style — matching the universal 5-feature finding. For
Writing, the direction is more semantically distributed: no shape feature
correlates above |0.35|, which is why the 17-feature probe lags the embedding
probe most on Writing.

---

## Style-controlled ratings (Arena-style)

Analogous to Arena's (arena.ai) "style-controlled ELO": fit a
Bradley-Terry-with-covariates logistic regression per benchmark —

    log P(A beats B) / P(B beats A)
        = skill_A − skill_B + β · (style_A − style_B)

where model indicators are fit as fixed effects and style deltas are 17
shape-feature differences between the two responses. Coefficients on model
indicators are **style-controlled skill ratings**; the shift between the
model-only baseline and the style-controlled fit tells us which models'
raw ratings were inflated (or depressed) by style privilege alone.

### Target selection

The 3 pairwise-judge benchmarks contribute 4 targets, each fit on a single
judge to avoid averaging across judges with slightly different preferences:

| Target | Judge | n battles |
|---|---|---:|
| EQ | Gemini 3.0 Flash Preview (2025-12-17) | 4,805 |
| Writing | Grok 4 Fast | 12,207 |
| Tone / `signal_density` criterion | Grok 4.1 Fast | 3,202 |
| Tone / `conversational_confidence` criterion | Grok 4.1 Fast | 3,202 |

The tonebench's two criteria are treated as two separate targets because
their per-criterion winner distributions differ substantially — treating
them jointly would average out interpretable axes (earlier finding:
sign-level agreement between the two criteria is only 11/17 features).

### Accuracy changes when style is added to model-identity

| Target | Model-only | + Style | Δ |
|---|---:|---:|---:|
| EQ / Gemini | 0.821 | 0.816 | **−0.005** |
| Writing / Grok | 0.772 | 0.771 | **−0.001** |
| Tone / signal_density | 0.759 | 0.774 | **+0.015** |
| **Tone / conv_confidence** | **0.758** | **0.800** | **+0.042** |

Once model identity is known, style deltas barely help predict outcomes on
EQ and Writing: each model writes with consistent style, so "which model"
and "its style" are nearly redundant. On the tonebench criteria, style
deltas matter per-battle — especially for `conversational_confidence`,
where style adds +4.2pp. Confidence is itself a shape property, so
within-model style variation across prompts drives a significant fraction
of battle outcomes.

### Style coefficients across the 4 targets (standardized)

| Feature | EQ/Gemini | Writing/Grok | Tone/signal | Tone/conf |
|---|---:|---:|---:|---:|
| em_dashes_per_100w | **+0.53** | +0.23 | +0.21 | +0.18 |
| bold_markers_per_1kw | **−0.47** | −0.11 | −0.29 | **−0.74** |
| italic_markers_per_1kw | +0.34 | +0.09 | +0.28 | **+0.74** |
| hedge_rate | −0.31 | +0.01 | −0.23 | −0.28 |
| char_count | −0.37 | −0.13 | **+1.24** | −0.27 |
| word_count | +0.26 | +0.01 | **−2.29** | **−1.28** |
| sentence_count | +0.24 | +0.14 | +0.05 | **+0.91** |
| newlines | −0.01 | −0.05 | **+0.87** | +0.43 |
| numbered_lines_per_1kw | −0.01 | **+0.60** | −0.08 | −0.12 |
| first_person_rate | +0.25 | +0.07 | +0.03 | +0.12 |
| second_person_rate | +0.13 | +0.01 | −0.06 | **+0.39** |
| exclamation_rate | −0.00 | +0.06 | −0.23 | **+0.39** |

**Highlights:**
- **bold vs italic symmetry on tone_conf** (−0.74 vs +0.74): the
  "conversational confidence" criterion punishes bold emphasis
  (heavy-handed) and rewards italic emphasis (understated).
- **Writing/Grok's signature bias is numbered_lines** (+0.60) — Grok
  specifically rewards numbered lists in Writing battles; none of the other
  three targets do.
- **hedge_rate is penalized on all 3 non-Writing targets.** Writing is the
  exception, and even there only at +0.01 (near-noise).
- **em_dashes is rewarded on all 4 targets.** The single most universal
  style preference.
- Length-related coefficients (char_count, word_count, sentence_count) are
  multicollinear; see caveat below.

### EQ / Gemini — biggest ratings shifts from style control

**Style-inflated (lost most):**

| Shift | Raw | Controlled | Model |
|---:|---:|---:|---|
| −1.89 | +1.29 | −0.60 | Qwen 3 235B A22B Thinking |
| −1.81 | +0.35 | −1.45 | Qwen3 Next 80B A3B Instruct |
| −1.70 | +0.77 | −0.94 | LongCat Flash Chat |
| −1.59 | +0.76 | −0.83 | Qwen 3 235B A22B Nonthinking |
| −1.15 | +2.14 | +0.99 | Claude Opus 4.6 |
| −1.08 | +0.86 | −0.22 | Qwen3 Max |
| −1.02 | +3.44 | +2.42 | Claude Opus 4.6 Thinking |
| −1.00 | +0.11 | −0.89 | Grok 4.1 Fast |

**Style-disadvantaged (gained most):**

| Shift | Raw | Controlled | Model |
|---:|---:|---:|---|
| **+2.46** | −1.78 | +0.68 | Llama 3.1 Nemotron 70B |
| +1.10 | −1.71 | −0.61 | Mistral Large (2411) |
| +0.94 | −1.19 | −0.26 | Gemma 2 9B IT |
| +0.93 | +0.49 | +1.42 | Claude 4 Opus Thinking |
| +0.89 | −0.82 | +0.07 | Claude 3.5 Sonnet |
| +0.80 | +0.88 | +1.68 | Claude 4 Opus |
| +0.79 | +0.77 | +1.56 | Claude Opus 4.1 |
| +0.73 | +0.34 | +1.07 | Claude 4 Sonnet Thinking |
| +0.68 | +0.33 | +1.01 | Claude 4 Sonnet |

**Pattern**: the entire older Claude family (3.5 Sonnet, 4 Opus, 4 Opus
Thinking, 4 Sonnet, 4 Sonnet Thinking, Opus 4.1) is
**style-disadvantaged** — these models gain substantially when style is
controlled. Qwen-family + LongCat are **style-inflated**.

Notable wrinkle: **Claude Opus 4.6 (both variants) are style-inflated**,
opposite the older Claude family. Suggests Anthropic adjusted Claude Opus
4.6's default style toward what LLM judges reward.

### Writing / Grok — small shifts (Writing is less style-predictable)

Biggest losers: LongCat Flash Chat (−0.85), Qwen3 Next 80B A3B Instruct
(−0.80), QwQ 32B (−0.70), Llama 3.1 Nemotron Ultra 253B (−0.64), Claude
Haiku 4.5 Thinking (−0.64). Biggest winners: Qwen3.5 397B Nonthinking
(+0.37), Llama 3.1 8B (+0.36), DeepSeek R1 2025-05-28 (+0.35), llama-4-scout
(+0.35), GPT-5.4 Thinking (+0.34).

### Tonebench by criterion

**`signal_density`** (moderate style dependence):
- Biggest losers: Llama 4 Maverick (−0.55), GPT-5.4 Thinking (−0.44), GPT-5.2
  Chat (−0.43), Claude Sonnet 4.6 Thinking (−0.38), Claude Opus 4.5 Thinking
  Low (−0.37).
- Biggest winners: Gemini 2.5 Flash Preview Nonthinking (+0.62), Qwen3 Next
  80B A3B Instruct (+0.52), o3 Medium (+0.43), Qwen3.5 Plus Nonthinking
  (+0.39), Mistral Large 3 (+0.39).

**`conversational_confidence`** (highest style dependence of any target):
- Biggest losers: Qwen 3 30B A3B (−0.85), MiMo-V2-Pro (−0.61), Gemini 1.5
  Flash (−0.55), Gemma 2 9B IT (−0.54), Qwen3.5 27B Thinking (−0.54).
- Biggest winners: **Qwen3 Next 80B A3B Instruct (+1.10)**, LongCat Flash
  Chat (+1.05), Gemini 2.5 Flash Preview Nonthinking (+0.76), Kimi K2
  (+0.75), Olmo 3.1 32B (+0.65).

### Methodological caveat

The 17 style features are multicollinear — `word_count`, `sentence_count`,
and `avg_sentence_words` satisfy `word_count = sentence_count ×
avg_sentence_words`, so their individual partial coefficients are hard to
interpret. On tonebench the fit pushes `char_count +1.24` and `word_count
−2.29` for signal_density — these need to be read jointly as "more chars,
not more words" = "fewer longer/denser words", not as independent causal
effects. The **rating-shift analysis is robust** — it uses the joint style
contribution, not individual coefficients — but per-feature coefficients
should not be read as per-feature causal claims.

### Artifacts

One row per model with raw skill, controlled skill, and shift:
- `judge_bias/output/style_controlled_ratings_eq_gemini.csv`
- `judge_bias/output/style_controlled_ratings_writing_grok.csv`
- `judge_bias/output/style_controlled_ratings_tone_signal_density.csv`
- `judge_bias/output/style_controlled_ratings_tone_conv_confidence.csv`

Reproducer: `judge_bias/style_controlled_ratings.py`.

---

## What this does NOT establish

- **Whether the bias is "wrong".** Each benchmark has its own job — EQ judges
  empathy, Writing judges creative-writing quality, Style judges
  conversational-tone features. That a judge rewards response shape is the
  *point* when the task is shape evaluation. These findings are about
  consistency, predictability, and interpretability, not moral valence.
- **Whether humans would agree with the judges' preferences.** No human-rater
  ground truth was collected for these specific battles.
- **Whether other judges (Llama, Qwen, as-yet-untested Claude) fit the same
  pattern.** Claude Opus 4.6 Thinking has n = 100 EQ battles in history but
  isn't used for Writing, and n = 100 is too small for a stable direction
  estimate.
- **Per-response-embedding generalization to novel tasks.** All probes are
  trained and tested within a (judge, benchmark) slice. Cross-slice
  generalization is a separate question — the cosine-similarity analysis in
  Section 7 suggests EQ ↔ Style directions are similar (cos ≈ +0.999),
  EQ/Style vs Writing are opposite, so a probe trained on EQ should generalize
  to Style but not Writing.

## Reproducibility

```bash
# Embedding-direction probe — judges' preferred directions and favored models.
python3 judge_bias/embedding_direction_probe.py

# Interpretable-shape probe — what textual features predict winning.
python3 judge_bias/shape_feature_probe.py --benchmark eq
python3 judge_bias/shape_feature_probe.py --benchmark writing
python3 judge_bias/shape_feature_probe.py --benchmark style
python3 judge_bias/shape_feature_probe.py --benchmark style --per_criterion

# Cross-benchmark cosine similarity across all 5 (judge, benchmark) slices.
python3 judge_bias/cross_benchmark_shape.py

# Residualize judge preferences against Arena ELO, interpret residuals via style_*.
python3 judge_bias/residualize_vs_arena.py

# Linear-probe per-battle and per-model prediction accuracy (LPO-CV / LOO Ridge).
python3 judge_bias/predict_from_embeddings.py

# Ablation (17-feature vs 384-dim probe) + interpretation of embedding direction.
python3 judge_bias/ablate_and_interpret.py

# Style-controlled (Arena-style) skill ratings via Bradley-Terry with covariates.
python3 judge_bias/style_controlled_ratings.py
```

Both scripts read from `embeddings/cache/all_responses.parquet` and
`embeddings/cache/response_embeddings.parquet` (bge-small). Run
`embeddings/collect_responses.py` then `embed_responses.py` to rebuild from
scratch.

---

## Natural next steps (not yet done)

1. **Residualize against Arena ELO.** For models with known `lmarena_elo`,
   regress per-model "judge preference" (dot product of fingerprint with judge
   direction) against Arena ELO. The residual quantifies bias independent of
   skill. Large residual on a specific axis = judge rewards it *beyond* what
   skill explains.
2. **More judges.** Claude Opus has n = 100 EQ battles in history but isn't
   used in Writing. Extending the battle set (or introducing another judge
   family: Llama, Qwen) would either reinforce or qualify the shared-bias
   story.
3. **Per-criterion decomposition.** `criteria_json` contains per-criterion
   judge decisions (e.g. "demonstrated_empathy", "pragmatic_ei" on EQ;
   "writing_quality", "instruction_following" on Writing). The aggregate
   winner/loser collapses these; splitting by criterion could reveal that
   different criteria reward different shape features.
4. **Cross-reference the fingerprint direction with `style_` and `tone_`
   features.** The 17 features here are ad hoc; the project already contains a
   richer stylometry feature set. Regressing the judge direction onto the full
   style/tone matrix would produce interpretable coefficients in the existing
   feature vocabulary.
