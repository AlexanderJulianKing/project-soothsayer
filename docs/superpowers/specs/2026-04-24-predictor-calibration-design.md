# Predictor Calibration Redesign

**Date:** 2026-04-24
**Status:** Design approved, pending implementation plan
**Owner:** Alexander King

## Problem

`arena_predictor/predict.py` currently emits a prediction interval and two event probabilities (`num_one_prob`, `top_by_margin_prob`) per model in `predictions_best_model.csv`. The probabilities are not calibrated in a way that supports the end-user question — *"what is the probability this new model takes the #1 spot on lmarena?"*

Concrete defects in the current output (run `output_20260424_132648`):

1. **`sigma_hat` is constant across all 166 rows (= 21.81).** It comes from `compute_grouped_conformal_intervals`, which assigns each model to one of four coarse groups (top-tier × high-missing) and uses the OOF q95 residual for that group. In this run, every model landed in the same predicted-top group so sigma is effectively a scalar. A model in a dense, well-observed region of feature space gets the same uncertainty as an extrapolative frontier model.
2. **`num_one_prob` is named for "takes #1" but isn't calibrated for it.** It computes `P(score > max(y_train))` via t-CDF with the constant sigma. The threshold treats the current leader's score as known without error, which is acceptable here (ELO day-over-day swings are ≤5 pts). The real problem isn't the formula — it's that (a) the sigma feeding it is constant across all models (see defect 1), (b) the OOF sigma it relies on is ~8% too narrow for novel-frontier models (see defect 3), and (c) the column name suggests a mutually-exclusive "takes #1" allocation when it is a marginal "beats the current leader" probability. A calibrated marginal probability of beating the leader is the right decision surface — this design targets that, honestly named.
3. **Novel-frontier regime is systematically overconfident.** OOF RMSE is 13.61; walk-forward RMSE on the newest 20% is 14.69. Probabilities calibrated on OOF are ~8% too narrow in the regime that actually matters for new-model prediction.
4. **`num_one_prob` threshold in walk-forward evaluation is wrong.** The current code uses the full-sample `max(y_all[train_idx])`. For honest walk-forward scoring of the "takes #1" event at step *t*, the threshold must be the leader *available at step t* — the full-sample max leaks future information into the evaluation.

## Goal

Emit a per-model `p_beats_leader` in `predictions_best_model.csv` that is a calibrated marginal probability of `score > current_leader_score`, and a per-model `sigma_hat` that varies with local extrapolation risk and matches realized error magnitudes in the novel-frontier regime.

Explicit non-claim: `p_beats_leader` is a marginal, per-candidate probability. It does not form a coherent allocation over a multi-candidate field (the values do not sum to 1, and two candidates can both have `p_beats_leader > 0.5` without contradiction). That is the correct decision surface for the user's question ("is this specific new model going to beat the current leader?") but should not be misread as "probability of winning a multi-way race."

Success criterion: on a walk-forward holdout (newest 20% of models, re-fit at every step), the predictive distribution passes:

- PIT histogram approximately uniform (KS p-value not < 0.01)
- Coverage at 50/80/95% nominal levels contained within exact binomial 90% CIs
- Brier and log-loss for stepwise `y_t > max_leader_t` event strictly better than the current constant-sigma baseline, and the same on the top slice (`mu_t ≥ 1400`)

These are descriptive acceptance checks, not a CI gate — n≈25 is small enough that individual runs can fail by chance.

## Non-goals

- Treating the current #1's lmarena score as uncertain. Day-over-day swings are ≤5 pts, inside the predictor's own error band; plumbing leader uncertainty through adds code without changing decisions.
- Replacing the point predictor. RMSE is not the target; only the predictive distribution around the existing `mu` changes.
- Max-over-leaderboard probabilities (joint `P(new > all current)`). The user's decision surface is "beats current #1," which is a single-threshold question.

## Validation strategy

Two-stage, agreed with user before design:

- **Shape** (scale function + tail `t_df`) is fit on OOF residuals (n=127). Enough data to see how uncertainty varies with local signals.
- **Level** (one scalar multiplier `m`) is fit on walk-forward residuals (n≈25). Corrects the OOF-to-novel-frontier gap.

## Architecture

Preserve the existing predict → impute → PLS → OOF loop → final fit pipeline through `mu` and OOF residuals. Replace the grouped-conformal block with a three-stage calibration:

```
(A) Diagnostic gate         ──┐
   OOF: y_nb_std vs |e|       │   no-go → fall back to constant-sigma + WF scalar
   top-slice monotonicity     │
                              ▼
(B) OOF normalized conformal-style shape (n=127)
    s(x)         = max(y_nb_std(x), s_floor)
    t_df         = MLE fit of t on r_i = e_i / s(x_i)
    q_hat        = empirical q95(|r|) / t_ppf(0.975, t_df)
    sigma_oof(x) = q_hat × s(x)
                              │
                              ▼
(C) Walk-forward level correction (n≈25)
    fit scalar m by MLE on WF z-scores at fitted t_df
    ship sigma(x) = m × sigma_oof(x)
                              │
                              ▼
(D) Output: per-model sigma (as t-scale parameter),
    per-model intervals (mu ± t_crit × sigma), and
    p_beats_leader = 1 − t_cdf((max_leader − mu)/sigma, t_df)
```

The `y_nb_std` signal drives local scale variation. It is already computed inside `predict_single` (around `predict.py:709`, currently only written to the optional jackknife log). Using it here keeps the scale model aligned with the point model — same neighborhood, same k chosen by the sublinear power cutoff.

### Why Approach 2 (OOF normalized conformal-style calibration with KNN-local scale)

Three candidate approaches were considered:

1. **Scale regression.** Fit `|e| ~ f(missingness, y_nb_std, distance-to-1NN, tier)` with regularization. Rejected for now: at n=127 the multi-feature regression tends to learn interpolation quirks more than frontier uncertainty, and it has more knobs than its evidence base supports.
2. **OOF normalized conformal-style calibration with KNN-local scale** — this design. Single local scale signal, one empirical OOF quantile anchor, one walk-forward scalar. Reuses infrastructure already in the codebase. Note: this is not strictly split-conformal — the shape (t_df, q_hat, gate decision, s_floor) is fit on the same OOF rows whose empirical quantile is used as the anchor. Calling it "conformal-style" acknowledges the lineage without overclaiming the split-conformal coverage guarantee.
3. **Weighted jackknife+ / Mondrian conformal.** Most principled for extrapolation, but more code and does not naturally produce the smooth `P(score > threshold)` output we need. Rejected as premature.

If (2) under-covers at the frontier, the first planned extension is to add one extrapolation-asymmetry signal (e.g. `mu − max(y_nb)`, which captures how far Ridge is pushing the prediction beyond its neighborhood). Not a full multi-feature regression.

## Components

### (1) `compute_oof_nb_std`

Runs the existing LOO adaptive-KNN loop and persists, for each training row, the `y_nb_std` of the neighborhood chosen by the sublinear power cutoff. Also persists `y_nb_std` for every test row during the final fit. Currently `predict_single` returns `(prediction, std_estimate, k_used)` and logs `y_nb_std` to the optional jackknife log only; the change is to keep it as a vector on the main code path.

### (2) `diagnose_scale_signal(y_nb_std_oof, oof_residuals, top_mask)`

Returns `{pass: bool, reason: str, metrics: dict}`.

Metrics:
- `spearman_all = spearmanr(y_nb_std, |e|)` on all OOF rows
- `spearman_top = spearmanr(y_nb_std[top], |e|[top])` where `top = predicted_score ≥ 1400`
- `log_log_slope`, `log_log_r2` from `log(|e|+ε) ~ log(y_nb_std+ε)`
- `decile_lift = mean(|e|) in top y_nb_std decile / mean(|e|) in bottom decile`

Gate:

```
pass = (spearman_top ≥ 0.20) or (spearman_all ≥ 0.25 and decile_lift ≥ 1.3)
```

Thresholds are soft. All metrics are written to `calibration_diagnostics.csv` regardless of pass/fail, so we can inspect why the gate did what it did.

### (3) Local scale function

```
s(x) = max(y_nb_std(x), s_floor)
s_floor = percentile(y_nb_std_oof, 25)
```

No log transform, no per-feature regression. The floor prevents pathologically small sigmas in uniform neighborhoods.

### (4) OOF tail shape + quantile fit (non-circular)

Two steps, in order, no iteration needed:

```
# Step 1: fit tail shape on shape-normalized residuals
r_i = e_i / s(x_i)                                 for each OOF row
t_df, _loc, _scale = scipy.stats.t.fit(r_i, floc=0)
t_df = float(clip(t_df, 3.0, 200.0))

# Step 2: anchor the conformal quantile to empirical q95
q_hat = np.quantile(np.abs(r_i), 0.95) / scipy.stats.t.ppf(0.975, t_df)
```

`sigma_oof(x) = q_hat × s(x)` is then the t-distribution **scale parameter** at each point. (Not a half-width — see component (7) for interval formula.)

Rationale: fitting `t_df` directly on `r_i = e_i / s_i` removes the q_hat→t_df→q_hat circularity that the earlier draft would have needed a fixed-point for. Empirically anchoring `q_hat` to the 95th percentile of `|r|` preserves the usual conformal target (empirical 95% coverage on OOF in the limit `m = 1`). If `t_df` clips to 200 we are effectively Gaussian, which is fine.

### (5) Walk-forward level correction `m`

**Walk-forward refit policy (strict).** At step *t*, everything upstream of `m` must be fit on the prefix `[0..t-1]` only — no information from steps ≥ t leaks in. Specifically, the WF loop at step *t* refits:
1. ModelBankImputer (already done by `_walkforward_honest.py`)
2. PCA on pooled embeddings (already done)
3. PLS + scaler + adaptive KNN to get `mu_t` (already done)
4. Per-row `y_nb_std` over the prefix, producing `y_nb_std_oof_t` (new — requires a mini-LOO pass inside the prefix, or caching partial results)
5. `diagnose_scale_signal` → gate decision at step *t*
6. `s_floor_t`, `q_hat_t`, `t_df_t` per component (4)
7. `sigma_oof_t = q_hat_t × max(y_nb_std_for_row_t, s_floor_t)`

Then `z_t = (y_t − mu_t) / sigma_oof_t`, and each step carries its own `t_df_t`. `m` is fit across all WF steps by MLE using the per-step `t_df_t` in each likelihood term:

```
m = argmax_m  Σ_t [ log t_pdf(z_t / m; df=t_df_t) − log(m) ]
    subject to m ∈ [0.5, 3.0]
```

One-dimensional optimization. The final production run uses full-training calibration parameters (`q_hat`, `s_floor`, `t_df`, gate decision) combined with the WF-learned `m`: `sigma(x) = m × q_hat × s(x)`, evaluated with the full-training `t_df` (not per-step).

Step 4 is a strict nested LOO — at each WF step *t*, a LOO pass over the prefix `[0..t-1]` produces the prefix's OOF residuals and `y_nb_std` vector, which together feed the gate, `s_floor_t`, `q_hat_t`, `t_df_t` fits. This is expensive (O(t²) per step, O(n³) total) but the WF script is a separate honest-eval artifact that's run infrequently, and the strict policy is what makes `m` honest. An approximation that skipped the nested LOO would leave t_df_t and q_hat_t without data to fit on, so there is no cheap shortcut here.

Prior: the RMSE ratio 14.69/13.61 ≈ 1.08 suggests `m ≈ 1.08`, but `m` must be fit from z-scores / likelihood directly, not from the RMSE ratio, because once `s(x)` is heteroscedastic the RMSE-based estimate is not a consistent scale correction.

### (6) Threshold definition

**Batch mode (predict.py production run).** The leader threshold is:
```
max_leader = max(lmarena_Score) over rows where lmarena_Score is not NaN
             in the current prediction run.
```
i.e., the max over rows whose target is observed. Candidate rows with missing actuals (the models we're predicting for) do not contribute to the threshold. This is the correct batch-mode convention: at inference time, all scored models are "available," and the candidate's own predicted score is never part of its own threshold.

**Walk-forward evaluation (walkforward_calibration.py).** At step *t*, the threshold must be the stepwise leader:
```
max_leader_t = max(lmarena_Score) over training rows [0..t-1]
```
not the full-sample max. The full-sample max leaks future information into the evaluation (a model released later with a higher score makes the threshold retroactively harder). The fix is localized to the walk-forward script — predict.py's batch behavior is unaffected.

### (7) Output columns in `predictions_best_model.csv`

**`sigma_hat` is the t-distribution scale parameter, not a half-width.** Consumers who want intervals must scale by `t_crit`. Explicit formulas:

```
sigma_hat    = sigma(x) = m × q_hat × s(x)     [from (5)]
t_crit_95    = scipy.stats.t.ppf(0.975, t_df)  [two-sided 95%]
lower_bound  = mu − t_crit_95 × sigma_hat
upper_bound  = mu + t_crit_95 × sigma_hat
p_beats_leader = 1 − scipy.stats.t.cdf(
    (max_leader − mu) / sigma_hat,
    df=t_df,
)                                              [NaN on training rows]
top_by_margin_prob = 1 − scipy.stats.t.cdf(
    (max_leader + margin − mu) / sigma_hat,
    df=t_df,
)                                              [NaN on training rows]
```

`max_leader` is defined in component (6) (batch mode).

Net changes vs current output:
- `sigma_hat` is per-model, no longer constant.
- `sigma_hat` semantics change from "95% half-width" (current grouped-conformal emission where `std_new = sigma_hat / 1.96`) to "t-scale parameter." Downstream code that treats `sigma_hat` as a half-width needs to be updated — specifically the intervals in this file, which now require an explicit `t_crit` multiplication as shown above.
- `num_one_prob` removed.
- `p_beats_leader` added, NaN on training rows (where the event is meaningless).
- `top_by_margin_prob` unchanged in semantics but recomputed with the new per-model sigma and explicit t_df.

### (8) `calibration_diagnostics.csv`

Single-row summary. Columns:

- Gate: `gate_pass`, `gate_reason`, `spearman_all`, `spearman_top`, `log_log_slope`, `log_log_r2`, `decile_lift`
- Calibration parameters: `q_hat`, `t_df`, `s_floor`, `m`, `fallback_used`
- OOF readouts: `oof_coverage_95`, `oof_pit_ks_pvalue`
- Walk-forward readouts (if WF data present): `wf_pit_ks_pvalue`, `wf_coverage_50`, `wf_coverage_80`, `wf_coverage_95` (with binomial CI bounds), `wf_brier`, `wf_log_loss`, and the same slice of metrics restricted to `mu_t ≥ 1400` (`wf_top_*`)

### (9) Fallback path (gate fails)

If `diagnose_scale_signal` returns `pass=False`:

- `s(x) = 1` for all x (constant)
- `t_df` fit via `scipy.stats.t.fit(e_i, floc=0)` on raw OOF residuals; clip to `[3.0, 200.0]`; the `.fit` scale output is discarded (we anchor scale via `q_hat`)
- `q_hat = np.quantile(np.abs(e_i), 0.95) / scipy.stats.t.ppf(0.975, t_df)` — reduces to near-current constant-sigma behavior (without the group split that wasn't buying anything in the 2026-04-24 run)
- `m` still fit from walk-forward z-scores (using per-step `t_df_t` per component 5; in the WF fallback path `s=1` is applied at each step, so `r_t = e_t` and `t_df_t` is fit on raw prefix residuals the same way)
- Loud stderr log: `"local scale gate failed (reason=...); falling back to constant sigma × WF scalar"`
- `calibration_diagnostics.csv` records `fallback_used=true`

## Data flow + integration points

```
predict.py main flow:
  ... existing: impute → PLS → OOF loop → final fit → mu ...

  NEW:
  (i)   persist y_nb_std_oof (from the existing OOF loop)
  (ii)  diagnose_scale_signal() → gate result + metrics
  (iii) if gate pass:  sigma_oof(x) = q_hat × max(y_nb_std(x), floor)
        else:          sigma_oof(x) = q_hat × 1        (fallback)
  (iv)  if args.walkforward_calibration_path: load fitted_m from that file
        else:                                  m = 1.0
        (m is fit inside walkforward_calibration.py; predict.py only consumes it)
  (v)   sigma(x) = m × sigma_oof(x)
  (vi)  max_leader = max(lmarena_Score over observed-target rows, excluding candidates)
        p_beats_leader = 1 − t_cdf((max_leader − mu) / sigma, t_df); NaN on train rows
  (vii) write predictions_best_model.csv (per-model sigmas)
        + calibration_diagnostics.csv

walkforward_calibration.py (new, alongside arena_predictor/_walkforward_honest.py):
  Either extends _walkforward_honest.py or imports its core loop. At each
  step t, on the prefix [0..t-1] ONLY, refits imputer + PCA + PLS + predictor,
  then refits y_nb_std, gate, s_floor, q_hat, t_df per components (1)-(4).
  Captures (mu_t, sigma_oof_t, y_t, max_leader_t) where max_leader_t is the
  stepwise leader from component (6). Then fits scalar m by MLE per component (5).
  Emits:
    - wf_residuals.csv  (columns: step, mu_t, sigma_oof_t, y_t, max_leader_t,
                         t_df_t, fitted_m; consumed by predict.py via
                         --walkforward_calibration_path; predict.py reads
                         only `fitted_m`, the rest is for diagnostics)
    - walkforward_calibration_diagnostics.csv (PIT, coverage, Brier, log-loss)
```

### Two-run bootstrap

1. **Run 1** of predict.sh without `--walkforward_calibration_path`. `m = 1.0`, all other calibration stages fire using full-training calibration parameters (q_hat, s_floor, t_df). This is a standalone valid run; `p_beats_leader` is produced with an uncorrected sigma level.
2. Run `walkforward_calibration.py`. This does NOT reuse Run 1's calibration artifacts — it refits imputer, PCA, PLS, point predictor, y_nb_std, gate, s_floor, q_hat, t_df independently at each WF step on its own prefix (see component 5). The only output consumed downstream is the fitted `m`.
3. **Run 2** of predict.sh with `--walkforward_calibration_path <wf_residuals.csv>` to fold `m` into the shipped sigma. Run 2 is otherwise identical to Run 1 — same full-training q_hat, s_floor, t_df.

Keeping the main predictor deterministic and WF a separate honest-eval artifact is a feature — the WF fit is expensive and shouldn't block routine prediction runs. The strict separation (WF never reuses full-training calibration artifacts) preserves honesty of the `m` estimate.

## Error handling

Beyond the fallback path in (9):

- `y_nb_std = 0` for a row (all neighbors identical y): the `s_floor` handles it; no division by zero.
- `m` optimization hits a bound (≤0.5 or ≥3.0): log warning, clip and continue. A binding clip at 3.0 means WF is exposing a bigger problem than a multiplier can fix, and the diagnostics will show it — don't silently ship a tripled sigma without surfacing the issue.
- `t_df` MLE fails or returns non-finite: fall back to Gaussian (existing behavior preserved).

## Testing

### Unit tests (`tests/test_calibration.py`, new file)

- `diagnose_scale_signal` on synthetic data: constant-|e| input → gate fails; linearly-increasing-|e|-with-y_nb_std → gate passes.
- `sigma(x)` monotonicity: if `y_nb_std` increases, `sigma_hat` increases.
- Fallback collapse: when gate fails, `sigma_hat` in the output is constant across rows.
- `p_beats_leader` is NaN on rows corresponding to training models.

### OOF self-check (in-line assertion in `predict.py`)

After the final fit, compute PIT KS-p-value on OOF z-scores. If < 0.01, emit a stderr warning. Does not fail the run.

### Walk-forward honest readout (from `walkforward_calibration.py`)

- PIT histogram + KS-p-value
- Coverage at 50/80/95% with exact binomial 90% CIs
- Brier + log-loss for stepwise `y_t > max_leader_t`
- Same metrics on the top slice `mu_t ≥ 1400`

All emitted to `walkforward_calibration_diagnostics.csv`. Not a CI gate — n≈25 means individual runs can fail by chance. Acceptance is a human read of the diagnostics file.

### Regression smell test

Before/after comparison on `output_20260424_132648`. Verify that for Opus 4.7 Thinking (currently `num_one_prob=0.586`) the new `p_beats_leader` and `sigma_hat` are qualitatively defensible — no hard numeric target, but a nonsense result (e.g. `sigma_hat > 50` or `p_beats_leader > 0.95` for a model below max) means something is wrong.

## Numerical details

To keep implementation unambiguous:

- **Quantile rule.** `np.quantile(a, 0.95, method="linear")` (numpy ≥ 1.22 default). No finite-sample conformal adjustment — we're not claiming a split-conformal guarantee, so the `⌈(n+1)×0.95⌉/n` correction is not needed.
- **Epsilon for log-log diagnostics.** `ε = 1e-6` added to both sides: `log(|e|+ε) ~ log(y_nb_std+ε)`.
- **Spearman with small top slice.** `top = mu ≥ 1400`. If `|top| < 10`, `spearman_top` is emitted as `NaN` and the `(spearman_top ≥ 0.20)` sub-rule in the gate returns False. Diagnostics CSV records the slice size.
- **NaN/zero handling in Spearman.** Drop rows with NaN or zero in either input before computing; if < 10 rows remain, return NaN.
- **PIT formula.** For each OOF / WF point, `u_i = t_cdf((y_i − mu_i) / sigma_i, df=t_df_for_that_point)`. KS test: `scipy.stats.kstest(u, "uniform").pvalue`.
- **Log-loss clipping.** Probabilities clipped to `[1e-6, 1 − 1e-6]` before `log`.
- **Coverage at nominal level α.** `α-interval = mu ± t_ppf((1+α)/2, t_df) × sigma`. Empirical coverage = fraction of rows with `y ∈ [lo, hi]`. Reported with exact binomial 90% CI (Clopper-Pearson, `scipy.stats.binomtest(k, n).proportion_ci(0.9, method="exact")`).
- **Coverage CI column names in `calibration_diagnostics.csv`:** `wf_coverage_50`, `wf_coverage_50_ci_lo`, `wf_coverage_50_ci_hi`, and analogously for 80, 95. Same pattern prefixed `wf_top_` for the `mu ≥ 1400` slice.
- **t_df fit.** `scipy.stats.t.fit(r_i, floc=0)` → df, _, scale. Clip df to `[3.0, 200.0]`. The `scale` output from `.fit` is discarded; we anchor scale via `q_hat` instead.
- **m optimization.** `scipy.optimize.minimize_scalar(neg_log_lik, bounds=(0.5, 3.0), method="bounded")`, where `neg_log_lik(m) = -sum(scipy.stats.t.logpdf(z_t / m, df=t_df_t)) + len(z) × log(m)`. Each term uses **its own step's `t_df_t`**, not a shared value. The `len(z) × log(m)` term is the Jacobian of the z → z/m rescaling.
- **Top slice empty / degenerate.** If `n_top < 1` for any WF top-slice metric, emit `NaN` for that metric and record `wf_top_n = 0` in diagnostics. If the top slice has all-positive or all-negative `y_t > max_leader_t` outcomes, Brier is still defined (and near 0 or near 1 respectively) but log-loss is degenerate — emit `NaN` for log-loss in that case, not an infinity. Brier uses the clipped probability bounds from the log-loss clipping rule to keep it numerically consistent with log-loss.

## Open implementation questions

- `s_floor` at p25 is a first-cut choice. If it's binding for too many rows (e.g. > 30%), revisit with p10 or a signal-specific rule.
- The gate thresholds (0.20, 0.25, 1.3) are unprincipled defaults. They're designed to be soft enough that a working signal passes and tight enough that noise fails. First run of the diagnostic will tell us whether they need adjustment.
