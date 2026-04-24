# Predictor Calibration Redesign

**Date:** 2026-04-24
**Status:** Design approved, pending implementation plan
**Owner:** Alexander King

## Problem

`arena_predictor/predict.py` currently emits a prediction interval and two event probabilities (`num_one_prob`, `top_by_margin_prob`) per model in `predictions_best_model.csv`. The probabilities are not calibrated in a way that supports the end-user question — *"what is the probability this new model takes the #1 spot on lmarena?"*

Concrete defects in the current output (run `output_20260424_132648`):

1. **`sigma_hat` is constant across all 166 rows (= 21.81).** It comes from `compute_grouped_conformal_intervals`, which assigns each model to one of four coarse groups (top-tier × high-missing) and uses the OOF q95 residual for that group. In this run, every model landed in the same predicted-top group so sigma is effectively a scalar. A model in a dense, well-observed region of feature space gets the same uncertainty as an extrapolative frontier model.
2. **`num_one_prob` is a marginal exceedance probability, not a "takes #1" probability.** It computes `P(score > max(y_train))` via t-CDF with the constant sigma. The threshold treats the current leader's score as known without error, which is acceptable here (ELO day-over-day swings are ≤5 pts), but the marginal-exceedance framing means multiple candidates can simultaneously have `num_one_prob > 0.5` in a way that isn't coherent with "only one model can be #1 at a time."
3. **Novel-frontier regime is systematically overconfident.** OOF RMSE is 13.61; walk-forward RMSE on the newest 20% is 14.69. Probabilities calibrated on OOF are ~8% too narrow in the regime that actually matters for new-model prediction.
4. **`num_one_prob` threshold in walk-forward evaluation is wrong.** The current code uses the full-sample `max(y_all[train_idx])`. For honest walk-forward scoring of the "takes #1" event at step *t*, the threshold must be the leader *available at step t* — the full-sample max leaks future information into the evaluation.

## Goal

Emit a per-model `p_takes_one` in `predictions_best_model.csv` that is calibrated against the "new model beats current leader" event, and a per-model `sigma_hat` that varies with local extrapolation risk and matches realized error magnitudes in the novel-frontier regime.

Success criterion: on a walk-forward holdout (newest 20% of models, re-fit at every step), the predictive distribution passes:

- PIT histogram approximately uniform (KS p-value not < 0.01)
- Coverage at 50/80/95% nominal levels contained within exact binomial 90% CIs
- Brier and log-loss for stepwise `y_t > max_train_t` event strictly better than the current constant-sigma baseline, and the same on the top slice (`mu_t ≥ 1400`)

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
(B) Normalized conformal shape (OOF, n=127)
    sigma_oof(x) = q_hat × max(y_nb_std(x), floor)
    fit q_hat so empirical z-score q95 = t_crit
    keep existing t_df fitting for tail shape
                              │
                              ▼
(C) Walk-forward level correction (n≈25)
    fit scalar m by MLE on WF z-scores at fitted t_df
    ship sigma(x) = m × sigma_oof(x)
                              │
                              ▼
(D) Output: per-model sigma, per-model intervals, and
    p_takes_one using stepwise-corrected threshold
```

The `y_nb_std` signal drives local scale variation. It is already computed inside `predict_single` (around `predict.py:709`, currently only written to the optional jackknife log). Using it here keeps the scale model aligned with the point model — same neighborhood, same k chosen by the sublinear power cutoff.

### Why Approach 2 (normalized conformal with KNN-local scale)

Three candidate approaches were considered:

1. **Scale regression.** Fit `|e| ~ f(missingness, y_nb_std, distance-to-1NN, tier)` with regularization. Rejected for now: at n=127 the multi-feature regression tends to learn interpolation quirks more than frontier uncertainty, and it has more knobs than its evidence base supports.
2. **Normalized split-conformal with KNN-local scale** — this design. Single local scale signal, one OOF conformal quantile, one walk-forward scalar. Reuses infrastructure already in the codebase.
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

### (4) OOF quantile fit `q_hat`

```
z_i = |e_i| / s(x_i)                              for each OOF row
q_hat = quantile(z, 0.95) / t_crit(t_df)
```

`t_df` is fit by MLE on the normalized z-scores `e_i / s_i / q_hat` (same approach as current code, which clipped to 200 → effectively Gaussian). If the clip binds again post-redesign, that is expected and fine.

### (5) Walk-forward level correction `m`

For each walk-forward step *t*, recover `(mu_t, sigma_oof_t, y_t)` using only training data available at step *t* (the WF loop already re-fits imputation + PCA + PLS). Compute z-scores `z_t = (y_t − mu_t) / sigma_oof_t`.

Fit `m` by MLE under the t-distribution at the OOF-fit `t_df`:

```
m = argmax_m  Σ_t [ log t_pdf(z_t / m; df=t_df) − log(m) ]
    subject to m ∈ [0.5, 3.0]
```

One-dimensional optimization. Ship `sigma(x) = m × q_hat × s(x)`.

Prior: the RMSE ratio 14.69/13.61 ≈ 1.08 suggests `m ≈ 1.08`, but `m` must be fit from z-scores / likelihood directly, not from the RMSE ratio, because once `s(x)` is heteroscedastic the RMSE-based estimate is not a consistent scale correction.

### (6) Stepwise threshold correction (walk-forward eval only)

At walk-forward step *t*, the "takes #1" event is `y_t > max(y_train_available_at_step_t)`, not `y_t > max(y_all[train_idx])`. The main predictor (batch mode) keeps the full-sample max, which is correct there. The fix is localized to the walk-forward evaluation script — when scoring `p_takes_one` for Brier/log-loss on WF, the threshold must be the stepwise leader.

### (7) Output columns in `predictions_best_model.csv`

- `sigma_hat` — per-model, `= sigma(x)` from (5).
- `lower_bound`, `upper_bound` — per-model 95% intervals from per-model sigma.
- `p_takes_one` — `1 − t_cdf((max_train − mu) / sigma, t_df)`, NaN on rows that are training models. Replaces `num_one_prob`.
- `top_by_margin_prob` — same formula with threshold `max_train + margin`, NaN on training rows (unchanged).

`num_one_prob` is removed. `p_takes_one` is the clean replacement: explicitly NaN for training rows (where the event is meaningless) and semantically named.

### (8) `calibration_diagnostics.csv`

Single-row summary. Columns:

- Gate: `gate_pass`, `gate_reason`, `spearman_all`, `spearman_top`, `log_log_slope`, `log_log_r2`, `decile_lift`
- Calibration parameters: `q_hat`, `t_df`, `s_floor`, `m`, `fallback_used`
- OOF readouts: `oof_coverage_95`, `oof_pit_ks_pvalue`
- Walk-forward readouts (if WF data present): `wf_pit_ks_pvalue`, `wf_coverage_50`, `wf_coverage_80`, `wf_coverage_95` (with binomial CI bounds), `wf_brier`, `wf_log_loss`, and the same slice of metrics restricted to `mu_t ≥ 1400` (`wf_top_*`)

### (9) Fallback path (gate fails)

If `diagnose_scale_signal` returns `pass=False`:

- `s(x) = 1` for all x (constant)
- `q_hat = quantile(|e|_oof, 0.95) / t_crit(t_df)` — reduces to near-current constant-sigma behavior (without the group split that wasn't buying anything in the 2026-04-24 run)
- `m` still fit from walk-forward z-scores
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
  (iv)  if args.walkforward_calibration_path: load WF residuals, fit m
        else:                                  m = 1.0
  (v)   sigma(x) = m × sigma_oof(x)
  (vi)  p_takes_one = 1 − t_cdf((max_train − mu) / sigma, t_df); NaN on train rows
  (vii) write predictions_best_model.csv (per-model sigmas)
        + calibration_diagnostics.csv

walkforward_calibration.py (new, alongside arena_predictor/_walkforward_honest.py):
  Either extends _walkforward_honest.py or imports its core loop. At each
  step t, captures (mu_t, sigma_oof_t, y_t, max_train_t).
  Emits:
    - wf_residuals.csv  (consumed by predict.py via --walkforward_calibration_path)
    - walkforward_calibration_diagnostics.csv (PIT, coverage, Brier, log-loss)
```

### Two-run bootstrap

1. **Run 1** of predict.sh without `--walkforward_calibration_path`. `m = 1.0`, all other calibration stages fire. WF inputs produced by the run (OOF residuals, per-model `y_nb_std`, etc.) are already persisted.
2. Run `walkforward_calibration.py` to get WF residuals and fit `m`.
3. **Run 2** of predict.sh with `--walkforward_calibration_path ./wf_residuals.csv` to ship `m` in the final sigma.

Keeping the main predictor deterministic and WF a separate honest-eval artifact is a feature — the WF fit is expensive (re-fits imputation + PCA + PLS at every step) and shouldn't block routine prediction runs.

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
- `p_takes_one` is NaN on rows corresponding to training models.

### OOF self-check (in-line assertion in `predict.py`)

After the final fit, compute PIT KS-p-value on OOF z-scores. If < 0.01, emit a stderr warning. Does not fail the run.

### Walk-forward honest readout (from `walkforward_calibration.py`)

- PIT histogram + KS-p-value
- Coverage at 50/80/95% with exact binomial 90% CIs
- Brier + log-loss for stepwise `y_t > max_train_t`
- Same metrics on the top slice `mu_t ≥ 1400`

All emitted to `walkforward_calibration_diagnostics.csv`. Not a CI gate — n≈25 means individual runs can fail by chance. Acceptance is a human read of the diagnostics file.

### Regression smell test

Before/after comparison on `output_20260424_132648`. Verify that for Opus 4.7 Thinking (currently `num_one_prob=0.586`) the new `p_takes_one` and `sigma_hat` are qualitatively defensible — no hard numeric target, but a nonsense result (e.g. `sigma_hat > 50` or `p_takes_one > 0.95` for a model below max) means something is wrong.

## Open implementation questions

- `s_floor` at p25 is a first-cut choice. If it's binding for too many rows (e.g. > 30%), revisit with p10 or a signal-specific rule.
- The gate thresholds (0.20, 0.25, 1.3) are unprincipled defaults. They're designed to be soft enough that a working signal passes and tight enough that noise fails. First run of the diagnostic will tell us whether they need adjustment.
- Whether to fit `t_df` on the raw OOF z-scores (current code) or on `e / (s × q_hat)` (normalized z-scores post-scale-fit). The design specifies the latter, but it's circular — the t_df fit depends on q_hat, which depends on t_crit(t_df). Either fixed-point iterate (once is probably enough), or fit `t_df` on unnormalized z-scores and use that for the t_crit. Pick one in the plan; both are fine.
