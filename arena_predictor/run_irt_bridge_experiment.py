"""IRT ALT + minimal style bridge experiment.

Two-stage pipeline:
  Stage 1: IRT theta(k=4, poly2) → BayesianRidge → predict lmarena_Score
  Stage 2: lmarena_pred + style_predicted_delta + interaction → predict lmsys_Score

Uses honest LOO/KFold OOF predictions at both stages to avoid leakage.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from irt_features import fit_irt_2pl, compute_irt_features

CSV_PATH = "../benchmark_combiner/benchmarks/clean_combined_all_benches.csv"
ALT_TARGET = "lmarena_Score"
TARGET = "lmsys_Score"


def run_experiment(k=4, poly_degree=2, reg_lambda=0.0001, n_splits=5, n_repeats=10, seed=42):
    df = pd.read_csv(CSV_PATH)

    # Identify benchmark feature columns (exclude targets, IDs, style)
    exclude = {"model_name", TARGET, ALT_TARGET, "openbench_Reasoning"}
    feature_cols = [c for c in df.columns if c not in exclude
                    and df[c].dtype in (np.float64, np.int64, float, int)]

    print(f"Data: {len(df)} models, {len(feature_cols)} benchmark features")

    # Models with known targets
    has_alt = df[ALT_TARGET].notna()
    has_target = df[TARGET].notna()
    has_both = has_alt & has_target

    print(f"Models with lmarena: {has_alt.sum()}")
    print(f"Models with lmsys:   {has_target.sum()}")
    print(f"Models with both:    {has_both.sum()}")

    # ============================================================
    # Stage 1: IRT → lmarena
    # ============================================================
    print(f"\n=== Stage 1: IRT(k={k}, poly={poly_degree}) → lmarena ===")

    # Compute IRT features on ALL models
    irt_df = compute_irt_features(df, feature_cols, k=k, poly_degree=poly_degree,
                                  reg_lambda=reg_lambda, seed=seed)
    irt_cols = [c for c in irt_df.columns if c != "model_name"]
    print(f"IRT features: {len(irt_cols)} columns")

    # OOF prediction of lmarena from IRT features
    X_alt = irt_df.loc[has_alt, irt_cols].values
    y_alt = df.loc[has_alt, ALT_TARGET].values

    alt_oof = np.full(len(y_alt), np.nan)
    rng = np.random.RandomState(seed)

    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rng.randint(0, 2**31))
        for tr, va in kf.split(X_alt):
            pipe = make_pipeline(StandardScaler(), BayesianRidge())
            pipe.fit(X_alt[tr], y_alt[tr])
            pred = pipe.predict(X_alt[va])
            if np.isnan(alt_oof[va]).all():
                alt_oof[va] = pred
            else:
                # Running average across repeats
                count = rep  # how many previous repeats contributed
                alt_oof[va] = (alt_oof[va] * count + pred) / (count + 1)

    alt_rmse = np.sqrt(np.nanmean((alt_oof - y_alt) ** 2))
    alt_r = np.corrcoef(alt_oof[~np.isnan(alt_oof)], y_alt[~np.isnan(alt_oof)])[0, 1]
    print(f"ALT OOF RMSE: {alt_rmse:.2f}  R={alt_r:.4f}")

    # Predict lmarena for ALL models (fit on all known)
    pipe_full = make_pipeline(StandardScaler(), BayesianRidge())
    pipe_full.fit(X_alt, y_alt)
    lmarena_pred_all = pipe_full.predict(irt_df[irt_cols].values)

    # Use real lmarena where available, predicted otherwise
    lmarena_filled = df[ALT_TARGET].values.copy()
    lmarena_filled[~has_alt] = lmarena_pred_all[~has_alt]

    # ============================================================
    # Stage 2: lmarena + style_delta → lmsys
    # ============================================================
    print(f"\n=== Stage 2: lmarena + style_delta + interaction → lmsys ===")

    style_delta = df["style_predicted_delta"].values

    # Build stage 2 features
    X_stage2_all = np.column_stack([
        lmarena_filled,
        style_delta,
        lmarena_filled * style_delta,
    ])
    feature_names_s2 = ["lmarena", "style_delta", "lmarena*style_delta"]

    # OOF prediction of lmsys
    train_mask = has_target.values
    X_train_s2 = X_stage2_all[train_mask]
    y_train_s2 = df.loc[has_target, TARGET].values

    # But for OOF, we need to use OOF lmarena predictions (not real values)
    # Otherwise training models leak their real lmarena into the stage 2 prediction
    lmarena_oof_filled = lmarena_pred_all.copy()  # use predicted for all
    # Actually for honest OOF: need fold-internal ALT predictions too
    # Let's do nested OOF properly

    target_oof = np.full(len(y_train_s2), np.nan)
    target_idx = np.where(train_mask)[0]

    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rng.randint(0, 2**31))
        for tr, va in kf.split(X_train_s2):
            # Stage 1: fit ALT on training fold only
            alt_train_mask_fold = has_alt.values[target_idx[tr]]
            X_irt_tr = irt_df.iloc[target_idx[tr]].loc[alt_train_mask_fold, irt_cols].values
            y_alt_tr = df.iloc[target_idx[tr]].loc[alt_train_mask_fold, ALT_TARGET].values

            if len(X_irt_tr) < 10:
                continue

            pipe_alt = make_pipeline(StandardScaler(), BayesianRidge())
            pipe_alt.fit(X_irt_tr, y_alt_tr)

            # Predict lmarena for val fold
            lmarena_va = pipe_alt.predict(irt_df.iloc[target_idx[va]][irt_cols].values)

            # HONEST: use fold-predicted lmarena for val (simulates new model)
            lmarena_va_filled = lmarena_va  # always predicted

            # Training fold: use real lmarena where available
            lmarena_tr = pipe_alt.predict(irt_df.iloc[target_idx[tr]][irt_cols].values)
            lmarena_tr_filled = df.iloc[target_idx[tr]][ALT_TARGET].values.copy()
            missing_alt_tr = np.isnan(lmarena_tr_filled)
            lmarena_tr_filled[missing_alt_tr] = lmarena_tr[missing_alt_tr]

            # Stage 2 features
            sd_tr = style_delta[target_idx[tr]]
            sd_va = style_delta[target_idx[va]]

            X_s2_tr = np.column_stack([lmarena_tr_filled, sd_tr, lmarena_tr_filled * sd_tr])
            X_s2_va = np.column_stack([lmarena_va_filled, sd_va, lmarena_va_filled * sd_va])
            y_s2_tr = y_train_s2[tr]

            pipe_s2 = make_pipeline(StandardScaler(), BayesianRidge())
            pipe_s2.fit(X_s2_tr, y_s2_tr)
            pred_va = pipe_s2.predict(X_s2_va)

            if np.isnan(target_oof[va]).all():
                target_oof[va] = pred_va
            else:
                count = rep
                target_oof[va] = (target_oof[va] * count + pred_va) / (count + 1)

    target_rmse = np.sqrt(np.nanmean((target_oof - y_train_s2) ** 2))
    target_r = np.corrcoef(target_oof[~np.isnan(target_oof)],
                            y_train_s2[~np.isnan(target_oof)])[0, 1]

    # Top-50 RMSE
    top50_idx = np.argsort(y_train_s2)[-50:]
    top50_rmse = np.sqrt(np.nanmean((target_oof[top50_idx] - y_train_s2[top50_idx]) ** 2))

    print(f"Target OOF RMSE: {target_rmse:.2f}  R={target_r:.4f}")
    print(f"Top-50 RMSE:     {top50_rmse:.2f}")

    # Show worst predictions
    oof_df = pd.DataFrame({
        "model_name": df.loc[has_target, "model_name"].values,
        "actual": y_train_s2,
        "predicted": target_oof,
        "residual": target_oof - y_train_s2,
    }).dropna()
    oof_df["abs_err"] = oof_df["residual"].abs()

    print(f"\nWorst 10 predictions:")
    for _, r in oof_df.nlargest(10, "abs_err").iterrows():
        print(f"  {r['model_name']:45s} actual={r['actual']:.0f} pred={r['predicted']:.0f} err={r['residual']:+.0f}")

    # ============================================================
    # Stage 2b: with eqbench as additional feature
    # ============================================================
    print(f"\n=== Stage 2b: + eqbench_eq_elo ===")

    eq_elo = df["eqbench_eq_elo"].values if "eqbench_eq_elo" in df.columns else None
    if eq_elo is not None:
        target_oof_eq = np.full(len(y_train_s2), np.nan)

        for rep in range(n_repeats):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=rng.randint(0, 2**31))
            for tr, va in kf.split(X_train_s2):
                alt_train_mask_fold = has_alt.values[target_idx[tr]]
                X_irt_tr = irt_df.iloc[target_idx[tr]].loc[alt_train_mask_fold, irt_cols].values
                y_alt_tr = df.iloc[target_idx[tr]].loc[alt_train_mask_fold, ALT_TARGET].values

                if len(X_irt_tr) < 10:
                    continue

                pipe_alt = make_pipeline(StandardScaler(), BayesianRidge())
                pipe_alt.fit(X_irt_tr, y_alt_tr)

                lmarena_va = pipe_alt.predict(irt_df.iloc[target_idx[va]][irt_cols].values)
                lmarena_va_filled = df.iloc[target_idx[va]][ALT_TARGET].values.copy()
                missing_alt_va = np.isnan(lmarena_va_filled)
                lmarena_va_filled[missing_alt_va] = lmarena_va[missing_alt_va]

                lmarena_tr = pipe_alt.predict(irt_df.iloc[target_idx[tr]][irt_cols].values)
                lmarena_tr_filled = df.iloc[target_idx[tr]][ALT_TARGET].values.copy()
                missing_alt_tr = np.isnan(lmarena_tr_filled)
                lmarena_tr_filled[missing_alt_tr] = lmarena_tr[missing_alt_tr]

                sd_tr = style_delta[target_idx[tr]]
                sd_va = style_delta[target_idx[va]]
                eq_tr = eq_elo[target_idx[tr]]
                eq_va = eq_elo[target_idx[va]]

                # Fill NaN eqbench with column median from training fold
                eq_median = np.nanmedian(eq_tr)
                eq_tr_filled = np.where(np.isnan(eq_tr), eq_median, eq_tr)
                eq_va_filled = np.where(np.isnan(eq_va), eq_median, eq_va)

                X_s2_tr = np.column_stack([lmarena_tr_filled, sd_tr, lmarena_tr_filled * sd_tr,
                                            eq_tr_filled, eq_tr_filled * sd_tr])
                X_s2_va = np.column_stack([lmarena_va_filled, sd_va, lmarena_va_filled * sd_va,
                                            eq_va_filled, eq_va_filled * sd_va])

                pipe_s2 = make_pipeline(StandardScaler(), BayesianRidge())
                pipe_s2.fit(X_s2_tr, y_s2_tr)
                pred_va = pipe_s2.predict(X_s2_va)

                if np.isnan(target_oof_eq[va]).all():
                    target_oof_eq[va] = pred_va
                else:
                    count = rep
                    target_oof_eq[va] = (target_oof_eq[va] * count + pred_va) / (count + 1)

        eq_rmse = np.sqrt(np.nanmean((target_oof_eq - y_train_s2) ** 2))
        eq_top50 = np.sqrt(np.nanmean((target_oof_eq[top50_idx] - y_train_s2[top50_idx]) ** 2))
        print(f"Target OOF RMSE (with EQ): {eq_rmse:.2f}")
        print(f"Top-50 RMSE (with EQ):     {eq_top50:.2f}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Stage 1 (IRT → lmarena):              ALT RMSE = {alt_rmse:.2f}")
    print(f"Stage 2 (minimal bridge):              RMSE = {target_rmse:.2f}  top50 = {top50_rmse:.2f}")
    if eq_elo is not None:
        print(f"Stage 2b (+ eqbench):                  RMSE = {eq_rmse:.2f}  top50 = {eq_top50:.2f}")
    print(f"Current best (full pipeline restricted): RMSE = 19.16  top50 = 15.25")
    print(f"Old baseline (unrestricted):             RMSE = 20.11  top50 = 15.10")


if __name__ == "__main__":
    run_experiment()
