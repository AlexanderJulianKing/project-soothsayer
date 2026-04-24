"""Walk-forward honest-eval + m-fit for predictor calibration.

Extends arena_predictor/_walkforward_honest.py by (a) running a nested LOO inside
each WF step to produce per-step OOF residuals and y_nb_std, (b) fitting per-step
gate + t_df + q_hat + s_floor on that prefix, (c) fitting a global scalar m across
steps, and (d) emitting diagnostics (PIT, coverage, Brier, log-loss) + wf_residuals.csv
for downstream consumption by predict.py via --walkforward_calibration_path.

See docs/superpowers/specs/2026-04-24-predictor-calibration-design.md for design.
"""
from __future__ import annotations

import itertools
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Make predict.py and calibration.py importable
sys.path.insert(0, str(Path(__file__).parent))

from _walkforward_honest import build_pooled_embeddings  # noqa: E402
from calibration import (  # noqa: E402
    compute_local_scale,
    compute_p_above,
    compute_sigma,
    diagnose_scale_signal,
    fit_tail_shape_and_qhat,
)
from predict import ID_COL, TARGET, ALT_TARGET, predict_adaptive_knn, run_imputation  # noqa: E402


OUT_DIR = Path(__file__).parent / "analysis_output" / "walkforward_calibration"


def main():
    """Entry point. Body is implemented in Task 15 (WF loop) and extended in Task 16 (m-fit + diagnostics)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] output dir: {OUT_DIR}", flush=True)
    print("[scaffold] main() body not yet implemented; see Task 15.", flush=True)


if __name__ == "__main__":
    main()
