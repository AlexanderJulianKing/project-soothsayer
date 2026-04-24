"""Tests for arena_predictor/calibration.py."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'arena_predictor'))

import numpy as np
import pytest

from calibration import (
    GateResult,
    ShapeFit,
    compute_local_scale,
    compute_p_above,
    compute_p_beats_leader,
    compute_sigma,
    diagnose_scale_signal,
    fit_tail_shape_and_qhat,
)


RNG = np.random.default_rng(42)
