"""Stable, model-agnostic summaries for phase-level forecasting errors."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def summarize_phase_mse(phase_mse: Sequence[float]) -> dict[str, float | list[float]]:
    """Compute the preregistered spread statistics from phase MSE values."""
    mse = np.asarray(phase_mse, dtype=np.float64)
    if mse.ndim != 1 or mse.size == 0 or not np.isfinite(mse).all() or (mse <= 0).any():
        raise ValueError("phase_mse must be a non-empty vector of positive finite values")
    return {
        "phase_mse": mse.tolist(),
        "G_phase_pct": float((mse.max() - mse.min()) / mse.min() * 100),
        "CV_phase": float(mse.std() / mse.mean()),
        "best_mse": float(mse.min()),
        "worst_mse": float(mse.max()),
    }
