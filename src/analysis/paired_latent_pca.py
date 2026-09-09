"""Deterministic sampling and paired PCA for qualitative latent diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TercileSelection:
    window_ids: np.ndarray
    strata: np.ndarray


@dataclass(frozen=True)
class PairedPCA:
    window_ids: np.ndarray
    origins: np.ndarray
    coordinates: np.ndarray


def select_tercile_windows(
    window_ids: np.ndarray,
    forecast_mse: np.ndarray,
    *,
    per_tercile: int = 24,
    seed: int = 42,
) -> TercileSelection:
    """Select an equal deterministic sample from each ranked disagreement tercile."""
    windows = np.asarray(window_ids, dtype=np.int64).reshape(-1)
    disagreement = np.asarray(forecast_mse, dtype=np.float64).reshape(-1)
    if windows.shape != disagreement.shape or windows.size < 3:
        raise ValueError("window IDs and forecast MSE values must have matching length at least three")
    if np.unique(windows).size != windows.size:
        raise ValueError("window IDs must be unique")
    if not np.isfinite(disagreement).all():
        raise ValueError("forecast MSE values must be finite")
    ranked = np.argsort(disagreement, kind="mergesort")
    terciles = np.array_split(ranked, 3)
    if min(part.size for part in terciles) < per_tercile:
        raise ValueError("each disagreement tercile must contain the requested sample size")
    rng = np.random.default_rng(seed)
    selected_indices = []
    selected_strata = []
    for stratum, indices in enumerate(terciles):
        chosen = rng.choice(indices, size=per_tercile, replace=False)
        selected_indices.append(np.sort(chosen))
        selected_strata.append(np.full(per_tercile, stratum, dtype=np.int64))
    final_indices = np.concatenate(selected_indices)
    return TercileSelection(windows[final_indices], np.concatenate(selected_strata))


def validate_paired_latents(
    window_ids: np.ndarray,
    origin_zero: np.ndarray,
    origin_six: np.ndarray,
    *,
    complete_token_count: int,
) -> None:
    """Require equal, finite complete-token tensors for paired window rows."""
    windows = np.asarray(window_ids, dtype=np.int64).reshape(-1)
    first = np.asarray(origin_zero, dtype=np.float64)
    second = np.asarray(origin_six, dtype=np.float64)
    if first.ndim != 3 or second.ndim != 3 or first.shape != second.shape:
        raise ValueError("paired latents must have matching [windows, tokens, hidden] shape")
    if first.shape[0] != windows.size or np.unique(windows).size != windows.size:
        raise ValueError("paired latent rows must align to unique window IDs")
    if first.shape[1] != complete_token_count:
        raise ValueError("complete token count does not match the retained latent tensors")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("paired latent tensors must be finite")


def paired_pca_coordinates(
    window_ids: np.ndarray,
    origin_zero: np.ndarray,
    origin_six: np.ndarray,
) -> PairedPCA:
    """Fit a 3D PCA to concatenated paired pooled latent vectors."""
    windows = np.asarray(window_ids, dtype=np.int64).reshape(-1)
    first = np.asarray(origin_zero, dtype=np.float64)
    second = np.asarray(origin_six, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
        raise ValueError("pooled latents must have matching [windows, hidden] shape")
    if first.shape[0] != windows.size or first.shape[1] < 3:
        raise ValueError("pooled latents must align to window IDs and have at least three hidden dimensions")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("pooled latents must be finite")
    combined = np.concatenate((first, second), axis=0)
    centered = combined - combined.mean(axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ right_vectors[:3].T
    return PairedPCA(
        window_ids=np.concatenate((windows, windows)),
        origins=np.concatenate((np.zeros(windows.size, dtype=np.int64), np.full(windows.size, 6, dtype=np.int64))),
        coordinates=coordinates,
    )
