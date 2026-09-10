"""Token-axis spectral summaries for aligned-length latent sequences."""

from __future__ import annotations

import numpy as np


def _mean_band_distance(distance: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Average a per-frequency distance over a non-empty frequency interval."""
    return distance[:, start:stop, :].mean(axis=(1, 2))


def _rank(values: np.ndarray) -> np.ndarray:
    """Return average ranks, including ties created by bootstrap resampling."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        raise ValueError("Spearman correlation requires nonconstant paired values")
    return float(np.corrcoef(first, second)[0, 1])


def permutation_spearman_test(
    spectral_difference: np.ndarray,
    forecast_difference: np.ndarray,
    *,
    permutations: int = 1000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Test a paired spectral--forecast association against random window matching.

    The test keeps forecast discrepancies fixed and permutes spectral discrepancies
    across windows.  It returns a finite-resample-corrected two-sided Monte Carlo p value
    using the ``(exceedances + 1) / (permutations + 1)`` correction.
    """
    spectral = np.asarray(spectral_difference, dtype=np.float64).reshape(-1)
    forecast = np.asarray(forecast_difference, dtype=np.float64).reshape(-1)
    if spectral.shape != forecast.shape or spectral.size < 2:
        raise ValueError("paired spectral and forecast arrays must share length at least two")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    if not np.isfinite(spectral).all() or not np.isfinite(forecast).all():
        raise ValueError("paired values must be finite")

    ranked_spectral = _rank(spectral)
    ranked_forecast = _rank(forecast)
    observed = _pearson(ranked_spectral, ranked_forecast)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        null[index] = _pearson(ranked_spectral[rng.permutation(spectral.size)], ranked_forecast)
    exceedances = int(np.count_nonzero(np.abs(null) >= abs(observed)))
    return {
        "observed_rho": observed,
        "permutation_mean": float(null.mean()),
        "permutation_sd": float(null.std(ddof=1)),
        "exceedances": exceedances,
        "permutations": permutations,
        "p_two_sided": float((exceedances + 1) / (permutations + 1)),
    }


def summarize_latent_pair(first: np.ndarray, second: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-sample magnitude-spectrum distances for two latent sequences.

    Inputs have shape ``[samples, tokens, hidden_dim]`` and must already exclude
    partial or padded tokens. FFT is applied only along the token axis.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 3 or second.ndim != 3:
        raise ValueError("latent arrays must have shape [samples, tokens, hidden_dim]")
    if first.shape != second.shape:
        raise ValueError("latent arrays must have identical shapes")
    if first.shape[1] < 2:
        raise ValueError("at least two full tokens are required for a token-axis FFT")

    spectra_first = np.abs(np.fft.rfft(first, axis=1, norm="ortho"))
    spectra_second = np.abs(np.fft.rfft(second, axis=1, norm="ortho"))
    distance = np.abs(spectra_first - spectra_second)
    frequency_bins = distance.shape[1]
    low_stop = max(1, frequency_bins // 3)
    mid_stop = max(low_stop + 1, (2 * frequency_bins) // 3)
    non_dc_low = (
        _mean_band_distance(distance, 1, low_stop)
        if low_stop > 1
        else np.zeros(distance.shape[0], dtype=np.float64)
    )

    return {
        "spectral_l1": distance.mean(axis=(1, 2)),
        "dc_l1": distance[:, 0, :].mean(axis=1),
        "non_dc_low_l1": non_dc_low,
        "low_band_l1": _mean_band_distance(distance, 0, low_stop),
        "mid_band_l1": _mean_band_distance(distance, low_stop, mid_stop),
        "high_band_l1": _mean_band_distance(distance, mid_stop, frequency_bins),
    }
