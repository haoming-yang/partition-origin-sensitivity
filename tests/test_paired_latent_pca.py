import numpy as np
import pytest

from src.analysis.paired_latent_pca import (
    paired_pca_coordinates,
    select_tercile_windows,
    validate_paired_latents,
)


def test_select_tercile_windows_is_deterministic_and_balanced():
    windows = np.arange(180, dtype=np.int64)
    disagreement = np.linspace(0.001, 1.0, 180)

    first = select_tercile_windows(windows, disagreement, per_tercile=24, seed=42)
    second = select_tercile_windows(windows, disagreement, per_tercile=24, seed=42)

    assert np.array_equal(first.window_ids, second.window_ids)
    assert np.array_equal(first.strata, second.strata)
    assert first.window_ids.size == 72
    assert np.bincount(first.strata, minlength=3).tolist() == [24, 24, 24]


def test_paired_pca_coordinates_preserve_pair_rows_and_return_144_finite_points():
    origin_zero = np.arange(72 * 5, dtype=np.float64).reshape(72, 5)
    origin_six = origin_zero + np.linspace(0.1, 0.8, 72)[:, None]
    window_ids = np.arange(100, 172, dtype=np.int64)

    projected = paired_pca_coordinates(window_ids, origin_zero, origin_six)

    assert projected.coordinates.shape == (144, 3)
    assert np.all(np.isfinite(projected.coordinates))
    assert np.array_equal(projected.window_ids[:72], window_ids)
    assert np.array_equal(projected.window_ids[72:], window_ids)
    assert projected.origins.tolist() == [0] * 72 + [6] * 72


def test_validate_paired_latents_rejects_partial_token_counts_and_misaligned_windows():
    first = np.ones((2, 42, 3), dtype=np.float64)
    second = np.ones((2, 42, 3), dtype=np.float64)

    validate_paired_latents(np.array([10, 11]), first, second, complete_token_count=42)

    with pytest.raises(ValueError, match="complete token count"):
        validate_paired_latents(np.array([10, 11]), first[:, :-1], second[:, :-1], complete_token_count=42)
