import csv
from pathlib import Path

import numpy as np
import pytest

from tools.summarize_latent_artifacts import _seed_from_path, _spearman, summarize_latent_artifacts


def _write_spectrum(path, multiplier):
    rows = []
    for index, prediction in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
        rows.append({
            "window": index,
            "dc_l1": multiplier * prediction,
            "non_dc_low_l1": multiplier * prediction,
            "mid_band_l1": multiplier * (5.0 - prediction),
            "high_band_l1": multiplier * prediction,
            "prediction_mse": prediction,
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_layers(path, offset):
    rows = [
        {"window": 0, "prediction_mse": 1.0, "layer0_spectral_l1": 1.0 + offset, "layer1_spectral_l1": 3.0 + offset, "layer2_spectral_l1": 2.0 + offset},
        {"window": 1, "prediction_mse": 2.0, "layer0_spectral_l1": 2.0 + offset, "layer1_spectral_l1": 4.0 + offset, "layer2_spectral_l1": 3.0 + offset},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_seed_inference_reads_seed_from_artifact_parent_directory():
    assert _seed_from_path(Path("artifacts/latent_spectrum_etth1_seed42_o0_o6/window_metrics.csv")) == 42
    assert _seed_from_path(Path("artifacts/latent_spectrum_o0_o6/seed43/latent_spectrum_origin0_o6_p12.csv")) == 43


def test_summarize_latent_artifacts_preserves_frequency_and_layer_panel_order(tmp_path):
    spectrum_paths = []
    layer_paths = []
    for seed, offset in ((42, 0.0), (43, 0.5), (44, 1.0)):
        seed_root = tmp_path / f"seed{seed}"
        seed_root.mkdir()
        spectrum = seed_root / "latent_spectrum_origin0_o6_p12_s12_L512_H96.csv"
        layers = seed_root / "latent_layers_origin0_o6_p12_s12_L512_H96.csv"
        _write_spectrum(spectrum, seed)
        _write_layers(layers, offset)
        spectrum_paths.append(spectrum)
        layer_paths.append(layers)

    summary = summarize_latent_artifacts(spectrum_paths, layer_paths)

    assert summary["seeds"] == [42, 43, 44]
    assert list(summary["frequency"]) == ["DC", "Non-DC low", "Mid", "High"]
    assert list(summary["layers"]) == ["Post-position input", "First encoder block", "Final normalized output"]
    assert np.allclose(summary["frequency"]["DC"]["replicate_values"], [1.0, 1.0, 1.0])
    assert np.allclose(summary["frequency"]["Mid"]["replicate_values"], [-1.0, -1.0, -1.0])
    assert np.allclose(summary["layers"]["First encoder block"]["replicate_values"], [3.5, 4.0, 4.5])


def test_average_rank_spearman_handles_ties_without_artificial_ordering():
    assert _spearman(np.array([1.0, 1.0, 2.0]), np.array([10.0, 10.0, 20.0])) == pytest.approx(1.0)


def test_spearman_rejects_constant_series():
    with pytest.raises(ValueError, match="nonconstant"):
        _spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))
