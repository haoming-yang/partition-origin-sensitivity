from pathlib import Path

import pytest

from src.run import resolve_seeds
from src.utils.artifact_naming import artifact_path, metric_filename, seed_directory


def test_seed_is_configurable_and_cli_values_take_precedence():
    assert resolve_seeds({}, seed=7, seeds=None) == [7]
    assert resolve_seeds({}, seed=None, seeds="7,8") == [7, 8]
    assert resolve_seeds({"seed": 11}, seed=None, seeds=None) == [11]
    assert resolve_seeds({"seeds": [12, 13]}, seed=None, seeds=None) == [12, 13]
    with pytest.raises(ValueError, match="either --seed or --seeds"):
        resolve_seeds({}, seed=7, seeds="8")


def test_artifact_filename_contains_purpose_and_hyperparameters_but_no_seed():
    name = metric_filename(
        "latent_spectrum",
        origin_a=0,
        origin_b=6,
        patch_length=12,
        stride=12,
        context=512,
        horizon=96,
    )
    assert name == "latent_spectrum_origin0_o6_p12_s12_L512_H96.csv"
    assert "seed" not in name


def test_seed_directory_isolated_and_artifact_path_creates_it(tmp_path):
    root = seed_directory(tmp_path / "artifacts", 7)
    path = artifact_path(tmp_path / "artifacts", 7, "window_metrics.csv")
    assert root == tmp_path / "artifacts" / "seed7"
    assert path == root / "window_metrics.csv"
    assert path.parent.is_dir()
