import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_dry_run_defaults_without_data_or_gpu(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "src.run", "--config", "configs/core/canonical.yaml",
         "--dry-run", "--output-root", str(tmp_path / "unused")],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert len(report["jobs"]) == 15
    assert {job["seed"] for job in report["jobs"]} == {42, 43, 44}
    assert report["training_started"] is False
    assert not (tmp_path / "unused").exists()


def test_inference_policy_selects_validation_not_test_and_averages_predictions():
    from src.training import runner

    validation = {0: np.array([[[1.0]]]), 1: np.array([[[-1.0]]])}
    test = {0: np.array([[[4.0]]]), 1: np.array([[[-2.0]]])}
    metrics = runner.inference_policy_metrics(validation, np.zeros((1, 1, 1)),
                                              test, np.zeros((1, 1, 1)))
    assert metrics["selected_origin"] == 0
    assert metrics["E_star"] == 16.0
    assert metrics["E_mean"] == 10.0
    assert metrics["E_ens"] == 1.0
    assert metrics["inference_forward_count_ensemble"] == 2


def test_run_provenance_is_self_describing(tmp_path):
    from src.training.runner import save_run_artifacts, metric_rows

    predictions = {0: np.ones((1, 1, 1)), 1: np.ones((1, 1, 1)) * 2}
    targets = np.zeros((1, 1, 1))
    rows, _ = metric_rows(predictions, targets)
    save_run_artifacts(tmp_path, {"dataset": "synthetic", "seed": 17, "origins": [0, 1]},
                       [], [], rows, predictions, targets, {}, [])
    provenance = json.loads((tmp_path / "PROVENANCE.json").read_text())
    assert provenance["config"]["seed"] == 17
    assert provenance["config"]["dataset"] == "synthetic"
    assert provenance["config"]["origins"] == [0, 1]
    assert provenance["generated_at"]
    assert provenance["device"] == "cpu"
    assert "git_dirty" in provenance
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["origin_prediction_variance"] == 0.25
    assert summary["S_theta"] == 1.0


def test_identical_float32_forecasts_have_zero_dispersion(tmp_path):
    from src.training.runner import inference_policy_metrics, save_run_artifacts, metric_rows

    predictions = {r: np.full((1, 1, 1), 10000.3, dtype=np.float32) for r in range(12)}
    targets = np.zeros((1, 1, 1), dtype=np.float32)
    metrics = inference_policy_metrics(predictions, targets, predictions, targets)
    assert metrics["E_ens"] == metrics["E_mean"]
    rows, _ = metric_rows(predictions, targets)
    save_run_artifacts(tmp_path, {"dataset": "synthetic", "seed": 0},
                       [], [], rows, predictions, targets, {}, [])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["origin_prediction_variance"] == 0.0
    assert summary["S_theta"] == 0.0


def test_synthetic_smoke_produces_metrics_without_dataset(tmp_path):
    result = subprocess.run(
        [sys.executable, "tools/smoke_experiment.py", "--output", str(tmp_path / "smoke")],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    output = tmp_path / "smoke"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["dataset"] == "synthetic_not_paper_evidence"
    assert summary["seed"] == 0
    assert summary["metric_recomputation_pass"] is True
    assert np.isfinite(summary["G_origin"])
    assert (output / "PROVENANCE.json").is_file()


def test_sentinel_cli_runs_outside_checkout(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "tools/check_sentinel.py")],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["pass"] is True
