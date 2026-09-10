import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_audit(root, report):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/audit_artifacts.py"),
         "--root", str(root), "--out", str(report)],
        capture_output=True, text=True,
    )
    return result.returncode, json.loads(report.read_text(encoding="utf-8"))


@pytest.mark.parametrize("exists", [False, True])
def test_audit_rejects_no_results(tmp_path, exists):
    root = tmp_path / "results"
    if exists:
        root.mkdir()
    status, report = run_audit(root, tmp_path / "audit.json")
    assert status != 0
    assert report["pass"] is False
    assert report["summary_count"] == 0
    assert report["errors"]


@pytest.mark.parametrize("content", [
    "{}", '{"MSE_mean": 0.4}', '{"G_origin": 20}',
    '{"MSE_mean": null, "G_origin": 20}',
    '{"MSE_mean": "bad", "G_origin": 20}',
    '{"MSE_mean": NaN, "G_origin": 20}',
    '{"MSE_mean": 0.4, "G_origin": Infinity}',
    '{"MSE_mean": true, "G_origin": 20}',
    '{"MSE_mean": -1, "G_origin": 20}',
    json.dumps({"MSE_mean": 10 ** 400, "G_origin": 20}),
    "[]", "not-json",
])
def test_audit_reports_invalid_metrics_and_fails(tmp_path, content):
    root = tmp_path / "results"
    root.mkdir()
    (root / "summary.json").write_text(content, encoding="utf-8")
    status, report = run_audit(root, tmp_path / "audit.json")
    assert status != 0
    assert report["pass"] is False
    assert report["summary_count"] == 1
    assert report["reports"][0]["errors"]


@pytest.mark.parametrize("metrics", [
    {"MSE_mean": 0.4, "G_origin": 20.0},
    {"avg_mse": 0.4, "G_origin_pct": 20.0},
    {"MSE_mean": 0, "G_origin": 0},
])
def test_audit_accepts_supported_finite_metrics_without_mutating_inputs(tmp_path, metrics):
    root = tmp_path / "results"
    root.mkdir()
    source = root / "summary.json"
    source.write_text(json.dumps(metrics), encoding="utf-8")
    before = source.read_bytes()
    status, report = run_audit(root, tmp_path / "audit.json")
    assert status == 0
    assert report["pass"] is True
    assert report["summary_count"] == 1
    assert source.read_bytes() == before


def test_one_invalid_summary_fails_the_whole_audit(tmp_path):
    for name, content in [("valid", '{"MSE_mean": 1, "G_origin": 2}'), ("invalid", '{}')]:
        directory = tmp_path / "results" / name
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(content, encoding="utf-8")
    status, report = run_audit(tmp_path / "results", tmp_path / "audit.json")
    assert status != 0
    assert report["pass"] is False
    assert report["summary_count"] == 2
