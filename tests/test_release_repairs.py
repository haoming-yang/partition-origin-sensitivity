import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.analysis.frozen_mechanisms import rademacher_jacobian_energy

ROOT = Path(__file__).resolve().parents[1]


def test_tables_separate_protocols_and_compute_sample_sd(tmp_path):
    for name, stride, seed, value in [("a", 12, 42, 1), ("b", 12, 43, 3), ("c", 6, 42, 7)]:
        d = tmp_path / name
        d.mkdir()
        (d / "summary.json").write_text(json.dumps(dict(experiment_id="core", dataset="ETTh1", seed=seed, stride=stride, MSE_mean=value, G_origin=10, G_interior=5, CV_origin=.1, Delta_origin=.2)))
    a, t = tmp_path / "a.csv", tmp_path / "tables.csv"
    subprocess.run([sys.executable, str(ROOT / "tools/aggregate_results.py"), "--root", str(tmp_path), "--out", str(a)], check=True)
    subprocess.run([sys.executable, str(ROOT / "tools/make_tables.py"), "--input", str(a), "--output", str(t)], check=True)
    rows = list(csv.DictReader(t.open()))
    assert len(rows) == 2
    row = next(r for r in rows if r["n_runs"] == "2")
    assert float(row["MSE_mean_mean"]) == 2
    assert float(row["MSE_mean_sample_sd"]) == pytest.approx(2 ** .5)
    assert json.loads(row["seeds"]) == ["42", "43"]


def test_projection_generator_is_independent_of_global_rng():
    x = torch.ones(2, 3, 1)
    def f(z):
        return z.sum(1, keepdim=True).expand(-1, 2, -1)
    def evaluate():
        return rademacher_jacobian_energy(f, lambda z: f(z) * 2, x, projections=3, generator=torch.Generator().manual_seed(17))
    a = evaluate()
    torch.rand(100)
    b = evaluate()
    assert all(torch.equal(u, v) for u, v in zip(a, b))


def test_shell_dispatch_explicitly_uses_bash():
    text = (ROOT / "scripts/reproduce_all.sh").read_text()
    assert 'exec "$ROOT/scripts/' not in text


def test_frozen_release_reproduces_paper_anchors(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "tools/reproduce_frozen_paper.py"), "--output", str(tmp_path)], check=True)
    rows = list(csv.DictReader((tmp_path / "table3_runs.csv").open()))
    etth1 = [r for r in rows if r["dataset"] == "ETTh1"]
    assert [round(float(r["selected_change_pct"]), 2) for r in etth1] == [-15.76, -22.92, -12.44]
    audit = json.loads((tmp_path / "diagnostic_audit.json").read_text())
    assert round(audit["readout_weather_controlled"]["top_mass_percent"], 2) == 14.04
    assert round(audit["readout_weather_patchtst"]["top_mass_percent"], 2) == 13.74


def test_review_export_sanitizes_identity_and_keeps_weights(tmp_path):
    if not (ROOT / ".git").exists():
        pytest.skip("Re-export requires the public source checkout, not an anonymous export")
    weights = json.loads((ROOT / "artifacts/review_checkpoint_manifest.json").read_text())
    if any(not (ROOT / row["file"]).is_file() for row in weights):
        pytest.skip("Review export integration requires external checkpoint bundle")
    subprocess.run([sys.executable, str(ROOT / "tools/export_review_artifact.py"), "--output", str(tmp_path / "review")], check=True)
    out = tmp_path / "review"
    assert not (out / ".git").exists()
    text = (out / "README.md").read_text(encoding="utf-8")
    author = "Haoming" + " Yang"
    owner = "haoming" + "-yang"
    assert author not in text and owner not in text
    manifest = json.loads((out / "artifacts/review_checkpoint_manifest.json").read_text())
    assert all((out / row["file"]).is_file() for row in manifest)
