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


def test_shell_dispatch_lists_paper_commands_without_native_audit():
    result = subprocess.run(["bash", str(ROOT / "scripts/reproduce_all.sh"), "list"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert all(name in result.stdout.split() for name in ("core", "mixers", "patchtst", "poc", "tables"))
    assert "tier1-smoke" not in result.stdout


def test_frozen_release_reproduces_paper_anchors(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "tools/reproduce_frozen_paper.py"), "--output", str(tmp_path)], check=True)
    rows = list(csv.DictReader((tmp_path / "table3_runs.csv").open()))
    etth1 = [r for r in rows if r["dataset"] == "ETTh1"]
    assert [round(float(r["selected_change_pct"]), 2) for r in etth1] == [-15.76, -22.92, -12.44]
    audit = json.loads((tmp_path / "diagnostic_audit.json").read_text())
    assert round(audit["readout_weather_controlled"]["top_mass_percent"], 2) == 14.04
    assert round(audit["readout_weather_patchtst"]["top_mass_percent"], 2) == 13.74
    with (tmp_path / "core_cross_dataset_summary.csv").open(newline="") as stream:
        core = list(csv.DictReader(stream))
    with (tmp_path / "etth1_window_dispersion_summary.csv").open(newline="") as stream:
        windows = list(csv.DictReader(stream))
    with (tmp_path / "visualization_per_origin_normalized.csv").open(newline="") as stream:
        origins = list(csv.DictReader(stream))
    assert len(core) == 5 and len(windows) == 3 and len(origins) == 180
    assert float(core[0]["G_origin_pct_mean"]) == pytest.approx(22.698977985463767, abs=1e-10)
    assert float(windows[0]["mean"]) == pytest.approx(0.07746532537159544, abs=1e-10)
    assert float(origins[0]["origin_relative_to_run_mean_pct"]) == pytest.approx(15.219289889899578, abs=1e-10)
    for name in ("visualization_canonical_summary.csv", "visualization_canonical_aggregate.csv"):
        assert (tmp_path / name).is_file()
