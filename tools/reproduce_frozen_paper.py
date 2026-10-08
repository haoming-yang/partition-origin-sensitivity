import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.output_paths import validate_generated_output


def save(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, keys):
    result = []
    for dataset in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]:
        runs = [r for r in rows if r["dataset"] == dataset]
        if len(runs) != 3 or {r["seed"] for r in runs} != {42, 43, 44}:
            raise ValueError(f"Incomplete replicate set: {dataset}")
        row = dict(dataset=dataset, n_runs=3)
        for key in keys:
            values = [r[key] for r in runs]
            row[key + "_mean"] = statistics.mean(values)
            row[key + "_sample_sd"] = statistics.stdev(values)
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_reproduction")
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args()
    args.output = validate_generated_output(args.output)
    manifest = json.loads((ROOT / "artifacts/paper_release_sha256.json").read_text())
    for rel, expected in manifest.items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact checksum mismatch: {rel}")
    args.output.mkdir(parents=True, exist_ok=True)
    base = ROOT / "artifacts/paper_records/inference"
    runs, gaps = [], []
    canonical, normalized = [], []
    core_by_dataset = {}
    for dataset in ["etth1", "etth2", "ettm1", "ettm2", "weather"]:
        for seed in [42, 43, 44]:
            a = json.loads((base / "inference_baselines" / dataset / f"seed{seed}.json").read_text())
            b = json.loads((base / "dispersion_analysis" / dataset / f"seed{seed}.json").read_text())
            if a["checkpoint_sha256"] != b["checkpoint_sha256"]:
                raise ValueError("Unmatched checkpoint records")
            t = a["test"]
            row = dict(dataset=a["dataset"], seed=seed, E0=t["origin0"]["MSE"], E_star=t["validation_selected_origin"]["MSE"], E_mean=t["mean_single_origin"]["avg_mse"], E_ens=t["ensemble"]["MSE"], S_theta=b["S_theta"])
            row["selected_change_pct"] = 100 * (row["E_star"] / row["E0"] - 1)
            runs.append(row)
            gaps.append(dict(dataset=a["dataset"], seed=seed, **{k:b[k] for k in ["G_origin", "G_interior", "CV_origin", "Delta_origin"]}))
            canonical.append(dict(dataset=a["dataset"], seed=seed, MSE_mean=b["Avg_MSE"],
                                  G_origin_pct=b["G_origin"], G_interior_pct=b["G_interior"]))
            core_by_dataset.setdefault(a["dataset"], []).append(b)
            for entry in t["per_origin"]:
                normalized.append(dict(dataset=a["dataset"], seed=seed, origin=entry["origin"],
                                       MSE=entry["MSE"], run_mean_MSE=b["Avg_MSE"],
                                       origin_relative_to_run_mean_pct=100 * (entry["MSE"] / b["Avg_MSE"] - 1)))
    save(args.output / "table3_runs.csv", runs)
    save(args.output / "table3_summary.csv", summarize(runs, ["E0", "E_star", "E_mean", "E_ens", "S_theta"]))
    save(args.output / "table2_runs.csv", gaps)
    save(args.output / "table2_summary.csv", summarize(gaps, ["G_origin", "G_interior", "CV_origin", "Delta_origin"]))
    save(args.output / "visualization_canonical_summary.csv", canonical)
    save(args.output / "visualization_per_origin_normalized.csv", normalized)
    aggregate, core = [], []
    fields = (("MSE_mean", "Avg_MSE"), ("G_origin_pct", "G_origin"),
              ("G_interior_pct", "G_interior"))
    for dataset in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"):
        records = core_by_dataset[dataset]
        row = {"dataset": dataset}
        for name, source in fields:
            values = [record[source] for record in records]
            row[name + "_mean"] = statistics.mean(values)
            row[name + "_std"] = statistics.stdev(values)
        aggregate.append({**row, "seed_count": 3})
        core_row = dict(row)
        for name, source in (("Delta_origin", "Delta_origin"), ("S_theta", "S_theta")):
            values = [record[source] for record in records]
            core_row[name + "_mean"] = statistics.mean(values)
            core_row[name + "_std"] = statistics.stdev(values)
        core.append({**core_row, "run_count": 3})
    save(args.output / "visualization_canonical_aggregate.csv", aggregate)
    save(args.output / "core_cross_dataset_summary.csv", core)
    window_values = {seed: [] for seed in (42, 43, 44)}
    with (ROOT / "artifacts/paper_records/etth1_window_dispersion.csv").open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            window_values[int(record["seed"])].append(float(record["S_theta"]))
    window_summary = []
    for seed, values in window_values.items():
        array = np.asarray(values, dtype=np.float64)
        if array.size != 5805:
            raise ValueError(f"Incomplete ETTh1 window dispersion for seed {seed}")
        window_summary.append(dict(seed=seed, window_count=int(array.size), mean=float(array.mean()),
                                   median=float(np.median(array)), p90=float(np.percentile(array, 90)),
                                   p95=float(np.percentile(array, 95)), p99=float(np.percentile(array, 99)),
                                   min=float(array.min()), max=float(array.max())))
    save(args.output / "etth1_window_dispersion_summary.csv", window_summary)
    diagnostics = ROOT / "artifacts/frozen_diagnostics"
    audit = {}
    for name in ["readout_weather_controlled", "readout_weather_patchtst"]:
        with (diagnostics / name / "head_contributions.csv").open() as f:
            rows = list(csv.DictReader(f))
        values = [float(r["mean_abs_contribution"]) for r in rows]
        with (diagnostics / name / "window_metrics.csv").open() as f:
            windows = list(csv.DictReader(f))
        audit[name] = dict(top_token=int(rows[values.index(max(values))]["token"]), top_mass_percent=max(values)/sum(values)*100, windows=len(windows), max_raw_error=max(float(r["raw_reconstruction_relative_error"]) for r in windows))
    (args.output / "diagnostic_audit.json").write_text(json.dumps(audit, indent=2))
    if args.figure:
        subprocess.run([sys.executable, str(ROOT / "tools/plot_frozen_diagnostics.py"), "--controlled-jacobian", str(diagnostics / "jacobian_weather_controlled_512/jacobian_profile.csv"), "--patchtst-jacobian", str(diagnostics / "jacobian_weather_patchtst_512/jacobian_profile.csv"), "--controlled-contributions", str(diagnostics / "readout_weather_controlled/head_contributions.csv"), "--patchtst-contributions", str(diagnostics / "readout_weather_patchtst/head_contributions.csv"), "--output", str(args.output / "figure5.pdf")], check=True)
    print("Frozen Table 2/3 statistics and readout audit verified and regenerated")


if __name__ == "__main__":
    main()
