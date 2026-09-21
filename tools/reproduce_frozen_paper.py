import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    manifest = json.loads((ROOT / "artifacts/paper_release_sha256.json").read_text())
    for rel, expected in manifest.items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact checksum mismatch: {rel}")
    args.output.mkdir(parents=True, exist_ok=True)
    base = ROOT / "artifacts/paper_records/inference"
    runs, gaps = [], []
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
    save(args.output / "table3_runs.csv", runs)
    save(args.output / "table3_summary.csv", summarize(runs, ["E0", "E_star", "E_mean", "E_ens", "S_theta"]))
    save(args.output / "table2_runs.csv", gaps)
    save(args.output / "table2_summary.csv", summarize(gaps, ["G_origin", "G_interior", "CV_origin", "Delta_origin"]))
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
