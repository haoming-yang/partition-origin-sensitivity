"""Audit the complete 57-run supplementary experiment matrix.

This script is read-only with respect to experiment artifacts.  It writes only
the audit JSON and the final markdown report in this directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULTS = Path(os.environ.get("OUTPUT_ROOT", str(REPO_ROOT / "outputs")))
SOURCE = Path(os.environ.get("TIME_SERIES_LIBRARY_ROOT", str(REPO_ROOT / "third_party" / "time_series_library")))
DATASETS = {
    "ETTh1": SOURCE / "dataset" / "ETT-small" / "ETTh1.csv",
    "ETTh2": SOURCE / "dataset" / "ETT-small" / "ETTh2.csv",
    "ETTm1": SOURCE / "dataset" / "ETT-small" / "ETTm1.csv",
    "ETTm2": SOURCE / "dataset" / "ETT-small" / "ETTm2.csv",
    "Weather": SOURCE / "dataset" / "weather" / "weather.csv",
}
REQUIRED = (
    "config.json",
    "checkpoint.pt",
    "final_checkpoint.pt",
    "training_log.csv",
    "validation_log.csv",
    "per_origin_mse.csv",
    "per_origin_predictions.npz",
    "summary.json",
    "PROVENANCE.json",
)


def run_specs() -> list[dict]:
    specs: list[dict] = []
    for seed in (42, 43, 44):
        specs.append({"group": "old_patchtst", "path": RESULTS / "patchtst_origin" / f"seed{seed}", "dataset": "ETTh1", "seed": seed, "origins": 12, "expected_horizon": 96})
        specs.append({"group": "old_horizon192", "path": RESULTS / "horizon192" / f"seed{seed}", "dataset": "ETTh1", "seed": seed, "origins": 12, "expected_horizon": 192})
        specs.append({"group": "old_overlap_stride12", "path": RESULTS / "overlap_audit" / "stride12" / f"seed{seed}", "dataset": "ETTh1", "seed": seed, "origins": 12, "expected_horizon": 96})
        specs.append({"group": "old_overlap_stride6", "path": RESULTS / "overlap_audit" / "stride6" / f"seed{seed}", "dataset": "ETTh1", "seed": seed, "origins": 6, "expected_horizon": 96})
        for strategy in ("boundary_only", "random_origin", "all_origins"):
            specs.append({"group": f"old_train_origin_{strategy}", "path": RESULTS / "train_origin_strategy" / strategy / f"seed{seed}", "dataset": "ETTh1", "seed": seed, "origins": 12, "expected_horizon": 96})
    for dataset in ("ETTh2", "ETTm1", "ETTm2", "Weather"):
        for seed in (42, 43, 44):
            specs.append({"group": "new_patchtst", "path": RESULTS / "patchtst_origin" / dataset / f"seed{seed}", "dataset": dataset, "seed": seed, "origins": 12, "expected_horizon": 96})
            specs.append({"group": "new_horizon192", "path": RESULTS / "horizon192" / dataset / f"seed{seed}", "dataset": dataset, "seed": seed, "origins": 12, "expected_horizon": 192})
            specs.append({"group": "new_overlap_stride6", "path": RESULTS / "overlap_audit" / "stride6" / dataset / f"seed{seed}", "dataset": dataset, "seed": seed, "origins": 6, "expected_horizon": 96})
    return specs


def formal_metrics(mse: np.ndarray) -> dict[str, float]:
    interior = mse[1:]
    return {
        "MSE_mean": float(mse.mean()),
        "G_origin": float((mse.max() - mse.min()) / mse.min() * 100.0),
        "G_interior": float((interior.max() - interior.min()) / interior.min() * 100.0),
        "Delta_origin": float(mse.max() - mse.min()),
        "CV_origin": float(mse.std(ddof=0) / mse.mean()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_for_run(config: dict, dataset: str, horizon: int, windows: int) -> np.ndarray:
    raw = pd.read_csv(DATASETS[dataset]).iloc[:, 1:].to_numpy(dtype=np.float32)
    split = config.get("split")
    if split is None:
        if dataset.startswith("ETTh"):
            train_end, val_end = 8640, 11520
        elif dataset.startswith("ETTm"):
            train_end, val_end = 34560, 46080
        else:
            train_end = int(raw.shape[0] * 0.7)
            val_end = raw.shape[0] - int(raw.shape[0] * 0.2)
        test_start = val_end
        train_rows = (0, train_end)
    else:
        test_start = int(split["test_rows"][0])
        train_rows = tuple(split["train_rows"])
    target = np.stack([raw[test_start + i : test_start + i + horizon] for i in range(windows)], axis=0)
    if config.get("input_scale") == "train-row standardized":
        train_start, train_end = train_rows
        mean = raw[train_start:train_end].mean(axis=0, keepdims=True)
        std = raw[train_start:train_end].std(axis=0, keepdims=True)
        target = (target - mean) / np.where(std < 1e-8, 1.0, std)
    return target.astype(np.float32)


def audit_one(spec: dict) -> dict:
    root: Path = spec["path"]
    result = {k: spec[k] for k in ("group", "dataset", "seed", "origins", "expected_horizon")}
    result["path"] = str(root)
    result["missing_files"] = [name for name in REQUIRED if not (root / name).is_file()]
    if result["missing_files"]:
        result["status"] = "MISSING_ARTIFACT"
        return result
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    result["summary_status"] = summary.get("status")
    result["config_experiment_id"] = config.get("experiment_id")
    result["summary_experiment_id"] = summary.get("experiment_id")
    result["config_horizon"] = config.get("horizon")
    result["config_origins"] = config.get("origins")
    result["config_split"] = config.get("split")
    result["config_sha256"] = sha256(root / "config.json")
    pred = np.load(root / "per_origin_predictions.npz")
    keys = sorted(pred.files, key=lambda key: int(key.rsplit("_", 1)[1]))
    result["prediction_keys"] = keys
    result["prediction_shape"] = list(pred[keys[0]].shape) if keys else None
    result["finite_predictions"] = bool(all(np.isfinite(pred[key]).all() for key in keys))
    result["origin_count"] = len(keys)
    mse_record = pd.read_csv(root / "per_origin_mse.csv")["MSE"].to_numpy(dtype=np.float64)
    target = target_for_run(config, spec["dataset"], int(config["horizon"]), int(config["test_windows"]))
    raw_mse = np.array([np.mean((pred[key] - target) ** 2, dtype=np.float64) for key in keys])
    result["raw_recomputed_metrics"] = formal_metrics(raw_mse)
    result["summary_metrics"] = {key: summary.get(key) for key in ("MSE_mean", "G_origin", "G_interior", "Delta_origin", "CV_origin")}
    result["max_raw_vs_record_mse_abs_diff"] = float(np.max(np.abs(raw_mse - mse_record)))
    result["max_raw_vs_summary_metric_abs_diff"] = float(max(abs(raw_recomputed_metrics - float(summary[key])) for raw_recomputed_metrics, key in ((result["raw_recomputed_metrics"]["MSE_mean"], "MSE_mean"), (result["raw_recomputed_metrics"]["G_origin"], "G_origin"), (result["raw_recomputed_metrics"]["G_interior"], "G_interior"), (result["raw_recomputed_metrics"]["Delta_origin"], "Delta_origin"), (result["raw_recomputed_metrics"]["CV_origin"], "CV_origin"))))
    result["status"] = "PASS" if (
        summary.get("status") == "COMPLETE"
        and len(keys) == spec["origins"]
        and list(pred[keys[0]].shape[:2]) == [int(config["test_windows"]), int(config["horizon"])]
        and result["finite_predictions"]
        and result["max_raw_vs_record_mse_abs_diff"] < 1e-5
    ) else "AUDIT_FAIL"
    pred.close()
    return result


def process_status() -> str:
    try:
        completed = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"], capture_output=True, text=True, check=False)
        return completed.stdout.strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def main() -> None:
    results = [audit_one(spec) for spec in run_specs()]
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "expected_old_runs": 21,
        "expected_new_runs": 36,
        "expected_total_runs": 57,
        "run_count": len(results),
        "pass_count": sum(item.get("status") == "PASS" for item in results),
        "failure_count": sum(item.get("status") != "PASS" for item in results),
        "python_process_snapshot": process_status(),
        "runs": results,
    }
    report_root = RESULTS / "matrix_audit"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "final_matrix_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Final Complete Experiment Report",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "This report is an artifact-only audit. No paper prose, paper tables, figures, formal metric definitions, or existing run artifacts were modified.",
        "",
        "## Matrix status",
        "",
        f"- Existing supplementary runs: **{payload['expected_old_runs']}**",
        f"- Newly required runs: **{payload['expected_new_runs']}**",
        f"- Audited total: **{payload['run_count']}** (expected 57)",
        f"- PASS: **{payload['pass_count']}**; failures: **{payload['failure_count']}**",
        "",
        "The formal metrics were independently recomputed from saved per-origin predictions with the minimum-MSE denominator for `G_origin` and `G_interior`; `CV_origin` uses population standard deviation divided by mean. Official PatchTST raw-scale MSE is not compared numerically with standardized controlled MSE.",
        "",
        "## Run-level audit",
        "",
        "| Group | Dataset | Seed | Origins | Horizon | Status | Max raw-vs-record MSE difference |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for item in results:
        lines.append(f"| {item['group']} | {item['dataset']} | {item['seed']} | {item['origins']} | {item['expected_horizon']} | {item.get('status')} | {item.get('max_raw_vs_record_mse_abs_diff', 'NA')} |")
    lines += [
        "",
        "## Artifact requirements",
        "",
        "Each PASS run contains config, checkpoint(s), training/validation logs, per-origin predictions and metrics, `summary.json`, and `PROVENANCE.json`; predictions are finite and have the configured window/horizon shape.",
        "",
        "## Failure archive",
        "",
        "Any failed or superseded artifacts remain in place and are not silently deleted. The known first H=192 split-bug attempt is retained under `outputs/horizon192/seed42_failed_split_bug/`.",
        "",
        "## Process state",
        "",
        "The process snapshot captured during generation is stored in `final_matrix_audit.json`; a final no-remaining-process check is required before this report is accepted.",
    ]
    (report_root / "FINAL_COMPLETE_EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
