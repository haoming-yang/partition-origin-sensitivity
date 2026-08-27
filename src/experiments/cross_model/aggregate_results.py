"""Aggregate the persisted Tier1 matrix only after all required jobs finish."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_tier1_matrix import matrix_jobs


def require_complete_matrix(completed: set[tuple[str, str, int]]) -> None:
    expected = {(job.model, job.dataset, job.seed) for job in matrix_jobs()}
    missing = expected - completed
    if missing:
        raise ValueError(f"missing {len(missing)} required matrix jobs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for manifest in args.root.rglob("manifest.json"):
        record = json.loads(manifest.read_text(encoding="utf8"))
        rows.append(record)
    completed = {(row["model"], row["dataset"], row["seed"]) for row in rows}
    require_complete_matrix(completed)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = [
        {key: row[key] for key in ("model", "dataset", "seed", "G_phase_pct", "CV_phase", "best_mse", "worst_mse", "patch_period", "sequence_length")}
        for row in sorted(rows, key=lambda r: (r["model"], r["dataset"], r["seed"]))
    ]
    with (args.out / "tier1_summary.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.out / "tier1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
