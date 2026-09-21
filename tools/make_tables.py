import argparse
import csv
import json
import math
import statistics
from pathlib import Path

METRICS = ["MSE_mean", "G_origin", "G_interior", "CV_origin", "Delta_origin"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/aggregate.csv")
    parser.add_argument("--output", default="outputs/generated_tables.csv")
    args = parser.parse_args()
    with Path(args.input).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No runs to summarize")
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("experiment_id", "dataset", "protocol"))
        group = groups.setdefault(key, [])
        if any(r["seed"] == row["seed"] for r in group):
            raise ValueError(f"Duplicate seed in experiment group: {key}")
        group.append(row)
    output = []
    for key, runs in sorted(groups.items()):
        runs.sort(key=lambda r: r["seed"])
        row = dict(zip(("experiment_id", "dataset", "protocol"), key))
        row.update(n_runs=len(runs), seeds=json.dumps([r["seed"] for r in runs]), sources=json.dumps([r["path"] for r in runs]))
        for k in METRICS:
            values = [float(r[k]) for r in runs]
            if not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError(f"Invalid {k}")
            row[k + "_mean"] = statistics.mean(values)
            row[k + "_sample_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        output.append(row)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
