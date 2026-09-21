import argparse
import csv
import json
import math
from pathlib import Path

METRICS = ["MSE_mean", "G_origin", "G_interior", "CV_origin", "Delta_origin"]
PROTOCOL = ["model", "context", "horizon", "p", "patch_length", "stride", "epochs", "batch_size", "optimizer", "learning_rate", "weight_decay", "strategy", "use_position", "head_geometry", "input_scale", "source_status", "git_commit", "dropout", "scheduler", "origins", "train_windows", "validation_windows", "test_windows"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--out", default="outputs/aggregate.csv")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.rglob("summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not any(k in data for k in ("MSE_mean", "avg_mse")):
            continue
        config = path.with_name("config.json")
        settings = json.loads(config.read_text(encoding="utf-8")) if config.exists() else {}
        settings.update(data)
        if any(settings.get(k) is None for k in ("experiment_id", "dataset", "seed")):
            raise ValueError(f"Missing experiment identity: {path}")
        row = {k: settings[k] for k in ("experiment_id", "dataset", "seed")}
        row["path"] = path.relative_to(root).as_posix()
        row["protocol"] = json.dumps({k: settings[k] for k in PROTOCOL if k in settings}, sort_keys=True)
        for k in METRICS:
            value = data.get(k, data.get("avg_mse") if k == "MSE_mean" else None)
            if value is None or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"Invalid {k}: {path}")
            row[k] = value
        rows.append(row)
    if not rows:
        raise ValueError("No forecasting summaries found")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "experiment_id", "dataset", "seed", "protocol"] + METRICS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
