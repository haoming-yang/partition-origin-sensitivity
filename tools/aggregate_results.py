from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--out", default="results/aggregate.csv")
    args = parser.parse_args()
    rows = []
    root = Path(args.root)
    for path in sorted(root.rglob("summary.json")) if root.exists() else []:
        data = json.loads(path.read_text(encoding="utf8"))
        rows.append({"path": str(path), "experiment_id": data.get("experiment_id"), "dataset": data.get("dataset"), "seed": data.get("seed"), "MSE_mean": data.get("MSE_mean", data.get("avg_mse")), "G_origin": data.get("G_origin"), "G_interior": data.get("G_interior"), "CV_origin": data.get("CV_origin"), "Delta_origin": data.get("Delta_origin")})
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "experiment_id", "dataset", "seed", "MSE_mean", "G_origin", "G_interior", "CV_origin", "Delta_origin"]
    with output.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
