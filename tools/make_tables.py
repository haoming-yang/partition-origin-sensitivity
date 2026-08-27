from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/aggregate.csv")
    parser.add_argument("--output", default="outputs/table_core.csv")
    args = parser.parse_args()
    source = Path(args.input)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", newline="", encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    fields = ["dataset", "MSE_mean", "G_origin", "G_interior", "CV_origin", "Delta_origin"]
    with target.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    print(target)


if __name__ == "__main__":
    main()
