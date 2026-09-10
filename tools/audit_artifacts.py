from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def audit_summary(path: Path) -> dict:
    errors = []
    summary = {}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError("summary must be a JSON object")
    except (OSError, ValueError) as exc:
        errors.append(f"Cannot read summary: {exc}")
        summary = {}
    if not errors:
        for names in (("MSE_mean", "avg_mse"), ("G_origin", "G_origin_pct")):
            value = summary.get(names[0], summary.get(names[1]))
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"Missing or nonnumeric metric: {names[0]}")
            else:
                try:
                    valid = math.isfinite(value) and value >= 0
                except OverflowError:
                    valid = False
                if not valid:
                    errors.append(f"Metric must be finite and nonnegative: {names[0]}")
    return {
        "path": str(path),
        "finite_summary_metrics": not errors,
        "experiment_id": summary.get("experiment_id"),
        "dataset": summary.get("dataset"),
        "seed": summary.get("seed"),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--out", default="audit_report.json")
    args = parser.parse_args()
    root = Path(args.root)
    paths = sorted(root.rglob("summary.json")) if root.is_dir() else []
    errors = [] if paths else ["No summary.json files found in the results directory"]
    reports = [audit_summary(path) for path in paths]
    result = {
        "root": str(root), "summary_count": len(reports), "reports": reports,
        "errors": errors,
        "pass": not errors and all(item["finite_summary_metrics"] for item in reports),
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
