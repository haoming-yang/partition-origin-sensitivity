from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--out", default="audit_report.json")
    args = parser.parse_args()
    root = Path(args.root)
    reports = []
    for summary_path in sorted(root.rglob("summary.json")) if root.exists() else []:
        summary = json.loads(summary_path.read_text(encoding="utf8"))
        mse = summary.get("MSE_mean", summary.get("avg_mse"))
        gap = summary.get("G_origin", summary.get("G_origin_pct"))
        finite = all(math.isfinite(float(value)) for value in (mse, gap) if value is not None)
        reports.append({"path": str(summary_path), "finite_summary_metrics": finite, "experiment_id": summary.get("experiment_id"), "dataset": summary.get("dataset"), "seed": summary.get("seed")})
    result = {"root": str(root), "summary_count": len(reports), "reports": reports, "pass": all(item["finite_summary_metrics"] for item in reports)}
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
