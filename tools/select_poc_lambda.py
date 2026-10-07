from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.poc_selection import TIE_TOLERANCE, select_lambda
from src.utils.output_paths import validate_generated_output
from src.utils.provenance import runtime_metadata


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(input_root: Path, output: Path) -> dict:
    output = validate_generated_output(output)
    protocol_path = input_root / "LAMBDA_SELECTION_PROTOCOL.json"
    protocol = read(protocol_path)
    candidates = [0.01, 0.03, 0.1, 0.3, 1.0]
    if protocol["candidate_lambdas"] != candidates:
        raise ValueError("Candidate set differs from the recorded protocol")
    if protocol["eligibility_rule"] != "ValAvgMSE_C <= 1.01 * ValAvgMSE_B":
        raise ValueError("Eligibility rule differs from the recorded historical 1.01 threshold")
    baseline_path = input_root / "B_seed42_validation.json"
    baseline = read(baseline_path)
    if baseline.get("test_evaluation_performed") is not False:
        raise ValueError("Baseline must be a validation-only record")
    b_mse = float(baseline["validation"]["avg_mse"])
    inputs = [protocol_path, baseline_path]
    rows = []
    for value in candidates:
        path = input_root / f"lambda_{str(value).replace('.', 'p')}" / "summary.json"
        summary = read(path)
        if summary.get("test_evaluation_performed") is not False or float(summary["lambda"]) != value:
            raise ValueError(f"Candidate must match lambda={value} and be validation-only: {path}")
        inputs.append(path)
        rows.append(dict(lambda_value=value, validation_mse=float(summary["validation"]["avg_mse"]),
                         validation_s=float(summary["validation"]["S_theta"])))
    chosen = select_lambda(rows, b_mse, 1.01)
    for row in rows:
        row["eligible"] = row["validation_mse"] <= b_mse * 1.01
    pair_differences = [abs(a["validation_s"] - b["validation_s"])
                        for i, a in enumerate(rows) for b in rows[i + 1:]]
    result = {
        "status": "FUTURE_SELECTION_VALIDATION_ONLY" if chosen is not None else "NO_ACCEPTABLE_LAMBDA",
        "selected_lambda": chosen, "baseline_validation_mse": b_mse,
        "eligibility_multiplier": 1.01, "eligibility_threshold": b_mse * 1.01,
        "tie_tolerance": TIE_TOLERANCE,
        "tie_rule": "Among eligible candidates within absolute 1e-12 of the global minimum validation S_theta, choose smaller lambda",
        "minimum_candidate_s_difference": min(pair_differences), "candidates": rows,
        "historical_record_rewritten": False, "test_metrics_used": False,
        "inputs_sha256": {str(path.relative_to(input_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_metadata(ROOT),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Future validation-only lambda selection; never rewrite the historical freeze")
    parser.add_argument("--input-root", type=Path, default=ROOT / "artifacts/poc_stage4/batch2_poc/lambda_selection")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/poc_selection/selection.json")
    args = parser.parse_args()
    print(json.dumps(run(args.input_root, args.output), indent=2))


if __name__ == "__main__":
    main()
