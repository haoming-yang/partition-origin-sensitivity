from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE = Path(os.environ.get("PARTITION_ORIGIN_STAGE4_ROOT", REPO_ROOT / "outputs" / "poc_stage4"))
LAMBDA_DIR = STAGE / "batch2_poc" / "lambda_selection"
PROTOCOL = LAMBDA_DIR / "LAMBDA_SELECTION_PROTOCOL.json"
BVAL = LAMBDA_DIR / "B_seed42_validation.json"
CANDIDATES = [0.01, 0.03, 0.1, 0.3, 1.0]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    protocol = read_json(PROTOCOL)
    if protocol["candidate_lambdas"] != CANDIDATES:
        raise RuntimeError("Candidate set changed from frozen protocol")
    b = read_json(BVAL)
    if b.get("test_evaluation_performed") is not False:
        raise RuntimeError("B validation artifact claims test evaluation")
    b_m = b["validation"]
    b_avg = float(b_m["avg_mse"])
    threshold = b_avg * 1.01
    rows = []
    for lam in CANDIDATES:
        d = LAMBDA_DIR / f"lambda_{str(lam).replace('.', 'p')}" / "summary.json"
        if not d.exists():
            raise FileNotFoundError(d)
        s = read_json(d)
        if s.get("test_evaluation_performed") is not False:
            raise RuntimeError(f"Test evaluation flag is not false: {d}")
        if float(s["lambda"]) != lam:
            raise RuntimeError(f"Lambda mismatch in {d}")
        vm = s["validation"]
        row = {
            "lambda": lam,
            "validation_avg_mse": float(vm["avg_mse"]),
            "validation_avg_mae": float(vm["avg_mae"]),
            "validation_S_theta": float(vm["S_theta"]),
            "validation_G_origin": float(vm["G_origin"]),
            "validation_CV_origin": float(vm["CV_origin"]),
            "validation_Delta_origin": float(vm["Delta_origin"]),
            "validation_identity_absolute_error": float(vm["identity_absolute_error"]),
            "eligible": bool(float(vm["avg_mse"]) <= threshold),
            "summary_sha256": sha(d),
        }
        rows.append(row)
    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        selected = None
        status = "NO_ACCEPTABLE_LAMBDA"
    else:

        eligible.sort(key=lambda r: (r["validation_S_theta"], r["lambda"]))
        selected = eligible[0]["lambda"]
        status = "FROZEN_POC_LAMBDA"
    schedule_hash = protocol["matched_controls"]["origin_pair_schedule_sha256"]
    out = LAMBDA_DIR.parent / "FROZEN_POC_LAMBDA.json"
    result = {
        "experiment_id": "STAGE4_BATCH2_POC_LAMBDA_FREEZE_V1",
        "status": status,
        "frozen": selected is not None,
        "selected_lambda": selected,
        "candidate_lambdas": CANDIDATES,
        "selection_dataset": "ETTh1",
        "selection_seed": 42,
        "test_evaluation_before_freeze": False,
        "b_validation_avg_mse": b_avg,
        "eligibility_threshold": threshold,
        "selection_rule": protocol["selection_rule"],
        "candidates": rows,
        "origin_pair_schedule": protocol["matched_controls"]["origin_pair_schedule_path"],
        "origin_pair_schedule_sha256": schedule_hash,
        "lambda_selection_protocol_sha256": sha(PROTOCOL),
        "pilot_runner_sha256": sha(STAGE / "scripts" / "poc_lambda_pilot.py"),
        "evaluator_sha256": sha(STAGE / "scripts" / "stage4_inference.py"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "all_candidates_validation_only": True,
            "seed43_or_seed44_used_for_selection": False,
            "test_metrics_used_for_selection": False,
            "candidate_set_expanded": False,
        },
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    digest = sha(out)
    (out.parent / "FROZEN_POC_LAMBDA.sha256").write_text(
        f"{digest}  {out.name}\n", encoding="utf-8"
    )
    report = STAGE / "reports" / "POC_LAMBDA_SELECTION.md"
    lines = [
        "# POC Lambda Selection",
        "",
        f"- Status: `{status}`",
        f"- Selected lambda: `{selected}`",
        f"- B validation Avg MSE: `{b_avg:.12f}`",
        f"- Eligibility threshold: `{threshold:.12f}`",
        "- Test evaluation before freeze: `NO`",
        "- Selection seed: `42`",
        "",
        "| lambda | Val Avg MSE | Val S_theta | Val G_origin | Val CV_origin | Val Delta_origin | eligible |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in sorted(rows, key=lambda x: x["lambda"]):
        lines.append(
            f"| {r['lambda']} | {r['validation_avg_mse']:.12f} | "
            f"{r['validation_S_theta']:.12f} | {r['validation_G_origin']:.6f} | "
            f"{r['validation_CV_origin']:.12f} | {r['validation_Delta_origin']:.12f} | "
            f"{r['eligible']} |"
        )
    lines += [
        "",
        f"- `FROZEN_POC_LAMBDA.json` SHA-256: `{digest}`",
        f"- Lambda protocol SHA-256: `{result['lambda_selection_protocol_sha256']}`",
        f"- Origin-pair schedule SHA-256: `{schedule_hash}`",
        "- No test metrics were read or used in this selection.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "selected_lambda": selected, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
