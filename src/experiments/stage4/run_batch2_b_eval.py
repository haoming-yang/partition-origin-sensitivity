from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import stage4_inference as base
import poc_lambda_pilot as pilot


OUT = base.STAGE / "batch2_poc" / "formal_C" / "B_reused_evaluation_corrected"


def evaluate_b(seed: int, out: Path):
    values, _, val, _, split = base.load_data("ETTh1")
    source = base.E_STAGE3 / "attribution_abc" / "b_two_view_supervised" / f"seed{seed}"
    ckpt = source / "checkpoint.pt"
    summary_path = source / "summary.json"
    schedule_path = base.E_STAGE3 / "attribution_abc" / f"origin_pair_schedule_seed{seed}.json"
    if not ckpt.exists() or not summary_path.exists():
        raise FileNotFoundError(source)
    model, initial_hash = pilot.make_model(seed, values.shape[1])
    model.load_state_dict(torch.load(ckpt, map_location=pilot.DEVICE, weights_only=True))
    model.eval()
    vl = torch.utils.data.DataLoader(base.Windows(values, val), batch_size=base.BATCH, shuffle=False, pin_memory=pilot.DEVICE.type == "cuda")
    test = base.load_data("ETTh1")[3]
    tl = torch.utils.data.DataLoader(base.Windows(values, test), batch_size=base.BATCH, shuffle=False, pin_memory=pilot.DEVICE.type == "cuda")
    preflight = pilot.preflight_once(model, values, vl)
    metrics = pilot.dispersion_metrics(model, tl, 12)
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = {
        "experiment_id": "STAGE4_BATCH2_B_REUSED_EVALUATION_V1",
        "condition": "B_two_view_supervised",
        "dataset": "ETTh1", "p": 12, "seed": seed,
        "source_run": str(source),
        "checkpoint_sha256": base.sha256(ckpt),
        "source_summary_sha256": base.sha256(summary_path),
        "origin_pair_schedule": str(schedule_path),
        "origin_pair_schedule_sha256": base.sha256(schedule_path),
        "evaluator_sha256": base.sha256(Path(__file__).resolve().parent / "stage4_inference.py"),
        "initial_parameter_hash_reconstructed": initial_hash,
        "split": split, "preflight": preflight,
        "S_theta": metrics["S_theta"],
        "origin_prediction_variance": metrics["origin_prediction_variance"],
        "identity_rhs": metrics["identity_rhs"],
        "identity_absolute_error": metrics["identity_absolute_error"],
        "identity_relative_error": metrics["identity_relative_error"],
        "G_origin": source_summary["G_origin"],
        "CV_origin": source_summary["CV_origin"],
        "Delta_origin": source_summary["Delta_origin"],
        "G_interior": source_summary["G_interior"],
        "Avg_MSE": source_summary["avg_mse"],
        "Avg_MAE": source_summary["avg_mae"],
        "test_evaluation_performed": True,
        "test_evaluation_after_lambda_freeze": True,
        "source_checkpoint_is_stage3_B": True,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in (42, 43, 44):
        out = OUT / f"seed{seed}.json"
        d = evaluate_b(seed, out)
        rows.append({"seed": seed, "S_theta": d["S_theta"], "Avg_MSE": d["Avg_MSE"]})
        print(json.dumps(rows[-1]), flush=True)
    (OUT / "STATUS.json").write_text(json.dumps({"condition": "B_two_view_supervised", "dataset": "ETTh1", "seeds": [42, 43, 44], "test_evaluation_after_lambda_freeze": True}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
