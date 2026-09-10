"""Exactly one full-split ETTh1 run for the approved three-run pilot.

This module imports the original matched-backbone definitions and changes only
the ETTh1 window index construction from the old 1024/16/16 cap to the complete
train/validation/test window sets. It is intentionally not a general benchmark
launcher.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    import controlled_backbone_runner as base
except ModuleNotFoundError:
    from tools.fullsplit import controlled_backbone_runner as base


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_full_etth1():
    raw = pd.read_csv(base.DATA["ETTh1"]).iloc[:, 1:].astype("float32").to_numpy()
    train_end, val_end, test_end = 8640, 11520, len(raw)
    mean = raw[:train_end].mean(0, keepdims=True)
    std = raw[:train_end].std(0, keepdims=True)
    values = ((raw - mean) / np.where(std < 1e-8, 1, std)).astype("float32")
    train = np.arange(0, train_end - base.CONTEXT - base.HORIZON + 1)
    validation = np.arange(train_end - base.CONTEXT, val_end - base.CONTEXT - base.HORIZON + 1)
    test = np.arange(val_end - base.CONTEXT, test_end - base.CONTEXT - base.HORIZON + 1)
    return values, train, validation, test


@torch.inference_mode()
def evaluate_full(model, loader, device):
    """Evaluate all origins while accumulating every target batch.

    The legacy evaluator was safe only for the old validation/test cap because
    16 windows fit in one batch. Full ETTh1 splits require concatenating target
    batches explicitly; otherwise the ensemble calculation compares all
    predictions with only the first target batch.
    """
    model.eval()
    rows, predictions, target_chunks = [], [], []
    for origin in range(base.PATCH):
        squared_error = absolute_error = count = 0
        origin_predictions = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x, origin)
            error = prediction - y
            squared_error += error.square().sum().item()
            absolute_error += error.abs().sum().item()
            count += error.numel()
            origin_predictions.append(prediction.cpu().numpy())
            if origin == 0:
                target_chunks.append(y.cpu().numpy())
        predictions.append(np.concatenate(origin_predictions, axis=0))
        rows.append({"origin": origin, "MSE": squared_error / count, "MAE": absolute_error / count, "n_windows": len(loader.dataset)})
    prediction_array = np.stack(predictions)
    target_array = np.concatenate(target_chunks, axis=0)
    errors = np.array([row["MSE"] for row in rows])
    ensemble = float(np.mean((prediction_array.mean(0) - target_array) ** 2))
    metrics = {
        "MSEmean": float(errors.mean()),
        "MSE0": float(errors[0]),
        "MSEens": ensemble,
        "G_origin": float((errors.max() - errors.min()) / errors.min() * 100),
        "CV_origin": float(errors.std() / errors.mean()),
        "Delta_origin": float(errors.max() - errors.min()),
        "G_interior": float((errors[1:].max() - errors[1:].min()) / errors[1:].min() * 100),
        "S_theta": float(np.mean((prediction_array - prediction_array.mean(0)) ** 2)),
    }
    return rows, metrics, prediction_array, target_array


def run(model_name: str, seed: int, out: Path):
    if model_name not in ("Transformer", "MLP", "Conv"):
        raise ValueError(model_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    values, train, validation, test = load_full_etth1()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = {"Transformer": base.Transformer, "MLP": base.MLP, "Conv": base.Conv}[model_name](values.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    train_loader = DataLoader(base.Windows(values, train), 32, shuffle=True, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(base.Windows(values, validation), 32, shuffle=False)
    test_loader = DataLoader(base.Windows(values, test), 32, shuffle=False)
    out.mkdir(parents=True, exist_ok=True)
    experiment_id = f"FULLSPLIT_CONTROLLED_{model_name.upper()}_ETTH1_P12_S{seed}_V2"
    if (len(train), len(validation), len(test)) != (8033, 2785, 5805):
        raise RuntimeError(f"NONCANONICAL_ETTH1_COUNTS {(len(train), len(validation), len(test))}")
    config = {
        "experiment_id": experiment_id,
        "dataset": "ETTh1",
        "model": model_name,
        "seed": seed,
        "context": base.CONTEXT,
        "horizon": base.HORIZON,
        "patch_length": base.PATCH,
        "stride": base.STRIDE,
        "epochs": 5,
        "batch_size": 32,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "training_policy": "random-origin-one-origin-per-batch",
        "checkpoint_selection": "minimum mean validation MSE over all 12 origins; later epoch on tie",
        "split": {"train_windows": int(len(train)), "validation_windows": int(len(validation)), "test_windows": int(len(test)), "train_rows": [0, 8640], "validation_rows": [8640, 11520], "test_rows": [11520, int(len(values))]},
        "old_cap_forbidden": True,
        "source_runner": "tools/fullsplit/controlled_backbone_runner.py",
        "source_runner_sha256": sha256(Path(base.__file__).resolve()),
    }
    np.savez(out / "window_indices.npz", train=train, validation=validation, test=test)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best_state = None
    best_val = float("inf")
    selected_epoch = 0
    start = time.perf_counter()
    peak = 0
    for epoch in range(1, 6):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            origin = random.randrange(base.PATCH)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x, origin) - y).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * x.shape[0]
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated())
        _, validation_metrics, _, _ = evaluate_full(model, validation_loader, device)
        history.append({"epoch": epoch, "train_loss": total / len(train), "val_MSEmean": validation_metrics["MSEmean"]})
        if validation_metrics["MSEmean"] <= best_val:
            best_val = validation_metrics["MSEmean"]
            selected_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    validation_rows, validation_metrics, validation_predictions, validation_targets = evaluate_full(model, validation_loader, device)
    selected_origin = int(np.argmin([row["MSE"] for row in validation_rows]))
    test_rows, metrics, test_predictions, test_targets = evaluate_full(model, test_loader, device)
    metrics["MSE_selected"] = float(test_rows[selected_origin]["MSE"])
    metrics["selected_origin"] = selected_origin
    metrics["MSE_mean"] = metrics["MSEmean"]
    metrics["MSE_0"] = metrics["MSE0"]
    metrics["MSE_ensemble"] = metrics["MSEens"]
    torch.save({"state_dict": model.state_dict(), "config": config}, out / "checkpoint.pt")
    with (out / "training_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(out / "validation_predictions.npz", predictions=validation_predictions, targets=validation_targets)
    np.savez_compressed(out / "test_predictions.npz", predictions=test_predictions, targets=test_targets)
    (out / "per_origin_validation.json").write_text(json.dumps(validation_rows, indent=2), encoding="utf-8")
    (out / "per_origin_test.json").write_text(json.dumps(test_rows, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "validation_metrics.json").write_text(json.dumps(validation_metrics, indent=2), encoding="utf-8")
    run_meta = {
        "experiment_id": experiment_id,
        "model": model_name,
        "dataset": "ETTh1",
        "seed": seed,
        "selected_epoch": selected_epoch,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "wall_seconds": time.perf_counter() - start,
        "peak_vram_bytes": peak,
        "device": str(device),
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        "train_windows": len(train),
        "validation_windows": len(validation),
        "test_windows": len(test),
        "test_origins": len(test_rows),
        "data_rows": int(len(values)),
        "row_boundaries": {"train": [0, 8640], "validation": [8640, 11520], "test": [11520, int(len(values))]},
    }
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(json.dumps({"model": model_name, "status": "PASS", "metrics": metrics, "run_meta": run_meta}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("Transformer", "MLP", "Conv"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.seed, args.out)
