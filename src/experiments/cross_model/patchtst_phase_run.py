"""Train native PatchTST with random phase and evaluate every patch origin."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from adapters import tier1_specs
from native_models import build_model, native_forecast
from phase_protocol import audit_phase_layout, padded_length, phase_pad_torch
from result_schema import summarize_phase_mse


CONTEXT = 512
HORIZON = 96
PATCH = 16
CHANNEL_BLOCK = 7
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(REPO_ROOT / "data")))
DATASETS = {
    "ETTh1": DATA_ROOT / "ETT-small" / "ETTh1.csv",
    "Weather": DATA_ROOT / "weather" / "weather.csv",
    "Electricity": DATA_ROOT / "electricity" / "electricity.csv",
}


def split_starts(dataset: str, n_rows: int, context: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Match the previous gate's train/test split definitions."""
    if dataset == "ETTh1":
        train_end, test_start, test_end = 8640, 11520, 14400
    else:
        train_end, test_start, test_end = int(0.7 * n_rows), n_rows - int(0.2 * n_rows), n_rows
    return (
        np.arange(0, train_end - context - horizon + 1),
        np.arange(test_start - context, test_end - context - horizon + 1),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_values(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = pd.read_csv(DATASETS[dataset]).iloc[:, 1:].astype("float32").to_numpy()
    train, test = split_starts(dataset, len(raw), CONTEXT, HORIZON)
    train_end = 8640 if dataset == "ETTh1" else int(0.7 * len(raw))
    mean = raw[:train_end].mean(axis=0, keepdims=True)
    std = raw[:train_end].std(axis=0, keepdims=True)
    values = (raw - mean) / np.where(std < 1e-8, 1, std)
    return values.astype(np.float32), train, test


class Windows(Dataset):
    def __init__(self, values: np.ndarray, starts: np.ndarray):
        self.values = values
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        return (
            torch.from_numpy(self.values[start : start + CONTEXT]),
            torch.from_numpy(self.values[start + CONTEXT : start + CONTEXT + HORIZON]),
        )


def channel_groups(channels: int) -> list[tuple[np.ndarray, int]]:
    groups: list[tuple[np.ndarray, int]] = []
    for first in range(0, channels, CHANNEL_BLOCK):
        actual = min(CHANNEL_BLOCK, channels - first)
        indices = np.arange(first, first + actual)
        if actual < CHANNEL_BLOCK:
            indices = np.pad(indices, (0, CHANNEL_BLOCK - actual), mode="edge")
        groups.append((indices, actual))
    return groups


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, channels: int, patch: int, total_length: int, model_name: str) -> list[float]:
    model.eval()
    totals = np.zeros(patch, dtype=np.float64)
    counts = np.zeros(patch, dtype=np.int64)
    for phase in range(patch):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            for indices, actual in channel_groups(channels):
                idx = torch.as_tensor(indices, device=device)
                prediction = native_forecast(model_name, model, phase_pad_torch(x.index_select(2, idx), phase, patch, total_length))
                error = (prediction[:, :, :actual] - y.index_select(2, idx)[:, :, :actual]).square()
                totals[phase] += error.sum().item()
                counts[phase] += error.numel()
    return (totals / counts).tolist()


def run(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    spec = tier1_specs()[args.model]
    patch = spec.patch_size
    sequence_length = padded_length(CONTEXT, patch, alignment=spec.input_alignment)
    values, train_starts, test_starts = load_values(args.dataset)
    if args.max_train_windows:
        train_starts = train_starts[np.linspace(0, len(train_starts) - 1, args.max_train_windows, dtype=int)]
    if args.max_test_windows:
        test_starts = test_starts[: args.max_test_windows]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, seq_len=sequence_length, pred_len=HORIZON, channels=CHANNEL_BLOCK).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_loader = DataLoader(Windows(values, train_starts), batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    test_loader = DataLoader(Windows(values, test_starts), batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
    history: list[float] = []
    for _ in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            first = random.randrange(0, values.shape[1] - CHANNEL_BLOCK + 1)
            x, y = x[:, :, first : first + CHANNEL_BLOCK], y[:, :, first : first + CHANNEL_BLOCK]
            phase = random.randrange(patch)
            optimizer.zero_grad(set_to_none=True)
            loss = (native_forecast(args.model, model, phase_pad_torch(x, phase, patch, sequence_length)) - y).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * x.shape[0]
        history.append(total / len(train_starts))
    phase_mse = evaluate(model, test_loader, device, values.shape[1], patch, sequence_length, args.model)
    summary: dict[str, object] = summarize_phase_mse(phase_mse)
    summary.update(
        {
            "model": args.model,
            "dataset": args.dataset,
            "seed": args.seed,
            "epochs": args.epochs,
            "protocol": "same 512 standardized observations once; native zero boundary padding; random training phase",
            "patch_period": patch,
            "sequence_length": sequence_length,
            "phase_layout_audit": audit_phase_layout(CONTEXT, patch, total_length=sequence_length),
            "history": history,
            "device": str(device),
            "n_train": int(len(train_starts)),
            "n_test": int(len(test_starts)),
        }
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "phase_metrics.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "mse"])
        writer.writeheader()
        writer.writerows({"phase": phase, "mse": mse} for phase, mse in enumerate(phase_mse))
    (out / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("PatchTST", "PatchMixer", "PatchMLP", "Pathformer", "HDMixer", "DeformableTST"), required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--max-test-windows", type=int)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
