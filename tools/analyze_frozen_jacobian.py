import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.frozen_mechanisms import rademacher_jacobian_energy
from src.training import runner
from tools.analyze_frozen_mechanisms import build_model, configure_data_root


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_loader(data: runner.DataBundle, args: argparse.Namespace, kind: str, device: torch.device) -> DataLoader:
    starts = data.test if not args.max_windows else data.test[: args.max_windows]
    values = data.raw if kind == "patchtst" else data.standardized
    return DataLoader(
        runner.Windows(values, starts, args.context, args.horizon),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    configure_data_root(args.data_root.resolve())
    data = runner.load_data(args.dataset, args.horizon)
    device = runner.device()
    model = build_model(args.model, data, args, device)
    loader = make_loader(data, args, args.model, device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    accumulated = np.zeros((3, args.context), dtype=np.float64)
    denominator = 0
    window_count = 0
    for x, _target in loader:
        x = x.to(device, non_blocking=True)
        padded_a, observed_a = runner.partition(x, args.origin_a, args.context, args.patch_length, args.stride)
        padded_b, observed_b = runner.partition(x, args.origin_b, args.context, args.patch_length, args.stride)
        with torch.enable_grad():
            energy_a, energy_b, energy_difference = rademacher_jacobian_energy(
                lambda z: model(*runner.partition(z, args.origin_a, args.context, args.patch_length, args.stride)),
                lambda z: model(*runner.partition(z, args.origin_b, args.context, args.patch_length, args.stride)),
                x,
                projections=args.projections,
            )
        batch_channels = x.shape[0] * x.shape[2]
        accumulated[0] += energy_a.detach().sum(dim=(0, 2)).cpu().numpy()
        accumulated[1] += energy_b.detach().sum(dim=(0, 2)).cpu().numpy()
        accumulated[2] += energy_difference.detach().sum(dim=(0, 2)).cpu().numpy()
        denominator += batch_channels
        window_count += x.shape[0]
    if window_count == 0:
        raise RuntimeError("no test windows were processed")
    profiles = accumulated / denominator
    rows = [
        {
            "time_index": index,
            "origin_a_jacobian_energy": float(profiles[0, index]),
            "origin_b_jacobian_energy": float(profiles[1, index]),
            "origin_difference_jacobian_energy": float(profiles[2, index]),
        }
        for index in range(args.context)
    ]
    write_csv(output / "jacobian_profile.csv", rows)
    positions = np.arange(args.context, dtype=np.float64)
    center = (positions[None, :] * profiles).sum(axis=1) / profiles.sum(axis=1)
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint.resolve()),
        "data_root": str(args.data_root.resolve()),
        "data_file": str(data.path.resolve()),
        "input_scale": "raw" if args.model == "patchtst" else "training-standardized",
        "context": args.context,
        "horizon": args.horizon,
        "patch_length": args.patch_length,
        "stride": args.stride,
        "origin_pair": [args.origin_a, args.origin_b],
        "windows": window_count,
        "batch_size": args.batch_size,
        "projections": args.projections,
        "gradient_quantity": "Rademacher estimate of per-input-position Jacobian energy",
        "center_of_mass": {
            "origin_a": float(center[0]),
            "origin_b": float(center[1]),
            "origin_difference": float(center[2]),
        },
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        "python": platform.python_version(),
        "artifact": str((output / "jacobian_profile.csv").resolve()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("patchtst", "controlled"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--origin-a", type=int, default=0)
    parser.add_argument("--origin-b", type=int, default=6)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--patch-length", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--projections", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
