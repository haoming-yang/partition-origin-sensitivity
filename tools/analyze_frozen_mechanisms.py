import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.frozen_mechanisms import (
    disagreement_gradient,
    finite_difference_response,
    linear_head_delta_contributions,
    masked_location_statistics,
    summarize_position_gradient,
    summarize_token_contributions,
)
from src.training import runner


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_data_root(data_root: Path) -> None:
    runner.DATA_ROOT = data_root
    runner.DATASET_PATHS = {
        "ETTh1": data_root / "ETT-small" / "ETTh1.csv",
        "ETTh2": data_root / "ETT-small" / "ETTh2.csv",
        "ETTm1": data_root / "ETT-small" / "ETTm1.csv",
        "ETTm2": data_root / "ETT-small" / "ETTm2.csv",
        "Weather": data_root / "weather" / "weather.csv",
    }


def build_model(kind: str, data: runner.DataBundle, args: argparse.Namespace, dev: torch.device) -> nn.Module:
    if kind == "patchtst":
        model = runner.OfficialPatchTSTAdapter(args.context, args.horizon, args.patch_length, args.stride, data.channels)
    else:
        model = runner.ControlledTransformerSupplement(args.context, args.horizon, args.patch_length, args.stride, data.channels)
    model = model.to(dev)
    state = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def head_module(model: nn.Module, kind: str) -> nn.Module:
    return model.inner.head if kind == "patchtst" else model.head


def head_linear(head: nn.Module) -> nn.Linear:
    layers = [module for module in head.modules() if isinstance(module, nn.Linear)]
    if len(layers) != 1:
        raise RuntimeError("expected one linear prediction head")
    return layers[0]


def captured_forward(model: nn.Module, head: nn.Module, values: torch.Tensor, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    holder: dict[str, torch.Tensor] = {}

    def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        holder["features"] = inputs[0]

    handle = head.register_forward_pre_hook(capture)
    try:
        prediction = model(values, observed)
    finally:
        handle.remove()
    return holder["features"], prediction


def mechanism_features(features: torch.Tensor, kind: str) -> torch.Tensor:
    return features


def head_output(head: nn.Module, features: torch.Tensor) -> torch.Tensor:
    return head(features)


def make_loader(data: runner.DataBundle, args: argparse.Namespace, kind: str, dev: torch.device) -> DataLoader:
    starts = data.test if not args.max_windows else data.test[: args.max_windows]
    values = data.raw if kind == "patchtst" else data.standardized
    return DataLoader(
        runner.Windows(values, starts, args.context, args.horizon),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=dev.type == "cuda",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    configure_data_root(args.data_root.resolve())
    data = runner.load_data(args.dataset, args.horizon)
    dev = runner.device()
    model = build_model(args.model, data, args, dev)
    head = head_module(model, args.model)
    linear = head_linear(head)
    loader = make_loader(data, args, args.model, dev)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    finite_indices = [int(value) for value in args.finite_indices.split(",") if value.strip()]
    token_abs = None
    token_signed = None
    gradient_abs = np.zeros(args.context, dtype=np.float64)
    gradient_signed = np.zeros(args.context, dtype=np.float64)
    finite_response = np.zeros(len(finite_indices), dtype=np.float64)
    gradient_count = 0
    token_count = 0
    finite_count = 0
    pair_rows: list[dict[str, object]] = []
    window_index = 0
    for x, _target in loader:
        x = x.to(dev, non_blocking=True)
        padded_a, observed_a = runner.partition(x, args.origin_a, args.context, args.patch_length, args.stride)
        padded_b, observed_b = runner.partition(x, args.origin_b, args.context, args.patch_length, args.stride)
        features_a, prediction_a = captured_forward(model, head, padded_a, observed_a)
        features_b, prediction_b = captured_forward(model, head, padded_b, observed_b)
        raw_features_a = features_a
        raw_features_b = features_b
        features_a = mechanism_features(raw_features_a, args.model)
        features_b = mechanism_features(raw_features_b, args.model)
        contributions, head_delta = linear_head_delta_contributions(
            linear,
            features_a,
            features_b,
            feature_order="feature_token" if args.model == "patchtst" else "token_feature",
        )
        actual_head_delta = (head_output(head, raw_features_b) - head_output(head, raw_features_a)).permute(0, 2, 1)
        head_delta_prediction = head_delta.permute(0, 2, 1)
        head_reconstruction_error = (head_delta_prediction - actual_head_delta).norm(dim=(1, 2)) / (actual_head_delta.norm(dim=(1, 2)) + 1e-8)
        if args.model == "patchtst":
            mean_a, stdev_a = masked_location_statistics(padded_a, observed_a)
            mean_b, stdev_b = masked_location_statistics(padded_b, observed_b)
            if not torch.allclose(mean_a, mean_b, rtol=1e-5, atol=1e-5) or not torch.allclose(stdev_a, stdev_b, rtol=1e-5, atol=1e-5):
                raise RuntimeError("origin changes the adapter statistics")
            raw_contributions = contributions * stdev_a[:, :, None, None]
            raw_head_delta = head_delta_prediction * stdev_a[:, None, :]
        else:
            raw_contributions = contributions
            raw_head_delta = head_delta_prediction
        raw_reconstruction_error = (raw_head_delta - (prediction_b - prediction_a)).norm(dim=(1, 2)) / ((prediction_b - prediction_a).norm(dim=(1, 2)) + 1e-8)
        if token_abs is None:
            token_abs = np.zeros(contributions.shape[2], dtype=np.float64)
            token_signed = np.zeros(contributions.shape[2], dtype=np.float64)
        token_batch_abs, token_batch_signed, token_batch_count = summarize_token_contributions(raw_contributions.detach())
        token_abs += (token_batch_abs * token_batch_count).cpu().numpy()
        token_signed += (token_batch_signed * token_batch_count).cpu().numpy()
        token_count += token_batch_count
        prediction_delta = prediction_b - prediction_a
        direct_residual = (prediction_delta - raw_head_delta).detach().square().mean(dim=(1, 2)).cpu().numpy()
        with torch.enable_grad():
            value, gradient = disagreement_gradient(
                lambda z: model(*runner.partition(z, args.origin_a, args.context, args.patch_length, args.stride)),
                lambda z: model(*runner.partition(z, args.origin_b, args.context, args.patch_length, args.stride)),
                x,
            )
        gradient_batch_abs, gradient_batch_signed, gradient_batch_count = summarize_position_gradient(gradient)
        gradient_abs += (gradient_batch_abs * gradient_batch_count).cpu().numpy()
        gradient_signed += (gradient_batch_signed * gradient_batch_count).cpu().numpy()
        gradient_count += gradient_batch_count
        if finite_indices and finite_count < args.finite_max_windows:
            response = finite_difference_response(
                lambda z: model(*runner.partition(z, args.origin_a, args.context, args.patch_length, args.stride)),
                lambda z: model(*runner.partition(z, args.origin_b, args.context, args.patch_length, args.stride)),
                x.detach(), finite_indices, args.finite_delta,
            )
            finite_response += response.detach().cpu().numpy()
            finite_count += 1
        objective = value.cpu().numpy()
        for offset in range(x.shape[0]):
            pair_rows.append({
                "window": window_index + offset,
                "origin_a": args.origin_a,
                "origin_b": args.origin_b,
                "disagreement_objective": float(objective[offset]),
                "gradient_l2": float(gradient[offset].norm().item()),
                "head_delta_l2": float(head_delta_prediction[offset].norm().item()),
                "direct_prediction_delta_l2": float(prediction_delta[offset].detach().norm().item()),
                "head_delta_residual_mse": float(direct_residual[offset]),
                "head_reconstruction_relative_error": float(head_reconstruction_error[offset].item()),
                "raw_reconstruction_relative_error": float(raw_reconstruction_error[offset].item()),
            })
        window_index += x.shape[0]
    if token_abs is None or gradient_count == 0:
        raise RuntimeError("no test windows were processed")
    token_rows = [
        {"token": index, "mean_abs_contribution": float(token_abs[index] / token_count), "mean_signed_contribution": float(token_signed[index] / token_count)}
        for index in range(len(token_abs))
    ]
    gradient_rows = [
        {
            "time_index": index,
            "mean_abs_gradient": float(gradient_abs[index] / gradient_count),
            "mean_signed_gradient": float(gradient_signed[index] / gradient_count),
            "finite_difference_response": "",
        }
        for index in range(args.context)
    ]
    if finite_indices and finite_count:
        finite_values = finite_response / finite_count
        for index, value in zip(finite_indices, finite_values):
            gradient_rows[index]["finite_difference_response"] = float(value)
    write_csv(output / "head_contributions.csv", token_rows)
    write_csv(output / "gradient_profile.csv", gradient_rows)
    write_csv(output / "window_metrics.csv", pair_rows)
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
        "windows": window_index,
        "batch_size": args.batch_size,
        "finite_difference_indices": finite_indices,
        "finite_difference_delta": args.finite_delta,
        "finite_difference_batches": finite_count,
        "head_attribution_space": "linear prediction-head pre-denormalization output",
        "gradient_objective": "mean squared prediction disagreement between the two origins",
        "device": str(dev),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        "python": platform.python_version(),
        "artifacts": {
            "head_contributions": str((output / "head_contributions.csv").resolve()),
            "gradient_profile": str((output / "gradient_profile.csv").resolve()),
            "window_metrics": str((output / "window_metrics.csv").resolve()),
        },
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
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--finite-indices", default="0,64,128,256,384,511")
    parser.add_argument("--finite-delta", type=float, default=1e-3)
    parser.add_argument("--finite-max-windows", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
