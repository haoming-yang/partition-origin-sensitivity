"""Layerwise, read-only latent-spectrum profile for frozen canonical checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.layerwise import dense_token_tensor, summarize_layer_pairs
from src.utils.artifact_naming import artifact_path, metric_filename
from tools.analyze_latent_spectrum import (
    _bootstrap_spearman_ci,
    _full_token_latents,
    _pearson,
    _rank,
    dataset_path_for,
)
from src.experiments.canonical.phenomenon_run import (
    BATCH,
    ControlledTransformerV2Family,
    Windows,
    load_dataset,
    partition,
)

LAYER_NAMES = ("layer0", "layer1", "layer2")


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _register_hooks(model: ControlledTransformerV2Family, captured: dict[str, torch.Tensor]):
    """Capture post-position input, first-block output, and final-normalized output."""
    return [
        model.encoder.layers[0].register_forward_pre_hook(
            lambda _module, inputs: captured.__setitem__("layer0", inputs[0].detach())
        ),
        model.encoder.layers[0].register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("layer1", output.detach())
        ),
        model.norm.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("layer2", output.detach())
        ),
    ]


def _expanded_valid_tokens(observed: torch.Tensor, patch_length: int, channels: int) -> torch.Tensor:
    valid = observed.unfold(-1, patch_length, patch_length).sum(-1).gt(0)
    return valid[:, None, :].expand(-1, channels, -1).reshape(-1, valid.shape[-1])


def run(args: argparse.Namespace) -> dict[str, object]:
    data_root = Path(args.data_root).resolve()
    import src.experiments.canonical.phenomenon_run as canonical

    canonical.DATA_ROOT = data_root
    canonical.DATASETS["ETTh1"]["path"] = dataset_path_for(data_root)
    values, _, _, test, _ = load_dataset("ETTh1")
    channels = values.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ControlledTransformerV2Family(args.patch_length, channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval()
    loader = DataLoader(Windows(values, test), batch_size=BATCH, shuffle=False, pin_memory=device.type == "cuda")
    captured: dict[str, torch.Tensor] = {}
    handles = _register_hooks(model, captured)
    rows: list[dict[str, float]] = []
    complete_token_counts: list[int] = []
    try:
        with torch.inference_mode():
            for x, _target in loader:
                x = x.to(device)
                origin_states: dict[int, tuple[dict[str, np.ndarray], torch.Tensor, int]] = {}
                for origin in (args.origin_a, args.origin_b):
                    captured.clear()
                    padded, observed = partition(x, origin, args.patch_length)
                    prediction = model(padded, observed)
                    if set(captured) != set(LAYER_NAMES):
                        raise RuntimeError(f"incomplete hook capture: {sorted(captured)}")
                    expanded_valid = _expanded_valid_tokens(observed, args.patch_length, channels)
                    latents = {
                        name: _full_token_latents(dense_token_tensor(captured[name], expanded_valid), observed, args.patch_length, channels)
                        for name in LAYER_NAMES
                    }
                    full_count = int(observed.unfold(-1, args.patch_length, args.patch_length).sum(-1).eq(args.patch_length).sum(1)[0])
                    origin_states[origin] = (latents, prediction.detach(), full_count)

                first, prediction_a, count_a = origin_states[args.origin_a]
                second, prediction_b, count_b = origin_states[args.origin_b]
                if count_a != count_b:
                    raise RuntimeError("origins do not yield the same complete-token count")
                complete_token_counts.append(count_a)
                layer_summary = summarize_layer_pairs({name: (first[name], second[name]) for name in LAYER_NAMES})
                prediction_mse = (prediction_a - prediction_b).square().mean(dim=(1, 2)).cpu().numpy()
                for index in range(prediction_mse.size):
                    row = {
                        "seed": "" if args.seed is None else int(args.seed),
                        "window": float(len(rows)),
                        "prediction_mse": float(prediction_mse[index]),
                    }
                    row.update({f"{name}_spectral_l1": float(layer_summary[name]["spectral_l1"][index]) for name in LAYER_NAMES})
                    rows.append(row)
    finally:
        for handle in handles:
            handle.remove()

    prediction = np.array([row["prediction_mse"] for row in rows])
    summary_layers = {}
    for name in LAYER_NAMES:
        distance = np.array([row[f"{name}_spectral_l1"] for row in rows])
        spearman = _pearson(_rank(distance), _rank(prediction))
        ci_low, ci_high = _bootstrap_spearman_ci(distance, prediction, np.random.default_rng(0))
        summary_layers[name] = {
            "median_spectral_l1": float(np.median(distance)),
            "spearman_spectral_vs_prediction": spearman,
            "spearman_bootstrap_95_ci": [ci_low, ci_high],
        }
    result = {
        "dataset": "ETTh1",
        "seed": args.seed,
        "checkpoint": "frozen-checkpoint.pt",
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "origins": [args.origin_a, args.origin_b],
        "windows": len(rows),
        "complete_tokens_per_window": sorted(set(complete_token_counts)),
        "layers": summary_layers,
        "interpretation": "Layer 0 is the post-position encoder input; Layers 1 and 2 are the first-block and final-normalized outputs. This is associative, not causal mediation evidence.",
    }
    output = artifact_path(
        Path(args.output).resolve(),
        args.seed,
        metric_filename(
            "latent_layers",
            origin_a=args.origin_a,
            origin_b=args.origin_b,
            patch_length=args.patch_length,
            context=512,
            horizon=96,
        ),
    )
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_name(output.stem + "_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Training seed recorded for this frozen checkpoint")
    parser.add_argument("--origin-a", type=int, default=0)
    parser.add_argument("--origin-b", type=int, default=6)
    parser.add_argument("--patch-length", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
