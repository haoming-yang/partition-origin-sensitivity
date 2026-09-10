"""Read-only spectral diagnostic for frozen protocol-adapted PatchTST checkpoints.

The script preserves the PatchTST audit protocol: raw ETTh1 inputs, the
mask-aware outer adapter, and the official model components.  It compares
origins 0 and 6 on identical test windows, retaining only complete outer patch
tokens and excluding the official embedder's replication-padded final token.
"""

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

from src.analysis.latent_spectral import summarize_latent_pair
from src.utils.artifact_naming import artifact_path, metric_filename


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
from src.analysis.patchtst import full_patchtst_token_latents
from src.training import runner
from tools.analyze_latent_spectrum import (
    _bootstrap_spearman_ci,
    _pearson,
    _rank,
    dataset_path_for,
)


def _encode_and_predict(
    model: runner.OfficialPatchTSTAdapter,
    padded: torch.Tensor,
    observed: torch.Tensor,
    patch_length: int,
) -> tuple[np.ndarray, torch.Tensor]:
    """Mirror the adapter forward pass while retaining the final encoder output."""
    safe = padded * observed.unsqueeze(-1).to(padded.dtype)
    count = observed.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1).to(padded.dtype)
    means = safe.sum(1, keepdim=True) / count
    centered = (safe - means) * observed.unsqueeze(-1).to(padded.dtype)
    stdev = torch.sqrt((centered * centered).sum(1, keepdim=True) / count + 1e-5)
    normalized = centered / stdev
    encoded, n_vars = model.inner.patch_embedding(normalized.permute(0, 2, 1))
    encoded, _ = model.inner.encoder(encoded)
    latent = full_patchtst_token_latents(encoded, observed, patch_length, n_vars)
    head_input = torch.reshape(encoded, (-1, n_vars, encoded.shape[-2], encoded.shape[-1]))
    head_input = head_input.permute(0, 1, 3, 2)
    prediction = model.inner.head(head_input).permute(0, 2, 1)
    prediction = prediction * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
    return latent.detach().cpu().numpy(), prediction.detach()


def run(args: argparse.Namespace) -> dict[str, object]:
    data_root = Path(args.data_root).resolve()
    runner.DATASET_PATHS["ETTh1"] = dataset_path_for(data_root)
    data = runner.load_data("ETTh1", args.horizon)
    if args.max_windows:
        test = data.test[: args.max_windows]
    else:
        test = data.test
    device = runner.device()
    model = runner.OfficialPatchTSTAdapter(
        args.context, args.horizon, args.patch_length, args.stride, data.channels
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    loader = DataLoader(
        runner.Windows(data.raw, test, args.context, args.horizon),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    rows: list[dict[str, float]] = []
    complete_counts: list[int] = []
    with torch.inference_mode():
        for x, _target in loader:
            x = x.to(device, non_blocking=True)
            states: dict[int, tuple[np.ndarray, torch.Tensor, int]] = {}
            for origin in (args.origin_a, args.origin_b):
                padded, observed = runner.partition(
                    x, origin, args.context, args.patch_length, args.stride
                )
                latent, prediction = _encode_and_predict(model, padded, observed, args.patch_length)
                token_count = int(
                    observed.reshape(observed.shape[0], -1, args.patch_length)
                    .all(dim=-1)
                    .sum(dim=-1)[0]
                )
                states[origin] = (latent, prediction, token_count)
            first, prediction_a, count_a = states[args.origin_a]
            second, prediction_b, count_b = states[args.origin_b]
            if count_a != count_b:
                raise RuntimeError("origins do not yield the same complete-token count")
            complete_counts.append(count_a)
            spectrum = summarize_latent_pair(first, second)
            prediction_mse = (prediction_a - prediction_b).square().mean(dim=(1, 2)).cpu().numpy()
            for index in range(first.shape[0]):
                rows.append(
                    {
                        "seed": "" if args.seed is None else int(args.seed),
                        "window": float(len(rows)),
                        "spectral_l1": float(spectrum["spectral_l1"][index]),
                        "dc_l1": float(spectrum["dc_l1"][index]),
                        "non_dc_low_l1": float(spectrum["non_dc_low_l1"][index]),
                        "low_band_l1": float(spectrum["low_band_l1"][index]),
                        "mid_band_l1": float(spectrum["mid_band_l1"][index]),
                        "high_band_l1": float(spectrum["high_band_l1"][index]),
                        "prediction_mse": float(prediction_mse[index]),
                    }
                )
    spectral = np.array([row["spectral_l1"] for row in rows])
    prediction = np.array([row["prediction_mse"] for row in rows])
    rng = np.random.default_rng(0)
    ci_low, ci_high = _bootstrap_spearman_ci(spectral, prediction, rng)
    bands = {
        key: {
            "median_l1": float(np.median([row[key] for row in rows])),
            "spearman_vs_prediction": _pearson(
                _rank(np.array([row[key] for row in rows])), _rank(prediction)
            ),
        }
        for key in ("dc_l1", "non_dc_low_l1", "low_band_l1", "mid_band_l1", "high_band_l1")
    }
    result = {
        "dataset": "ETTh1",
        "seed": args.seed,
        "checkpoint": "frozen-checkpoint.pt",
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "protocol": "official_Time-Series-Library_PatchTST_with_mask_aware_outer_adapter",
        "input_scale": "raw ETTh1 values",
        "origins": [args.origin_a, args.origin_b],
        "windows": len(rows),
        "complete_tokens_per_window": sorted(set(complete_counts)),
        "raw_input_max_abs_difference": 0.0,
        "median_spectral_l1": float(np.median(spectral)),
        "median_prediction_mse": float(np.median(prediction)),
        "spearman_spectral_vs_prediction": _pearson(_rank(spectral), _rank(prediction)),
        "spearman_bootstrap_95_ci": [ci_low, ci_high],
        "bands": bands,
        "interpretation": "Protocol-bound representation-level association only; not a causal mediation test and not numerically comparable to standardized controlled-model MSE.",
    }
    output = artifact_path(
        Path(args.output).resolve(),
        args.seed,
        metric_filename(
            "patchtst_latent_spectrum",
            origin_a=args.origin_a,
            origin_b=args.origin_b,
            patch_length=args.patch_length,
            stride=args.stride,
            context=args.context,
            horizon=args.horizon,
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
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing ETT-small/ETTh1.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Training seed recorded for this frozen checkpoint")
    parser.add_argument("--origin-a", type=int, default=0)
    parser.add_argument("--origin-b", type=int, default=6)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--patch-length", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
