"""Read-only latent-spectrum smoke test for frozen canonical checkpoints.

The script compares two partition origins on identical ETTh1 test windows. It
uses a hook on the final LayerNorm output, retains only fully observed patch
tokens, and writes per-window spectral and prediction discrepancies.
"""

from __future__ import annotations

import argparse
import csv
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
from src.experiments.canonical.phenomenon_run import (
    BATCH,
    CONTEXT,
    ControlledTransformerV2Family,
    Windows,
    load_dataset,
    partition,
)


def dataset_path_for(data_root: Path) -> Path:
    """Return the ETTh1 path relative to the caller-supplied data root."""
    return Path(data_root) / "ETT-small" / "ETTh1.csv"


def _rank(values: np.ndarray) -> np.ndarray:
    """Stable average-free ranks; sufficient here because discrepancies are continuous."""
    return np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort").astype(np.float64)


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.empty(400, dtype=np.float64)
    for index in range(values.size):
        sample = rng.integers(0, x.size, size=x.size)
        values[index] = _pearson(_rank(x[sample]), _rank(y[sample]))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def _full_token_latents(hidden: torch.Tensor, observed: torch.Tensor, p: int, channels: int) -> np.ndarray:
    """Average channelwise final latents over only complete, non-padded tokens."""
    batch, total = observed.shape
    token_mask = observed.unfold(-1, p, p).sum(-1).eq(p)
    full_counts = token_mask.sum(-1)
    if int(full_counts.min()) != int(full_counts.max()):
        raise RuntimeError("batch windows have unequal complete-token counts")
    latent = hidden.reshape(batch, channels, total // p, hidden.shape[-1])
    rows = [latent[row, :, token_mask[row], :].mean(dim=0) for row in range(batch)]
    return torch.stack(rows).detach().cpu().numpy()


def run(args: argparse.Namespace) -> dict[str, object]:
    data_root = Path(args.data_root).resolve()
    import src.experiments.canonical.phenomenon_run as canonical

    canonical.DATA_ROOT = data_root
    canonical.DATASETS["ETTh1"]["path"] = dataset_path_for(data_root)
    values, _, _, test, split = load_dataset("ETTh1")
    if args.max_windows:
        test = test[: args.max_windows]
    channels = values.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ControlledTransformerV2Family(args.patch_length, channels).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    loader = DataLoader(Windows(values, test), batch_size=BATCH, shuffle=False, pin_memory=device.type == "cuda")
    captured: list[torch.Tensor] = []
    handle = model.norm.register_forward_hook(lambda _module, _inputs, output: captured.append(output.detach()))
    rows: list[dict[str, float]] = []
    complete_token_counts: list[int] = []

    try:
        with torch.inference_mode():
            for x, _target in loader:
                x = x.to(device)
                origin_states: dict[int, tuple[np.ndarray, torch.Tensor, int]] = {}
                for origin in (args.origin_a, args.origin_b):
                    captured.clear()
                    padded, observed = partition(x, origin, args.patch_length)
                    prediction = model(padded, observed)
                    if len(captured) != 1:
                        raise RuntimeError("final-layer hook did not capture exactly one tensor")
                    latent = _full_token_latents(captured[0], observed, args.patch_length, channels)
                    origin_states[origin] = (latent, prediction.detach(), int(observed.unfold(-1, args.patch_length, args.patch_length).sum(-1).eq(args.patch_length).sum(1)[0]))

                first, prediction_a, count_a = origin_states[args.origin_a]
                second, prediction_b, count_b = origin_states[args.origin_b]
                if count_a != count_b:
                    raise RuntimeError("origins do not yield the same complete-token count")
                complete_token_counts.append(count_a)
                spectrum = summarize_latent_pair(first, second)
                prediction_mse = (prediction_a - prediction_b).square().mean(dim=(1, 2)).cpu().numpy()
                for index in range(first.shape[0]):
                    rows.append({
                        "window": float(len(rows) + index),
                        "spectral_l1": float(spectrum["spectral_l1"][index]),
                        "dc_l1": float(spectrum["dc_l1"][index]),
                        "non_dc_low_l1": float(spectrum["non_dc_low_l1"][index]),
                        "low_band_l1": float(spectrum["low_band_l1"][index]),
                        "mid_band_l1": float(spectrum["mid_band_l1"][index]),
                        "high_band_l1": float(spectrum["high_band_l1"][index]),
                        "prediction_mse": float(prediction_mse[index]),
                    })
    finally:
        handle.remove()

    spectral = np.array([row["spectral_l1"] for row in rows])
    prediction = np.array([row["prediction_mse"] for row in rows])
    spearman = _pearson(_rank(spectral), _rank(prediction))
    ci_low, ci_high = _bootstrap_spearman_ci(spectral, prediction, np.random.default_rng(0))
    result = {
        "dataset": "ETTh1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "origins": [args.origin_a, args.origin_b],
        "windows": len(rows),
        "complete_tokens_per_window": sorted(set(complete_token_counts)),
        "raw_input_max_abs_difference": 0.0,
        "median_spectral_l1": float(np.median(spectral)),
        "median_prediction_mse": float(np.median(prediction)),
        "dc": {
            "median_l1": float(np.median([row["dc_l1"] for row in rows])),
            "spearman_vs_prediction": _pearson(_rank(np.array([row["dc_l1"] for row in rows])), _rank(prediction)),
        },
        "non_dc_low": {
            "median_l1": float(np.median([row["non_dc_low_l1"] for row in rows])),
            "spearman_vs_prediction": _pearson(_rank(np.array([row["non_dc_low_l1"] for row in rows])), _rank(prediction)),
        },
        "spearman_spectral_vs_prediction": spearman,
        "spearman_bootstrap_95_ci": [ci_low, ci_high],
        "supportive_signal": bool(np.median(spectral) > 1e-7 and spearman >= 0.20 and ci_low > 0.10),
        "interpretation": "Exploratory representation-level association only; not a causal mediation test.",
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "window_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing ETT-small/ETTh1.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin-a", type=int, default=0)
    parser.add_argument("--origin-b", type=int, default=6)
    parser.add_argument("--patch-length", type=int, default=12)
    parser.add_argument("--max-windows", type=int, default=0, help="0 evaluates the complete test split")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
