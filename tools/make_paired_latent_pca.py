"""Create a qualitative paired-PCA figure from a frozen ETTh1 checkpoint."""

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

from src.analysis.layerwise import dense_token_tensor
from src.analysis.paired_latent_pca import (
    PairedPCA,
    paired_pca_coordinates,
    select_tercile_windows,
    validate_paired_latents,
)
from tools.analyze_latent_layers import LAYER_NAMES, _expanded_valid_tokens, _register_hooks
from tools.analyze_latent_spectrum import _full_token_latents, dataset_path_for
from src.experiments.canonical.phenomenon_run import (
    BATCH,
    ControlledTransformerV2Family,
    Windows,
    load_dataset,
    partition,
)

PANEL_TITLES = {
    "layer0": "(a) Post-position embedding",
    "layer1": "(b) First encoder block",
    "layer2": "(c) Final normalized latent",
}


def render_paired_pca_figure(
    records: dict[str, PairedPCA],
    forecast_mse: np.ndarray,
    output: Path,
) -> None:
    """Render three layer-specific, paired 3D PCA panels as a vector PDF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable

    if set(records) != set(LAYER_NAMES):
        raise ValueError("records must contain exactly the three configured latent layers")
    disagreement = np.asarray(forecast_mse, dtype=np.float64).reshape(-1)
    if disagreement.size != 72 or not np.isfinite(disagreement).all() or np.any(disagreement <= 0):
        raise ValueError("forecast MSE must contain 72 positive finite values")
    vmin = float(disagreement.min())
    vmax = float(disagreement.max())
    if vmin == vmax:
        vmin *= 0.9
        vmax *= 1.1
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")
    figure = plt.figure(figsize=(11.4, 3.55), constrained_layout=True)
    axes = []
    for panel_index, layer in enumerate(LAYER_NAMES, start=1):
        record = records[layer]
        if record.coordinates.shape != (144, 3) or record.window_ids.size != 144:
            raise ValueError("each layer must contain 144 projected rows")
        if not np.array_equal(record.window_ids[:72], record.window_ids[72:]):
            raise ValueError("each projected layer must preserve 72 paired window IDs")
        axis = figure.add_subplot(1, 3, panel_index, projection="3d")
        axes.append(axis)
        for index in range(72):
            color = cmap(norm(disagreement[index]))
            first = record.coordinates[index]
            second = record.coordinates[index + 72]
            axis.plot(
                (first[0], second[0]),
                (first[1], second[1]),
                (first[2], second[2]),
                color=color,
                alpha=0.28,
                linewidth=0.45,
                zorder=1,
            )
        origin_zero = record.coordinates[:72]
        origin_six = record.coordinates[72:]
        colors = cmap(norm(disagreement))
        axis.scatter(
            origin_zero[:, 0], origin_zero[:, 1], origin_zero[:, 2],
            c=colors, marker="o", s=11, edgecolors="none", alpha=0.84, depthshade=False, zorder=3,
        )
        axis.scatter(
            origin_six[:, 0], origin_six[:, 1], origin_six[:, 2],
            facecolors="white", edgecolors=colors, marker="o", s=17, linewidths=0.75,
            alpha=0.92, depthshade=False, zorder=4,
        )
        axis.set_title(PANEL_TITLES[layer], fontsize=8.5, pad=7)
        axis.set_xlabel("PC1", fontsize=7.5, labelpad=-5)
        axis.set_ylabel("PC2", fontsize=7.5, labelpad=-5)
        axis.set_zlabel("PC3", fontsize=7.5, labelpad=-4)
        axis.tick_params(labelsize=5.5, pad=-1)
        axis.view_init(elev=20, azim=-55)
        axis.set_facecolor("white")
        axis.xaxis.pane.fill = False
        axis.yaxis.pane.fill = False
        axis.zaxis.pane.fill = False
        axis.grid(True, alpha=0.08, linewidth=0.35)
        for axis_dimension in (axis.xaxis, axis.yaxis, axis.zaxis):
            axis_dimension._axinfo["grid"]["color"] = (0.70, 0.70, 0.70, 0.35)
            axis_dimension._axinfo["grid"]["linewidth"] = 0.35
    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#595959", markeredgecolor="none", markersize=5, label="Origin $r=0$"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#595959", markersize=5, label="Origin $r=6$"),
        ],
        loc="upper left", fontsize=7, frameon=False, handletextpad=0.3, borderpad=0.2,
    )
    colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.72, pad=0.045)
    colorbar.set_label("Forecast disagreement magnitude\n(per-window MSE)", fontsize=8)
    colorbar.ax.tick_params(labelsize=6)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)


def _pooled_latents(latents: np.ndarray) -> np.ndarray:
    return np.asarray(latents, dtype=np.float64).mean(axis=1)


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
    full_latents: dict[int, dict[str, list[np.ndarray]]] = {
        args.origin_a: {layer: [] for layer in LAYER_NAMES},
        args.origin_b: {layer: [] for layer in LAYER_NAMES},
    }
    forecast_mse_batches: list[np.ndarray] = []
    complete_counts: list[int] = []
    try:
        with torch.inference_mode():
            for x, _target in loader:
                x = x.to(device)
                states: dict[int, tuple[dict[str, np.ndarray], torch.Tensor, int]] = {}
                for origin in (args.origin_a, args.origin_b):
                    captured.clear()
                    padded, observed = partition(x, origin, args.patch_length)
                    prediction = model(padded, observed)
                    if set(captured) != set(LAYER_NAMES):
                        raise RuntimeError(f"incomplete hook capture: {sorted(captured)}")
                    valid = _expanded_valid_tokens(observed, args.patch_length, channels)
                    latents = {
                        layer: _full_token_latents(
                            dense_token_tensor(captured[layer], valid), observed, args.patch_length, channels
                        )
                        for layer in LAYER_NAMES
                    }
                    count = int(observed.unfold(-1, args.patch_length, args.patch_length).sum(-1).eq(args.patch_length).sum(1)[0])
                    states[origin] = (latents, prediction.detach(), count)
                first, prediction_zero, count_zero = states[args.origin_a]
                second, prediction_six, count_six = states[args.origin_b]
                if count_zero != count_six:
                    raise RuntimeError("origins do not yield the same complete-token count")
                window_ids = np.arange(sum(batch.size for batch in forecast_mse_batches), sum(batch.size for batch in forecast_mse_batches) + prediction_zero.shape[0], dtype=np.int64)
                for layer in LAYER_NAMES:
                    validate_paired_latents(window_ids, first[layer], second[layer], complete_token_count=count_zero)
                    full_latents[args.origin_a][layer].append(_pooled_latents(first[layer]))
                    full_latents[args.origin_b][layer].append(_pooled_latents(second[layer]))
                forecast_mse_batches.append((prediction_zero - prediction_six).square().mean(dim=(1, 2)).cpu().numpy())
                complete_counts.append(count_zero)
    finally:
        for handle in handles:
            handle.remove()

    forecast_mse = np.concatenate(forecast_mse_batches)
    all_window_ids = np.arange(forecast_mse.size, dtype=np.int64)
    selection = select_tercile_windows(
        all_window_ids, forecast_mse, per_tercile=args.per_tercile, seed=args.sampling_seed
    )
    selected_indices = selection.window_ids
    selected_mse = forecast_mse[selected_indices]
    records = {}
    saved_arrays: dict[str, np.ndarray] = {
        "window_ids": selection.window_ids,
        "strata": selection.strata,
        "forecast_mse": selected_mse,
    }
    coordinate_rows: list[dict[str, object]] = []
    for layer in LAYER_NAMES:
        first = np.concatenate(full_latents[args.origin_a][layer], axis=0)[selected_indices]
        second = np.concatenate(full_latents[args.origin_b][layer], axis=0)[selected_indices]
        record = paired_pca_coordinates(selection.window_ids, first, second)
        records[layer] = record
        saved_arrays[f"{layer}_origin_{args.origin_a}"] = first
        saved_arrays[f"{layer}_origin_{args.origin_b}"] = second
        saved_arrays[f"{layer}_coordinates"] = record.coordinates
        saved_arrays[f"{layer}_origins"] = record.origins
        for index, (window, origin, point) in enumerate(zip(record.window_ids, record.origins, record.coordinates)):
            source_index = index % selection.window_ids.size
            coordinate_rows.append({
                "layer": layer,
                "window_id": int(window),
                "origin": int(origin),
                "stratum": int(selection.strata[source_index]),
                "forecast_mse": float(selected_mse[source_index]),
                "pc1": float(point[0]),
                "pc2": float(point[1]),
                "pc3": float(point[2]),
            })
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "pooled_latents_and_pca.npz", **saved_arrays)
    with (output / "selected_windows.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["window_id", "stratum", "forecast_mse"])
        writer.writeheader()
        writer.writerows({"window_id": int(window), "stratum": int(stratum), "forecast_mse": float(mse)} for window, stratum, mse in zip(selection.window_ids, selection.strata, selected_mse))
    with (output / "pca_coordinates.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(coordinate_rows[0]))
        writer.writeheader()
        writer.writerows(coordinate_rows)
    figure_output = Path(args.figure_output).resolve()
    render_paired_pca_figure(records, selected_mse, figure_output)
    summary = {
        "dataset": "ETTh1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "origins": [args.origin_a, args.origin_b],
        "test_windows": int(forecast_mse.size),
        "selected_windows": int(selection.window_ids.size),
        "per_tercile": int(args.per_tercile),
        "sampling_seed": int(args.sampling_seed),
        "complete_tokens_per_window": sorted(set(complete_counts)),
        "layers": list(LAYER_NAMES),
        "figure": str(figure_output),
        "interpretation": "Qualitative, layer-specific PCA projections of pooled complete-token representations; not a full-dimensional distance or causal analysis.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    parser.add_argument("--origin-a", type=int, default=0)
    parser.add_argument("--origin-b", type=int, default=6)
    parser.add_argument("--patch-length", type=int, default=12)
    parser.add_argument("--per-tercile", type=int, default=24)
    parser.add_argument("--sampling-seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
