"""Render frequency-wise and layer-wise summaries from frozen latent artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


FREQUENCY_COLUMNS = {
    "DC": "dc_l1",
    "Non-DC low": "non_dc_low_l1",
    "Mid": "mid_band_l1",
    "High": "high_band_l1",
}
LAYER_COLUMNS = {
    "Post-position input": "layer0_spectral_l1",
    "First encoder block": "layer1_spectral_l1",
    "Final normalized output": "layer2_spectral_l1",
}
SEED_COLORS = ("#0072B2", "#D55E00", "#009E73")


def _seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", str(path))
    if match:
        return int(match.group(1))
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        raise ValueError(f"could not infer a seed from {path}")
    return int(matches[-1])


def _read_rows(path: Path) -> list[dict[str, float]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [{name: float(value) for name, value in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"artifact contains no rows: {path}")
    return rows


def _rank(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort").astype(np.float64)


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    ranked_first = _rank(first)
    ranked_second = _rank(second)
    if ranked_first.size < 2 or np.std(ranked_first) == 0.0 or np.std(ranked_second) == 0.0:
        raise ValueError("Spearman correlation requires nonconstant paired values")
    return float(np.corrcoef(ranked_first, ranked_second)[0, 1])


def _mean_and_sample_sd(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "seed_values": [float(value) for value in array],
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
    }


def summarize_latent_artifacts(
    spectrum_paths: list[Path],
    layer_paths: list[Path],
) -> dict[str, object]:
    """Compute per-seed frequency associations and layer median discrepancies."""
    if len(spectrum_paths) != 3 or len(layer_paths) != 3:
        raise ValueError("exactly three spectrum and three layer artifacts are required")
    spectrum_by_seed = {_seed_from_path(Path(path)): _read_rows(Path(path)) for path in spectrum_paths}
    layer_by_seed = {_seed_from_path(Path(path)): _read_rows(Path(path)) for path in layer_paths}
    if set(spectrum_by_seed) != set(layer_by_seed) or len(spectrum_by_seed) != 3:
        raise ValueError("spectrum and layer artifacts must contain the same three unique seeds")
    seeds = sorted(spectrum_by_seed)
    frequency = {}
    for label, column in FREQUENCY_COLUMNS.items():
        values = []
        for seed in seeds:
            rows = spectrum_by_seed[seed]
            values.append(_spearman(
                np.array([row[column] for row in rows]),
                np.array([row["prediction_mse"] for row in rows]),
            ))
        frequency[label] = _mean_and_sample_sd(values)
    layers = {}
    for label, column in LAYER_COLUMNS.items():
        values = []
        for seed in seeds:
            rows = layer_by_seed[seed]
            values.append(float(np.median([row[column] for row in rows])))
        layers[label] = _mean_and_sample_sd(values)
    return {"seeds": seeds, "frequency": frequency, "layers": layers}


def render_latent_summary_figure(summary: dict[str, object], output: Path) -> None:
    """Render the two-panel publication figure as a vector PDF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = summary["seeds"]
    frequency = summary["frequency"]
    layers = summary["layers"]
    if len(seeds) != 3 or list(frequency) != list(FREQUENCY_COLUMNS) or list(layers) != list(LAYER_COLUMNS):
        raise ValueError("summary must preserve the configured three replicates and panel orders")
    figure, (frequency_axis, layer_axis) = plt.subplots(
        1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": (1.02, 1.0), "wspace": 0.42}
    )
    accent = "#174A6E"
    individual = "#B8C3CC"
    frequency_positions = np.arange(len(frequency), dtype=float)
    for seed_index, seed in enumerate(seeds):
        values = [frequency[label]["seed_values"][seed_index] for label in frequency]
        frequency_axis.scatter(
            values, frequency_positions,
            s=12, color=individual, edgecolors="white", linewidths=0.35,
            alpha=0.92, zorder=2,
        )
    frequency_means = np.array([frequency[label]["mean"] for label in frequency])
    frequency_sd = np.array([frequency[label]["sample_sd"] for label in frequency])
    frequency_axis.errorbar(
        frequency_means, frequency_positions, xerr=frequency_sd,
        fmt="none", ecolor=accent, elinewidth=1.1, capsize=2.0, capthick=1.1, zorder=3,
    )
    frequency_axis.scatter(
        frequency_means, frequency_positions, s=25, color=accent,
        edgecolors="white", linewidths=0.55, zorder=4,
    )
    frequency_axis.set_yticks(
        frequency_positions,
        ["DC  ($m=0$)", "Non-DC low  ($m=1$--$6$)", "Mid  ($m=7$--$13$)", "High  ($m=14$--$21$)"],
    )
    frequency_axis.set_xlabel("Spearman $\\rho$", fontsize=7.5)
    frequency_axis.set_ylabel("")
    frequency_axis.set_title("(a) Frequency association", loc="left", fontweight="bold", fontsize=8.5)
    frequency_axis.set_xlim(0.0, 0.65)
    frequency_axis.set_xticks((0.0, 0.2, 0.4, 0.6))
    frequency_axis.invert_yaxis()
    layer_positions = np.arange(len(layers), dtype=float)
    for seed_index, seed in enumerate(seeds):
        values = [layers[label]["seed_values"][seed_index] for label in layers]
        layer_axis.scatter(
            layer_positions, values, s=13, color=individual,
            edgecolors="white", linewidths=0.35, alpha=0.95, zorder=2,
        )
        layer_axis.plot(layer_positions, values, color=individual, linewidth=0.8, alpha=0.85, zorder=1)
    layer_means = np.array([layers[label]["mean"] for label in layers])
    layer_sd = np.array([layers[label]["sample_sd"] for label in layers])
    layer_axis.errorbar(
        layer_positions, layer_means, yerr=layer_sd,
        fmt="none", ecolor=accent, elinewidth=1.1, capsize=2.0, capthick=1.1, zorder=3,
    )
    layer_axis.plot(layer_positions, layer_means, color=accent, linewidth=1.8, zorder=4)
    layer_axis.scatter(
        layer_positions, layer_means, s=25, color=accent,
        edgecolors="white", linewidths=0.55, zorder=5,
    )
    layer_axis.set_xticks(layer_positions, ["Post-position\ninput", "First\nblock", "Final\nnorm."])
    layer_axis.set_xlabel("Encoder location", fontsize=7.5)
    layer_axis.set_ylabel("Median spectral discrepancy", fontsize=7.5)
    layer_axis.set_title("(b) Across encoder stages", loc="left", fontweight="bold", fontsize=8.5)
    layer_axis.set_ylim(0.092, 0.146)
    layer_axis.set_yticks((0.10, 0.12, 0.14))
    for axis in (frequency_axis, layer_axis):
        axis.set_facecolor("white")
        axis.grid(axis="both", color="#D9DDE1", linewidth=0.45, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=6.8)
    figure.text(
        0.5, 0.985, "Dark marks and line: mean $\\pm$ SD   ·   light marks and lines: individual checkpoints",
        ha="center", va="top", fontsize=6.5, color="#5E6972",
    )
    figure.subplots_adjust(top=0.84, bottom=0.24, left=0.12, right=0.985)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", action="append", type=Path, required=True)
    parser.add_argument("--layers", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = summarize_latent_artifacts(arguments.spectrum, arguments.layers)
    render_latent_summary_figure(result, arguments.output)
    arguments.summary_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.summary_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
