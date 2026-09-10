"""Summarize frozen latent CSV artifacts without rendering figures."""

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


def _seed_from_path(path: Path) -> int:
    for candidate in (path.parent.name, path.name):
        match = re.search(r"(?:seed|replicate)(\d+)", candidate)
        if match:
            return int(match.group(1))
    raise ValueError(f"seed is missing from artifact path: {path}")


def _read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed: dict[str, object] = {}
            for name, value in row.items():
                parsed[name] = None if name == "seed" and value == "" else float(value)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"artifact contains no rows: {path}")
    return rows


def _seed_from_rows(path: Path, rows: list[dict[str, object]]) -> int:
    values = {row.get("seed") for row in rows if row.get("seed") not in (None, "")}
    if len(values) > 1:
        raise ValueError(f"artifact contains multiple seed values: {path}")
    if values:
        seed = int(next(iter(values)))
        if seed != _seed_from_path(path):
            raise ValueError(f"CSV seed and artifact directory disagree: {path}")
        return seed
    return _seed_from_path(path)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        raise ValueError("Spearman correlation requires nonconstant paired values")
    return float(np.corrcoef(_rank(first), _rank(second))[0, 1])


def _aggregate(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "replicate_values": [float(value) for value in array],
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
    }


def summarize_latent_artifacts(spectrum_paths: list[Path], layer_paths: list[Path]) -> dict[str, object]:
    if len(spectrum_paths) != 3 or len(layer_paths) != 3:
        raise ValueError("exactly three spectrum and three layer artifacts are required")
    spectrum_by_seed = {}
    for path in spectrum_paths:
        rows = _read_rows(path)
        spectrum_by_seed[_seed_from_rows(path, rows)] = rows
    layer_by_seed = {}
    for path in layer_paths:
        rows = _read_rows(path)
        layer_by_seed[_seed_from_rows(path, rows)] = rows
    if set(spectrum_by_seed) != set(layer_by_seed) or len(spectrum_by_seed) != 3:
        raise ValueError("spectrum and layer artifacts must contain the same three unique seeds")
    seeds = sorted(spectrum_by_seed)
    frequency = {}
    for label, column in FREQUENCY_COLUMNS.items():
        frequency[label] = _aggregate([
            _spearman(
                np.asarray([row[column] for row in spectrum_by_seed[seed]], dtype=np.float64),
                np.asarray([row["prediction_mse"] for row in spectrum_by_seed[seed]], dtype=np.float64),
            )
            for seed in seeds
        ])
    layers = {}
    for label, column in LAYER_COLUMNS.items():
        layers[label] = _aggregate([
            float(np.median([row[column] for row in layer_by_seed[seed]])) for seed in seeds
        ])
    return {"seeds": seeds, "frequency": frequency, "layers": layers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", action="append", type=Path, required=True)
    parser.add_argument("--layers", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_latent_artifacts(args.spectrum, args.layers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
