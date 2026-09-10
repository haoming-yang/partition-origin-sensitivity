"""Permutation significance test for saved latent-spectrum window diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.latent_spectral import permutation_spearman_test


def read_window_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read paired spectral and forecast discrepancies from one saved run."""
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"no rows in {path}")
    spectral = np.array([float(row["spectral_l1"]) for row in rows], dtype=np.float64)
    forecast = np.array([float(row["prediction_mse"]) for row in rows], dtype=np.float64)
    return spectral, forecast


def seed_from_artifact(path: Path) -> int | None:
    """Read a training seed from the new parent directory or old filename."""
    for candidate in (path.parent.name, path.name):
        match = re.search(r"(?:seed|replicate)(\d+)", candidate)
        if match:
            return int(match.group(1))
    return None


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, path in enumerate(args.input):
        spectral, forecast = read_window_metrics(path)
        result = permutation_spearman_test(
            spectral,
            forecast,
            permutations=args.permutations,
            seed=args.seed + index,
        )
        try:
            public_input = str(path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            public_input = path.name
        records.append({"input": public_input.replace("\\", "/"), "seed": seed_from_artifact(path), "windows": int(spectral.size), **result})
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", "--permutation-seed", dest="seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
