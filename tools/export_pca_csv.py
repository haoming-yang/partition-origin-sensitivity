from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.utils.output_paths import validate_generated_output


def export_pca_csv(source: Path, output: Path) -> tuple[Path, Path]:
    selected_path = validate_generated_output(output / "selected_windows.csv")
    coordinates_path = validate_generated_output(output / "pca_coordinates.csv")
    with np.load(source, allow_pickle=False) as data:
        window_ids = data["window_ids"]
        strata = data["strata"]
        forecast_mse = data["forecast_mse"]
        count = len(window_ids)
        if len(strata) != count or len(forecast_mse) != count:
            raise ValueError("PCA window metadata have inconsistent lengths")
        layers = []
        for layer in ("layer0", "layer1", "layer2"):
            points = data[layer + "_coordinates"]
            origins = data[layer + "_origins"]
            if points.shape != (2 * count, 3) or origins.shape != (2 * count,):
                raise ValueError(f"Invalid paired coordinates for {layer}")
            layers.append((layer, points, origins))
        output.mkdir(parents=True, exist_ok=True)
        with selected_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("window_id", "stratum", "forecast_mse"))
            for index in range(count):
                writer.writerow((int(window_ids[index]), int(strata[index]), float(forecast_mse[index])))
        with coordinates_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("layer", "window_id", "origin", "stratum", "forecast_mse", "pc1", "pc2", "pc3"))
            for layer, points, origins in layers:
                for index in range(2 * count):
                    window = index % count
                    writer.writerow((layer, int(window_ids[window]), int(origins[index]),
                                     int(strata[window]), float(forecast_mse[window]),
                                     *[float(value) for value in points[index]]))
    return selected_path, coordinates_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the saved paired PCA NPZ as readable CSVs")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/pca_csv")
    args = parser.parse_args()
    selected, coordinates = export_pca_csv(args.input, args.output)
    print(selected)
    print(coordinates)


if __name__ == "__main__":
    main()
