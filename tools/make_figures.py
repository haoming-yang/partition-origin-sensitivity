from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/aggregate.csv")
    parser.add_argument("--output", default="results/figures")
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for figure generation") from exc
    rows = list(csv.DictReader(Path(args.input).open("r", newline="", encoding="utf8")))
    groups = {}
    for row in rows:
        if row.get("dataset"):
            groups.setdefault(row["dataset"], []).append(float(row["G_origin"]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    names = list(groups)
    means = [sum(groups[name]) / len(groups[name]) for name in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(names, means, marker="o", color="#24527a")
    ax.set_ylabel("Origin gap (%)")
    ax.set_xlabel("Dataset")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "origin_gap_summary.pdf")
    fig.savefig(output / "origin_gap_summary.png", dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
