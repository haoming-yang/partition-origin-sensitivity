import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def normalized(values):
    values = values.astype(float)
    total = values.sum()
    return values / total if total else values


def read_jacobian(path):
    frame = pd.read_csv(path)
    columns = ["origin_a_jacobian_energy", "origin_b_jacobian_energy", "origin_difference_jacobian_energy"]
    return frame["time_index"].to_numpy(), {column: normalized(frame[column]) for column in columns}


def read_contributions(path):
    frame = pd.read_csv(path)
    return frame["token"].to_numpy(), normalized(frame["mean_abs_contribution"])


def draw_row(axes, label, jacobian_path, contribution_path, color):
    x, profiles = read_jacobian(jacobian_path)
    token, mass = read_contributions(contribution_path)
    axes[0].plot(x, profiles["origin_a_jacobian_energy"], color="#0072B2", linewidth=1.1, label="origin 0")
    axes[0].plot(x, profiles["origin_b_jacobian_energy"], color="#D55E00", linewidth=1.1, label="origin 6")
    axes[1].plot(x, profiles["origin_difference_jacobian_energy"], color=color, linewidth=1.2)
    axes[2].plot(token, mass, color=color, linewidth=1.2)
    axes[0].set_ylabel(label, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=7, length=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlled-jacobian", required=True)
    parser.add_argument("--controlled-contributions", required=True)
    parser.add_argument("--patchtst-jacobian", required=True)
    parser.add_argument("--patchtst-contributions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.linewidth": 0.65})
    figure, axes = plt.subplots(2, 3, figsize=(7.1, 4.25), squeeze=False)
    draw_row(axes[0], "Controlled\nWeather", args.controlled_jacobian, args.controlled_contributions, "#009E73")
    draw_row(axes[1], "PatchTST\nWeather", args.patchtst_jacobian, args.patchtst_contributions, "#CC79A7")
    axes[0][0].legend(frameon=False, fontsize=7, loc="upper left", ncol=2, handlelength=1.8)
    axes[0][0].set_title("Ordinary Jacobian energy", fontsize=8, pad=5)
    axes[0][1].set_title("Origin-response Jacobian energy", fontsize=8, pad=5)
    axes[0][2].set_title("Readout contribution mass", fontsize=8, pad=5)
    for row in axes:
        row[0].set_xlabel("history position", fontsize=7)
        row[1].set_xlabel("history position", fontsize=7)
        row[2].set_xlabel("token index", fontsize=7)
        row[0].set_xlim(0, 511)
        row[1].set_xlim(0, 511)
        row[2].set_ylim(bottom=0)
    axes[0][0].set_ylabel("Controlled\nWeather", fontsize=8)
    axes[1][0].set_ylabel("PatchTST\nWeather", fontsize=8)
    figure.text(0.015, 0.5, "normalized profile", rotation=90, va="center", ha="center", fontsize=8)
    figure.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.13, wspace=0.28, hspace=0.48)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
