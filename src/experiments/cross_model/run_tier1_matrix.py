"""Serial launcher for the complete Tier1 cross-model phase-sensitivity matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


MODELS = ("PatchTST", "PatchMixer", "PatchMLP", "Pathformer", "HDMixer", "DeformableTST")
DATASETS = ("ETTh1", "Weather", "Electricity")
SEEDS = (42, 43, 44)
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Job:
    model: str
    dataset: str
    seed: int


def matrix_jobs() -> list[Job]:
    return [Job(model, dataset, seed) for model in MODELS for dataset in DATASETS for seed in SEEDS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-windows", type=int, default=1024)
    parser.add_argument("--max-test-windows", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    runner = ROOT / "patchtst_phase_run.py"
    for job in matrix_jobs():
        out = args.out / job.model / job.dataset / f"seed{job.seed}"
        if (out / "manifest.json").is_file():
            print(f"skip completed {job}", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(runner), "--model", job.model, "--dataset", job.dataset,
            "--seed", str(job.seed), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--max-train-windows", str(args.max_train_windows), "--max-test-windows", str(args.max_test_windows),
            "--out", str(out),
        ]
        print(f"start {job}", flush=True)
        with (out / "run.log").open("w", encoding="utf8") as log, (out / "run.err").open("w", encoding="utf8") as err:
            subprocess.run(command, check=True, stdout=log, stderr=err)
        print(f"done {job}", flush=True)


if __name__ == "__main__":
    main()
