from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training import runner


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run(args: argparse.Namespace) -> None:
    if args.seed is not None and args.seeds is not None:
        raise ValueError("use either --seed or --seeds, not both")
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds:
        seeds = parse_ints(args.seeds)
    else:
        raise ValueError("a seed is required; pass --seed or --seeds")
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    results = []
    for dataset in datasets:
        for seed in seeds:
            if args.experiment == "core":
                out = runner.OUT_ROOT / "core" / dataset / f"seed{seed}"
                results.append(runner.train_controlled(seed, out, "random_origin", 512, 96, 12, 12, dataset=dataset))
            elif args.experiment == "overlap":
                out = runner.OUT_ROOT / "extensions" / "overlap_stride6" / dataset / f"seed{seed}"
                results.append(runner.train_controlled(seed, out, "random_origin", 512, 96, 12, 6, dataset=dataset))
            elif args.experiment == "h192":
                out = runner.OUT_ROOT / "extensions" / "horizon192" / dataset / f"seed{seed}"
                results.append(runner.train_controlled(seed, out, "random_origin", 512, 192, 12, 12, dataset=dataset))
            elif args.experiment == "training-policy":
                for strategy in ("boundary_only", "random_origin", "all_origins"):
                    out = runner.OUT_ROOT / "training_policy" / strategy / dataset / f"seed{seed}"
                    results.append(runner.train_controlled(seed, out, strategy, 512, 96, 12, 12, dataset=dataset))
            elif args.experiment == "patchtst":
                out = runner.OUT_ROOT / "patchtst" / dataset / f"seed{seed}"
                results.append(runner.train_patchtst(seed, out, 512, 96, 12, 12, 10, dataset))
    print(json.dumps(results, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("core", "overlap", "h192", "training-policy", "patchtst"), required=True)
    parser.add_argument("--datasets", default="ETTh1,ETTh2,ETTm1,ETTm2,Weather")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--seeds")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
