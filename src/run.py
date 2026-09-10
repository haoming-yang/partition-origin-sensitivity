"""Configuration-driven experiment entry point.

The command follows the public Time-Series-Library workflow: a small argparse
front end selects a configuration, data and output roots are externalized, and
the experiment implementation writes a self-describing run directory.  A
``--dry-run`` validates the matrix without importing a model or starting work.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not config.get("experiment_id"):
        raise ValueError(f"invalid experiment configuration: {path}")
    return config


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def resolve_seeds(config: dict, *, seed: int | None, seeds: str | None) -> list[int]:
    """Resolve one explicit seed, a comma-separated batch, or config values."""
    if seed is not None and seeds is not None:
        raise ValueError("use either --seed or --seeds, not both")
    if seed is not None:
        return [int(seed)]
    if seeds is not None:
        values = parse_ints(seeds)
        if values:
            return values
    configured = config.get("seed", config.get("seeds", config.get("default_seeds")))
    if configured is None:
        raise ValueError("a seed is required; pass --seed or --seeds")
    if isinstance(configured, int):
        return [configured]
    if isinstance(configured, str):
        return parse_ints(configured)
    return [int(value) for value in configured]


def config_jobs(config: dict, seeds: list[int], datasets: list[str]) -> list[tuple[str, int]]:
    configured_datasets = config.get("dataset") or config.get("datasets") or datasets
    if isinstance(configured_datasets, str):
        configured_datasets = [configured_datasets]
    selected = [dataset for dataset in datasets if dataset in configured_datasets]
    return [(dataset, seed) for dataset in selected for seed in seeds]


def dry_run(config: dict, jobs: list[tuple[str, int] | dict]) -> dict:
    if jobs and isinstance(jobs[0], dict):
        normalized_jobs = jobs
    else:
        normalized_jobs = [
            {"dataset": dataset, "seed": seed}
            for dataset, seed in jobs
        ]
    return {
        "experiment_id": config["experiment_id"],
        "source_status": config.get("source_status", "SOURCE_PRESENT"),
        "jobs": normalized_jobs,
        "training_started": False,
    }


def run_controlled(config: dict, dataset: str, seed: int, output_root: Path) -> dict:
    from .training import runner

    experiment_id = config["experiment_id"]
    out = output_root / experiment_id.lower() / dataset.lower() / f"seed{seed}"
    strategy = config.get("strategy", "random_origin")
    return runner.train_controlled(
        seed=seed,
        out=out,
        strategy=strategy,
        context=int(config.get("context", 512)),
        horizon=int(config.get("horizon", 96)),
        patch_len=int(config.get("patch_len", 12)),
        stride=int(config.get("stride", config.get("patch_len", 12))),
        epochs=int(config.get("epochs", 15)),
        batch_size=int(config.get("batch_size", 32)),
        learning_rate=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        dataset=dataset,
        experiment_id=experiment_id,
        use_position=bool(config.get("use_position", True)),
        head_geometry=str(config.get("head_geometry", "flattened")),
        source_status=str(config.get("source_status", "SOURCE_PRESENT")),
    )


def run_phenomenon(config: dict, dataset: str, seed: int, output_root: Path) -> dict:
    from .experiments.canonical.phenomenon_run import run

    patch = int(config.get("patch_len", config.get("patch_lengths", [12])[0]))
    out = output_root / config["experiment_id"].lower() / dataset.lower() / f"p{patch}" / f"seed{seed}"
    run(dataset, patch, seed, out)
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


def run_patch_lengths(config: dict, seeds: list[int], output_root: Path) -> list[dict]:
    results = []
    for dataset in config["datasets"]:
        for patch in config["patch_lengths"]:
            for seed in seeds:
                item = dict(config, patch_len=patch)
                results.append(run_phenomenon(item, dataset, seed, output_root))
    return results


def run_supplement(config: dict, dataset: str, seed: int, output_root: Path) -> object:
    """Dispatch source-backed supplement experiments to their intended runner."""
    from .training import runner

    experiment_id = config["experiment_id"]
    base = output_root / experiment_id.lower() / dataset.lower()
    common = dict(
        context=int(config.get("context", 512)),
        horizon=int(config.get("horizon", 96)),
        patch_len=int(config.get("patch_len", 12)),
        stride=int(config.get("stride", config.get("patch_len", 12))),
        epochs=int(config.get("epochs", 15)),
        batch_size=int(config.get("batch_size", 32)),
        learning_rate=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0.0 if experiment_id == "SUPPLEMENT_PATCHTST_ORIGIN" else 1e-4)),
        dataset=dataset,
        source_status=str(config.get("source_status", "SOURCE_PRESENT")),
    )
    if experiment_id == "SUPPLEMENT_PATCHTST_ORIGIN":
        return runner.train_patchtst(
            seed, base / f"seed{seed}", experiment_id=experiment_id, **common
        )
    if experiment_id == "SUPPLEMENT_TRAIN_ORIGIN_STRATEGY":
        results = {}
        for strategy in config.get("strategies", ("boundary_only", "random_origin", "all_origins")):
            results[strategy] = runner.train_controlled(
                seed, base / str(strategy) / f"seed{seed}", strategy,
                experiment_id=experiment_id, **common
            )
        return {"experiment_id": experiment_id, "dataset": dataset, "seed": seed, "strategies": results}
    return runner.train_controlled(
        seed, base / f"seed{seed}", str(config.get("strategy", "random_origin")),
        experiment_id=experiment_id, **common
    )


def run_cross_model(config: dict, output_root: Path) -> dict:
    script = ROOT / "src" / "experiments" / "cross_model" / "run_tier1_matrix.py"
    command = [sys.executable, str(script), "--out", str(output_root / config["experiment_id"].lower()),
               "--epochs", str(config.get("epochs", 5)),
               "--max-train-windows", str(config.get("max_train_windows", 1024)),
               "--max-test-windows", str(config.get("max_test_windows", 16)),
               "--batch-size", str(config.get("batch_size", 8))]
    subprocess.run(command, cwd=script.parent, check=True)
    return {"experiment_id": config["experiment_id"], "status": "COMPLETE"}


FULLSPLIT_PROTOCOL = {
    "context": 512,
    "horizon": 96,
    "patch_len": 12,
    "stride": 12,
    "epochs": 5,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "optimizer": "AdamW",
}


def validate_fullsplit_config(config: dict) -> str:
    """Validate fields that the recovered runner fixes by protocol."""
    for key, expected in FULLSPLIT_PROTOCOL.items():
        if key in config and config[key] != expected:
            raise ValueError(
                f"full-split protocol fixes {key}={expected}; got {config[key]}"
            )
    dataset = config.get("dataset", "ETTh1")
    if isinstance(dataset, list):
        if len(dataset) != 1:
            raise ValueError("full-split mixer runner supports only ETTh1")
        dataset = dataset[0]
    if str(dataset).lower() != "etth1":
        raise ValueError("full-split mixer runner supports only ETTh1")
    return "ETTh1"


def fullsplit_jobs(config: dict, seeds: list[int], datasets: list[str] | None = None) -> list[dict]:
    dataset = validate_fullsplit_config(config)
    selected_datasets = datasets if datasets is not None else [dataset]
    if not any(str(item).lower() == dataset.lower() for item in selected_datasets):
        return []
    models = tuple(config.get("models", ("Transformer", "MLP", "Conv")))
    return [
        {"model": str(model), "dataset": dataset, "seed": int(seed)}
        for model in models
        for seed in seeds
    ]


def run_fullsplit(
    config: dict,
    seeds: list[int],
    output_root: Path,
    datasets: list[str] | None = None,
) -> dict:
    """Launch the recovered complete-split ETTh1 mixer runner per model/seed."""
    script = ROOT / "tools" / "fullsplit" / "fullsplit_3run_runner.py"
    jobs = fullsplit_jobs(config, seeds, datasets)
    if not jobs:
        return {"experiment_id": config["experiment_id"], "status": "NO_JOBS", "jobs": []}
    calls = []
    for job in jobs:
        out = output_root / "fullsplit_cross_backbone" / job["model"] / "etth1" / f"seed{job['seed']}"
        command = [
            sys.executable,
            str(script),
            "--model", job["model"],
            "--seed", str(job["seed"]),
            "--out", str(out),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        calls.append({**job, "output": str(out)})
    return {"experiment_id": config["experiment_id"], "status": "COMPLETE", "jobs": calls}


def run(config: dict, seeds: list[int], datasets: list[str], output_root: Path) -> object:
    experiment_id = config["experiment_id"]
    if experiment_id == "PATCH_LENGTH_AUDIT_V1":
        return run_patch_lengths(config, seeds, output_root)
    if experiment_id == "CROSS_MODEL_PHASE_V1":
        return run_fullsplit(config, seeds, output_root, datasets)
    if experiment_id == "CANONICAL_PHENOMENON_27_V1":
        return [run_controlled(config, dataset, seed, output_root)
                for dataset, seed in config_jobs(config, seeds, datasets)]
    if experiment_id in {
        "SUPPLEMENT_HORIZON192",
        "SUPPLEMENT_OVERLAP_STRIDE6",
        "SUPPLEMENT_TRAIN_ORIGIN_STRATEGY",
        "SUPPLEMENT_PATCHTST_ORIGIN",
    }:
        return [run_supplement(config, dataset, seed, output_root)
                for dataset, seed in config_jobs(config, seeds, datasets)]
    if experiment_id == "NO_PE_CONTROL_V1":
        config = dict(config, use_position=False, context=512, horizon=96,
                      patch_len=12, stride=12, epochs=15,
                      strategy="random_origin")
    if experiment_id == "OPTIMIZATION_TRAJECTORY_V1":
        results = []
        for dataset, seed in config_jobs(dict(config, dataset="ETTh1"), seeds, ["ETTh1"]):
            results.append(run_controlled(dict(config, epochs=max(config["epochs"]),
                                               context=512, horizon=96, patch_len=12,
                                               stride=12, strategy="random_origin"), dataset, seed, output_root))
        return results
    if experiment_id == "MASK_HEAD_FACTORIAL_V1":
        results = []
        for head_geometry in ("flattened", "pooled"):
            local = dict(config, context=512, horizon=96, patch_len=12, stride=12,
                         epochs=5, strategy="random_origin", head_geometry=head_geometry,
                         source_status="RECONSTRUCTED_CONTROL")
            for dataset, seed in config_jobs(dict(local, dataset="ETTh1"), seeds, ["ETTh1"]):
                results.append(run_controlled(local, dataset, seed, output_root / head_geometry))
        return results
    results = []
    for dataset, seed in config_jobs(config, seeds, datasets):
        results.append(run_controlled(config, dataset, seed, output_root))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a registered partition-origin experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasets", default="ETTh1,ETTh2,ETTm1,ETTm2,Weather")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--seeds")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config if args.config.is_absolute() else ROOT / args.config)
    selected_seeds = resolve_seeds(config, seed=args.seed, seeds=args.seeds)
    selected_datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if config["experiment_id"] == "CROSS_MODEL_PHASE_V1":
        jobs = fullsplit_jobs(config, selected_seeds, selected_datasets)
    else:
        jobs = config_jobs(config, selected_seeds, selected_datasets)
    if args.dry_run:
        result = dry_run(config, jobs)
    else:
        status = str(config.get("source_status", "SOURCE_PRESENT")).upper()
        if status in {"FROZEN_ARTIFACT_ONLY", "ARTIFACT_DEPENDENT"}:
            message = (
                "This configuration is frozen-artifact-only; use the recorded artifact audit."
                if status == "FROZEN_ARTIFACT_ONLY" else
                "This configuration requires external frozen schedules/checkpoints; use scripts/run_poc.sh."
            )
            print(json.dumps({"status": status, "experiment_id": config["experiment_id"], "message": message}))
            raise SystemExit(2)
        result = run(config, selected_seeds, [item.strip() for item in args.datasets.split(",") if item.strip()], args.output_root)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
