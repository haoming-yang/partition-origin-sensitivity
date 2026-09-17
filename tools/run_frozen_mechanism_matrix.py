from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "tools" / "analyze_frozen_mechanisms.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_seed(config: dict, path: Path) -> int | None:
    if config.get("seed") is not None:
        return int(config["seed"])
    for part in reversed(path.parts):
        match = re.fullmatch(r"seed(\d+)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def classify_model(config: dict) -> str | None:
    model = str(config.get("model", "")).lower()
    if "patchtst" in model:
        return "patchtst"
    if "controlled-transformer" in model or "controlledtransformer" in model:
        return "controlled"
    return None


def discover_jobs(root: Path, requested: str, deduplicate: bool) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    seen: set[str] = set()
    for config_path in sorted(root.rglob("config.json")):
        checkpoint = config_path.parent / "checkpoint.pt"
        if not checkpoint.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        kind = classify_model(config)
        if kind is None or (requested != "all" and kind != requested):
            continue
        dataset = str(config.get("dataset", ""))
        if dataset not in {"ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"}:
            continue
        checkpoint_hash = sha256(checkpoint)
        if deduplicate and checkpoint_hash in seen:
            continue
        seen.add(checkpoint_hash)
        patch_length = int(config.get("patch_len", config.get("p", 12)))
        stride = int(config.get("stride", patch_length))
        jobs.append(
            {
                "model": kind,
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_hash,
                "dataset": dataset,
                "seed": infer_seed(config, config_path.parent),
                "context": int(config.get("context", 512)),
                "horizon": int(config.get("horizon", 96)),
                "patch_length": patch_length,
                "stride": stride,
                "experiment_id": config.get("experiment_id"),
                "config": config_path,
            }
        )
    return jobs


def command_for(job: dict[str, object], args: argparse.Namespace, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(ANALYZER),
        "--model",
        str(job["model"]),
        "--checkpoint",
        str(job["checkpoint"]),
        "--data-root",
        str(args.data_root.resolve()),
        "--dataset",
        str(job["dataset"]),
        "--output",
        str(output),
        "--context",
        str(job["context"]),
        "--horizon",
        str(job["horizon"]),
        "--patch-length",
        str(job["patch_length"]),
        "--stride",
        str(job["stride"]),
        "--batch-size",
        str(args.batch_size),
        "--finite-indices",
        args.finite_indices,
        "--finite-max-windows",
        str(args.finite_max_windows),
    ]
    if job["seed"] is not None:
        command.extend(["--seed", str(job["seed"])])
    if args.max_windows:
        command.extend(["--max-windows", str(args.max_windows)])
    return command


def run(args: argparse.Namespace) -> dict[str, object]:
    jobs = discover_jobs(args.archive_root.resolve(), args.model, args.deduplicate)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "archive_root": str(args.archive_root.resolve()),
        "data_root": str(args.data_root.resolve()),
        "requested_model": args.model,
        "deduplicate": args.deduplicate,
        "jobs": [],
    }
    for index, job in enumerate(jobs):
        name = f"{index:03d}_{job['model']}_{job['dataset']}_s{job['seed'] if job['seed'] is not None else 'na'}_p{job['patch_length']}_h{job['horizon']}_{str(job['checkpoint_sha256'])[:12]}"
        output = output_root / name
        command = command_for(job, args, output)
        record = {key: str(value) if isinstance(value, Path) else value for key, value in job.items() if key not in {"config"}}
        record["output"] = str(output)
        record["command"] = command
        if args.dry_run:
            record["status"] = "DRY_RUN"
        else:
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            record["status"] = "PASS" if completed.returncode == 0 else "FAIL"
            record["returncode"] = completed.returncode
            record["stdout_tail"] = completed.stdout[-2000:]
            record["stderr_tail"] = completed.stderr[-2000:]
        manifest["jobs"].append(record)
        print(json.dumps(record, default=str), flush=True)
    manifest["job_count"] = len(jobs)
    manifest["pass_count"] = sum(record.get("status") == "PASS" for record in manifest["jobs"])
    manifest["fail_count"] = sum(record.get("status") == "FAIL" for record in manifest["jobs"])
    (output_root / "matrix_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=("patchtst", "controlled", "all"), default="all")
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--finite-indices", default="0,64,128,256,384,511")
    parser.add_argument("--finite-max-windows", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, default=str))
