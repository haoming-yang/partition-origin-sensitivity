from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import stage4_inference as base


OUT = base.STAGE / "batch1_inference_and_fixed_origin" / "optional_patch_sizes"
EXPECTED_JOBS = []
JOBS = []
MISSING_JOBS = []
for dataset in ("ETTh1", "Weather"):
    for p in (8, 16):
        for seed in (42, 43, 44):
            EXPECTED_JOBS.append((dataset, p, seed))
            src = base.source_run(dataset, seed, p) / "checkpoint.pt"
            if src.exists():
                JOBS.append((dataset, p, seed, src))
            else:
                MISSING_JOBS.append(f"{dataset} p{p} seed{seed}")


def can_reuse_outputs(src: Path, baseline_path: Path, dispersion_path: Path) -> bool:
    if not baseline_path.is_file() or not dispersion_path.is_file():
        return False
    try:
        expected_hash = base.sha256(src)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        dispersion = json.loads(dispersion_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError):
        return False
    return (
        baseline.get("checkpoint_sha256") == expected_hash
        and dispersion.get("checkpoint_sha256") == expected_hash
        and "test" in baseline
        and "S_theta" in dispersion
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    statuses = []
    for dataset, p, seed, src in JOBS:
        b = OUT / "inference_baselines" / dataset / f"p{p}_seed{seed}.json"
        d = OUT / "dispersion_analysis" / dataset / f"p{p}_seed{seed}.json"
        if can_reuse_outputs(src, b, d):
            job_status = "REUSED_HASH_MATCHED"
        else:
            base.run_baseline(dataset, seed, b, p)
            base.run_dispersion(dataset, seed, d, p)
            job_status = "COMPLETE"
        statuses.append({"dataset": dataset, "p": p, "seed": seed, "checkpoint": str(src), "status": job_status})
        print(json.dumps(statuses[-1]), flush=True)
    (OUT / "STATUS.json").write_text(json.dumps({"experiment_id": "STAGE4_BATCH1_OPTIONAL_PATCH_SIZE_V1", "expected_jobs": len(EXPECTED_JOBS), "jobs": statuses, "missing_existing_checkpoint_jobs": MISSING_JOBS}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
