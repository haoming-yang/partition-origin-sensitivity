from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.provenance import file_fingerprint, manifest_path


def verify_manifest(manifest: Path, source_root: Path) -> tuple[int, list[str]]:
    errors = []
    try:
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            return 0, [f"Empty source manifest: {manifest}"]
    except (OSError, ValueError) as exc:
        return 0, [f"Cannot read source manifest {manifest}: {exc}"]
    seen = set()
    for row in rows:
        try:
            path = manifest_path(source_root, row["path"])
            if path in seen:
                raise ValueError("duplicate path")
            seen.add(path)
            mode = row["hash_mode"]
            if mode not in {"lf", "raw"}:
                raise ValueError(f"unknown hash mode: {mode}")
            size, digest = file_fingerprint(path, normalize_lf=mode == "lf")
            if size != int(row["bytes"]) or digest != row["sha256"].lower():
                raise ValueError("size or SHA256 mismatch")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{manifest}: {row.get('path', '<missing path>')}: {exc}")
    return len(rows), errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify current vendored-source release manifests")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    total = 0
    errors = []
    for name, prefix in (("patch_models", "tier1"), ("time_series_library", "")):
        root = args.root / "third_party" / name
        count, failures = verify_manifest(root / "SOURCE_MANIFEST.csv", root / prefix)
        total += count
        errors.extend(failures)
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        return 1
    print(f"OK: {total} source manifest entries verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
