"""Write a deterministic SHA-256 manifest for a vendored source tree."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.provenance import file_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-mode", choices=("lf", "raw"), default="lf")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error("--root must be an existing source directory")
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or ".git" in path.parts:
            continue
        if path.resolve() == args.output.resolve() or path.name in {
            "SOURCE_MANIFEST.csv", "SOURCE_MANIFEST.historical.csv",
        }:
            continue
        relative = path.relative_to(root).as_posix()
        size, sha256 = file_fingerprint(path, normalize_lf=args.hash_mode == "lf")
        rows.append({"path": relative, "bytes": size, "sha256": sha256, "hash_mode": args.hash_mode})
    if not rows:
        parser.error("source directory contains no manifestable files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256", "hash_mode"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"manifest_rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
