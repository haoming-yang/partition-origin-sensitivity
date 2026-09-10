"""Verify the tracked POC provenance manifest and optional external checkpoints."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.provenance import file_fingerprint, manifest_path


MANIFEST = REPO_ROOT / "artifacts" / "poc_stage3" / "SHA256SUMS.txt"


def _digest(path: Path, *, normalize_lf: bool = False) -> str:
    return file_fingerprint(path, normalize_lf=normalize_lf)[1]


def _entries() -> list[tuple[str, Path, bool]]:
    entries: list[tuple[str, Path, bool]] = []
    for number, raw in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        record = re.fullmatch(r"([0-9a-fA-F]{64})\s+(.+)", line)
        if record is None:
            raise ValueError(f"Malformed SHA256 record at line {number}")
        digest, rest = record.groups()
        external = rest.startswith("[external] ")
        if external:
            rest = rest[len("[external] ") :]
        entries.append((digest.lower(), manifest_path(REPO_ROOT, rest), external))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail when an external checkpoint is not present",
    )
    args = parser.parse_args()

    try:
        entries = _entries()
        if not entries:
            raise ValueError("manifest contains no file records")
        if len({path for _, path, _ in entries}) != len(entries):
            raise ValueError("manifest contains duplicate paths")
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1

    missing_external = 0
    failures = 0
    for expected, path, external in entries:
        if not path.is_file():
            if external:
                missing_external += 1
                print(f"MISSING external: {path.relative_to(REPO_ROOT)}")
                continue
            failures += 1
            print(f"MISSING tracked: {path.relative_to(REPO_ROOT)}")
            continue
        actual = _digest(path, normalize_lf=not external)
        if actual != expected:
            failures += 1
            kind = "external" if external else "tracked"
            print(f"MISMATCH {kind}: {path.relative_to(REPO_ROOT)}")

    if failures:
        print(f"FAIL: {failures} manifest entries are missing or mismatched")
        return 1
    if missing_external:
        message = f"OK: tracked entries verified; {missing_external} external checkpoint(s) absent"
        print(message)
        return 1 if args.strict else 0
    print("OK: all manifest entries verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
