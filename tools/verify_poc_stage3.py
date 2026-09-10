"""Verify the tracked POC provenance manifest and optional external checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "artifacts" / "poc_stage3" / "SHA256SUMS.txt"


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _entries() -> list[tuple[str, Path, bool]]:
    entries: list[tuple[str, Path, bool]] = []
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or not re.match(r"^[0-9a-fA-F]{64}\s+", line):
            continue
        digest, rest = line.split(maxsplit=1)
        external = rest.startswith("[external] ")
        if external:
            rest = rest[len("[external] ") :]
        entries.append((digest, REPO_ROOT / rest.replace("/", "\\"), external))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail when an external checkpoint is not present",
    )
    args = parser.parse_args()

    missing_external = 0
    failures = 0
    for expected, path, external in _entries():
        if not path.is_file():
            if external:
                missing_external += 1
                print(f"MISSING external: {path.relative_to(REPO_ROOT)}")
                continue
            failures += 1
            print(f"MISSING tracked: {path.relative_to(REPO_ROOT)}")
            continue
        actual = _digest(path)
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
