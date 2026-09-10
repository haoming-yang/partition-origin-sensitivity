"""Emit a deterministic SHA-256 manifest for JSON run records.

This is a release/audit manifest, not a repeated-forward determinism test. It
does not start training or inference and therefore cannot establish that two
fresh executions produce identical outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON manifest")
    args = parser.parse_args()
    root = Path(args.root)
    files = sorted(path for path in root.rglob("*.json") if path.is_file()) if root.exists() else []
    records = [{"path": str(path.relative_to(root)), "sha256": digest(path)} for path in files]
    payload = {"root": str(root), "files": records, "count": len(records)}
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
