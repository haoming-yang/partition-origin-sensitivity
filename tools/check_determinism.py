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
    args = parser.parse_args()
    root = Path(args.root)
    files = sorted(path for path in root.rglob("*.json") if path.is_file()) if root.exists() else []
    records = [{"path": str(path), "sha256": digest(path)} for path in files]
    print(json.dumps({"files": records, "count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
