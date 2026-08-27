from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from patching.phase_protocol import audit_phase_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--patch", type=int, default=12)
    parser.add_argument("--total-length", type=int)
    args = parser.parse_args()
    rows = audit_phase_layout(args.context, args.patch, args.total_length)
    passed = all(row["exactly_once"] for row in rows)
    result = {"pass": passed, "context": args.context, "patch": args.patch, "records": rows}
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
