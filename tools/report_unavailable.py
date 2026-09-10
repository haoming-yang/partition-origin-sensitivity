from __future__ import annotations

import argparse
import json


MESSAGES = {
    "optimization": "The exact 5/10/15/20/25/30 optimization runner is not present in the supplied source tree; frozen checkpoint records were not converted into a guessed implementation.",
    "heads": "The exact mask-head factorial training entrypoint is not present in the supplied source tree; frozen artifacts were not converted into a guessed implementation.",
    "pe-control": "The exact positional-encoding control training entrypoint is not present in the supplied source tree; frozen artifacts were not converted into a guessed implementation.",
    "poc": "The POC Stage-4 runner and schedules are included; execution requires the external frozen checkpoint bundle described in README.md.",
    "patch-length": "The exact 18-run patch-length training entrypoint is not present in the supplied source tree; frozen artifacts were not converted into a guessed implementation.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(MESSAGES), required=True)
    args = parser.parse_args()
    status = "ARTIFACT_DEPENDENT" if args.experiment == "poc" else "SOURCE_UNAVAILABLE"
    print(json.dumps({"status": status, "experiment": args.experiment, "message": MESSAGES[args.experiment]}))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
