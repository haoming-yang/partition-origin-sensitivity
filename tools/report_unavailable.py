from __future__ import annotations

import argparse
import json


MESSAGES = {
    "mixers": "The exact full-split Transformer/MLP/Conv training entrypoint is not present in the supplied source tree; frozen artifacts and configs were not converted into a guessed implementation.",
    "optimization": "The exact 5/10/15/20/25/30 optimization runner is not present in the supplied source tree; frozen checkpoint records were not converted into a guessed implementation.",
    "heads": "The exact mask-head factorial training entrypoint is not present in the supplied source tree; frozen artifacts were not converted into a guessed implementation.",
    "pe-control": "The exact positional-encoding control training entrypoint is not present in the supplied source tree; frozen artifacts were not converted into a guessed implementation.",
    "poc": "The exact POC schedule and training entrypoint are not present in the supplied source tree; frozen artifacts were not converted into a guessed implementation.",
    "patch-length": "The exact 18-run patch-length training entrypoint is not present in the supplied source tree; frozen artifacts were not converted into a guessed implementation.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(MESSAGES), required=True)
    args = parser.parse_args()
    print(json.dumps({"status": "SOURCE_UNAVAILABLE", "experiment": args.experiment, "message": MESSAGES[args.experiment]}))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
