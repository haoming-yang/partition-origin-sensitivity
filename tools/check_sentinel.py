from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.runner import ControlledTransformerSupplement, partition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--patch", type=int, default=12)
    parser.add_argument("--channels", type=int, default=7)
    args = parser.parse_args()
    model = ControlledTransformerSupplement(args.context, 96, args.patch, args.patch, args.channels).eval()
    x = torch.randn(2, args.context, args.channels)
    base, mask = partition(x, 0, args.context, args.patch, args.patch, 0.0)
    changed, changed_mask = partition(x, 0, args.context, args.patch, args.patch, 1000.0)
    with torch.inference_mode():
        first = model(base, mask)
        second = model(changed, changed_mask)
    delta = float((first - second).abs().max())
    result = {"max_abs_difference": delta, "pass": delta <= 1e-6}
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
