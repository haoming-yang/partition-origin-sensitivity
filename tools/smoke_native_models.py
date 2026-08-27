"""Instantiate and forward all vendored Tier-1 models without training."""

from __future__ import annotations

import argparse
import json

import torch

from src.experiments.cross_model.adapters import tier1_specs
from src.experiments.cross_model.native_models import build_model, native_forecast


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *tier1_specs().keys()], default="all")
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--channels", type=int, default=7)
    args = parser.parse_args()

    names = list(tier1_specs()) if args.model == "all" else [args.model]
    reports = []
    failures = []
    for name in names:
        spec = tier1_specs()[name]
        input_len = spec.input_length(args.context_len)
        try:
            model = build_model(name, input_len, args.horizon, args.channels)
            model.eval()
            x = torch.zeros(1, input_len, args.channels)
            with torch.inference_mode():
                y = native_forecast(name, model, x)
            expected = (1, args.horizon, args.channels)
            if tuple(y.shape) != expected:
                raise RuntimeError(f"unexpected output shape {tuple(y.shape)}; expected {expected}")
            reports.append({"model": name, "input_length": input_len, "output_shape": list(y.shape), "status": "PASS"})
        except Exception as exc:  # pragma: no cover - exercised by source snapshots
            failures.append(name)
            reports.append({"model": name, "input_length": input_len, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})

    if args.model == "all":
        from src.training.runner import OfficialPatchTSTAdapter
        from src.training.runner import partition as controlled_partition

        try:
            adapter = OfficialPatchTSTAdapter(args.context_len, args.horizon, 12, 12, args.channels)
            x = torch.zeros(1, args.context_len, args.channels)
            padded, observed = controlled_partition(x, 0, args.context_len, 12, 12)
            with torch.inference_mode():
                y = adapter(padded, observed)
            expected = (1, args.horizon, args.channels)
            if tuple(y.shape) != expected:
                raise RuntimeError(f"unexpected output shape {tuple(y.shape)}; expected {expected}")
            reports.append({"model": "OfficialPatchTSTAdapter", "input_length": int(padded.shape[1]), "output_shape": list(y.shape), "status": "PASS"})
        except Exception as exc:  # pragma: no cover - exercised by source snapshots
            failures.append("OfficialPatchTSTAdapter")
            reports.append({"model": "OfficialPatchTSTAdapter", "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})

    print(json.dumps(reports, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
