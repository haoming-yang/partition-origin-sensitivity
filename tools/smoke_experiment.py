from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.training import runner


def run(output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Smoke output must be new or empty: {output}")
    runner.seed_all(0)
    x, y = torch.randn(2, 512, 2), torch.randn(2, 96, 2)
    model = runner.ControlledTransformerSupplement(512, 96, 12, 12, 2).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    values, mask = runner.partition(x, 0, 512, 12, 12)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = (model(values, mask) - y).square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    predictions = {}
    model.eval()
    with torch.inference_mode():
        for origin in range(12):
            values, mask = runner.partition(x, origin, 512, 12, 12)
            predictions[origin] = model(values, mask).numpy()
    targets = y.numpy()
    rows, _ = runner.metric_rows(predictions, targets)
    config = {"experiment_id": "SYNTHETIC_SMOKE_NOT_PAPER", "dataset": "synthetic_not_paper_evidence",
              "seed": 0, "device": "cpu", "context": 512, "horizon": 96,
              "patch_len": 12, "stride": 12, "origins": list(range(12)), "updates": 1}
    return runner.save_run_artifacts(output, config, [{"loss": float(loss.detach())}], [],
                                     rows, predictions, targets, {}, [Path(__file__)])


def main():
    parser = argparse.ArgumentParser(description="CPU synthetic training/evaluation check; not paper evidence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
