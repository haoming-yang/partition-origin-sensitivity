#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m src.run --config "$ROOT/configs/optimization/etth1_epochs_5_30.yaml" --datasets ETTh1 --seeds "${SEEDS:-42,43,44}"
