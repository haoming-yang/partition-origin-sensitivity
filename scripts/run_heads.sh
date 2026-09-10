#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/defaults.sh"
python -m src.run --config "$ROOT/configs/heads/etth1_mask_head_factorial.yaml" --datasets ETTh1 --seeds "$DEFAULT_SEEDS"
