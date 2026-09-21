#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/defaults.sh"
python -m src.run --config "$ROOT/configs/heads/etth1_mask_head_factorial.yaml" --datasets ETTh1 --seeds "$DEFAULT_SEEDS"
python -m src.run --config "$ROOT/configs/heads/weather_mask_head_factorial.yaml" --datasets Weather --seeds "$DEFAULT_SEEDS"
