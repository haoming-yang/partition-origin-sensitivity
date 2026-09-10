#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/defaults.sh"
python -m src.run --config "$ROOT/configs/extensions/etth1_overlap_s6.yaml" --datasets "${DATASETS:-ETTh1}" --seeds "$DEFAULT_SEEDS"
