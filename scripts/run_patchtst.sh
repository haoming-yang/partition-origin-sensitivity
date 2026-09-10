#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/defaults.sh"
python -m src.run --config "$ROOT/configs/patchtst/etth1_official_source.yaml" --datasets "${DATASETS:-ETTh1}" --seeds "$DEFAULT_SEEDS"
