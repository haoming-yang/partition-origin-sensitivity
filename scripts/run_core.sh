#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/defaults.sh"
python -m src.run --config "$ROOT/configs/core/canonical.yaml" --datasets "${DATASETS:-ETTh1,ETTh2,ETTm1,ETTm2,Weather}" --seeds "$DEFAULT_SEEDS"
