#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/defaults.sh"
python -m src.experiments.stage4.poc_formal_c_runner --seeds "$DEFAULT_SEEDS"
