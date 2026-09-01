#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m pytest -q "$ROOT/tests"
python -m src.run --config "$ROOT/configs/core/canonical.yaml" --dry-run
python "$ROOT/tools/check_reconstruction.py" --context 512 --patch 12
python "$ROOT/tools/check_sentinel.py"
