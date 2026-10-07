#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/defaults.sh"
python -m pytest -q "$ROOT/tests/test_metrics.py" "$ROOT/tests/test_phase_protocol.py" "$ROOT/tests/test_protocol_and_metrics.py"
python -m src.run --config "$ROOT/configs/core/canonical.yaml" --seeds "$DEFAULT_SEEDS" --dry-run
python "$ROOT/tools/check_reconstruction.py" --context 512 --patch 12
python "$ROOT/tools/check_sentinel.py"
python "$ROOT/tools/smoke_experiment.py" --output "${SMOKE_OUTPUT_ROOT:-$ROOT/outputs/smoke/$(date -u +%Y%m%dT%H%M%S)-$$}"
