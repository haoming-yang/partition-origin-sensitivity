#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/tools/check_reconstruction.py" --context 512 --patch 12
python "$ROOT/tools/check_sentinel.py"
python "$ROOT/tools/audit_artifacts.py" --root "${OUTPUT_ROOT:-$ROOT/outputs}" --out "${OUTPUT_ROOT:-$ROOT/outputs}/audit_report.json"
python "$ROOT/tools/check_determinism.py" --root "${OUTPUT_ROOT:-$ROOT/outputs}"
python "$ROOT/tools/check_anonymity.py" --root "$ROOT"
