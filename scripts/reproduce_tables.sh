#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${OUTPUT_ROOT:-$ROOT/outputs}"
if [[ "${1:-frozen}" == "frozen" ]]; then
  exec python "$ROOT/tools/reproduce_frozen_paper.py" --output "$OUT/paper_reproduction"
fi
if [[ "${1:-}" != "new-runs" ]]; then
  printf "Usage: reproduce_tables.sh [frozen|new-runs]\n"
  exit 2
fi
python "$ROOT/tools/aggregate_results.py" --root "$OUT" --out "$OUT/aggregate.csv"
python "$ROOT/tools/make_tables.py" --input "$OUT/aggregate.csv" --output "$OUT/generated_tables.csv"
