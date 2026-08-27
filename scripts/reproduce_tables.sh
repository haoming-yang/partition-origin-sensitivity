#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUTPUT_ROOT:-$ROOT/outputs}"
python "$ROOT/tools/aggregate_results.py" --root "$OUT" --out "$OUT/aggregate.csv"
python "$ROOT/tools/make_tables.py" --input "$OUT/aggregate.csv" --output "$ROOT/results/generated_tables.csv"
