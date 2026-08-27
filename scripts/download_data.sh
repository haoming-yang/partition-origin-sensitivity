#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_ROOT:-$ROOT/data}"
missing=0
for path in "$DATA_DIR/ETT-small/ETTh1.csv" "$DATA_DIR/ETT-small/ETTh2.csv" "$DATA_DIR/ETT-small/ETTm1.csv" "$DATA_DIR/ETT-small/ETTm2.csv" "$DATA_DIR/weather/weather.csv"; do
  if [[ ! -f "$path" ]]; then
    printf 'MISSING %s\n' "$path"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  printf 'Place datasets from their original public sources under DATA_ROOT.\n'
  exit 1
fi
printf 'DATA_PRESENT %s\n' "$DATA_DIR"
