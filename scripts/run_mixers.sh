#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS_VALUE="${SEEDS:-42,43,44}"
OUTPUT_ROOT="${MIXER_OUTPUT_ROOT:-$ROOT/outputs/fullsplit_cross_backbone}"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS_VALUE"

for model in Transformer MLP Conv; do
  for seed in "${SEED_LIST[@]}"; do
    python "$ROOT/tools/fullsplit/fullsplit_3run_runner.py" \
      --model "$model" \
      --seed "$seed" \
      --out "$OUTPUT_ROOT/$model/etth1/seed$seed"
  done
done
