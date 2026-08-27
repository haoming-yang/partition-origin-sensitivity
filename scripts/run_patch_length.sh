#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m src.run --config "$ROOT/configs/patch_length/etth1_weather_p8_p12_p16.yaml" --datasets "${DATASETS:-ETTh1,Weather}" --seeds "${SEEDS:-42,43,44}"
