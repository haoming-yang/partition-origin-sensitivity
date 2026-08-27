#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m src.run --config "$ROOT/configs/positional_encoding/etth1_no_pe.yaml" --datasets ETTh1 --seeds "${SEEDS:-42,43,44}"
