#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/defaults.sh"

SPECTRUM_ROOT="${SPECTRUM_ROOT:-$ROOT/artifacts/latent_spectrum_etth1_o0_o6_p12_L512_H96}"
LAYERS_ROOT="${LAYERS_ROOT:-$ROOT/artifacts/latent_layers_etth1_o0_o6_p12_L512_H96}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts}"
PERMUTATIONS="${PERMUTATIONS:-1000}"

IFS=',' read -r -a seeds <<< "$DEFAULT_SEEDS"
if [[ "${#seeds[@]}" -ne 3 ]]; then
  echo "summarize_latent.sh expects exactly three replicate seeds; got: $DEFAULT_SEEDS" >&2
  exit 2
fi
spectrum_args=()
layer_args=()
permutation_args=()
for seed in "${seeds[@]}"; do
  spectrum="$SPECTRUM_ROOT/seed${seed}/latent_spectrum_origin0_o6_p12_L512_H96.csv"
  layers="$LAYERS_ROOT/seed${seed}/latent_layers_origin0_o6_p12_L512_H96.csv"
  [[ -f "$spectrum" ]] || { echo "Missing spectrum artifact: $spectrum" >&2; exit 1; }
  [[ -f "$layers" ]] || { echo "Missing layer artifact: $layers" >&2; exit 1; }
  spectrum_args+=(--spectrum "$spectrum")
  layer_args+=(--layers "$layers")
  permutation_args+=(--input "$spectrum")
done

python "$ROOT/tools/summarize_latent_artifacts.py" \
  "${spectrum_args[@]}" "${layer_args[@]}" \
  --output "$OUTPUT_ROOT/latent_summary_etth1_o0_o6.json"

python "$ROOT/tools/permutation_latent_spectrum.py" \
  "${permutation_args[@]}" \
  --output "$OUTPUT_ROOT/latent_spectrum_etth1_o0_o6_permutation.json" \
  --permutations "$PERMUTATIONS" --seed 0
