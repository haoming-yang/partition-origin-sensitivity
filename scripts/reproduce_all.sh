#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/defaults.sh"
command="${1:-}"
shift || true
run_config() {
  local config="$1" datasets="$2"
  shift 2
  python -m src.run --config "$ROOT/configs/$config" --datasets "$datasets" --seeds "$DEFAULT_SEEDS" "$@"
}
case "$command" in
  list) printf '%s\n' core overlap h192 training-policy patchtst mixers optimization heads pe-control poc patch-length audits tables latent ;;
  core) run_config core/canonical.yaml "${DATASETS:-ETTh1,ETTh2,ETTm1,ETTm2,Weather}" "$@" ;;
  overlap) run_config extensions/etth1_overlap_s6.yaml "${DATASETS:-ETTh1}" "$@" ;;
  h192) run_config extensions/etth1_h192.yaml "${DATASETS:-ETTh1}" "$@" ;;
  training-policy) run_config training_policy/etth1_origin_strategies.yaml "${DATASETS:-ETTh1}" "$@" ;;
  patchtst) run_config patchtst/etth1_official_source.yaml "${DATASETS:-ETTh1}" "$@" ;;
  mixers) exec bash "$ROOT/scripts/run_mixers.sh" ;;
  optimization) run_config optimization/etth1_epochs_5_30.yaml ETTh1 "$@" ;;
  heads)
    run_config heads/etth1_mask_head_factorial.yaml ETTh1 "$@"
    run_config heads/weather_mask_head_factorial.yaml Weather "$@"
    ;;
  pe-control) run_config positional_encoding/etth1_no_pe.yaml ETTh1 "$@" ;;
  poc) exec bash "$ROOT/scripts/run_poc.sh" ;;
  patch-length) run_config patch_length/etth1_weather_p8_p12_p16.yaml "${DATASETS:-ETTh1,Weather}" "$@" ;;
  audits) exec bash "$ROOT/scripts/run_audits.sh" ;;
  tables) exec bash "$ROOT/scripts/reproduce_tables.sh" "$@" ;;
  latent) exec bash "$ROOT/scripts/summarize_latent.sh" ;;
  *) printf 'Usage: %s {%s}\n' "$0" "$(bash "$0" list | paste -sd '|')"; exit 2 ;;
esac
