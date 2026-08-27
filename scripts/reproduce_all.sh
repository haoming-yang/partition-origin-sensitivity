#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  core) exec "$ROOT/scripts/run_core.sh" ;;
  overlap) exec "$ROOT/scripts/run_overlap.sh" ;;
  h192) exec "$ROOT/scripts/run_h192.sh" ;;
  training-policy) exec "$ROOT/scripts/run_training_policy.sh" ;;
  patchtst) exec "$ROOT/scripts/run_patchtst.sh" ;;
  mixers) exec "$ROOT/scripts/run_mixers.sh" ;;
  optimization) exec "$ROOT/scripts/run_optimization.sh" ;;
  heads) exec "$ROOT/scripts/run_heads.sh" ;;
  pe-control) exec "$ROOT/scripts/run_pe_control.sh" ;;
  poc) exec "$ROOT/scripts/run_poc.sh" ;;
  patch-length) exec "$ROOT/scripts/run_patch_length.sh" ;;
  audits) exec "$ROOT/scripts/run_audits.sh" ;;
  tables) exec "$ROOT/scripts/reproduce_tables.sh" ;;
  figures) exec "$ROOT/scripts/reproduce_figures.sh" ;;
  tier1-smoke) exec "$ROOT/scripts/run_tier1_smoke.sh" ;;
  *) printf 'Usage: %s {core|overlap|h192|training-policy|patchtst|mixers|optimization|heads|pe-control|poc|patch-length|audits|tables|figures|tier1-smoke}\n' "$0"; exit 2 ;;
esac
