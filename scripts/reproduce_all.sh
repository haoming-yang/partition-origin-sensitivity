#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
case "${1:-}" in
  core) exec bash "$ROOT/scripts/run_core.sh" ;;
  overlap) exec bash "$ROOT/scripts/run_overlap.sh" ;;
  h192) exec bash "$ROOT/scripts/run_h192.sh" ;;
  training-policy) exec bash "$ROOT/scripts/run_training_policy.sh" ;;
  patchtst) exec bash "$ROOT/scripts/run_patchtst.sh" ;;
  mixers) exec bash "$ROOT/scripts/run_mixers.sh" ;;
  optimization) exec bash "$ROOT/scripts/run_optimization.sh" ;;
  heads) exec bash "$ROOT/scripts/run_heads.sh" ;;
  pe-control) exec bash "$ROOT/scripts/run_pe_control.sh" ;;
  poc) exec bash "$ROOT/scripts/run_poc.sh" ;;
  patch-length) exec bash "$ROOT/scripts/run_patch_length.sh" ;;
  audits) exec bash "$ROOT/scripts/run_audits.sh" ;;
  tables) exec bash "$ROOT/scripts/reproduce_tables.sh" ;;
  tier1-smoke) exec bash "$ROOT/scripts/run_tier1_smoke.sh" ;;
  *) printf 'Usage: %s {core|overlap|h192|training-policy|patchtst|mixers|optimization|heads|pe-control|poc|patch-length|audits|tables|tier1-smoke}\n' "$0"; exit 2 ;;
esac
