# Source availability audit

## Source states

The repository includes the controlled supplementary runner, canonical runner, reference validator, phase protocol, result schema, frozen-matrix audit utility, and isolated snapshots of the six Tier-1 model sources. The copied sources are isolated under `third_party/`; the original projects are not modified.

Each configuration declares one of these states:

- `SOURCE_PRESENT`: the runner and its required model/data protocol are present in this repository.
- `RECONSTRUCTED_CONTROL`: an explicit configuration-driven control runner is present, but the exact historical source identity for the frozen paper values was not established. Its outputs must not be substituted for frozen manuscript values without a separate audit.
- `FROZEN_ARTIFACT_ONLY`: only frozen records were available; no training source is claimed.
- `ARTIFACT_DEPENDENT`: source code is present, but the run additionally requires external frozen schedules/checkpoints supplied through environment variables.

The canonical measurement, overlap/H=192 extensions, origin-training strategies, isolated PatchTST adapter, and patch-length runner are source-backed. The optimization trajectory, mask-by-head factorial, and no-PE control are runnable reconstructed controls. The full-split Transformer/MLP/Conv mixer comparison remains frozen-artifact-only because the exact historical runner was not identified. The POC source and schedules are included under `artifacts/poc_stage3/` and `artifacts/poc_stage4/`; the eight binary checkpoints remain external and are verified by the repository SHA256 manifest.

## Policy

Unavailable experiments are surfaced as machine-readable `SOURCE_UNAVAILABLE` without creating output directories or side effects. A reconstructed control is deliberately labeled and is not a claim of historical-value reproduction. Any future addition of an exact runner must include its source identity, configuration, and independent artifact audit before it is marked source-backed.
