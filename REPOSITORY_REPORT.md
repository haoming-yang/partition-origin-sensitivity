# Repository report

## Target

The clean supplementary repository is `partition-origin-sensitivity` on the Desktop. It was created as a new Git repository with branch `main`. The original manuscript, experiment archive, and Time-Series-Library directories were not modified.

## Scientific and operational boundary

No training, inference, or re-evaluation was started during assembly. Frozen paper values were not edited. Formal origin gaps retain the minimum-MSE denominator. Mean-normalized quantities are not used as formal metrics.

## Included components

```text
configs/
  core/                 Five canonical dataset configurations
  extensions/           Overlap and H=192 configurations
  heads/                Frozen mask-head audit descriptors
  mixers/               Frozen full-split mixer descriptor
  optimization/         Frozen trajectory descriptor
  patch_length/         Frozen patch-length descriptor
  patchtst/             Official-source PatchTST configuration
  poc/                  Frozen POC descriptors
  positional_encoding/  Frozen no-PE descriptor
  training_policy/      Origin-training strategy configuration
data/                   README only; datasets excluded
docs/                   Experiment map, source audit, reproducibility
results/                Frozen paper summary and README
scripts/                Setup, reproduction, audit, table, figure commands
src/                    Controlled runner, evaluation, patching, metrics
third_party/            Isolated official PatchTST source snapshot
tools/                  Audit, aggregation, table, figure utilities
```

## Paper-to-code status

Source-backed runners are present for the canonical measurement, overlap audit, H=192 audit, training-origin strategies, isolated PatchTST adapter, patch-length audit, and native Tier-1 model source harness. Explicit reconstructed controls are present for optimization, mask-head, and no-PE configurations; these are labeled `RECONSTRUCTED_CONTROL` and must not be read as exact reproductions of frozen historical values. The full-split Transformer/MLP/Conv comparison remains `SOURCE_UNAVAILABLE`. The POC source is `ARTIFACT_DEPENDENT` and requires external Stage-3/Stage-4 artifacts.

## Static verification

The following checks passed in the tested Python 3.9/CUDA 12.1 environment
described by `environment.yml`, without starting training:

| Check | Result |
|---|---|
| Python `compileall` for `src` and `tools` | PASS |
| CLI help and unavailable-source help | PASS |
| 512/12 phase reconstruction, 12 origins, exact-once coverage | PASS |
| Padding sentinel influence test | PASS; maximum absolute difference 0.0 |
| YAML parsing | PASS; 18 files |
| Shell syntax | PASS; all scripts |
| Anonymity scan | PASS; 0 issues |
| Empty-output artifact audit | PASS |
| Empty-output table aggregation | PASS |
| Empty-output figure generation | PASS |
| Six-model native import/forward smoke | PASS; model-specific native input lengths |
| Vendored SHA-256 manifests | PASS; 381 Tier-1 files and 10 fork-snapshot files |

Observed validation stack: Python 3.9.16, NumPy 1.26.4, pandas 1.5.3, Matplotlib 3.7.1, PyTorch 2.5.1+cu121, PyYAML 6.0.3.

The release dependency files now pin the validated non-CUDA package versions;
`environment.yml` pins the validated Python/PyTorch/CUDA stack and installs
`reformer-pytorch` through pip.

## Reproduction caveat

The runnable commands are source reproduction entrypoints only where the configuration is marked `SOURCE_PRESENT`. `RECONSTRUCTED_CONTROL` outputs are explicitly separated from frozen manuscript values. Data and external Stage-4 artifacts must be supplied separately. The exact protocol and artifact identity must be audited before any frozen result is replaced.

## Git hygiene

Data, checkpoints, outputs, archives, caches, logs, and generated diagnostics are ignored by `.gitignore`. The official upstream license is copied to `third_party/time_series_library/UPSTREAM_LICENSE`. Release commit and remote state are recorded by Git metadata during handoff.
