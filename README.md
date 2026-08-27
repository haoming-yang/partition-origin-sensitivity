# Same Observations, Different Forecasts

This repository contains the reproducibility package for **Same Observations, Different Forecasts: Partition-Origin Sensitivity in Patch-Based Time-Series Forecasting**.

The scientific object is partition-origin sensitivity under an observation-equivalent fixed-observation intervention. The primary evidence is the canonical five-dataset measurement. Optimization, mixer, head, ranking, and consistency analyses are bounded diagnostics or probes.

## Scope

The repository is intentionally source-conservative. It includes the verified controlled runner, phase reconstruction audit, metric schema, isolated snapshots of the six Tier-1 model sources, paper summaries, configurations, and audit utilities. Configurations distinguish source-backed runners from reconstructed controls and artifact-dependent workflows; missing historical source is never silently inferred from frozen numbers.

## Layout

```text
configs/                         Protocol configurations
data/                            User-provided datasets, never committed
docs/                            Experiment map, provenance, and audit notes
results/                         Paper-facing summaries and generated outputs
scripts/                         Reproduction and audit entrypoints
src/                             Controlled training, evaluation, metrics, patching
  third_party/time_series_library/ Time-Series-Library fork snapshot
  third_party/patch_models/      Isolated Tier-1 model snapshots
tools/                           Static audits and post-processing utilities
```

## Environment

The validated environment is the user's `sdsd_torch` environment. Install the declared dependencies with:

```bash
python -m pip install -r requirements.txt
```

Set `DATA_ROOT`, `OUTPUT_ROOT`, and optionally `DEVICE=cpu` or `DEVICE=cuda`. The expected datasets are documented in `data/README.md`.

## Reproduction

The following commands are available:

```bash
bash scripts/reproduce_all.sh core
bash scripts/reproduce_all.sh overlap
bash scripts/reproduce_all.sh h192
bash scripts/reproduce_all.sh training-policy
bash scripts/reproduce_all.sh patchtst
bash scripts/reproduce_all.sh audits
bash scripts/reproduce_all.sh tables
bash scripts/reproduce_all.sh figures
bash scripts/run_tier1_smoke.sh
```

The runnable commands execute training when explicitly selected. No command is run during repository assembly. The audit-only commands do not train.

The full-split Transformer/MLP/Conv mixer comparison remains `SOURCE_UNAVAILABLE` because its exact historical runner was not identified. Optimization, mask-head, and no-PE entries are runnable `RECONSTRUCTED_CONTROL` implementations and are not replacements for frozen paper values. The POC source is `ARTIFACT_DEPENDENT` and requires external schedules/checkpoints. See `docs/source_audit.md`.

## Data and outputs

Datasets are not redistributed. Place them below `data/` or point `DATA_ROOT` to an external directory. Generated checkpoints and outputs are ignored by Git. Paper summaries copied from the manuscript project are marked as frozen summaries and are not regenerated from guessed values.

## Paper mapping

The complete paper-to-code mapping is in `docs/experiments.md`. Reproducibility and provenance rules are in `docs/reproducibility.md`.

## License

The repository-level code is released under the MIT License. Files under `third_party/` retain their source provenance; see the corresponding provenance files and use each snapshot in accordance with its source license.
