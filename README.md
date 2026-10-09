# Partition-Origin Sensitivity

This repository studies whether changing the origin of a fixed patch lattice
changes forecasting results while the observed history and target stay fixed.
It provides the paper's experiment code and frozen records for results whose
original inputs are not all available.

The illustration is schematic: the same observations can be assigned to
different patches and produce different forecasts from a fixed model.

<p align="center">
  <img src="docs/figures/partition_origin_effect_case.png" alt="Identical observations assigned to different partition origins can yield different forecasts" width="100%">
</p>

## Quick Start

Use Python 3.11 on CPU for the quick checks. From the repository root, create
an environment, activate it using your shell's command, then install the pinned
dependencies. Ensure that the active `python` refers to Python 3.11.

```bash
python -m venv .venv
```

After activating `.venv`:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

Run the fast checks. They do not require datasets, checkpoints, or a GPU:

```bash
python -m pytest -q
python -m src.run --config configs/core/canonical.yaml --dry-run
bash scripts/smoke_test.sh
```

These checks need no dataset, checkpoint, or GPU; they do not rerun paper-scale training.
[`environment.yml`](environment.yml) is the CUDA reference environment.

## Reproducing results

The paper's default seeds are **42, 43, and 44**. Override them with
`--seed`/`--seeds`, or `SEEDS` for shell entrypoints. New results go to
`outputs/`; [`artifacts/`](artifacts/) holds frozen evidence and must not be
used as a generated-output directory.

After placing the five public CSV datasets in the layout described by
[`data/README.md`](data/README.md), start the canonical experiment:

```bash
python -m src.run --config configs/core/canonical.yaml --seeds 42,43,44
```

This launches 15 training jobs. It is not a smoke test. Use
[`docs/reproduction_map.md`](docs/reproduction_map.md) to locate every other
paper experiment, its command and configuration, expected output, and
reproduction status. The compact dispatcher lists its commands with
`bash scripts/reproduce_all.sh list`.

Frozen summaries and summaries of newly generated runs can be produced with:

```bash
bash scripts/reproduce_all.sh tables
bash scripts/reproduce_all.sh tables new-runs
```

These commands write generated files under `outputs/`; they do not overwrite
the tracked evidence in `artifacts/`. The frozen path recomputes summaries from
archived records, while `new-runs` aggregates results already present in the
output tree. Neither command by itself reruns model training.

To render the archived diagnostic visualization, in a Bash-compatible shell:

```bash
MPLBACKEND=Agg python tools/reproduce_frozen_paper.py --output outputs/paper_reproduction --figure
```

## Independent random-projection verification

[Independent Random-Projection Robustness Check](verification/random_projection_robustness/README.md)
accompanies Supplementary Section 3.6. It covers four model--dataset
configurations and ten projection seeds, providing diagnostic code, frozen
checkpoints, projection records, per-seed paired results, and reproduction
instructions. Raw dataset CSVs are not distributed; use the fixed-version
sources and SHA-256 checks in the package's download instructions.

The independent check does not recover the historical Jacobian projection RNG
state, and global CUDA bitwise determinism was not verified.

## Reproduction boundaries

Not every historical result can be regenerated from this checkout. Some
analyses require datasets or checkpoints that are not included; some historical
controls are available only as frozen records or are explicitly marked as
reconstructed controls. Missing inputs stop the corresponding run rather than
triggering a substitute experiment. The reproduction map and
[`docs/reproducibility.md`](docs/reproducibility.md) describe these boundaries,
the recorded protocols, and the expected sources of numerical variation.

The frozen POC protocol uses **1% validation-error eligibility slack**. All
archived candidates also satisfy zero slack, so the selected value remains
**\(\lambda=1.0\)**. See [`configs/poc/README.md`](configs/poc/README.md) for
the exact selection and input requirements.

## Diagnostics and data

Post-hoc diagnostics require matching data and checkpoints; they do not
retrain models. Frozen latent summaries can be regenerated with
`bash scripts/summarize_latent.sh`. The saved PCA NPZ can be exported to CSV:

```bash
python tools/export_pca_csv.py --input artifacts/latent_pca_etth1_seed42_o0_o6/pooled_latents_and_pca.npz --output outputs/pca_csv
```

Datasets are not included. Put the CSVs under `data/` or set `DATA_ROOT` to an
external data directory. The expected paths, split definitions, and window
counts are documented in [`data/README.md`](data/README.md). Dataset licenses
should be checked at their respective sources before redistribution.

## Repository layout

```text
artifacts/       Frozen compact records and provenance; do not use for new outputs
configs/         Experiment and protocol configurations
data/            Dataset layout instructions; datasets are not included
docs/            Reproduction map and provenance guidance
scripts/         Compact dispatcher, smoke, and specialist entrypoints
src/             Core runners, patching, evaluation, and metrics
tests/           Protocol, analysis, and release-contract tests
third_party/     Isolated upstream source snapshots and license notices
tools/           Audits, post-hoc diagnostics, and historical protocol helpers
outputs/         Default destination for generated runs and summaries (Git-ignored)
```

## Further documentation

- [Paper-result reproduction map](docs/reproduction_map.md)
- [Reproducibility and provenance](docs/reproducibility.md)
- [Release integrity and historical provenance](docs/provenance_checks.md)
- [Dataset paths and split protocol](data/README.md)
- [POC protocol and required inputs](configs/poc/README.md)
- [Third-party licenses](docs/THIRD_PARTY.md)

## Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Repository
code is released under the MIT License. Files under `third_party/` retain their
upstream provenance and licensing terms; see [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).
