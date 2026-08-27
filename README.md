<div align="center">

# Same Observations, Different Forecasts

### Partition-Origin Sensitivity in Patch-Based Time-Series Forecasting

<a href="https://github.com/haoming-yang/partition-origin-sensitivity"><img src="https://img.shields.io/badge/reproducibility-source--conservative-1f6feb?style=for-the-badge" alt="source conservative"></a>
<a href="environment.yml"><img src="https://img.shields.io/badge/Python-3.9-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9"></a>
<a href="environment.yml"><img src="https://img.shields.io/badge/PyTorch-2.5.1%2BCUDA%2012.1-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch CUDA"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2ea44f?style=for-the-badge" alt="MIT license"></a>

**A controlled audit of one hidden degree of freedom in discrete forecasting representations.**

</div>

> **Scientific object:** Partition-origin sensitivity under an observation-equivalent, fixed-observation intervention.
>
> **Primary evidence:** Canonical five-dataset measurement.
>
> **Secondary evidence:** Optimization, mixer, head, ranking, and consistency diagnostics with explicit provenance boundaries.

## The central question

```text
same observed history X
          |
          v
different partition origin r
          |  same forecaster, same target, same observations
          v
possibly different forecast Yr
```

This repository provides the code, configurations, source snapshots, metric
definitions, and audits needed to study that intervention without silently
replacing frozen manuscript values.

## Evidence status at a glance

| Status | Meaning |
|---|---|
| `SOURCE_PRESENT` | Runner and required model/data protocol are included |
| `RECONSTRUCTED_CONTROL` | Executable control; not an identity claim for historical frozen values |
| `ARTIFACT_DEPENDENT` | Source is included but external schedules/checkpoints are required |
| `FROZEN_ARTIFACT_ONLY` | Historical records are preserved; exact training source was not identified |

The full-split Transformer/MLP/Conv comparison is intentionally fail-closed as
`FROZEN_ARTIFACT_ONLY`. Optimization, mask-head, and no-PE entries are
explicit `RECONSTRUCTED_CONTROL` implementations. The POC workflow is
`ARTIFACT_DEPENDENT`. See the complete [experiment execution matrix](docs/experiment_execution_matrix.md).

## Repository map

```text
configs/                         Protocol configurations
data/                            Dataset instructions; data never committed
docs/                            Experiment map, provenance, and audit notes
results/                         Frozen paper summaries and generated outputs
scripts/                         Reproduction and audit entrypoints
src/                             Training, evaluation, metrics, and patching
third_party/time_series_library/ Time-Series-Library fork snapshot
third_party/patch_models/        Six isolated Tier-1 model snapshots
tools/                           Static audits and post-processing utilities
```

## Environment setup

The repository provides a public, reproducible environment specification. The
tested package versions are:

```text
Python        3.9.16
PyTorch       2.5.1+cu121
NumPy         1.26.4
pandas        1.5.3
PyYAML        6.0.3
Matplotlib    3.7.1
scikit-learn  1.6.1
einops        0.8.1
reformer      1.4.4
timm          0.3.2
```

`requirements.txt` pins the non-CUDA Python packages listed above.
`environment.yml` additionally specifies Python, PyTorch, and CUDA. Keeping
the CUDA stack in the environment file avoids machine-specific pip wheel
selection.

```bash
conda env create -f environment.yml
conda activate partition-origin-sensitivity
python -m pip install -r requirements.txt
```

Set `DATA_ROOT`, `OUTPUT_ROOT`, and optionally `DEVICE=cpu` or `DEVICE=cuda`.
Expected dataset paths are documented in [data/README.md](data/README.md).

## Reproduction entrypoints

```bash
# source-backed experiments
bash scripts/reproduce_all.sh core
bash scripts/reproduce_all.sh overlap
bash scripts/reproduce_all.sh h192
bash scripts/reproduce_all.sh training-policy
bash scripts/reproduce_all.sh patchtst
bash scripts/reproduce_all.sh patch-length

# no-training source/model audit
bash scripts/reproduce_all.sh tier1-smoke
bash scripts/reproduce_all.sh audits

# explicit reconstructed controls
bash scripts/reproduce_all.sh optimization
bash scripts/reproduce_all.sh heads
bash scripts/reproduce_all.sh pe-control

# post-processing
bash scripts/reproduce_all.sh tables
bash scripts/reproduce_all.sh figures
```

The commands execute training only when explicitly selected. Repository
assembly and validation did not start training; validation used dry-runs,
static checks, and import/forward smoke tests.

## Model-source audit

The six Tier-1 snapshots preserve their recorded repository URLs, commits,
licenses, and copied-file SHA-256 manifest in
[third_party/patch_models/PROVENANCE.md](third_party/patch_models/PROVENANCE.md).
The Time-Series-Library fork snapshot has separate provenance in
[third_party/time_series_library/PROVENANCE.md](third_party/time_series_library/PROVENANCE.md).

The no-training native smoke command checks PatchTST, PatchMixer, PatchMLP,
Pathformer, HDMixer, DeformableTST, and the isolated official PatchTST adapter.
It uses protocol-valid model-specific input lengths rather than forcing every
architecture into one incompatible tensor shape.

## Data, outputs, and frozen evidence

Datasets, checkpoints, logs, archives, and generated outputs are excluded from
Git. Put datasets below `data/` or point `DATA_ROOT` to an external directory.
Frozen paper summaries are archival records and are not regenerated from
guessed values.

Formal metrics retain the paper definitions:

```text
G_origin   = (max(MSE_r) - min(MSE_r)) / min(MSE_r) * 100
G_interior = (max(MSE_r,r>=1) - min(MSE_r,r>=1)) / min(MSE_r,r>=1) * 100
```

Visualization-only mean normalization is never substituted for either formal
gap. Full provenance rules are in [docs/reproducibility.md](docs/reproducibility.md).

## License

Repository-level code is released under the MIT License. Files under
`third_party/` retain their source provenance and must be used in accordance
with their respective source licenses.
