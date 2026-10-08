# Reproducibility and provenance

This document supports reproduction of the accompanying manuscript.
Use [reproduction_map.md](reproduction_map.md) for paper-result commands and
source/frozen-record boundaries. A successful command is not a historical
numerical reproduction claim.

Core and extension outputs record actual seed, dataset, split/window counts,
context/horizon/patch/stride/origins, training configuration and metric definitions.
`config.json` is the effective configuration, not an unconsumed YAML copy.
`summary.json` holds metrics; `PROVENANCE.json` embeds that configuration,
source and dataset SHA256 values, Git commit/dirty status, software versions,
device and generation timestamp. `validation_per_origin.csv` from controlled
runs records the selected checkpoint's validation sweep. Checkpoints are
selected before the final test sweep; validation MSE, never test MSE, chooses
the inference origin (exact ties choose the smaller origin).

Full-split mixers use `run_meta.json` and `config.json` instead of the core
filenames, with commit/dirty status/time/software/device and effective config.
Post-hoc diagnostics write these runtime fields alongside checkpoint hashes
and their existing analysis settings. ETTh1 formal POC adds runtime metadata
to `PROVENANCE_FORMAL.json`; immutable historical records are not rewritten.
`git_dirty=true` means HEAD alone does not identify the local edits; preserve
the source hashes and working-tree diff alongside a result.

Formal metrics are kept distinct from visualization-only quantities:

```text
G_origin = (max_r MSE_r - min_r MSE_r) / min_r MSE_r * 100
G_interior = (max_{r>=1} MSE_r - min_{r>=1} MSE_r) / min_{r>=1} MSE_r * 100
S_theta = 2 * n_origins / (n_origins - 1) * origin_prediction_variance
```

No mean-normalized visualization quantity is used to compute either formal gap.
The core summary records both variance and uniform-distinct-pair S_theta.
The recovered mixer runner's legacy field named `S_theta` is actually origin
prediction variance, without the pairwise factor. Its historical field is
preserved, not redefined; do not pool it with the core/Stage4 S_theta. Main
Table 4 compares origin gaps, not this legacy diagnostic field.

E0/E_star are fixed/validation-selected inference-policy errors; E_mean is
an error average, and E_ens is MSE after averaging predictions. Ensemble
requires one forward per origin; selection also requires a validation sweep.
Metrics stay standardized except for the explicitly raw-scale adapted PatchTST.
Population SD applies across origins; sample SD applies across fits.

The repository does not include data, checkpoints, or user-specific paths.
Generated files are ignored by Git. Checkpoint placement is described in the
README; missing input files do not authorize substitute experiments.

`artifacts/` is immutable historical evidence, not a generated-output root.
Latent summary/permutation defaults are `outputs/latent_summary/`. These tools,
seed-isolated analysis outputs and table summary tools reject paths under the
repository's frozen directory and refuse to overwrite tracked destinations.
Configuration validation and generated-output protections are enforced by the
experiment entrypoints. See [reproduction_map.md](reproduction_map.md) for
execution boundaries and result availability.

## Environment and numerical identity

`environment.yml` declares Python 3.9.16/PyTorch 2.5.1/CUDA runtime 12.1.
The CPU CI specification uses Python 3.11/PyTorch 2.5.1. These are pinned
specifications; this audit did not install or benchmark their CUDA combination.
The local engineering checks used Python 3.13.5/PyTorch 2.13.0+cpu,
NumPy 2.2.6/Pandas 3.0.5. Passing these checks is not validation of a historical
training environment. A separate clean CPU environment was installed and checked
with Python 3.11.17, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu, NumPy 1.26.4,
and pandas 1.5.3; `pip check`, the tests, dry-run and CPU smoke passed.
This validates engineering checks, not full training or CUDA execution.
No scientific dependency was upgraded for this cleanup. The optional native
six-model audit and its `timm` compatibility shim were removed; the paper's
controlled models and adapted PatchTST do not import `timm`.
Capture `python -m pip freeze` and `nvidia-smi` with new runs as well as provenance.

There is no established fresh-training numerical tolerance or measured runtime
for this release. Frozen-table arithmetic/printed rounding and stochastic
diagnostic resampling have different acceptance boundaries, documented in the map.

The [reproduction map](reproduction_map.md) gives each result's source status.
`source-rerun` means the declared runner and configuration are present, not
that fresh training reproduces a historical checkpoint. `reconstructed-control`
marks a runnable control whose exact historical source identity is unresolved.
`artifact-dependent` requires external checkpoints, while
`frozen-artifact-only` has no exact rerun path in this checkout. The complete
ETTm2 POC transfer and historical A/B/C table remain frozen-artifact-only;
the compact POC tree does not contain every full-precision table record.
Anonymity checks remain available through `scripts/run_audits.sh`.
