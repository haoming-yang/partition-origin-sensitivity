# Same Observations, Different Forecasts

Code and configuration for the paper **“Same Observations, Different
Forecasts: Partition-Origin Effects in Patch-Based Time-Series Forecasting.”**

The repository studies whether changing only the origin of a one-dimensional
patch lattice can change a forecast when the observed history, target, and
forecasting model are fixed.

Paper: the accompanying manuscript and its frozen numerical results are
released separately from this code repository.

## What is included

- the canonical five-dataset origin-sensitivity protocol;
- controlled extensions for optimization, training-origin policy, overlap,
  horizon 192, patch length, positional encoding, and the official PatchTST
  adapter;
- the patching, masking, evaluation, and formal metric implementations;
- isolated third-party model snapshots used by the native source audit;
- configuration files and audit tools for checking protocol behavior.

The repository does not include datasets, checkpoints, full prediction dumps,
or a second copy of the paper's result tables. Historical results that depend
on unavailable training artifacts are identified in the
[experiment matrix](docs/experiment_execution_matrix.md); they are never
silently replaced by a new run.

## Install

The public environment specification is in [environment.yml](environment.yml).
For a Conda installation:

```bash
conda env create -f environment.yml
conda activate partition-origin-sensitivity
python -m pip install -r requirements.txt
```

Put the public datasets under `data/`, or set `DATA_ROOT` to an external data
directory. The expected layout and split definitions are documented in
[data/README.md](data/README.md).

## Verify the checkout

The following checks do not train a model and do not require datasets:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m src.run --config configs/core/canonical.yaml --dry-run
bash scripts/smoke_test.sh
```

The smoke script performs protocol tests, a dry-run, reconstruction checks,
and padding-sentinel checks. The separate native source smoke check performs
import/forward validation only:

```bash
bash scripts/run_tier1_smoke.sh
```

## Reproduce experiments

After the datasets are available, use the explicit entrypoints below. Training
starts only when one of these commands is selected.

```bash
bash scripts/run_core.sh
bash scripts/run_overlap.sh
bash scripts/run_h192.sh
bash scripts/run_training_policy.sh
bash scripts/run_patch_length.sh
bash scripts/run_patchtst.sh
```

These controls are intentionally separate from the historical paper records:

```bash
bash scripts/run_optimization.sh
bash scripts/run_heads.sh
bash scripts/run_pe_control.sh
bash scripts/run_poc.sh
```

See [docs/experiments.md](docs/experiments.md) for the experiment-to-code
map, [docs/reproducibility.md](docs/reproducibility.md) for metric and
provenance details, and [docs/source_audit.md](docs/source_audit.md) for the
source-availability boundary.

## Formal metrics

The reported origin gaps use the minimum-MSE denominator:

```text
G_origin   = (max(MSE_r) - min(MSE_r)) / min(MSE_r) * 100
G_interior = (max(MSE_r,r>=1) - min(MSE_r,r>=1)) / min(MSE_r,r>=1) * 100
```

The run-mean normalization used by visual diagnostics is a separate quantity
and is not substituted for either formal gap.

## Repository layout

```text
configs/                  Protocol configurations
data/                     Dataset layout instructions; data is ignored
docs/                     Reproduction, source, and experiment documentation
scripts/                  Reproduction entrypoints
src/                      Core runners, patching, evaluation, and metrics
tests/                    Fast protocol and release-contract tests
third_party/              Isolated source snapshots with provenance notices
tools/                    Static audits and post-processing utilities
```

## Citation and license

If you use this repository, cite the paper using [CITATION.cff](CITATION.cff).
Repository-level code is released under the MIT License. Files under
`third_party/` retain their upstream provenance and licensing requirements;
see [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).
