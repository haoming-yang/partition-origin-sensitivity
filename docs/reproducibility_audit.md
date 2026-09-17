# Reproducibility audit

This release uses a small root layout with lowercase, descriptive directory names, following the compact research-repository style visible in the [TimesNet repository](https://github.com/thuml/TimesNet). New experiment code should use lowercase snake_case names. Provenance directories such as `third_party/time_series_library`, `artifacts/poc_stage3`, and `tools/fullsplit` retain their stable names because they are part of source and artifact identity.

## Clean checkout requirements

The repository contains code, configuration, compact diagnostic records, source snapshots, and provenance manifests. It does not contain the public datasets, model checkpoints, or full prediction dumps. Exact paper-value reproduction therefore requires the external inputs listed below.

| Reproduction level | Entry point | External inputs | Result status |
|---|---|---|---|
| Protocol and implementation checks | `scripts/smoke_test.sh` | None | Reproduces phase reconstruction, sentinel, dry-run, and tests |
| Canonical training | `scripts/run_core.sh` | Five files under `DATA_ROOT` | Re-trains the source-backed controlled protocol |
| Source-backed extensions | `scripts/run_overlap.sh`, `scripts/run_h192.sh`, `scripts/run_training_policy.sh`, `scripts/run_patch_length.sh`, `scripts/run_patchtst.sh` | Dataset files; PatchTST source snapshot is tracked | Re-runs the declared configurations |
| Full-split mixer comparison | `scripts/run_mixers.sh` | ETTh1 and optional `PARTITION_ORIGIN_DATA_ROOT` | Re-runs the fixed five-epoch Transformer/MLP/Conv protocol |
| Reconstructed controls | `scripts/run_optimization.sh`, `scripts/run_heads.sh`, `scripts/run_pe_control.sh` | Dataset files | Produces labeled controls, not replacements for frozen historical values |
| POC | `scripts/run_poc.sh` | Eight checkpoint files at the paths in `README.md` | Audits or runs only when external checkpoints are present |
| Frozen latent summaries | `scripts/summarize_latent.sh` | Tracked compact artifact CSV files | Reproduces summaries and random-pairing checks without training |

## External datasets

Set `DATA_ROOT` to a directory containing:

```text
ETT-small/ETTh1.csv
ETT-small/ETTh2.csv
ETT-small/ETTm1.csv
ETT-small/ETTm2.csv
weather/weather.csv
```

`data/README.md` records the split boundaries and the train-only standardization rule. Dataset licenses and checksums must be verified by the user before redistribution.

## External checkpoints

The repository tracks POC schedules, provenance records, and SHA256 expectations. The eight binary POC checkpoints are intentionally external; `python tools/verify_poc_stage3.py` reports their absence without failing, while `--strict` requires all eight. No missing checkpoint is silently replaced by a new run.

The canonical and diagnostic training checkpoints are also external. Tracked compact artifacts can be audited and summarized, but they cannot be used to claim that a fresh training run exactly reproduces a historical checkpoint without the original checkpoint and protocol record.

## Entrypoint contract

Every public shell entrypoint resolves its own repository root and changes into it before invoking Python. Commands can therefore be called from any working directory. `DATA_ROOT`, `OUTPUT_ROOT`, `SEEDS`, `MIXER_OUTPUT_ROOT`, `PARTITION_ORIGIN_DATA_ROOT`, `SPECTRUM_ROOT`, `LAYERS_ROOT`, and `PERMUTATIONS` retain their documented meanings.

The shell entrypoints target Bash, Git Bash, or WSL. On Windows PowerShell, run the equivalent Python commands from the README or enter a Bash-compatible shell.

The public environment is defined by `environment.yml`. A local Conda environment may use a different name; no runner depends on the environment name. Use the environment's Python to run the commands, for example `python -m pytest -q` and `python -m src.run --config configs/core/canonical.yaml --seeds 42,43,44 --dry-run`.

## Verification commands

The following commands are read-only unless an output directory is explicitly selected:

```bash
python -m pytest -q
python -m src.run --config configs/core/canonical.yaml --seeds 42,43,44 --dry-run
python tools/check_reconstruction.py --context 512 --patch 12
python tools/check_sentinel.py
python tools/verify_source_manifests.py
python tools/verify_poc_stage3.py
python tools/check_anonymity.py --root .
```

`python tools/audit_artifacts.py --root outputs --out outputs/audit_report.json` validates generated summaries. It does not establish numerical equivalence with historical paper values.

## Publication boundary

Before publishing a run, preserve the exact Git revision, configuration, seed list, data checksum, checkpoint provenance, environment versions, and output manifest. A successful dry-run proves only that the job matrix is valid; it does not prove that a training result matches the manuscript.
