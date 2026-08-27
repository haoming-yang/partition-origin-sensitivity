# Repository assembly report

## Scope

This repository was assembled as a new Desktop directory for GitHub and TMLR supplementary use. The original manuscript and experiment directories were not modified. No training, inference, or re-evaluation was started during assembly.

## Included source

The included source is listed in the repository tree and in `docs/source_audit.md`. The core runner uses repository-relative defaults and accepts `DATA_ROOT`, `OUTPUT_ROOT`, and `DEVICE` environment variables. The recorded Time-Series-Library fork snapshot and six Tier-1 model sources are isolated under `third_party/`.

## Paper mapping

See `docs/experiments.md` for the experiment-by-experiment mapping. Exact source gaps are explicit and are not filled with guessed implementations.

## Verification record

Static checks cover Python compilation, shell syntax, configuration parsing, package help, layout reconstruction, padding-mask sentinel behavior, artifact summary audit, aggregate generation, and repository anonymity scanning. The final command results are recorded in the handoff response and should be refreshed before publication after the environment is installed.

The validated environment is `sdsd_torch`: Python 3.9.16, PyTorch 2.5.1+cu121,
NumPy 1.26.4, pandas 1.5.3, PyYAML 6.0.3, Matplotlib 3.7.1,
scikit-learn 1.6.1, einops 0.8.1, reformer-pytorch 1.4.4, and timm 0.3.2.

## Publication hygiene

Data, checkpoints, outputs, archives, caches, logs, and user-specific files are ignored. The repository-level license is MIT. Upstream third-party code remains separated from original paper code and should retain its upstream license when published.
