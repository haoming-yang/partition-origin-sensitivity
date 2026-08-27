# Repository assembly report

## Scope

This repository was assembled as a new Desktop directory for GitHub and TMLR supplementary use. The original manuscript and experiment directories were not modified. No training, inference, or re-evaluation was started during assembly.

## Included source

The included source is listed in the repository tree and in `docs/source_audit.md`. The core runner uses repository-relative defaults and accepts `DATA_ROOT`, `OUTPUT_ROOT`, and `DEVICE` environment variables. Official PatchTST components are isolated under `third_party/time_series_library/`.

## Paper mapping

See `docs/experiments.md` for the experiment-by-experiment mapping. Exact source gaps are explicit and are not filled with guessed implementations.

## Verification record

Static checks cover Python compilation, shell syntax, configuration parsing, package help, layout reconstruction, padding-mask sentinel behavior, artifact summary audit, aggregate generation, and repository anonymity scanning. The final command results are recorded in the handoff response and should be refreshed before publication after the environment is installed.

## Publication hygiene

Data, checkpoints, outputs, archives, caches, logs, and user-specific files are ignored. The repository-level license is MIT. Upstream third-party code remains separated from original paper code and should retain its upstream license when published.
