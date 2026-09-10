# Experiment execution matrix

This table separates runnable source from historical-value identity. A command
being executable does not authorize replacing frozen manuscript numbers with a
new run.

| Paper evidence | Configuration / command | Repository status | Meaning |
|---|---|---|---|
| Canonical five-dataset measurement | `configs/core/canonical.yaml` / `scripts/run_core.sh` | `SOURCE_PRESENT` | Configuration-driven controlled runner with the canonical split and metric definitions |
| H=192 extension | `configs/extensions/etth1_h192.yaml` / `scripts/run_h192.sh` | `SOURCE_PRESENT` | Source-backed controlled extension |
| Stride-6 overlap extension | `configs/extensions/etth1_overlap_s6.yaml` / `scripts/run_overlap.sh` | `SOURCE_PRESENT` | Source-backed relaxed lattice audit |
| Training-origin strategies | `configs/training_policy/etth1_origin_strategies.yaml` / `scripts/run_training_policy.sh` | `SOURCE_PRESENT` | Boundary, random-origin, and all-origin controls |
| Public PatchTST check | `configs/patchtst/etth1_official_source.yaml` / `scripts/run_patchtst.sh` | `SOURCE_PRESENT` | Isolated adapter around the recorded Time-Series-Library fork snapshot |
| Patch-length audit | `configs/patch_length/etth1_weather_p8_p12_p16.yaml` / `scripts/run_patch_length.sh` | `SOURCE_PRESENT` | Canonical runner over the recorded patch lengths |
| Optimization trajectory | `configs/optimization/etth1_epochs_5_30.yaml` / `scripts/run_optimization.sh` | `RECONSTRUCTED_CONTROL` | Executable control implementation; not an identity claim for frozen historical values |
| Mask-head factorial | `configs/heads/*.yaml` / `scripts/run_heads.sh` | `RECONSTRUCTED_CONTROL` | Executable flattened/pooled control; not a replacement for the frozen mask-head records |
| No-PE control | `configs/positional_encoding/etth1_no_pe.yaml` / `scripts/run_pe_control.sh` | `RECONSTRUCTED_CONTROL` | Executable control with position parameters disabled |
| Full-split Transformer/MLP/Conv comparison | `configs/mixers/etth1_fullsplit_5epoch.yaml` / `scripts/run_mixers.sh` | `SOURCE_PRESENT` | Complete ETTh1 runner under `tools/fullsplit/`; default batch uses seeds 42, 43, 44 and writes isolated model/seed outputs |
| ETTh1/ETTm2 POC | `scripts/run_poc.sh` | `ARTIFACT_DEPENDENT` | Included Stage-4 source and schedules require eight external checkpoints |

The six native Tier-1 sources have a no-training import/forward check:
`scripts/run_tier1_smoke.sh`. It uses model-specific native input lengths; in
particular, Pathformer uses 672 for a 512-step context because its 96-origin
phase lattice and internal 16/12/8/32 scales require a common divisible length.

Validation of this repository used dry-runs, parser/compile checks, phase
reconstruction, sentinel checks, and model import/forward smoke only. No paper
training run was started during assembly.
