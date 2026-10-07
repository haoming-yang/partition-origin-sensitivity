# Paper result to command map

This map links the accompanying manuscript's results to commands,
configurations, outputs, reference values, tolerances, and reproduction boundaries.
It follows the current SIGMOD main-paper numbering (Sections 3--6,
Tables 1--9, Figures 1--5) and supplementary identifiers. Section/table labels,
not page numbers, are the stable identifiers. It is not a claim that new
training has reproduced historical numerical values.

## Status, tolerance, and time conventions

- **source-rerun**: implementation and configuration are present; datasets must
  be supplied. This means the declared protocol can run, not bitwise identity
  with a historical training trajectory.
- **artifact-dependent**: runnable analysis/training also requires the named
  external checkpoint. Missing inputs stop execution, without replacement.
- **frozen-artifact-only**: historical records can be inspected or summarized;
  the exact original experiment cannot currently be regenerated from this checkout.
- **reconstructed-control**: runnable new control, not the historical result.

`F` tolerance: frozen CSV/JSON full-precision recomputation should agree within
`1e-10` absolute for summary arithmetic; a printed value should agree within
half of its final reported decimal unit (e.g. 0.005 for two decimals). This is
not a training tolerance. `N` tolerance: no empirically validated cross-device
training acceptance interval exists. Report actual differences; neither sample
SD nor an invented percentage is a pass threshold. `J` tolerance: original
Jacobian projection RNG states were not archived, so fresh seeded estimates
cannot be asserted equal to those realized profiles. Readout reconstruction
errors are recorded and should be audited, not silently tolerated away.

`CPU-check` time: seconds to minutes for compact records/protocol checks, not a
formal hardware-independent bound. `unmeasured` time: no fresh full-run GPU or
CPU timing was collected for this release; do not interpret the epoch budget
as a measured runtime. Use a separate output directory and recorded wall time
to budget training on your hardware. No training is started by dry-run.

## Common protocol and paths

Unless stated otherwise: seeds **42,43,44**, `L=512`, `H=96`, `p=12`, patch
stride `12`, window stride `1`, origins `0..11`, 44 masked padded tokens.
Diagnostics at origins 0/6 retain 42 complete tokens. These are different token
counts, not interchangeable settings. `G_origin` uses min MSE in its denominator;
`G_interior` restricts to nonzero origins and does not remove boundary effects.
Within-origin-set SD is population SD; across fits it is sample SD.

Set `DATA_ROOT` or use `data/` with the five CSVs listed in
[data/README.md](../data/README.md). Mixer lookup additionally accepts
`PARTITION_ORIGIN_DATA_ROOT` (takes precedence over `DATA_ROOT`). Outputs from
`src.run` use `--output-root` (default `outputs/`); wrappers resolve the repo root.
`<id>` below is the lowercased `experiment_id` from the config.
Datasets/checkpoints are not included. Never overwrite frozen records with new results.

## Main-paper evidence

| Paper result | Command and configuration | Data / seeds / budget | Output | Status / tolerance / time |
|---|---|---|---|---|
| §3 construction, Figure 1 | `python tools/check_reconstruction.py --context 512 --patch 12`; `python tools/check_sentinel.py` | synthetic; no training | stdout; Figure 1 curves are schematic | source-rerun protocol check; exact reconstruction, zero sentinel difference; CPU-check |
| §4.1, Table 1 splits | `python -m src.run --config configs/core/canonical.yaml --dry-run`; loader checks in tests | all five; defaults 42/43/44 | stdout 15 jobs; generated `config.json` records actual window counts | source-rerun; ETTh1 8033/2785/5805; CPU-check for dry-run |
| §4.2, Tables 2--3, Figures 2--3; A3/A4a/A4b | `python -m src.run --config configs/core/canonical.yaml`; frozen tables: `python tools/reproduce_frozen_paper.py --output outputs/paper_reproduction` | all five; 42/43/44; 15 epochs | `outputs/canonical_phenomenon_27_v1/<dataset>/seed<seed>/`; frozen `table2_*`, `table3_*` CSVs | source-rerun + frozen summaries; N for training, F for frozen; unmeasured / CPU-check |
| §4.3, Table 4; A5 mixers | `python -m src.run --config configs/mixers/etth1_fullsplit_5epoch.yaml` or `bash scripts/run_mixers.sh` | ETTh1; 42/43/44; Transformer/MLP/Conv; 5 epochs | `outputs/fullsplit_cross_backbone/<model>/etth1/seed<seed>/metrics.json`, `run_meta.json` | source-rerun recovered historical runner; N; unmeasured |
| §4.3, Table 5; A6 patch lengths | `bash scripts/run_patch_length.sh`; `configs/patch_length/etth1_weather_p8_p12_p16.yaml` | ETTh1/Weather; 42/43/44; p=8/12/16, stride=p, origins 0..p-1; 15 epochs | `outputs/patch_length_audit_v1/<dataset>/p<p>/seed<seed>/` | source-rerun; N; unmeasured |
| §4.3, Table 6; A7 extensions | `bash scripts/run_overlap.sh`; `configs/extensions/etth1_overlap_s6.yaml`; `bash scripts/run_h192.sh`; `configs/extensions/etth1_h192.yaml` | ETTh1; 42/43/44; stride=6 with origins 0..5 OR H=192; 15 epochs | `outputs/<id>/etth1/seed<seed>/` | source-rerun; geometry/capacity differs in overlap; N; unmeasured |
| §4.3, Figure 4(a); A8 optimization | `bash scripts/run_optimization.sh`; `configs/optimization/etth1_epochs_5_30.yaml` | ETTh1; 42/43/44; up to 30 epochs | `outputs/optimization_trajectory_v1/etth1/seed<seed>/`; original aggregate `artifacts/paper_records/visualization_convergence_summary.csv` | historical frozen-artifact-only; new reconstructed-control; N / F; unmeasured / CPU-check |
| §4.3, Figure 4(b,c) head/mask | `bash scripts/run_heads.sh`; `configs/heads/*.yaml` | ETTh1/Weather; 42/43/44; 4 cells; 1024/16/16 capped windows; 5 epochs | `outputs/<head>_<mask>/mask_head_factorial_v1/<dataset>/seed<seed>/` | historical frozen-artifact-only (single contrast and 3-run means); new reconstructed-control; N; unmeasured |
| §4.3, Table 7; A9 training-origin policies | `bash scripts/run_training_policy.sh`; `configs/training_policy/etth1_origin_strategies.yaml` | ETTh1; 42/43/44; boundary/random/all; 15 epochs | `outputs/supplement_train_origin_strategy/etth1/<strategy>/seed<seed>/` | source-rerun; all-origin is not compute-matched; N; unmeasured |
| §4.4 padding/no-PE | padding commands above; `bash scripts/run_pe_control.sh`; `configs/positional_encoding/etth1_no_pe.yaml` | synthetic padding checks; ETTh1 no-PE 42/43/44, 15 epochs | stdout; `outputs/no_pe_control_v1/etth1/seed<seed>/` | padding source-rerun; historical no-PE frozen-artifact-only + reconstructed-control; N; CPU-check / unmeasured |
| §4.4 adapted PatchTST | `bash scripts/run_patchtst.sh`; `configs/patchtst/etth1_official_source.yaml` | ETTh1; 42/43/44; L512/p12/s12, 10 epochs, Adam | `outputs/supplement_patchtst_origin/etth1/seed<seed>/` | source-rerun; raw-scale mask-aware adapter; not native unmodified PatchTST, not the separate L336/p16/s8 8-phase study; N; unmeasured |
| §5.1, Table 8 spectrum/layers/permutation | `bash scripts/summarize_latent.sh`; frozen-checkpoint commands below | ETTh1; 42/43/44; origins 0/6, 42 complete tokens | default `outputs/latent_summary/latent_summary_etth1_o0_o6.json`, `outputs/latent_summary/latent_spectrum_etth1_o0_o6_permutation.json`; fresh analysis `seed<seed>/*.csv` | frozen summaries runnable; fresh diagnostics artifact-dependent; F; CPU-check / unmeasured inference |
| §5.2--5.3, Figure 5; supplementary Jacobian/readout | `python tools/reproduce_frozen_paper.py --output outputs/paper_reproduction --figure`; fresh commands below | Weather Figure 5; controlled and adapted PatchTST seed42; additional archived datasets/pairs | `diagnostic_audit.json`, figure assets under selected output; fresh `jacobian_profile.csv`, `head_contributions.csv`, `window_metrics.csv` | frozen-artifact-only for original Jacobian draws; checkpoint-dependent seeded resampling/readout; F / J; CPU-check / unmeasured inference |
| Supplementary Figure S1 PCA | inspect `artifacts/latent_pca_etth1_seed42_o0_o6/` and `docs/figures/paired_latent_pca_etth1_seed42_o0_o6.pdf` | ETTh1 seed42; frozen 72 paired windows | frozen coordinate CSV and PDF | frozen-artifact-only; no PCA generator entrypoint claimed; exact coordinates; CPU-check |
| §6.1--6.2, Table 9; A11/A12/A13 POC | `bash scripts/run_poc.sh`; `configs/poc/etth1_poc.yaml` documents scope, not a generic A/B/C dispatcher | ETTh1 C: 42 reused pilot, 43/44 train; 15 epochs; lambda1.0 | `outputs/poc_stage4/batch2_poc/formal_C/seed<seed>/formal_summary.json` | artifact-dependent ETTh1 C; ETTm2 transfer and complete historical A/B/C table frozen-artifact-only; N / F; unmeasured |

Reference values, not thresholds to tune toward: Table 2 mean gaps are
22.70/17.30/12.57/25.12/2.66% for ETTh1/ETTh2/ETTm1/ETTm2/Weather.
Table 4 mean gaps are 58.00/24.99/26.11% for Transformer/MLP/Conv.
Table 7 boundary/all-origin gaps are 316.35/20.25%. Table 8 non-DC low/high
Spearman associations are 0.363/0.545; adapted PatchTST overall correlation is
0.260. Inspect full-precision records and sample SD, not just these rounded means.
Policy errors: validation chooses `argmin ValMSE(r)`, exact ties choose smaller
r. `E_mean` averages errors; `E_ens` averages predictions before MSE. The
ensemble costs 12 forwards; fixed/validation-selected inference costs one
after selection. Validation sweep costs are additional, not hidden.

## Checkpoint-dependent diagnostic commands

Use actual matching checkpoint files; `--seed` is training metadata, not a
way to turn one checkpoint into another replicate.

```bash
python tools/analyze_latent_spectrum.py --checkpoint checkpoints/controlled.pt --data-root data --seed 42 --output outputs/latent_spectrum
python tools/analyze_latent_layers.py --checkpoint checkpoints/controlled.pt --data-root data --seed 42 --output outputs/latent_layers
python tools/analyze_patchtst_latent_spectrum.py --checkpoint checkpoints/patchtst.pt --data-root data --seed 42 --output outputs/patchtst_spectrum
python tools/analyze_frozen_jacobian.py --model controlled --checkpoint checkpoints/controlled_weather.pt --data-root data --dataset Weather --seed 42 --projection-seed 0 --output outputs/jacobian_weather
python tools/analyze_frozen_mechanisms.py --model controlled --checkpoint checkpoints/controlled_weather.pt --data-root data --dataset Weather --seed 42 --output outputs/readout_weather
```

For the adapted PatchTST path, use `--model patchtst` and the corresponding
checkpoint. These commands accept CLI data/checkpoint/output paths and do not
train. Follow recorded window/projection/batch settings for each historical
profile; the example default is not a promise of historical projection identity.
See [POC protocol and boundaries](../configs/poc/README.md) and
[execution matrix](experiment_execution_matrix.md) for unresolved dependencies.

The compact POC tree does not contain the complete full-precision ETTh1 C
seed43/44 or ETTm2 A/B/C result records. For these historical tables,
`frozen-artifact-only` denotes the absence of an exact rerun path, not a claim
that every original raw result is available for independent recomputation.
Binary checkpoint hashes require the explicit strict-verifier step; overridden
paths must be compared against the corresponding manifest entry separately.
