# Review artifact and exact-result boundaries

The public repository retains its author metadata. Use a separate export for review:

```bash
python tools/export_review_artifact.py --output /path/outside/checkout/review_artifact
```

The export includes 8 POC input checkpoints and 4 diagnostic checkpoints when installed locally, each checked against `artifacts/review_checkpoint_manifest.json`. They are binary artifacts excluded from Git, not inferred or replaced by retraining. A prepared review package contains them in `checkpoints/`; extract that package and run:

```bash
python tools/verify_poc_stage3.py --strict
python tools/verify_source_manifests.py
python tools/check_anonymity.py --root . --double-blind
python tools/reproduce_frozen_paper.py --output outputs/paper_reproduction --figure
```

The package is prepared locally; no public download URL has been published by this tooling. Upload the anonymous package through the chosen anonymous artifact host and supply its URL in the submission system. A public Git clone alone does not contain the weights. The export fails if a required weight is absent or mismatched.

## Scope

- Tables 2 and 3: regenerate three-run means/sample SD and policy changes from full-precision frozen inference and dispersion records; no training or dataset needed.
- Figure 5: regenerate the archived normalized Jacobian and readout curves and audit the contribution mass/reconstruction error; no inference needed.
- Figure 2/3 source CSVs and multi-pair gradient profiles: preserved in `artifacts/paper_records` and `artifacts/frozen_diagnostics` with SHA-256 manifests.
- Other tables: use their documented experiment-specific entrypoints and frozen records; the generic new-run summary is not a claim to reproduce all paper tables.
- POC: eight historically required binary inputs are included in the review package. Rerunning training also requires the public datasets and the documented environment.
- Frozen diagnostics: four seed-42 checkpoint hashes are included. Jacobian records describe the first 512 chronological windows and original batch sizes. Original projection RNG states were not archived; retain the CSVs as the original realized estimates. New `--projection-seed` runs are separately seeded resamples, not replacements for those historical values.
- Core training weights beyond these diagnostic and POC inputs are not included. Core code supports new training, while the archived inference records support historical table recomputation.

## Definitions

Readout CSV values are absolute contributions averaged across windows, channels and horizon entries. Normalize these values over tokens, after aggregation. PatchTST contributions include per-window channel scaling. This is not an L2-norm share. Ordinary and difference Jacobian energies are averaged over projections, windows and input channels before positional normalization.

`--seed` on the Jacobian CLI identifies the checkpoint training seed. `--projection-seed` controls a dedicated generator independent of model initialization/global random draws. The projection seed, batch size, window selection, device and environment must all be recorded. No cross-device bitwise identity is promised.

`reproduce_tables.sh new-runs` groups by experiment, dataset and recorded protocol, rejects repeated seeds, retains source paths and emits sample SD. Singleton SD is missing rather than zero. It does not silently combine incomplete or invalid runs. `reproduce_tables.sh frozen` is the default archived Table 2/3 path.
