# Vendored source patches

The six Tier-1 source trees were copied as isolated snapshots. No model
architecture was replaced by a shared implementation. Initial source edits
repaired import paths required by Windows' case-insensitive filesystem:

- module imports such as `layers.PatchTST_layers` were normalized to the
  existing lowercase filenames;
- `layers.box_coder1D` was normalized to the existing `layers/box_coder1d.py`;
- cache files and `.git` directories were excluded from the snapshot.

Commit `380e13a` subsequently removed explanatory comments and normalized
trailing whitespace in source files. The later provenance repair restores the
file-level Facebook copyright notice in HDMixer's `layers/box_coder1d.py` and
the GluonTS/Amazon Apache-2.0 header in PatchMLP's `utils/timefeatures.py`.
Copyright, license, and attribution notices are not ordinary explanatory
comments and must be retained. Restoring these file-level notices does not
resolve the separate snapshot-wide license questions in `docs/THIRD_PARTY.md`.

`third_party/patch_models/SOURCE_MANIFEST.historical.csv` preserves the earlier
copied-file hashes. `SOURCE_MANIFEST.csv` describes current release contents
with explicit hash modes. The distinction and verification commands are in
`docs/provenance_checks.md`; current hashes are not an upstream byte-identity
or retraining-equivalence claim.
