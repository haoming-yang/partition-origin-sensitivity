# Vendored source patches

The six Tier-1 source trees were copied as isolated snapshots. No model
architecture was replaced by a shared implementation. The only source edits
made after copying were import-path repairs required by Windows' case-
insensitive filesystem:

- module imports such as `layers.PatchTST_layers` were normalized to the
  existing lowercase filenames;
- `layers.box_coder1D` was normalized to the existing `layers/box_coder1d.py`;
- cache files and `.git` directories were excluded from the snapshot.

The exact file-level hashes are recorded in
`third_party/patch_models/SOURCE_MANIFEST.csv`.
