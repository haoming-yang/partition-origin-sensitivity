# Release integrity and historical provenance

Current release integrity and historical provenance are separate checks.
Updating a release hash must follow inspection of the corresponding diff; it
does not establish that training reproduces a historical experiment.

## Current release checks

```bash
python tools/verify_poc_stage3.py
python tools/verify_source_manifests.py
```

Both commands are read-only and run in CI. The POC check validates tracked
source/protocol records and any external checkpoints present at their listed
paths. Missing external checkpoints are reported but optional by default;
`--strict` requires all eight. A present checkpoint with a wrong hash always
fails, even without `--strict`.

Tracked POC files are text: their SHA256 is computed after replacing CRLF with
LF, without stripping whitespace or changing any other bytes. External
checkpoints use exact raw bytes. Each vendored-source CSV entry declares
`hash_mode`: `lf` uses the same newline normalization, and `raw` hashes exact
bytes. The `bytes` column counts the bytes actually hashed. This avoids
checkout-specific newline changes being mistaken for source modifications.

The verifier rejects empty manifests, missing files, and changed contents.
POC header lines start with `#`; every nonblank, noncomment line must be a
well-formed SHA256 record. Corrupt records are errors, not silently skipped.
Hashes attest only to the listed files, not to upstream endorsement, complete
license clearance, or numerical equivalence of a newly trained model.

## Preserved historical manifests

The previous manifests were copied byte-for-byte before repairing release
checks. They retain all recorded historical digests, including the eight
external checkpoint values; no checkpoint was regenerated or uploaded.
`.gitattributes` preserves these archive files without newline conversion.

| Archive | SHA256 of archived manifest file |
|---|---|
| `artifacts/poc_stage3/SHA256SUMS.historical.txt` | `c40e53b85d8247ed31441e6c6785f52b74c7d6222584e3a9901cc18b35636956` |
| `third_party/patch_models/SOURCE_MANIFEST.historical.csv` | `95b9eb7747fd7a4219cc62953e9f57931f57b740dc52aa20f5f8519cdf466b50` |
| `third_party/time_series_library/SOURCE_MANIFEST.historical.csv` | `1807dcdacfbeedc4592fb705b59b5dbf52f1f2eb41fed05d3b7d2ab0d4815145` |

The archived lists use the historical byte conventions and are not checks of
the current modified source. For an intentional source change, inspect the
diff, update only the affected current release records using the declared
hash mode, leave the archived manifests and external checkpoint values alone,
and rerun both verifiers and the test suite. Source changes since copying,
including comment cleanup and restored notices, are recorded in
`docs/model_source_patches.md`.

The source-list generator emits the current CSV schema and defaults to `lf`.
For example, generate a candidate outside the source tree before reviewing it:

```bash
python tools/generate_source_manifest.py --root third_party/patch_models/tier1 --output candidate_sources.csv
```

Use `--hash-mode raw` only for a tree requiring exact binary-byte hashes.
The generator excludes cache files, its own output, and existing source
manifests so repeated generation does not hash the manifest into itself.
