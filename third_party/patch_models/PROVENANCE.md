# Tier-1 model source provenance

This directory contains isolated source snapshots copied from the local
experiment archive. The snapshot keeps the original model implementations and
their local support files; it is not a claim that these repositories are
official upstream dependencies of the main Time-Series-Library fork.

| Model | Source repository recorded in local `.git` | Source commit | Snapshot origin |
|---|---|---|---|
| DeformableTST | `https://github.com/luodhhh/DeformableTST.git` | `0eea1f89df35e5ff870f07f88820239bb602e9a4` | local `experiments/kdd/patch_models/tier1/deformabletst` |
| HDMixer | `https://github.com/hqh0728/HDMixer.git` | `da17f94b63b869633556b6bf65a5c68e3f322e2b` | local `experiments/kdd/patch_models/tier1/hdmixer` |
| PatchMixer | `https://github.com/Zeying-Gong/PatchMixer.git` | `cfc6c1386e7fe1633f92ef4b258ff1a4649008b4` | local `experiments/kdd/patch_models/tier1/patchmixer` |
| PatchMLP | `https://github.com/TangPeiwang/PatchMLP.git` | `b36bbc92ecfc4732acaabb6d5e8c4ff487876f5d` | local `experiments/kdd/patch_models/tier1/patchmlp` |
| PatchTST | `https://github.com/yuqinie98/PatchTST.git` | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` | local `experiments/kdd/patch_models/tier1/patchtst` |
| Pathformer | `https://github.com/decisionintelligence/pathformer.git` | `ea85d82932215e171357da47b3bc82d502344758` | local `experiments/kdd/patch_models/tier1/pathformer` |

The original model trees had local cache deletions in their working trees;
those cache files were not copied. The earlier copied-file and SHA-256 list is
preserved in `SOURCE_MANIFEST.historical.csv`. `SOURCE_MANIFEST.csv` verifies
the current release files, including import-path repairs, subsequent comment
cleanup, and restored file-level legal notices. These edits and their limits
are recorded in `docs/model_source_patches.md`.

The manifest currently covers 381 copied files, including the available source
license files. For `hash_mode=lf`, the byte count and SHA256 cover the file after
CRLF-to-LF normalization; no other bytes are normalized.
