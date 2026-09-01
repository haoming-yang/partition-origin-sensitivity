# Third-party source and licensing

The `third_party/` directory contains isolated snapshots used by the native
model source audit. These files are not original contributions of this
repository and must be used under their upstream terms.

| Snapshot | Upstream repository | Local license file | Release action |
|---|---|---:|---|
| DeformableTST | [luodhhh/DeformableTST](https://github.com/luodhhh/DeformableTST) | present | retain with notice |
| HDMixer | [hqh0728/HDMixer](https://github.com/hqh0728/HDMixer) | not found in snapshot | verify before redistribution |
| PatchMixer | [Zeying-Gong/PatchMixer](https://github.com/Zeying-Gong/PatchMixer) | present | retain with notice |
| PatchMLP | [TangPeiwang/PatchMLP](https://github.com/TangPeiwang/PatchMLP) | not found in snapshot | verify before redistribution |
| PatchTST | [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) | present | retain with notice |
| Pathformer | [decisionintelligence/pathformer](https://github.com/decisionintelligence/pathformer) | not found in snapshot | verify before redistribution |

The Time-Series-Library snapshot has a separate provenance notice at
`third_party/time_series_library/PROVENANCE.md`. Before a tagged public
release, every redistributed snapshot must have a matching upstream license
notice or be removed from the release artifact.
