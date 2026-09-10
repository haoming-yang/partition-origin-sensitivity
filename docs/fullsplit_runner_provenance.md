# Full-split runner provenance

The full-split ETTh1 Transformer/MLP/Conv runner was recovered from the
historical experiment workspace and is now included under `tools/fullsplit/`.
The recovered source used these SHA256 values before public-repository
adaptation:

| File | Historical SHA256 |
|---|---|
| `fullsplit_3run_runner.py` | `24e9560bde893918c206fbab47bd0a5d96b3258f8507df383917435489efc667` |
| `controlled_backbone_runner.py` | `2c31ae8cad31e9160dd09084b45a3d81346f8fc342d6199a0b32203525892cad` |

The public copy changes the data lookup from a machine-specific absolute path
to `PARTITION_ORIGIN_DATA_ROOT`, with a repository-relative `data/` fallback.
It adds a package-import fallback for repository execution, records a
repository-relative source path in run metadata, and accepts arbitrary integer
seeds; the batch wrapper defaults to the paper replicate set `42,43,44`. No
model was retrained during this source recovery, so the frozen manuscript
values remain the reference results until a separate rerun is performed.
