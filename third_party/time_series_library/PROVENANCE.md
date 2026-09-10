# Time-Series-Library snapshot provenance

This is a local fork snapshot used by the isolated PatchTST adapter. It is
not labeled as canonical upstream Time-Series-Library.

- Recorded remote: `https://github.com/haoming-yang/Time-Series-Library.git`
- Recorded commit: `61f68df6965b8d4061a08d0d09b6d69dba8728c8`
- Worktree status at audit: clean
- Role in this repository: source snapshot for `models/PatchTST.py` and its
  required layers/utilities

The snapshot retains its upstream license notice in `UPSTREAM_LICENSE`.

`SOURCE_MANIFEST.historical.csv` retains the earlier source hashes.
`SOURCE_MANIFEST.csv` describes the current release after comment cleanup.
For `hash_mode=lf`, the byte count and SHA256 normalize CRLF to LF only.
Current release integrity does not assert byte identity with the recorded fork
commit. See `docs/provenance_checks.md` for the verification procedure.
