# Third-party source and licensing

The adapted PatchTST experiment uses the isolated Time-Series-Library fork
snapshot under `third_party/time_series_library/`. Its recorded source commit,
local changes, and integrity manifests are described in
[`PROVENANCE.md`](../third_party/time_series_library/PROVENANCE.md). The
upstream license notice is retained as
[`UPSTREAM_LICENSE`](../third_party/time_series_library/UPSTREAM_LICENSE).

The paper's Transformer, MLP, and Conv mixer comparison instead uses the
recovered full-split runner under `tools/fullsplit/`; it does not depend on the
separate six-model native source audit. Source recovery details are in
[`provenance_checks.md`](provenance_checks.md).

The repository MIT license does not replace the upstream terms applicable to
the vendored source. Dataset licenses must be checked at their own sources;
the CSV datasets are not redistributed here.
