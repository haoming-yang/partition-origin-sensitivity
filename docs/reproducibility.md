# Reproducibility and provenance

Every generated run records the experiment identifier, dataset, split, context, horizon, patch length, stride, origins, optimizer, training budget, metric definitions, source hashes, software versions, and device information.

Formal metrics are kept distinct from visualization-only quantities:

```text
G_origin = (max_r MSE_r - min_r MSE_r) / min_r MSE_r * 100
G_interior = (max_{r>=1} MSE_r - min_{r>=1} MSE_r) / min_{r>=1} MSE_r * 100
```

No mean-normalized visualization quantity is used to compute either formal gap. The repository does not include data, checkpoints, or user-specific paths. Generated files are ignored by Git.

The exact source and artifact availability audit is recorded in `docs/source_audit.md`. Anonymity checks are available through `scripts/run_audits.sh`.
