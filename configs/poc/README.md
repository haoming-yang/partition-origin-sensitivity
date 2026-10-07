# POC design and execution boundary

The historical design is tracked as text, not an author-machine dependency:

- `artifacts/poc_stage4/batch2_poc/lambda_selection/LAMBDA_SELECTION_PROTOCOL.json`
  is the immutable pilot protocol: ETTh1 seed42, lambda grid
  0.01/0.03/0.1/0.3/1.0, L512/H96/p12/stride12, 15 epochs, batch32,
  AdamW lr1e-4/wd1e-4, no scheduler, clip1.0, standardized-space evaluation.
- `artifacts/poc_stage3/attribution_abc/origin_pair_schedule_seed{42,43,44}.json`
  records data-order indices and distinct batch-shared origin pairs for every
  update. Do not generate a different schedule and call it the historical run.
- The checkpoint rule is minimum mean validation MSE across all 12 origins;
  exact epoch ties select the later epoch. The initialization hashes and schedule
  hashes are matched against tracked B provenance.
- Lambda eligibility in the frozen protocol is `ValAvgMSE_C <= 1.01 * ValAvgMSE_B`.
  Among eligible values, select minimum validation S_theta; a tie within the
  recorded 1e-12 comparison chooses the smaller lambda. No test data is used.
  B reference is 0.7489891514129009; threshold is 0.7564790429270299.
  The frozen selection is lambda1.0; the grid is not expanded afterward.

## Paper wording discrepancy

The current manuscript states the stricter `ValMSE_C <= ValMSE_B`, while the
pre-pilot protocol records a 1% eligibility slack. All five saved candidates
also satisfy the stricter criterion, so the selected lambda is unchanged.
Nevertheless, the two rules are not identical. This release preserves the
historical record and runner rather than rewriting the experiment definition.
The author should resolve the wording/protocol discrepancy separately.

## Runnable scope

`bash scripts/run_poc.sh` invokes the formal **ETTh1 C** runner. With data and
matching checkpoints present it reuses the selected seed42 pilot checkpoint
and trains seeds43/44 using tracked schedules, then evaluates frozen models.
It does not retrain A or B, run ETTm2 transfer, or reproduce the whole Table 9.
`python -m src.run --config configs/poc/etth1_poc.yaml --dry-run` lists the
metadata jobs only; actual generic dispatch intentionally fails and points to
the specialized entrypoint. ETTm2's config is `FROZEN_ARTIFACT_ONLY`.

`tools/poc_stage3/stage3_b_runner.py` retains historical B source, but is not
the public formal batch entrypoint and its hash/source boundary must be audited
before any replacement of frozen B records. Complete ETTm2 matched A/B/C
training source, schedules, and weight tree are not established in this checkout.
The original Table 9/A12/A13 values are not reconstructed or fabricated.

## Inputs and failure behavior

Place binary files under `checkpoints/poc_stage3/...` and
`checkpoints/poc_stage4/...` at the exact paths in the README. Verify them with
`python tools/verify_poc_stage3.py --strict` before training; absent or mismatched
files fail. The non-strict form verifies all tracked records but merely reports
absent checkpoints, so it is not a complete preflight for formal training.
The runner fails on missing frozen lambda/checkpoint/schedule, schedule or
initial-parameter hash mismatch, test-leakage flags or nonfinite loss; it does
not silently train substitutes. Binary checkpoint integrity is checked by the
strict verifier above, not automatically enforced by the training runner.
When overriding checkpoint paths, compare their SHA-256 with the manifest's
corresponding canonical checkpoint entry before invoking the runner.

The pilot can be run through `python -m src.experiments.stage4.poc_lambda_pilot`
only after data and B checkpoint placement; do not execute
`tools/poc_stage3/freeze_lambda_selection.py` against tracked records. That
historical snapshot is retained for audit, not to overwrite frozen inputs.

For future validation-only selection, use the guarded entrypoint:

```bash
python -m tools.select_poc_lambda --input-root outputs/poc_stage4/batch2_poc/lambda_selection --output outputs/poc_selection/selection.json
```

Its default input root is the frozen pilot directory, allowing an independent
check without new training; its default output is `outputs/poc_selection/selection.json`.
Among eligible candidates, find the global minimum validation S_theta, then
choose the smaller lambda among values at most `1e-12` above that minimum.
This avoids order-dependent chains of pairwise approximate ties. The legacy
`tools/poc_stage3/freeze_lambda_selection.py` used raw-value sorting despite the
recorded tolerance; it remains unchanged because its source hash is frozen.
The new tool does not rewrite `FROZEN_POC_LAMBDA.json` or change eligibility.
The five historical candidates' minimum S_theta separation is
`0.0018594912662968435`, far above `1e-12`; all are eligible and lambda1.0
is unchanged. No paper value is recomputed or replaced by this check.

Environment variables: `DATA_ROOT`, `STAGE_ROOT` for new output
(default `outputs/poc_stage4`), `STAGE4_INPUT_ROOT` for protocol inputs,
`STAGE3_ROOT` for schedules/B records, `PARTITION_ORIGIN_POC_CHECKPOINT_ROOT`
for pilot B weights, and `PARTITION_ORIGIN_POC_LAMBDA_CHECKPOINT_ROOT` for
the selected pilot weights. These do not alter the protocol. Use a fresh
STAGE_ROOT for each rerun and preserve hashes and provenance.
