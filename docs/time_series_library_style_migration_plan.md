# Time-Series-Library Style Migration Plan

## 1. Objective

This plan migrates the implementation style of `partition-origin-sensitivity` toward the recognizable engineering conventions of Time-Series-Library while preserving the paper's frozen scientific protocol, artifact schema, numerical definitions, and provenance boundary.

The migration is a code-organization and interface-alignment task. It is not an experiment rerun, model redesign, metric revision, benchmark integration, or reconstruction of unavailable source.

## 2. Repositories and scope

Read-only style reference:

```text
E:\Deep Learning\Code\Time-Series-Library
```

Migration target:

```text
C:\Users\杨昊明\Desktop\partition-origin-sensitivity
```

The reference repository remains read-only. All future implementation work is confined to the migration target.

## 3. Reference files inspected

### 3.1 Model layer

- `models/PatchTST.py`
- `models/iTransformer.py`
- `models/DLinear.py`
- `models/Autoformer.py`
- `models/Informer.py`
- `models/MultiPatchFormer.py`
- `models/MambaSimple.py`
- `models/TimeMixer.py`

### 3.2 Shared layers

- `layers/Embed.py`
- `layers/Transformer_EncDec.py`
- `layers/SelfAttention_Family.py`
- `layers/StandardNorm.py`
- `layers/Autoformer_EncDec.py`

### 3.3 Experiment layer

- `exp/exp_basic.py`
- `exp/exp_long_term_forecasting.py`
- `exp/exp_short_term_forecasting.py`
- `exp/exp_imputation.py`

### 3.4 Data layer

- `data_provider/data_factory.py`
- `data_provider/data_loader.py`

### 3.5 Entry points and utilities

- `run.py`
- `utils/metrics.py`
- `utils/tools.py`
- `utils/print_args.py`
- representative files under `scripts/long_term_forecast/` and `scripts/short_term_forecast/`

### 3.6 Quantitative style profile

The inspected source snapshot contains approximately 96 Python files, 842 function definitions, and 41 `class Model(nn.Module)` implementations. Type annotations are rare, `os.path` is common, `pathlib` is absent, direct `print` calls are common, and experiment execution is organized around `Exp_*` classes and a central argparse namespace.

This confirms that the transferable identity is architectural and interface-based, not strict formatter-level imitation.

## 4. Time-Series-Library style model

### 4.1 Structural conventions to adopt

1. A task-facing model module exposes `class Model(nn.Module)`.
2. The constructor receives one configuration namespace named `configs`.
3. Model hyperparameters use established names such as `seq_len`, `pred_len`, `enc_in`, `d_model`, `n_heads`, `e_layers`, `d_ff`, `dropout`, and `activation`.
4. Forecasting logic is placed in a `forecast(...)` method.
5. `forward(...)` dispatches to the task method and returns the task output.
6. Reusable neural components live under `layers/`; model assembly lives under `models/`.
7. Data selection is centralized through `data_provider(args, flag)`.
8. Training and evaluation lifecycle code lives in `Exp_*` classes derived from `Exp_Basic`.
9. The CLI builds one argparse namespace and selects an experiment class from `task_name`.
10. Shell launchers invoke one Python entrypoint with explicit long-form arguments.
11. Tensor shape changes are visible near `permute`, `reshape`, `unfold`, and head operations.
12. Model registration is explicit and discoverable.

### 4.2 Naming conventions to adopt

| Concept | Target convention |
|---|---|
| Input history length | `seq_len` |
| Forecast horizon | `pred_len` |
| Number of channels | `enc_in` |
| Patch length | `patch_len` |
| Patch stride | `stride` |
| Training epochs | `train_epochs` |
| Learning rate | `learning_rate` |
| Validation method | `vali(...)` |
| Experiment base | `Exp_Basic` |
| Origin experiment | `Exp_Origin_Sensitivity` |
| Model implementation | `class Model(nn.Module)` |
| Model registry key | `OriginTransformer`, `PatchTSTOrigin` |
| Data split selector | `flag in {'train', 'val', 'test'}` |

### 4.3 Readability conventions to adopt

- Imports grouped into standard library, third-party packages, then local modules.
- One import per line except established compact imports that materially improve clarity.
- Two blank lines between top-level definitions.
- Short comments describe tensor semantics or scientific protocol, not obvious Python operations.
- Shape comments use one stable vocabulary: `[batch, length, channels]`, `[batch, channels, patches, width]`, and `[batch, horizon, channels]`.
- Public experiment and model methods receive concise docstrings when behavior is not evident from the Time-Series-Library interface.
- Configuration names match command-line names exactly.
- Error messages name the violated protocol field and observed value.

## 5. Upstream traits that must not be copied mechanically

The following traits occur in the reference repository but are historical implementation details rather than desirable style:

- hard-coded absolute dataset paths;
- global `warnings.filterwarnings('ignore')`;
- import-time mutation of `sys.path`;
- unstructured result text files as the only provenance record;
- unrestricted global model imports that make optional dependencies mandatory;
- one-line compound statements;
- mixed quote styles and inconsistent class naming;
- repeated train/validation/test logic;
- test-set loss evaluated during every training epoch when not required by the frozen protocol;
- silent fallthrough in task dispatch;
- broad mutable argparse namespaces without validation;
- copying upstream files into paper-owned modules without provenance separation.

The target will preserve its stronger use of repository-relative paths, environment overrides, structured JSON/CSV artifacts, source hashes, explicit masks, deterministic controls, and unavailable-source failures.

## 6. Current target audit

### 6.1 Strengths to preserve

- Formal metrics are explicit and use the paper's minimum-MSE denominator.
- Origin reconstruction and sentinel masking have dedicated audits.
- Output artifacts include configuration, provenance, predictions, logs, and summaries.
- `DATA_ROOT`, `OUTPUT_ROOT`, and `DEVICE` avoid hard-coded user paths.
- Missing original runners terminate explicitly rather than fabricating implementations.
- Third-party PatchTST source is isolated under `third_party/`.
- The repository already separates configs, scripts, results, data instructions, and tools.

### 6.2 Style and architecture mismatches

1. `src/training/runner.py` is a monolithic file combining data loading, patching, metrics, models, training, evaluation, artifact writing, third-party loading, and CLI dispatch.
2. The primary controlled model is named `ControlledTransformerSupplement` rather than being exposed through a Time-Series-Library-style `Model(configs)` interface.
3. `src/cli.py` dispatches directly to training functions rather than selecting an `Exp_*` class.
4. Data loading does not use a `data_provider(args, flag)` factory.
5. The target has `src/training`, `src/evaluation`, and `src/patching`, while the reference vocabulary is `models`, `layers`, `data_provider`, `exp`, and `utils`.
6. `canonical_run.py` and `canonical_reference.py` duplicate substantial logic and use a separate interface from `runner.py`.
7. Model construction, origin policy, and artifact policy are tightly coupled.
8. Some copied files use compact multi-import lines while newer files use typed, modern Python conventions.
9. `sys.path` mutation is currently used by several command-line tools.
10. Shell scripts use a safer Bash style than upstream, but their argument surface does not resemble the familiar Time-Series-Library launch format.
11. The vendored Time-Series-Library files currently trace to the clean local fork `https://github.com/haoming-yang/Time-Series-Library.git` at commit `61f68df6965b8d4061a08d0d09b6d69dba8728c8`; this is not sufficient to call the snapshot an official upstream revision. The copied license identifies `ANCHOR authors`, so provenance must be corrected before public release.

## 7. Migration principle

Adopt Time-Series-Library's recognizable interfaces and module boundaries while keeping the target's stronger reproducibility safeguards.

The migration should produce code that a Time-Series-Library user can navigate immediately:

```text
configuration -> run.py -> Exp_Origin_Sensitivity -> data_provider
                                      |-> models.Model
                                      |-> origin protocol
                                      |-> metrics and artifacts
```

The numerical path must remain independently testable at every boundary.

## 8. Proposed target architecture

```text
partition-origin-sensitivity/
├─ configs/
├─ data/
├─ docs/
├─ outputs/
├─ scripts/
├─ src/
│  ├─ run.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ OriginTransformer.py
│  │  └─ PatchTSTOrigin.py
│  ├─ layers/
│  │  ├─ __init__.py
│  │  └─ OriginPatching.py
│  ├─ data_provider/
│  │  ├─ __init__.py
│  │  ├─ data_factory.py
│  │  └─ data_loader.py
│  ├─ exp/
│  │  ├─ __init__.py
│  │  ├─ exp_basic.py
│  │  └─ exp_origin_sensitivity.py
│  ├─ utils/
│  │  ├─ __init__.py
│  │  ├─ origin_metrics.py
│  │  ├─ origin_protocol.py
│  │  ├─ artifacts.py
│  │  ├─ reproducibility.py
│  │  ├─ third_party_loader.py
│  │  └─ tools.py
├─ third_party/
├─ schemas/
├─ tests/
│  └─ fixtures/
└─ tools/
```

Immutable baseline oracles remain byte-for-byte at their recorded locations until their hashes, dependencies, fixtures, and invocation procedure have been archived. They must not be moved merely for visual organization because their behavior and provenance depend on `__file__`, relative paths, and source hashes.

## 9. Detailed file migration map

| Current source | Destination | Responsibility after migration |
|---|---|---|
| `src/cli.py` | `src/run.py` | package-mode argparse, config loading, experiment selection, train/test dispatch |
| `runner.py:ControlledTransformerSupplement` | `src/models/OriginTransformer.py` | `Model(configs)`, `forecast`, `forward` |
| `runner.py:OfficialPatchTSTAdapter` | `src/models/PatchTSTOrigin.py` | mask-aware adapter for the verified vendored fork snapshot through `Model(configs)` |
| `runner.py:partition`, `phase_count`, `total_length`, `patch_count` | `src/layers/OriginPatching.py` | sole production origin transformation and mask construction |
| `phase_protocol.py` audit functions | `src/utils/origin_protocol.py` | stride-aware NumPy reconstruction and exact-once audits; compatibility import for audit tools |
| `runner.py:DataBundle`, `Windows`, `load_data` | `src/data_provider/data_loader.py` | dataset classes, split boundaries, standardization |
| dataset selection dictionary | `src/data_provider/data_factory.py` | `data_provider(args, flag)` |
| `runner.py:train_controlled`, `train_patchtst`, `evaluate_model` | `src/exp/exp_origin_sensitivity.py` | training, validation, origin-wise testing |
| device and seed helpers | `src/utils/tools.py` | deterministic seed and device acquisition |
| `formal_metrics`, `metric_rows`, `result_schema.py` | `src/utils/origin_metrics.py` | one canonical metric implementation |
| artifact writers and SHA-256 helpers | `src/utils/artifacts.py` | schema validation, atomic output, provenance |
| authoritative canonical source selected in Phase 0B | unchanged immutable location | baseline parity oracle only; never imported by production code |
| non-authoritative duplicate canonical source | immutable audit archive | obsolete source evidence, not runnable production code |
| evaluation validator | `tools/validate_reference.py` | subprocess and artifact validation |

Before implementation, this table must be expanded into an ownership ledger with the columns `responsibility`, `sole implementation`, `compatibility path`, `removal phase`, and `parity fixture`. Every compatibility wrapper must contain dispatch only and no protocol, metric, preprocessing, or artifact logic.

## 10. Interface contracts

### 10.1 Model contract

Every paper-owned task model must expose:

```python
class Model(nn.Module):
    def __init__(self, configs):
        ...

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        ...

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        ...
```

The Time-Series-Library-compatible `forward` signature retains `mask=None` syntactically, but the origin-sensitivity task must reject `None` before computation. The observation mask must be Boolean, nonempty, shape-matched to `x_enc`, and must never be inferred from zero values. Internally the variable is named `observed_mask` to avoid confusing it with Time-Series-Library imputation masks.

### 10.2 Tensor contract

- `context_len`: number of real observed history steps; `--seq_len` maps to this field and is 512 in frozen protocols.
- `padded_len`: outer origin-layout length after observation-equivalent padding.
- `inner_seq_len`: length passed into a wrapped upstream model; for PatchTST this equals `padded_len`, not `context_len`.
- `patch_num_outer`: patch count induced by the controlled outer lattice.
- `patch_num_native`: upstream PatchTST internal patch count including native right replication padding.
- Raw history: `[batch, context_len, enc_in]`.
- Target: `[batch, pred_len, enc_in]`.
- Observation mask: `[batch, padded_len]`, Boolean.
- Padded model input: `[batch, padded_len, enc_in]`.
- Patched values: `[batch, enc_in, patch_num, patch_len]`.
- Model output: `[batch, pred_len, enc_in]`.
- Origins: integers in the configured origin lattice, with exact order preserved in artifacts.

`data_provider` returns raw history. `Exp_Origin_Sensitivity` constructs the origin layout and passes padded values plus the same-length Boolean mask to the model. Model-boundary assertions require `x_enc.shape[1] == mask.shape[1] == padded_len`.

### 10.3 Data contract

`data_provider(args, flag)` returns `(dataset, loader)`. The dataset exposes split metadata, scaler metadata, and complete start-index arrays. The frozen batch item remains `(batch_x, batch_y)`; timestamp marks and decoder inputs are accepted at the model interface only for familiarity and remain `None` unless separately justified by evidence.

Preprocessing is model-family specific. Controlled models receive NumPy float32 train-row-standardized values using the current mean, standard deviation, and safe-standard-deviation logic. PatchTST receives raw values and applies the existing mask-aware per-window normalization in its adapter. The migration must not substitute `StandardScaler`, timestamp features, augmentation, inverse transforms, or Time-Series-Library split constants.

Baseline loader behavior is fixed to `shuffle=True` for training, `False` for validation and test, `drop_last=False`, and `num_workers=0` unless a separately validated protocol version says otherwise.

### 10.4 Experiment contract

`Exp_Origin_Sensitivity` provides:

- `_build_model()`;
- `_get_data(flag)`;
- `_select_optimizer()`;
- `_select_criterion()`;
- `vali(...)`;
- `train(setting)`;
- `test(setting, test=0)`;
- `test_origins(setting, test=0)`.

Origin strategy selection belongs to the experiment layer, not the model layer.

### 10.5 Artifact contract

The migration must preserve or version explicitly:

- `config.json`;
- `PROVENANCE.json`;
- `training_log.csv`;
- `validation_log.csv`;
- `per_origin_mse.csv`;
- `per_origin_predictions.npz`;
- `summary.json`;
- selected and final checkpoints.

No filename or field may change silently. Legacy schema v1 is frozen as an executable schema. Added fields such as seed, checkpoint hashes, transitive source hashes, or corrected experiment identity require schema v2 and an explicit migration record; they are not described as byte-level parity.

## 11. CLI design

The primary command should resemble Time-Series-Library while exposing paper-specific controls:

```text
python -u -m src.run
  --task_name origin_sensitivity
  --is_training 1
  --model_id ETTh1_L512_H96_p12
  --model OriginTransformer
  --data ETTh1
  --root_path <dataset-root>
  --data_path ETTh1.csv
  --seq_len 512
  --pred_len 96
  --patch_len 12
  --stride 12
  --origin_strategy random_origin
  --train_epochs 15
  --seed 42
```

Additional required controls:

- `--experiment_id`;
- `--origins`;
- `--output_root`;
- `--device`;
- `--deterministic`;
- `--save_predictions`;
- `--config` for loading the existing YAML record;
- `--dry_run` for validation without training.

CLI values must override YAML only when the override is explicit and recorded in the resolved configuration.

All paper-owned imports must be package-qualified, such as `src.models`, `src.layers`, and `src.utils`. No repository-root `models`, `layers`, or `utils` compatibility package may be created because it would collide with the vendored Time-Series-Library snapshot. PatchTST loading must run through a dedicated loader that verifies the `module.__file__` of every transitive third-party import.

## 12. Configuration migration

1. Treat current YAML files as human-readable records until Phase 0 resolves every YAML/runner conflict; do not call them executable authority before that audit.
2. Define explicit aliases: `context -> seq_len`, `horizon -> pred_len`, `epochs -> train_epochs`, `dataset -> data`, and `seeds -> repeated scalar seed runs`.
3. Add a configuration-resolution phase before `Exp_Basic` or model construction.
4. Define one schema per experiment family with required fields, optional fields, types, and supported values.
5. Add model-family profiles sourced from current code. `OriginTransformer` retains `d_model=64`, `n_heads=4`, `e_layers=2`, and `d_ff=256`; PatchTST retains `512/8/2/2048`. Generic Time-Series-Library defaults must not overwrite either profile.
6. Resolve `enc_in` from validated dataset metadata before `_build_model()` and record it as a derived value.
7. Preserve original YAML keys and resolved canonical keys, recording each value's source as YAML, explicit CLI, model profile, or dataset metadata.
8. Store the fully resolved configuration, scalar seed, config path, config hash, experiment identity, model identity, and source identity in every run output.
9. Mark `source_status: frozen_artifact_only` as non-runnable before model, data, device, or output construction.
10. Do not translate missing scientific fields from manuscript numbers by inference.

## 13. Comment and documentation style

### 13.1 Keep

- paper links in model docstrings;
- shape comments at nontrivial tensor transitions;
- short section comments such as `Embedding`, `Encoder`, and `Prediction Head`;
- comments explaining why masks or origin lattices differ from native PatchTST behavior.

### 13.2 Remove or avoid

- comments narrating routine assignments;
- temporary migration notes in production files;
- internal conversation language;
- comments claiming equivalence without an automated test;
- comments that repeat the paper rather than explain code behavior.

### 13.3 Scientific terminology

Use the paper's stable terms consistently:

- `partition origin`;
- `observation-equivalent`;
- `origin lattice`;
- `formal origin gap`;
- `interior origin`;
- `observation mask`;
- `visualization-only normalization`.

Do not rename the scientific object to generic `phase augmentation` or `shift robustness`.

## 14. Migration phases

### Phase 0A — Establish a recoverable baseline

The current repository has an unborn `main` branch and untracked files, so a new branch alone is not a rollback point.

1. Review the assembled repository contents.
2. Create an immutable baseline commit or an external immutable archive plus a complete SHA-256 manifest.
3. Tag the reviewed baseline, for example `pre-tsl-style-migration`.
4. Record repository identity, tree, environment, CLI help, static checks, and all untracked-file hashes.
5. Inventory actual checkpoints, predictions, configs, logs, and summaries for each runnable family.
6. Confirm that no training process is running.

Exit gate: a recoverable baseline and source manifest exist. If trained fixtures cannot be located, the plan explicitly downgrades claims to synthetic fixed-batch and state-dict parity rather than claiming full artifact or training parity.

### Phase 0B — Resolve protocol authority and current defects

1. Create a protocol authority ledger for every runnable and frozen-only family.
2. Record authoritative experiment ID, model, architecture profile, split, scaler, start indices, preprocessing, origin lattice and order, optimizer, clipping, checkpoint rule, output path, filenames, and schema.
3. Resolve the current conflict in which core, overlap, and H=192 runs can be labeled `SUPPLEMENT_TRAIN_ORIGIN_STRATEGY` despite different YAML identities.
4. Require scalar seed in resolved config, summary, provenance, and output identity.
5. Classify `runner.py`, `canonical_run.py`, and `canonical_reference.py` as exactly one of authoritative implementation, immutable oracle, compatibility wrapper, or obsolete source.
6. Repair or replace missing oracle dependencies such as absent `configs/reference.json` and broken validator paths before using an oracle.
7. Freeze a machine-readable legacy artifact schema.
8. Change publication audits so zero eligible runs cannot pass unless an explicit developer-only `--allow-empty` is supplied.
9. Create a vendoring manifest for the recorded fork URL and commit, clean/dirty state, retrieval date, copied file list, transitive runtime imports, per-file hashes, license identity, and adapter hash. If canonical upstream identity is later verified, record it as a separate field rather than rewriting the fork provenance.

Exit gate: every YAML/runner disagreement is documented; every runnable family has one authority; broken oracle paths are resolved; empty audits fail by default.

### Phase 1 — Make configurations executable before architectural refactoring

1. Implement family-specific YAML schemas and backward-compatible aliases.
2. Resolve YAML, explicit CLI overrides, model profile, and dataset-derived metadata into one namespace before `Exp_Basic` construction.
3. Use suppressed argparse defaults or an equivalent mechanism to distinguish explicit overrides from defaults.
4. Add dry-run validation that constructs no model, data loader, device, or output directory.
5. Reject `source_status: frozen_artifact_only` with stable nonzero exit status and machine-readable `SOURCE_UNAVAILABLE`.
6. Test every shipped YAML against an expected resolved-config fixture.

Exit gate: all runnable YAMLs resolve field-for-field to the protocol authority ledger; unavailable YAMLs fail before side effects.

### Phase 2 — Add the package-safe Time-Series-Library-shaped skeleton

1. Add `src.models`, `src.layers`, `src.data_provider`, `src.exp`, and expanded `src.utils` packages.
2. Add `src.run` and invoke it only as `python -u -m src.run`.
3. Use package-qualified paper-owned imports everywhere.
4. Add a lazy model registry.
5. Add a dedicated third-party loader with transitive module-path validation.
6. Keep current runners and immutable oracles untouched.

Exit gate: CLI help and dry-run work; importing paper-owned and vendored models in either order resolves every module to the expected file in fresh subprocesses.

### Phase 3 — Migrate origin protocol and labeled metrics

1. Declare the current runner's stride-aware `phase_count`, `total_length`, `patch_count`, and `partition` behavior as the numerical migration oracle.
2. Implement one production origin transformation returning padded values and Boolean mask.
3. Make audit utilities stride-aware before allowing them to replace any runner behavior.
4. Define formal metrics over explicit `(origin, MSE, MAE)` records.
5. Define interior origins by origin identity, never array position.
6. Keep visualization-only normalization in a separate, unmistakably named function.
7. Add dispatch-only compatibility imports for current audit tools.

Required fixtures:

```text
L=512, p=12, stride=12: origins 0..11, padded_len 528, patch_count 44
L=512, p=12, stride=6:  origins 0..5,  padded_len 522, patch_count 86
```

Exit gate: bitwise padded-value and mask parity for every origin; formal scalar metrics are invariant to record ordering; origin identities remain correct.

### Phase 4 — Migrate the data layer without adopting upstream data semantics

1. Implement dataset classes with familiar Time-Series-Library argument names but frozen paper behavior.
2. Implement `data_provider(args, flag)`.
3. Freeze complete train, validation, and test start-index arrays.
4. Freeze scaler input rows, float dtype, mean, standard deviation, safe standard deviation, and standardized-array digest.
5. Require `seq_len == 512` for frozen configurations unless a separately versioned protocol supports another value.
6. Preserve the current ETTm end row of 57600 and the current Weather split.
7. Resolve channel count before model construction.

Expected H=96 windows under the current loader:

```text
ETTh1/ETTh2: 8033 / 2785 / 5805
ETTm1/ETTm2: 33953 / 11425 / 11425
Weather:     36280 / 5175 / 10444
```

Expected ETTh1 H=192 windows: `7937 / 2689 / 5709`.

Exit gate: `np.array_equal` for start arrays, scaler arrays, standardized values, raw target windows, and dataset order.

### Phase 5 — Migrate model modules with family-specific contracts

1. Implement `src.models.OriginTransformer.Model(configs)` with the exact controlled architecture profile.
2. Keep private head components in the model file unless genuine reuse is demonstrated, avoiding unnecessary state-dict renaming.
3. Preserve concatenated mask features and key-padding behavior for OriginTransformer.
4. Implement `src.models.PatchTSTOrigin.Model(configs)` around the vendored fork snapshot.
5. Preserve PatchTST's mask-aware outer normalization, unchanged native patch embedding, native right replication padding, encoder, and head.
6. Do not add an upstream attention mask or call native `forward/forecast` in a way that bypasses the adapter.
7. Require mask shape, dtype, and observed-count validation.
8. Verify `context_len`, `padded_len`, `inner_seq_len`, outer patch count, and native patch count independently.

Exit gate: identical ordered state-dict keys, shapes, dtypes, and seeded tensor values; strict checkpoint loading without remapping; fixed-input output parity on CPU and declared tolerance on CUDA.

### Phase 6 — Migrate experiment lifecycle and training trace

1. Add `Exp_Basic` after config and dataset metadata resolution.
2. Add `Exp_Origin_Sensitivity` with train, validation, selected-checkpoint, final-checkpoint, and origin-test methods.
3. Keep origin strategy in the experiment layer.
4. Preserve DataLoader construction order, shuffle flags, Python/NumPy/PyTorch RNG streams, origin sampling order, optimizer family, optimizer parameters, gradient clipping, and thread settings.
5. Preserve `<=` checkpoint tie behavior, which selects the latest equal-best epoch.
6. Preserve saving `final_checkpoint.pt` before restoring and saving selected `checkpoint.pt`.
7. Add a bounded synthetic two-step optimizer trace, not a paper experiment rerun.

The trace records batch indices, sampled origins, losses, unclipped and clipped gradient norms, parameter hashes after each step, optimizer-state hashes, validation records, and checkpoint decision.

Exit gate: exact CPU parity for the synthetic trace and a tie fixture `[0.5, 0.4, 0.4, 0.6]` selecting epoch 3.

### Phase 7 — Migrate artifacts and provenance with schema versioning

1. Centralize structured artifact writers.
2. Write new files atomically without overwriting a completed frozen run.
3. Validate exact filenames, JSON paths and types, CSV headers and order, NPZ keys/dtypes/shapes, checkpoint roles, and hashes.
4. Keep schema v1 immutable and create schema v2 for added or corrected fields.
5. Compare NPZ arrays semantically because archive metadata can differ; compare untouched frozen artifacts byte-for-byte.
6. Record config source, seed, experiment identity, output identity, checkpoint hashes, code hashes, library versions, device, and complete third-party source manifest.
7. Make duplicate run identities and incomplete expected matrices fatal.

Exit gate: schema tests pass; existing completed outputs are never mutated; output path identifiers equal identifiers inside every record.

### Phase 8 — Migrate launch scripts and publication UX

1. Update launchers to call `python -u -m src.run`.
2. Retain `set -euo pipefail`, repository-relative paths, and environment overrides.
3. Add direct Python examples and PowerShell equivalents where supported.
4. Add CPU dry-run and bounded one-batch smoke commands.
5. Document runnable, frozen-artifact-only, and unavailable families in one matrix.
6. Document expected output tree, runtime, GPU memory, disk use, dataset sources, licenses, and checksums where verified.
7. Pin or export the exact tested environment for release.

Exit gate: every launcher passes syntax and dry-run checks; unavailable launchers produce `SOURCE_UNAVAILABLE` without side effects.

### Phase 9 — Compatibility, cleanup, and clean-clone validation

1. Convert enumerated old entrypoints into dispatch-only wrappers.
2. Test every old import path, CLI path, and checkpoint path listed in the ownership ledger.
3. Remove duplicate active implementations only after all parity gates pass.
4. Preserve immutable oracle bytes, hashes, fixtures, and procedure in the audit archive.
5. Update README and paper-to-code map.
6. Perform clean-clone validation on Linux and Windows.

Exit gate: one active implementation exists per responsibility; no wrapper contains scientific logic; the documented clean-clone workflow passes.

## 15. Verification matrix

### 15.1 Static checks

- Python compilation for every paper-owned module;
- import smoke test for every model registry entry;
- Bash syntax for every launcher;
- YAML parse and schema validation;
- no absolute user paths, email addresses, tokens, or private service identifiers;
- no duplicate module implementing the same metric or partition rule;
- no import cycle among `models`, `layers`, `data_provider`, `exp`, and `utils`.
- fresh-subprocess import-order test for paper-owned and vendored modules;
- every runtime-loaded third-party module resolves below the vendored root;
- zero-run publication audits fail unless `--allow-empty` is explicit.

### 15.2 Protocol parity

- origin set and order identical;
- exact-once observation coverage for the canonical non-overlap protocol;
- stride-6 relaxed lattice preserved as a separate protocol;
- padding sentinel has zero influence under the observation mask;
- temporal order preserved;
- train/validation/test row boundaries identical;
- window counts identical;
- scaler fit rows identical;
- checkpoint selection rule identical.
- full start-index arrays identical, not only counts;
- NumPy float32 scaler arrays and standardized-array digest identical;
- `seq_len=512` enforced for frozen protocols;
- stride-12 and stride-6 lattices tested independently;
- mask omission, wrong dtype, wrong shape, or zero observed count raises.

### 15.3 Numerical parity

- identical per-origin predictions from identical weights and inputs within tolerance;
- identical MSE and MAE definitions;
- identical `Gamma_o`, `Gamma_i`, `Delta_o`, and `CV_o` values;
- formal gaps continue to use minimum MSE as denominator;
- visualization-only run-mean normalization remains isolated;
- fixed artifacts remain byte-unchanged.
- formal metrics remain invariant under permutation of labeled origin records;
- ordered state-dict keys, shapes, dtypes, and tensor values match;
- strict old-to-new and new-to-old checkpoint loading requires no key remapping;
- synthetic two-step optimizer traces match on the same CPU/PyTorch build;
- checkpoint tie behavior and final-versus-selected save order match.

### 15.4 Artifact parity

- all required files produced;
- JSON keys and CSV columns preserved;
- replicate, dataset, origin, and experiment identifiers present;
- source hashes resolve to repository files;
- unavailable experiments remain unavailable rather than silently mapped to another runner.
- missing required files, fields, seeds, origins, datasets, or duplicate identities fail;
- NPZ keys, dtypes, shapes, and arrays match semantically;
- completed frozen output directories are never overwritten;
- schema corrections are versioned rather than called parity.

### 15.5 Usability checks

- README command executes from repository root;
- `--help` groups arguments by basic, data, forecasting, origin protocol, optimization, and device settings;
- dry run prints the resolved configuration and planned output path;
- missing data error lists the exact expected path;
- CPU smoke path works;
- CUDA selection remains explicit and auditable.
- direct Python, Bash, and supported PowerShell commands agree;
- runnable/frozen/unavailable status is visible before execution;
- clean-clone workflows are tested on Linux and Windows;
- publication release identifies exact environment, data sources, and verified resource expectations.

## 16. Acceptance criteria

The migration is complete only when all conditions hold:

1. A Time-Series-Library user can locate the model, layer, data, experiment, utility, and entrypoint code without reading the monolithic legacy runner.
2. Paper-owned models expose `Model(configs)` and the standard forecast-oriented `forward` signature.
3. Experiment execution goes through `Exp_Origin_Sensitivity`.
4. Dataset construction goes through `data_provider(args, flag)`.
5. Existing YAML configurations and experiment identifiers remain valid.
6. No frozen numerical value or artifact is changed.
7. All parity and provenance gates pass.
8. No unavailable runner is fabricated.
9. The old monolithic runner is no longer the documented primary entrypoint.
10. The reference Time-Series-Library repository remains unchanged.
11. Package-qualified imports eliminate collisions with vendored `layers` and `utils` modules.
12. The protocol authority ledger resolves all YAML, runner, artifact, and output-path identities.
13. All publication audits fail on empty or incomplete evidence by default.
14. Vendored PatchTST is described according to verified provenance, not assumed official status.

## 17. Risks and controls

| Risk | Control |
|---|---|
| Refactor changes numerical behavior | Old/new fixed-batch parity before each deletion |
| Mask semantics become implicit | Mandatory mask argument and sentinel test |
| Time-Series-Library naming obscures paper terminology | Keep origin-specific module and argument names |
| Model state dict becomes incompatible | Compare parameter names, shapes, initialization, and outputs |
| Split behavior drifts | Assert exact row boundaries and window indices |
| CLI/YAML precedence becomes ambiguous | Store resolved config and override provenance |
| Upstream optional imports break the package | Lazy model registry and isolated third-party adapter |
| Legacy and new implementations diverge | Short-lived wrappers and a fixed removal deadline |
| Style imitation reduces code quality | Adopt interfaces, reject documented upstream weaknesses |
| Missing-source experiments appear runnable | Preserve explicit `SOURCE_UNAVAILABLE` gates |
| `src.layers` collides with vendored `layers` | Package-mode entrypoint, qualified imports, subprocess import-order tests |
| Time-Series-Library ETTm split changes frozen data | Freeze current boundaries and complete start arrays |
| `StandardScaler` changes float values | Preserve NumPy float32 scaler implementation and digest |
| Arbitrary origin order changes interior metrics | Compute from labeled records and test permutation invariance |
| Generic defaults redesign a model family | Resolve explicit per-model architecture profiles |
| Mask semantics drift across model families | Maintain distinct controlled and PatchTST mask policies |
| Empty outputs falsely pass audits | Fail closed unless explicit developer-only `--allow-empty` |
| Correcting missing provenance is mislabeled parity | Introduce schema v2 and correction record |

## 18. Rollback strategy

1. Create a reviewed baseline commit or immutable archive before creating the migration branch.
2. Tag the baseline and keep a complete SHA-256 manifest, including currently untracked files.
3. Do not edit, move, or delete immutable oracle files before their bytes and dependencies are archived.
4. Do not delete active legacy code before all parity gates pass.
5. Keep dispatch-only compatibility wrappers for one release boundary.
6. If any protocol or numerical parity check fails, stop at the current phase and restore the documented primary entrypoint to the last passing implementation.
7. Never use frozen paper values to force a parity test to pass.

## 19. Recommended execution order

The recommended order is:

```text
recoverable baseline
-> protocol authority and schema audit
-> executable configuration
-> package-safe skeleton
-> protocol and labeled metrics
-> data provider
-> model interface
-> experiment lifecycle and synthetic trace
-> artifacts
-> CLI and scripts
-> wrappers, cleanup, and clean-clone validation
```

This order moves deterministic, easily testable components first and delays the training lifecycle until the underlying contracts are stable.

## 20. Explicit non-goals

- no training during the style-migration pass;
- no new experiment;
- no architecture change;
- no hyperparameter change;
- no output-value change;
- no paper revision;
- no direct merge into Time-Series-Library;
- no replacement of structured provenance with upstream-style text logs;
- no speculative implementation of missing experiment source;
- no modification of the reference repository.

## 21. Planned review questions

The independent review must answer:

1. Does the plan reproduce recognizable Time-Series-Library organization without copying its weaker historical practices?
2. Is every current responsibility mapped to exactly one destination?
3. Can the migration preserve frozen scientific semantics and state-dict compatibility?
4. Are the parity gates strong enough to detect split, mask, origin, metric, checkpoint, and artifact drift?
5. Does the plan leave any permanent duplicate implementation?
6. Is the CLI familiar to Time-Series-Library users while remaining explicit about the paper's origin protocol?
7. Are unavailable-source experiments still impossible to run accidentally?
8. Is the phased rollback strategy adequate?

## 22. Review status

Independent review completed by three read-only sub-agents covering Time-Series-Library fidelity, scientific reproducibility, and repository maintenance.

### 22.1 Accepted blocker findings

- establish a real baseline before branching;
- resolve YAML/runner/experiment-ID/seed authority before refactoring;
- classify the three current runner/canonical implementations explicitly;
- use `python -u -m src.run` and package-qualified imports;
- prevent collisions with vendored absolute `layers` and `utils` imports;
- distinguish `context_len`, `padded_len`, `inner_seq_len`, and both patch counts;
- preserve separate controlled-model and PatchTST preprocessing and mask policies;
- freeze current ETTm, Weather, scaler, dtype, and start-index behavior;
- make origin metrics label-aware and order-invariant;
- require strict state-dict, optimizer-trace, checkpoint-tie, and artifact-schema parity;
- fail audits on empty or incomplete evidence;
- replace unsupported `official` claims with verified vendored-source provenance;
- make `SOURCE_UNAVAILABLE` machine-readable and side-effect free.

### 22.2 Accepted simplifications

- remove the proposed shared `OriginHeads.py`; keep model-specific heads private unless reused;
- choose `src.utils.origin_protocol` as the sole audit implementation rather than leaving an `or tools` ambiguity;
- do not move immutable oracle files into a cosmetic `legacy/` tree.

### 22.3 Clarified verification level

Full trained-artifact parity is permitted only when complete independently validated checkpoints and predictions are located. Without them, acceptance is explicitly limited to source-manifest, layout, data-array, state-dict, fixed-batch, synthetic optimizer-trace, schema, and static parity. This limitation must be reported rather than concealed.

### 22.4 Final status

Review-integrated plan complete. Implementation must not begin until Phase 0A and Phase 0B prerequisites are satisfied.
