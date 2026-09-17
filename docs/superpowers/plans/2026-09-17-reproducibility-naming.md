# Reproducibility and Repository Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the public repository reproducible from a clean checkout, remove machine-specific execution assumptions, and keep directory/file naming consistent with a simple TimesNet-style research release.

**Architecture:** Preserve the current experiment and artifact layout. Add a single repository-root execution contract, make every public shell entrypoint resolve and enter the repository root, document exact data/checkpoint availability boundaries, and validate artifact/config naming with existing tests. Rename nothing unless a concrete inconsistency is proven and all references can be updated safely.

**Tech Stack:** Python 3.9+, PyTorch/Conda, Bash entrypoints, pytest, YAML configuration, Git manifests.

**Spec:** User request: audit the repository checkout, learn from `https://github.com/thuml/TimesNet` directory naming, ensure others can reproduce results, and do not add code comments.

## Global Constraints

- Do not change scientific results, tracked artifact values, or experiment protocols.
- Do not add comments to Python, shell, or configuration code.
- Do not embed absolute local paths in executable code or public documentation.
- Keep existing historical artifacts and untracked diagnostic work intact.
- State clearly which results require external datasets/checkpoints.

### Task 1: Audit repository contracts

**Files:**
- Inspect `README.md`, `docs/reproducibility.md`, `docs/source_audit.md`, `scripts/*.sh`, `src`, `tools`, `tests`, and manifests.
- Create or update `docs/reproducibility_audit.md` with verified clean-checkout prerequisites and result availability.

**Steps:**
- [x] Check tracked/untracked state, absolute paths, shell working-directory assumptions, environment names, and missing external inputs.
- [x] Map each public reproduction command to its config, data requirement, output root, and expected completion artifact.
- [x] Record naming findings against the TimesNet reference without changing scientific identifiers.
- [x] Run the read-only repository audit tools and capture their results in the audit document.

### Task 2: Normalize execution entrypoints

**Files:**
- Modify `scripts/setup.sh`, `scripts/reproduce_all.sh`, `scripts/reproduce_tables.sh`, and any public runner that invokes Python without entering the repository root.
- Add a small executable helper only if the existing scripts cannot share one root-resolution mechanism.

**Interfaces:**
- Every script must work when invoked from an arbitrary current directory.
- `SEEDS`, `DATA_ROOT`, `OUTPUT_ROOT`, and existing protocol variables retain their current meaning.

**Steps:**
- [x] Resolve `ROOT` from the script location.
- [x] Run Python modules/files with `Set-Location`/`cd` to `ROOT` or explicit `PYTHONPATH`.
- [x] Preserve existing command arguments and output locations.
- [x] Add no code comments.

### Task 3: Make clean-checkout reproduction explicit

**Files:**
- Modify `README.md`, `docs/reproducibility.md`, and `data/README.md` only where verified audit results require clarification.
- Add or update a machine-readable availability manifest under `docs` or `artifacts` if needed.

**Steps:**
- [x] Separate no-data protocol verification, data-backed retraining, frozen-checkpoint diagnostics, and checkpoint-dependent POC reproduction.
- [x] Document exact external input locations and expected validation commands.
- [x] Explain that repository-tracked compact artifacts reproduce analyses but not unavailable training weights/data.
- [x] Keep naming simple, lowercase, descriptive, and stable; avoid renaming identifiers that are part of paper provenance.

### Task 4: Add regression checks

**Files:**
- Modify or add tests under `tests/` for arbitrary working-directory script invocation, path portability, and required reproducibility metadata.

**Steps:**
- [x] Add regression checks for shell-root contracts and dry-run metadata.
- [x] Check that no tracked executable or configuration file contains a local absolute path.
- [x] Check that public README commands reference existing files and configs.
- [x] Run the full test suite and relevant smoke/audit commands.

### Task 5: Final verification

**Files:**
- No further source changes unless a verification failure identifies a required fix.

**Steps:**
- [x] Run `pytest -q` in the project environment.
- [x] Run dry-run and read-only audit commands without datasets or checkpoints.
- [x] Run the full public smoke script and source/provenance checks.
- [x] Review the diff, ensure no comments were added, and report limits and exact commands.
