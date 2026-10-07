from __future__ import annotations

import math


EXPERIMENT_IDS = (
    "CANONICAL_PHENOMENON_27_V1", "CROSS_MODEL_PHASE_V1", "PATCH_LENGTH_AUDIT_V1",
    "NO_PE_CONTROL_V1", "OPTIMIZATION_TRAJECTORY_V1", "MASK_HEAD_FACTORIAL_V1",
    "SUPPLEMENT_HORIZON192", "SUPPLEMENT_OVERLAP_STRIDE6",
    "SUPPLEMENT_TRAIN_ORIGIN_STRATEGY", "SUPPLEMENT_PATCHTST_ORIGIN",
    "POC_ETTH1_V1", "POC_ETTM2_V1",
)
SOURCE_STATUSES = {
    "NO_PE_CONTROL_V1": "RECONSTRUCTED_CONTROL", "OPTIMIZATION_TRAJECTORY_V1": "RECONSTRUCTED_CONTROL",
    "MASK_HEAD_FACTORIAL_V1": "RECONSTRUCTED_CONTROL", "POC_ETTH1_V1": "ARTIFACT_DEPENDENT",
    "POC_ETTM2_V1": "FROZEN_ARTIFACT_ONLY",
}
COMMON = {"experiment_id", "dataset", "datasets", "seed", "seeds", "default_seeds", "source_status"}
TRAINING = {"context", "horizon", "patch_len", "stride", "epochs", "batch_size",
            "learning_rate", "weight_decay", "optimizer", "gradient_clip", "scheduler",
            "padding", "standardization", "formal_gap_denominator", "origins"}
CONTROLLED = {"strategy", "use_position", "head_geometry", "use_mask", "max_train_windows",
              "max_validation_windows", "max_test_windows", "diagnostic_split"}
EXTRA = {
    "CANONICAL_PHENOMENON_27_V1": CONTROLLED,
    "NO_PE_CONTROL_V1": CONTROLLED,
    "OPTIMIZATION_TRAJECTORY_V1": CONTROLLED,
    "MASK_HEAD_FACTORIAL_V1": {"diagnostic_split"},
    "CROSS_MODEL_PHASE_V1": {"models", "runner", "data_root_env", "split", "train_windows", "validation_windows", "test_windows"},
    "PATCH_LENGTH_AUDIT_V1": {"patch_lengths"},
    "SUPPLEMENT_HORIZON192": {"strategy"},
    "SUPPLEMENT_OVERLAP_STRIDE6": {"strategy"},
    "SUPPLEMENT_TRAIN_ORIGIN_STRATEGY": {"strategies"},
    "SUPPLEMENT_PATCHTST_ORIGIN": {"outer_adapter"},
    "POC_ETTH1_V1": {"conditions", "lambda_candidates", "historical_conditions", "runnable_condition",
                      "runner", "eligibility_multiplier", "protocol_record", "frozen_lambda_record", "schedule_root"},
    "POC_ETTM2_V1": {"conditions", "reason"},
}


def validate_config(config: dict) -> None:
    experiment = config.get("experiment_id")
    if experiment not in EXPERIMENT_IDS:
        raise ValueError(f"Unknown experiment_id {experiment!r}; legal values: {', '.join(EXPERIMENT_IDS)}")
    allowed = COMMON | TRAINING | EXTRA[experiment]
    if experiment == "POC_ETTM2_V1":
        allowed = COMMON | EXTRA[experiment]
    for field in config:
        if field not in allowed:
            raise ValueError(f"Unsupported config field {field!r} for {experiment}; it would otherwise be ignored")
    fixed = {
        "context": 512, "horizon": 192 if experiment == "SUPPLEMENT_HORIZON192" else 96,
        "patch_len": 12, "stride": 6 if experiment == "SUPPLEMENT_OVERLAP_STRIDE6" else 12,
        "optimizer": "Adam" if experiment == "SUPPLEMENT_PATCHTST_ORIGIN" else "AdamW",
        "gradient_clip": 1.0, "scheduler": "none", "padding": "zero_with_observation_mask",
        "standardization": "train_rows_only", "formal_gap_denominator": "minimum_mse",
    }
    if experiment in {"PATCH_LENGTH_AUDIT_V1", "CROSS_MODEL_PHASE_V1", "POC_ETTH1_V1"}:
        fixed.update(epochs=5 if experiment == "CROSS_MODEL_PHASE_V1" else 15,
                     batch_size=32, learning_rate=1e-4, weight_decay=1e-4)
    if experiment == "PATCH_LENGTH_AUDIT_V1":
        allowed -= {"patch_len", "stride", "origins"}
    if experiment == "NO_PE_CONTROL_V1":
        fixed.update(use_position=False, epochs=15, strategy="random_origin")
    if experiment == "OPTIMIZATION_TRAJECTORY_V1":
        fixed.update(strategy="random_origin")
    if experiment == "MASK_HEAD_FACTORIAL_V1":
        fixed.update(epochs=5, diagnostic_split="capped_test_setting")
    if experiment == "SUPPLEMENT_PATCHTST_ORIGIN":
        fixed.update(outer_adapter="mask_aware_normalization", standardization="raw_with_mask_aware_normalization")
    if experiment == "CROSS_MODEL_PHASE_V1":
        fixed.update(runner="tools/fullsplit/fullsplit_3run_runner.py", split="full_canonical",
                     data_root_env="PARTITION_ORIGIN_DATA_ROOT",
                     train_windows=8033, validation_windows=2785, test_windows=5805)
    if experiment == "POC_ETTH1_V1":
        fixed.update(eligibility_multiplier=1.01, conditions=["A", "B", "C"],
                     historical_conditions=["A", "B", "C"], runnable_condition="C",
                     lambda_candidates=[0.01, 0.03, 0.1, 0.3, 1.0],
                     runner="src.experiments.stage4.poc_formal_c_runner",
                     protocol_record="artifacts/poc_stage4/batch2_poc/lambda_selection/LAMBDA_SELECTION_PROTOCOL.json",
                     frozen_lambda_record="artifacts/poc_stage4/batch2_poc/FROZEN_POC_LAMBDA.json",
                     schedule_root="artifacts/poc_stage3/attribution_abc")
    if experiment == "POC_ETTM2_V1":
        fixed.update(conditions=["A", "B", "C"])
    for field, value in config.items():
        if field not in allowed:
            raise ValueError(f"Unsupported config field {field!r} for {experiment}")
        if field in fixed and value != fixed[field]:
            raise ValueError(f"{experiment} protocol fixes {field}={fixed[field]!r}; got {value!r}")
    for field in ("batch_size", "max_train_windows", "max_validation_windows", "max_test_windows"):
        if field in config and (type(config[field]) is not int or config[field] <= 0):
            raise ValueError(f"{field} must be a positive integer")
    if "epochs" in config:
        epochs = config["epochs"]
        values = epochs if isinstance(epochs, list) and experiment == "OPTIMIZATION_TRAJECTORY_V1" else [epochs]
        if not values or any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("epochs must be positive integers")
    for field in ("learning_rate", "weight_decay"):
        if field in config:
            value = config[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (field == "learning_rate" and value == 0):
                raise ValueError(f"{field} must be finite and {'positive' if field == 'learning_rate' else 'nonnegative'}")
    for field in ("use_position", "use_mask"):
        if field in config and type(config[field]) is not bool:
            raise ValueError(f"{field} must be boolean")
    if "origins" in config and config["origins"] != list(range(fixed["stride"])):
        raise ValueError(f"origins must include the full fixed range 0..{fixed['stride'] - 1}")
    for field, choices in (("strategy", {"boundary_only", "random_origin", "all_origins"}),
                           ("head_geometry", {"flattened", "pooled"}),
                           ("diagnostic_split", {"full", "capped_test_setting"})):
        if field in config and config[field] not in choices:
            raise ValueError(f"Unsupported {field}: {config[field]!r}")
    for field, choices in (("models", {"Transformer", "MLP", "Conv"}),
                           ("strategies", {"boundary_only", "random_origin", "all_origins"})):
        if field in config and (not isinstance(config[field], list) or not config[field] or any(v not in choices for v in config[field])):
            raise ValueError(f"{field} must be a nonempty list from {sorted(choices)}")
    if "patch_lengths" in config and (not isinstance(config["patch_lengths"], list) or not config["patch_lengths"] or any(type(p) is not int or p not in (8, 12, 16) for p in config["patch_lengths"])):
        raise ValueError("patch_lengths supports the historical grid [8, 12, 16]")
    if "source_status" in config:
        expected = SOURCE_STATUSES.get(experiment, "SOURCE_PRESENT")
        if config["source_status"] != expected:
            raise ValueError(f"source_status is fixed to {expected} for {experiment}")
    datasets = config.get("datasets", config.get("dataset", []))
    if "dataset" in config and "datasets" in config:
        raise ValueError("Specify either dataset or datasets, not both")
    datasets = [datasets] if isinstance(datasets, str) else datasets
    supported = {"etth1", "etth2", "ettm1", "ettm2", "weather"}
    if not isinstance(datasets, list) or any(not isinstance(d, str) or d.lower() not in supported for d in datasets):
        raise ValueError("dataset/datasets must name ETTh1, ETTh2, ETTm1, ETTm2, or Weather")
    if experiment in {"CROSS_MODEL_PHASE_V1", "OPTIMIZATION_TRAJECTORY_V1", "POC_ETTH1_V1"} and any(d.lower() != "etth1" for d in datasets):
        raise ValueError("dataset is fixed to ETTh1 for this experiment")
