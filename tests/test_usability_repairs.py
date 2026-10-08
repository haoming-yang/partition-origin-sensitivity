import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_permutation_refuses_frozen_destination_before_reading_inputs(tmp_path, monkeypatch):
    from tools import permutation_latent_spectrum as tool

    frozen = tmp_path / "artifacts" / "record.json"
    frozen.parent.mkdir()
    frozen.write_bytes(b"frozen evidence")
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    args = argparse.Namespace(input=[tmp_path / "missing.csv"], output=frozen,
                              permutations=2, seed=0)
    with pytest.raises(ValueError, match="frozen"):
        tool.run(args)
    assert frozen.read_bytes() == b"frozen evidence"


def test_latent_summary_refuses_frozen_destination_before_reading_inputs(tmp_path, monkeypatch):
    from tools import summarize_latent_artifacts as tool

    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path, raising=False)
    output = tmp_path / "artifacts" / "new.json"
    monkeypatch.setattr(sys, "argv", ["summary", "--spectrum", "missing.csv", "--layers",
                                     "missing.csv", "--output", str(output)])
    with pytest.raises(ValueError, match="frozen"):
        tool.main()
    assert not output.exists()


def test_generated_output_refuses_git_tracked_file(tmp_path):
    from src.utils.output_paths import validate_generated_output

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    output = tmp_path / "historical.json"
    output.write_bytes(b"immutable")
    subprocess.run(["git", "add", "historical.json"], cwd=tmp_path, check=True)
    with pytest.raises(ValueError, match="tracked"):
        validate_generated_output(output, tmp_path)
    assert output.read_bytes() == b"immutable"
    assert validate_generated_output(tmp_path / "outputs" / "result.json", tmp_path) == (
        tmp_path / "outputs" / "result.json"
    )


def test_summary_wrapper_writes_to_explicit_safe_output(tmp_path):
    import shutil

    bash = shutil.which("bash")
    if not bash:
        pytest.skip("Bash is required for the public wrapper")
    # Keep real dispatch, but route the two generated files into a fresh root.
    import os
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    env["OUTPUT_ROOT"] = str(tmp_path / "generated")
    env["PERMUTATIONS"] = "2"
    result = subprocess.run([bash, "scripts/summarize_latent.sh"], cwd=ROOT,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "generated" / "latent_summary_etth1_o0_o6.json").is_file()


def test_summary_wrapper_defaults_outside_frozen_tree():
    import os
    import shutil

    bash = shutil.which("bash")
    if not bash:
        pytest.skip("Bash is required for the public wrapper")
    env = os.environ.copy()
    env.pop("OUTPUT_ROOT", None)
    env["SEEDS"] = "42,43,44"
    command = 'function python() { printf "%s\\n" "$@"; }; export -f python; bash scripts/summarize_latent.sh'
    result = subprocess.run([bash, "-c", command], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "/outputs/latent_summary/latent_summary_etth1_o0_o6.json" in result.stdout
    assert "/outputs/latent_summary/latent_spectrum_etth1_o0_o6_permutation.json" in result.stdout


def test_wrapper_rejects_frozen_directory(tmp_path):
    import os
    import shutil

    bash = shutil.which("bash")
    if not bash:
        pytest.skip("Bash is required for the public wrapper")
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    env["OUTPUT_ROOT"] = str(ROOT / "artifacts")
    # On the broken version, missing seeds ensure no real frozen file is written.
    env["SEEDS"] = "901,902,903"
    result = subprocess.run([bash, "scripts/summarize_latent.sh"], cwd=ROOT,
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "frozen" in result.stderr.lower()


def test_unknown_id_does_not_reach_training(monkeypatch, tmp_path):
    from src import run

    def unexpected_training(*args, **kwargs):
        pytest.fail("Unknown ID reached training")

    monkeypatch.setattr(run, "run_controlled", unexpected_training)
    with pytest.raises(ValueError, match="Unknown experiment_id.*CANONICAL_PHENOMENON"):
        run.run({"experiment_id": "CANONICAL_PHENOMENOM_27_V1"}, [42], ["ETTh1"], tmp_path)


@pytest.mark.parametrize("field,value", [
    ("optimizer", "SGD"), ("gradient_clip", 7.0),
    ("padding", "unmasked_replication"), ("standardization", "test_rows"),
    ("formal_gap_denominator", "mean_mse"), ("origins", [0, 1]),
    ("scheduler", "cosine"), ("learning_rate_typo", 0.01),
    ("context", 336), ("batch_size", 0),
])
def test_unsupported_config_fails_during_load(tmp_path, field, value):
    from src import run
    import yaml

    config = {"experiment_id": "CANONICAL_PHENOMENON_27_V1", "dataset": "ETTh1", field: value}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=field):
        run.load_config(path)


@pytest.mark.parametrize("experiment,field,value", [
    ("PATCH_LENGTH_AUDIT_V1", "epochs", 3),
    ("NO_PE_CONTROL_V1", "use_position", True),
    ("MASK_HEAD_FACTORIAL_V1", "epochs", 20),
    ("POC_ETTH1_V1", "eligibility_multiplier", 1.0),
    ("SUPPLEMENT_PATCHTST_ORIGIN", "optimizer", "AdamW"),
])
def test_specialized_fixed_fields_reject_overrides(tmp_path, experiment, field, value):
    from src import run
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"experiment_id": experiment, field: value}))
    with pytest.raises(ValueError, match=field):
        run.load_config(path)


def test_patch_length_weather_filter_matches_dry_run(monkeypatch, tmp_path):
    from src import run

    config = run.load_config(ROOT / "configs/patch_length/etth1_weather_p8_p12_p16.yaml")
    # Training is replaced; the public dispatcher/filter is real.
    def record(config, dataset, seed, output_root):
        return {"dataset": dataset, "seed": seed, "patch_len": config["patch_len"]}

    monkeypatch.setattr(run, "run_phenomenon", record)
    results = run.run(config, [17], ["Weather"], tmp_path)
    assert results == [
        {"dataset": "Weather", "seed": 17, "patch_len": 8},
        {"dataset": "Weather", "seed": 17, "patch_len": 12},
        {"dataset": "Weather", "seed": 17, "patch_len": 16},
    ]
    command = [sys.executable, "-m", "src.run", "--config",
               "configs/patch_length/etth1_weather_p8_p12_p16.yaml", "--seed", "17",
               "--datasets", "Weather", "--dry-run", "--output-root", str(tmp_path / "unused")]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["jobs"] == results
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("offset,expected", [(5e-13, 0.01), (2e-12, 1.0)])
def test_poc_future_selection_respects_tolerance(offset, expected):
    from src.analysis.poc_selection import select_lambda

    rows = [dict(lambda_value=0.01, validation_mse=0.7, validation_s=1 + offset),
            dict(lambda_value=1.0, validation_mse=0.7, validation_s=1.0)]
    assert select_lambda(rows, baseline_mse=0.7, eligibility_multiplier=1.01) == expected


def test_poc_future_selection_preserves_eligibility():
    from src.analysis.poc_selection import select_lambda

    rows = [dict(lambda_value=0.01, validation_mse=0.7, validation_s=2.0),
            dict(lambda_value=1.0, validation_mse=0.708, validation_s=1.0)]
    assert select_lambda(rows, baseline_mse=0.7, eligibility_multiplier=1.01) == 0.01
    assert select_lambda(rows, baseline_mse=0.6, eligibility_multiplier=1.01) is None


def test_poc_selector_rejects_frozen_output_first(tmp_path):
    result = subprocess.run([sys.executable, "-m", "tools.select_poc_lambda", "--input-root",
                             str(tmp_path / "missing"), "--output", str(ROOT / "artifacts/new_selection.json")],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "frozen" in result.stderr.lower()


def test_current_poc_candidates_are_not_tied_and_selection_is_unchanged(tmp_path):
    from tools.select_poc_lambda import run

    root = ROOT / "artifacts/poc_stage4/batch2_poc/lambda_selection"
    before = {path: path.read_bytes() for path in root.rglob("*.json")}
    result = run(root, tmp_path / "future_selection.json")
    assert result["selected_lambda"] == 1.0
    assert all(row["eligible"] for row in result["candidates"])
    assert result["minimum_candidate_s_difference"] > 1e-12
    assert before == {path: path.read_bytes() for path in before}


def test_seed_isolated_analysis_refuses_frozen_directory_before_creation():
    from src.utils.artifact_naming import artifact_path

    with pytest.raises(ValueError, match="frozen"):
        artifact_path(ROOT / "artifacts/usability_test_never_create", 902, "new.csv")
    assert not (ROOT / "artifacts/usability_test_never_create").exists()


@pytest.mark.parametrize("module,option", [("tools.aggregate_results", "--out"),
                                           ("tools.reproduce_frozen_paper", "--output")])
def test_other_summary_tools_refuse_frozen_directory(module, option):
    result = subprocess.run([sys.executable, "-m", module, option, str(ROOT / "artifacts")],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "frozen" in result.stderr.lower()


def test_unused_transfer_config_field_fails():
    from src.experiment_config import validate_config

    with pytest.raises(ValueError, match="batch_size"):
        validate_config(dict(experiment_id="POC_ETTM2_V1", batch_size=32))


@pytest.mark.parametrize("module", ["tools.analyze_latent_spectrum", "tools.analyze_latent_layers",
                                    "tools.analyze_patchtst_latent_spectrum"])
def test_latent_analysis_checks_destination_before_loading_data(module):
    import importlib

    tool = importlib.import_module(module)
    with pytest.raises(ValueError, match="frozen"):
        tool.run(argparse.Namespace(output=ROOT / "artifacts"))


def test_generated_output_fails_closed_on_git_error(tmp_path, monkeypatch):
    import src.utils.output_paths as paths

    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(paths.subprocess, "run", lambda *a, **k: argparse.Namespace(returncode=128))
    with pytest.raises(ValueError, match="Git"):
        paths.validate_generated_output(tmp_path / "outputs/new.json", tmp_path)


def test_dataset_case_validation_matches_task_generation():
    from src import run

    config = dict(experiment_id="CANONICAL_PHENOMENON_27_V1", dataset="etth1")
    assert run.experiment_jobs(config, [42], ["ETTh1"]) == [dict(dataset="ETTh1", seed=42)]


def test_patchtst_rejects_incorrect_input_scale_metadata():
    from src.experiment_config import validate_config

    with pytest.raises(ValueError, match="standardization"):
        validate_config(dict(experiment_id="SUPPLEMENT_PATCHTST_ORIGIN", standardization="train_rows_only"))


def test_generated_directory_rejects_descendant_link_to_frozen_file(tmp_path):
    from src.utils.output_paths import validate_generated_output

    frozen = tmp_path / "artifacts/old.json"
    frozen.parent.mkdir()
    frozen.write_bytes(b"frozen")
    output = tmp_path / "outputs"
    output.mkdir()
    try:
        (output / "new.json").symlink_to(frozen)
    except OSError:
        pytest.skip("OS does not allow creating symbolic links")
    with pytest.raises(ValueError, match="link"):
        validate_generated_output(output, tmp_path)
    assert frozen.read_bytes() == b"frozen"


def test_reconstructed_status_cannot_default_to_source_present():
    from src import run

    result = run.dry_run(dict(experiment_id="NO_PE_CONTROL_V1"), [])
    assert result["source_status"] == "RECONSTRUCTED_CONTROL"
