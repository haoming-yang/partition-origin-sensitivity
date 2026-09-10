from src import run as run_module


def test_run_controlled_passes_configured_training_hyperparameters(monkeypatch, tmp_path):
    captured = {}

    def fake_train_controlled(**kwargs):
        captured.update(kwargs)
        return {"status": "DRY"}

    monkeypatch.setattr("src.training.runner.train_controlled", fake_train_controlled)
    config = {
        "experiment_id": "CANONICAL_PHENOMENON_27_V1",
        "batch_size": 7,
        "learning_rate": 0.002,
        "weight_decay": 0.003,
        "epochs": 4,
    }

    result = run_module.run_controlled(config, "ETTh1", 91, tmp_path)

    assert result == {"status": "DRY"}
    assert captured["batch_size"] == 7
    assert captured["learning_rate"] == 0.002
    assert captured["weight_decay"] == 0.003
    assert captured["seed"] == 91


def test_run_supplement_preserves_patchtst_zero_weight_decay_default(monkeypatch, tmp_path):
    captured = {}

    def fake_train_patchtst(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "DRY"}

    monkeypatch.setattr("src.training.runner.train_patchtst", fake_train_patchtst)
    config = {
        "experiment_id": "SUPPLEMENT_PATCHTST_ORIGIN",
        "epochs": 10,
        "batch_size": 11,
        "learning_rate": 0.001,
    }

    result = run_module.run_supplement(config, "ETTh1", 42, tmp_path)

    assert result == {"status": "DRY"}
    assert captured["batch_size"] == 11
    assert captured["learning_rate"] == 0.001
    assert captured["weight_decay"] == 0.0


def test_mixer_config_uses_recovered_fullsplit_runner(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(run_module.subprocess, "run", fake_run)
    config = {
        "experiment_id": "CROSS_MODEL_PHASE_V1",
        "models": ["Transformer", "MLP", "Conv"],
        "dataset": "ETTH1",
        "epochs": 5,
    }

    result = run_module.run_fullsplit(config, [17], tmp_path)

    assert result["status"] == "COMPLETE"
    assert len(calls) == 3
    for model, (command, kwargs) in zip(config["models"], calls):
        assert command[1].replace("\\", "/").endswith("tools/fullsplit/fullsplit_3run_runner.py")
        assert command[command.index("--model") + 1] == model
        assert command[command.index("--seed") + 1] == "17"
        assert kwargs["cwd"] == run_module.ROOT


def test_mixer_config_reads_default_seeds():
    config = {"default_seeds": [42, 43, 44]}

    assert run_module.resolve_seeds(config, seed=None, seeds=None) == [42, 43, 44]


def test_mixer_config_skips_unselected_dataset(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(run_module.subprocess, "run", fake_run)
    config = {
        "experiment_id": "CROSS_MODEL_PHASE_V1",
        "models": ["Transformer", "MLP", "Conv"],
        "dataset": "ETTh1",
    }

    result = run_module.run_fullsplit(config, [17], tmp_path, datasets=["Weather"])

    assert result["status"] == "NO_JOBS"
    assert result["jobs"] == []
    assert calls == []


def test_mixer_config_rejects_unsupported_protocol_override(monkeypatch, tmp_path):
    import pytest

    calls = []
    monkeypatch.setattr(
        run_module.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    config = {
        "experiment_id": "CROSS_MODEL_PHASE_V1",
        "models": ["Transformer", "MLP", "Conv"],
        "dataset": "ETTh1",
        "epochs": 30,
    }

    with pytest.raises(ValueError, match="epochs=5"):
        run_module.run_fullsplit(config, [17], tmp_path)
    assert calls == []


def test_mixer_config_rejects_ignored_optimizer_overrides(tmp_path):
    import pytest

    for field, value in (("batch_size", 64), ("learning_rate", 0.01),
                         ("weight_decay", 0.0), ("optimizer", "SGD")):
        config = {
            "experiment_id": "CROSS_MODEL_PHASE_V1",
            "models": ["Transformer", "MLP", "Conv"],
            "dataset": "ETTh1",
            field: value,
        }
        with pytest.raises(ValueError, match=field):
            run_module.run_fullsplit(config, [17], tmp_path)


def test_fullsplit_runner_refuses_nonempty_output(tmp_path):
    import pytest
    from tools.fullsplit import fullsplit_3run_runner as runner

    output = tmp_path / "existing-run"
    output.mkdir()
    (output / "partial.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.prepare_output(output)
