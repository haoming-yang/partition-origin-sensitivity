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
