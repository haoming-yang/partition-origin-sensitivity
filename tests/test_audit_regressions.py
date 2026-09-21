import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import run
from src.training import runner
from tools import aggregate_results, export_review_artifact


def test_weather_head_dispatch_has_four_capped_conditions(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(run, "run_controlled", lambda c, d, s, o: calls.append((c, d, s, o)))
    config = {"experiment_id": "MASK_HEAD_FACTORIAL_V1", "dataset": "Weather", "diagnostic_split": "capped_test_setting"}
    run.run(config, [42], ["Weather"], tmp_path)
    assert len(calls) == 4
    assert {d for c, d, s, o in calls} == {"Weather"}
    assert {(c["head_geometry"], c["use_mask"]) for c, d, s, o in calls} == {(h, m) for h in ["flattened", "pooled"] for m in [True, False]}
    assert len({o for c, d, s, o in calls}) == 4
    for c, d, s, o in calls:
        assert (c["max_train_windows"], c["max_validation_windows"], c["max_test_windows"]) == (1024, 16, 16)


def test_patch_aliases_are_canonical_and_conflicts_fail():
    assert aggregate_results.protocol_settings({"patch_len": 8}) == aggregate_results.protocol_settings({"p": 8, "patch_length": 8})
    assert aggregate_results.protocol_settings({"patch_len": 8}) != aggregate_results.protocol_settings({"patch_len": 16})
    with pytest.raises(ValueError):
        aggregate_results.protocol_settings({"patch_len": 8, "p": 16})


def test_export_unchanged_text_preserves_crlf_bytes(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_bytes(b'{\r\n  "frozen": true\r\n}\r\n')
    export_review_artifact.write_export_text(a, b, a.read_text(encoding="utf-8"))
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()


def test_unmasked_model_has_value_only_tokenizer_and_valid_outputs():
    model = runner.ControlledTransformerSupplement(24, 3, 12, 12, channels=1, use_mask=False).eval()
    assert model.embed.in_features == 12
    x, observed = runner.partition(torch.randn(2, 24, 1), 0, 24, 12, 12)
    with torch.no_grad():
        y = model(x, observed)
    assert y.shape == (2, 3, 1)
    assert torch.isfinite(y).all()


def test_window_caps_select_exact_prefixes():
    data = runner.DataBundle("ETTh1", Path("unused"), np.zeros((20, 1)), np.zeros((20, 1)), np.arange(10), np.arange(10, 15), np.arange(15, 20), np.zeros(1), np.ones(1), {}, 1)
    capped = runner.cap_windows(data, 4, 2, 3)
    assert capped.train.tolist() == [0, 1, 2, 3]
    assert capped.validation.tolist() == [10, 11]
    assert capped.test.tolist() == [15, 16, 17]
    assert len(data.train) == 10


def test_run_controlled_passes_masks_and_caps(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(runner, "train_controlled", lambda **kw: captured.update(kw))
    run.run_controlled(dict(experiment_id="MASK_HEAD_FACTORIAL_V1", use_mask=False, max_train_windows=1024, max_validation_windows=16, max_test_windows=16, diagnostic_split="capped_test_setting"), "Weather", 42, tmp_path)
    assert captured["dataset"] == "Weather"
    assert captured["use_mask"] is False
    assert (captured["max_train_windows"], captured["max_validation_windows"], captured["max_test_windows"]) == (1024, 16, 16)


def test_head_dry_matrix_matches_dataset_selection():
    jobs = run.head_jobs(dict(dataset="Weather"), [42, 43, 44], ["Weather"])
    assert len(jobs) == 12
    assert {j["dataset"] for j in jobs} == {"Weather"}
    assert run.head_jobs(dict(dataset="Weather"), [42], ["ETTh1"]) == []
