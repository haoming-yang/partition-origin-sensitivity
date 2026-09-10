import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reference_config_has_no_machine_specific_dataset_path():
    config = json.loads((ROOT / "configs" / "reference.json").read_text(encoding="utf-8"))
    dataset_file = config["dataset_file"]
    assert not Path(dataset_file).is_absolute()
    assert "Deep Learning" not in dataset_file
    assert "杨昊明" not in dataset_file


def test_public_text_has_no_machine_specific_paths():
    candidates = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "tests" not in path.parts
        and "third_party" not in path.parts
        and path.suffix.lower() in {".md", ".py", ".yaml", ".yml", ".json", ".sh", ".txt", ".toml"}
    ]
    offenders = []
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        if "E:\\Deep Learning" in text or "C:\\Users\\" in text or "杨昊明" in text:
            offenders.append(str(path.relative_to(ROOT)))
            continue
        if path.suffix.lower() == ".json":
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                continue
            stack = [decoded]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
                elif isinstance(value, str) and (re.match(r"^[A-Za-z]:[\\/]", value) or "杨昊明" in value):
                    offenders.append(str(path.relative_to(ROOT)))
                    stack.clear()
                    break
    assert offenders == []


def test_canonical_config_dry_run_is_side_effect_free(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.run",
            "--config",
            str(ROOT / "configs" / "core" / "canonical.yaml"),
            "--output-root",
            str(tmp_path / "outputs"),
            "--seeds",
            "42,43,44",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"training_started": false' in result.stdout
    assert not (tmp_path / "outputs").exists()


def test_formal_gap_definitions_are_explicit():
    text = (ROOT / "docs" / "reproducibility.md").read_text(encoding="utf-8")
    assert "(max_r MSE_r - min_r MSE_r) / min_r MSE_r" in text
    assert "(max_{r>=1} MSE_r - min_{r>=1} MSE_r) / min_{r>=1} MSE_r" in text


def test_git_commit_provenance_is_available():
    from src.training.runner import current_git_commit

    commit = current_git_commit()
    assert commit is not None
    assert len(commit) == 40
    assert all(character in "0123456789abcdef" for character in commit)
