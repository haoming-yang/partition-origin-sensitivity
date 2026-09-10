import csv
import hashlib
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from tools import verify_poc_stage3 as poc


ROOT = Path(__file__).resolve().parents[1]


def test_poc_manifest_preserves_posix_path_components(monkeypatch):
    monkeypatch.setattr(poc, "REPO_ROOT", PurePosixPath("/checkout"))
    _, path, _ = poc._entries()[0]
    assert path == PurePosixPath("/checkout/artifacts/poc_stage3/attribution_abc/a_reused_from_canonical.json")


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
def test_poc_tracked_text_accepts_only_line_ending_variation(tmp_path, monkeypatch, ending):
    source = tmp_path / "source.py"
    source.write_bytes(b"x = 1" + ending)
    manifest = tmp_path / "SHA256SUMS.txt"
    digest = hashlib.sha256(b"x = 1\n").hexdigest()
    manifest.write_text(f"{digest}  source.py\n", encoding="utf-8")
    monkeypatch.setattr(poc, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(poc, "MANIFEST", manifest)
    monkeypatch.setattr(sys, "argv", ["verify_poc_stage3.py"])
    assert poc.main() == 0
    source.write_bytes(b"x = 2" + ending)
    assert poc.main() != 0


def test_poc_external_checkpoint_uses_exact_bytes(tmp_path, monkeypatch):
    source = tmp_path / "checkpoint.pt"
    content = b"binary\x00data\r\n"
    source.write_bytes(content)
    manifest = tmp_path / "SHA256SUMS.txt"
    manifest.write_text(f"{hashlib.sha256(content).hexdigest()}  [external] checkpoint.pt\n", encoding="utf-8")
    monkeypatch.setattr(poc, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(poc, "MANIFEST", manifest)
    monkeypatch.setattr(sys, "argv", ["verify_poc_stage3.py"])
    assert poc.main() == 0
    source.write_bytes(content.replace(b"\r\n", b"\n"))
    assert poc.main() != 0


def test_poc_absent_external_is_optional_unless_strict(tmp_path, monkeypatch):
    manifest = tmp_path / "SHA256SUMS.txt"
    manifest.write_text(f"{'a' * 64}  [external] checkpoint.pt\n", encoding="utf-8")
    monkeypatch.setattr(poc, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(poc, "MANIFEST", manifest)
    monkeypatch.setattr(sys, "argv", ["verify_poc_stage3.py"])
    assert poc.main() == 0
    monkeypatch.setattr(sys, "argv", ["verify_poc_stage3.py", "--strict"])
    assert poc.main() != 0


def test_poc_empty_manifest_does_not_pass(tmp_path, monkeypatch):
    manifest = tmp_path / "SHA256SUMS.txt"
    manifest.write_text("# Manifest header only\n", encoding="utf-8")
    monkeypatch.setattr(poc, "MANIFEST", manifest)
    monkeypatch.setattr(sys, "argv", ["verify_poc_stage3.py"])
    assert poc.main() != 0


def test_checked_in_poc_manifest_verifies_without_checkpoints():
    result = subprocess.run([sys.executable, str(ROOT / "tools/verify_poc_stage3.py")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_checked_in_source_manifests_match_release_contents():
    for name, prefix in [("patch_models", "tier1"), ("time_series_library", "")]:
        root = ROOT / "third_party" / name
        with (root / "SOURCE_MANIFEST.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows
        for row in rows:
            content = (root / prefix / row["path"]).read_bytes().replace(b"\r\n", b"\n")
            assert row.get("hash_mode") == "lf"
            assert len(content) == int(row["bytes"]), row["path"]
            assert hashlib.sha256(content).hexdigest() == row["sha256"], row["path"]


def source_fixture(root):
    manifests = []
    for name, prefix in [("patch_models", "tier1"), ("time_series_library", "")]:
        directory = root / "third_party" / name
        source = directory / prefix / "model.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"x = 1\r\n")
        manifest = directory / "SOURCE_MANIFEST.csv"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "bytes", "sha256", "hash_mode"])
            writer.writerow(["model.py", 6, hashlib.sha256(b"x = 1\n").hexdigest(), "lf"])
        manifests.append((source, manifest))
    return manifests


def run_source_verifier(root):
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/verify_source_manifests.py"), "--root", str(root)],
        capture_output=True, text=True,
    )


def test_source_verifier_accepts_newlines_but_rejects_changed_or_missing_files(tmp_path):
    source, _ = source_fixture(tmp_path)[0]
    result = run_source_verifier(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    source.write_bytes(b"x = 2\r\n")
    assert run_source_verifier(tmp_path).returncode != 0
    source.unlink()
    assert run_source_verifier(tmp_path).returncode != 0


def test_source_verifier_rejects_empty_manifest(tmp_path):
    _, manifest = source_fixture(tmp_path)[0]
    manifest.write_text("path,bytes,sha256,hash_mode\n", encoding="utf-8")
    assert run_source_verifier(tmp_path).returncode != 0


@pytest.mark.parametrize("path", ["../outside.py", "/outside.py", "C:/outside.py"])
def test_poc_verifier_rejects_paths_outside_repository(tmp_path, monkeypatch, path):
    manifest = tmp_path / "SHA256SUMS.txt"
    manifest.write_text(f"{'a' * 64}  {path}\n", encoding="utf-8")
    monkeypatch.setattr(poc, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(poc, "MANIFEST", manifest)
    monkeypatch.setattr(sys, "argv", ["verify_poc_stage3.py"])
    assert poc.main() != 0


@pytest.mark.parametrize("malformed", ["X" + "a" * 63 + "  missing.py", "a" * 63 + "  missing.py", "a" * 64])
@pytest.mark.parametrize("first", [False, True])
def test_poc_malformed_record_cannot_be_silently_skipped(tmp_path, monkeypatch, malformed, first):
    (tmp_path / "source.py").write_bytes(b"x = 1\n")
    digest = hashlib.sha256(b"x = 1\n").hexdigest()
    valid = f"{digest}  source.py"
    lines = [malformed, valid] if first else [valid, malformed]
    manifest = tmp_path / "SHA256SUMS.txt"
    manifest.write_text("# Test manifest\n" + "\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(poc, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(poc, "MANIFEST", manifest)
    monkeypatch.setattr(sys, "argv", ["verify_poc_stage3.py"])
    assert poc.main() != 0


@pytest.mark.parametrize("mode", ["lf", "raw"])
def test_source_generator_and_verifier_round_trip_is_idempotent(tmp_path, mode):
    from tools.verify_source_manifests import verify_manifest

    root = tmp_path / "source"
    root.mkdir()
    (root / "model.py").write_bytes(b"x = 1\r\n")
    archive = root / "SOURCE_MANIFEST.historical.csv"
    archive.write_bytes(b"unchanged historical record\r\n")
    output = root / "SOURCE_MANIFEST.csv"
    command = [sys.executable, str(ROOT / "tools/generate_source_manifest.py"),
               "--root", str(root), "--output", str(output)]
    if mode == "raw":
        command += ["--hash-mode", mode]
    for _ in range(2):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        with output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert rows[0].get("hash_mode") == mode
        expected = b"x = 1\n" if mode == "lf" else b"x = 1\r\n"
        assert rows[0]["sha256"] == hashlib.sha256(expected).hexdigest()
        assert int(rows[0]["bytes"]) == len(expected)
        assert verify_manifest(output, root) == (1, [])
    assert archive.read_bytes() == b"unchanged historical record\r\n"
