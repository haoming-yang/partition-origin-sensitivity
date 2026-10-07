from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath


def runtime_metadata(root: Path, device: str = "cpu") -> dict:
    versions = {}
    for package in ("torch", "numpy", "pandas", "PyYAML", "scipy", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    commit, dirty = None, None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                            text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit, "git_dirty": dirty, "device": device,
            "python": platform.python_version(), "software_versions": versions}


def manifest_path(root: PurePath, value: str) -> PurePath:
    path = PurePosixPath(value.replace("\\", "/"))
    if not path.parts or path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
        raise ValueError(f"Manifest path must be relative and contained in its root: {value}")
    return root.joinpath(*path.parts)


def file_fingerprint(path: Path, *, normalize_lf: bool = False) -> tuple[int, str]:
    digest = hashlib.sha256()
    if normalize_lf:
        content = path.read_bytes().replace(b"\r\n", b"\n")
        digest.update(content)
        return len(content), digest.hexdigest()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()
