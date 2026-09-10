from __future__ import annotations

import hashlib
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath


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
