"""Naming helpers for seed-isolated experiment artifacts."""

from __future__ import annotations

from pathlib import Path


def seed_directory(root: Path, seed: int | None) -> Path:
    """Return a seed-isolated output directory without encoding seed in files."""
    label = "unspecified" if seed is None else str(int(seed))
    return Path(root) / f"seed{label}"


def metric_filename(
    purpose: str,
    *,
    origin_a: int,
    origin_b: int,
    patch_length: int,
    context: int | None = None,
    horizon: int | None = None,
    stride: int | None = None,
    extension: str = "csv",
) -> str:
    """Build a purpose-and-hyperparameter filename with no seed token."""
    parts = [purpose, f"origin{int(origin_a)}_o{int(origin_b)}", f"p{int(patch_length)}"]
    if stride is not None:
        parts.append(f"s{int(stride)}")
    if context is not None:
        parts.append(f"L{int(context)}")
    if horizon is not None:
        parts.append(f"H{int(horizon)}")
    return "_".join(parts) + "." + extension.lstrip(".")


def artifact_path(root: Path, seed: int | None, filename: str) -> Path:
    output = seed_directory(Path(root), seed)
    output.mkdir(parents=True, exist_ok=True)
    return output / filename
