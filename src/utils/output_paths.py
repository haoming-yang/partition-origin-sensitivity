from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def validate_generated_output(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    root = repo_root.resolve()
    output = path.resolve()
    frozen = (root / "artifacts").resolve()
    if output == frozen or frozen in output.parents:
        raise ValueError(f"Refusing generated output in frozen artifacts: {output}; use outputs/ instead")
    try:
        relative = output.relative_to(root)
    except ValueError:
        relative = None
    if relative is not None and relative.parts and relative.parts[0] == ".git":
        raise ValueError(f"Refusing generated output in repository metadata: {output}")
    if relative is not None:
        try:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
                cwd=root, capture_output=True,
            )
        except OSError as exc:
            if (root / ".git").exists():
                raise ValueError("Git is required to check repository-tracked output protection") from exc
        else:
            if tracked.returncode == 0:
                raise ValueError(f"Refusing to overwrite repository-tracked file: {output}; use outputs/ instead")
            if tracked.returncode != 1 and (root / ".git").exists():
                raise ValueError("Git tracking check failed; refusing an unverified output destination")
    pending = [output] if output.is_dir() else []
    while pending:
        for child in pending.pop().iterdir():
            if child.is_symlink() or child.resolve() != child.absolute():
                raise ValueError(f"Refusing generated output tree containing a link: {child}; use a fresh plain directory")
            if child.is_dir():
                pending.append(child)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated destinations without creating files")
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        for path in args.paths:
            validate_generated_output(path)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
