import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = {".md", ".json", ".py", ".yml", ".yaml", ".toml", ".cff", ".txt", ".sh"}


def write_export_text(source, target, text):
    if text == source.read_text(encoding="utf-8"):
        shutil.copy2(source, target)
    else:
        target.write_bytes(text.encode("utf-8"))


def sanitize_review_text(rel, text):
    if rel == "CITATION.cff":
        text = re.sub(r"(?ms)^authors:\n(?:^[ \t].*\n)+", "", text)
        text = re.sub(r"(?m)^repository-code:.*\n", "", text)
    if rel == "pyproject.toml":
        text = re.sub(r"(?m)^authors\s*=\s*.*\n", "", text)
    if rel == "README.md":
        text = re.sub(r"<a href=\"https?://github\.com/[^\"]+\">.*?</a>", "", text, flags=re.S)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Export outside the source repository")
    if output.exists():
        raise ValueError("Export directory must not already exist")
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    output.mkdir(parents=True)
    for rel in sorted(set(paths)):
        if not rel or rel.startswith((".github/", "docs/superpowers/")):
            continue
        source = ROOT / rel
        if not source.is_file():
            continue
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() in TEXT and "third_party" not in source.parts and source.name != "check_anonymity.py":
            text = source.read_text(encoding="utf-8")
            text = sanitize_review_text(rel, text)
            write_export_text(source, target, text)
        else:
            shutil.copy2(source, target)
    for record in json.loads((ROOT / "artifacts/review_checkpoint_manifest.json").read_text()):
        source = ROOT / record["file"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        target = output / record["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*") if p.is_file()}
    (output / "REVIEW_SHA256.json").write_text(json.dumps(manifest, indent=2))
    print("Anonymous export created")


if __name__ == "__main__":
    main()
