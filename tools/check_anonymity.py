import argparse
import json
import re
from pathlib import Path

TEXT_SUFFIXES = {".md", ".txt", ".yaml", ".yml", ".json", ".py", ".sh", ".bib", ".csv", ".toml"}
PATTERNS = {
    "windows_path": re.compile(r"[A-Za-z]:\\\\Users\\\\|[A-Za-z]:/Users/"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "wandb": re.compile(r"\bwandb\b", re.IGNORECASE),
    "api_key": re.compile(r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]")
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    issues = []
    for path in args.root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name == "check_anonymity.py":
            continue
        if any(part in {"data", "outputs", "temp_outputs", "generated", "third_party", ".git"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                issues.append({"file": str(path), "pattern": name})
    result = {"root": str(args.root), "issue_count": len(issues), "issues": issues}
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()
