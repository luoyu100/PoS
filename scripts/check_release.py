"""Audit a source package for generated files, secrets, and portability problems.

This is a conservative static check, not a substitute for reviewing data rights
or running provider/benchmark integration tests. It never edits source files.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import sys

FORBIDDEN_PARTS = {
    ".git",
    "outputs",
    "run_logs",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".pos_workspaces",
    ".pos_artifacts",
    "node_modules",
    "backup",
    "backups",
}
FORBIDDEN_FILES = {
    ".DS_Store",
    ".env",
    "events.jsonl",
    "events.md",
    "result.json",
    "all_emails.json",
    "emails_export.json",
    "inbox_export.json",
}
REMOVED_IMPORTS = {
    "acon",
    "amem",
    "a_mem",
    "pace",
    "hiagent",
    "lhh",
    "longhorizon_harness",
}
PRIVATE_PATH = re.compile(r"/(?:Users|apsara|home)/[A-Za-z0-9_.-]+/")
CJK = re.compile(r"[\u3400-\u9fff]")
SECRET = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)"
)


def audit(root: Path) -> list[str]:
    """Return actionable package violations without printing sensitive contents."""
    errors = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            errors.append(f"Symlink is not allowed: {relative}")
            continue
        if set(relative.parts) & FORBIDDEN_PARTS:
            errors.append(f"Generated/private path: {relative}")
            continue
        if not path.is_file():
            continue
        if (
            path.name in FORBIDDEN_FILES
            or path.name.startswith("._")
            or path.suffix in {".pyc", ".pyo", ".bak", ".zip", ".pem", ".key"}
        ):
            errors.append(f"Forbidden package file: {relative}")
        if "cases" in relative.parts or (
            relative.parts[:2] == ("benchmarks", "ClinDiag")
            and "answer_key" in relative.parts
        ):
            errors.append(f"Raw dataset must not be packaged: {relative}")
        if path.suffix not in {".py", ".md", ".yaml", ".json", ".txt", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8")
        # Original third-party labels/notices are not translated or anonymized.
        third_party = relative.parts[:3] == (
            "benchmarks",
            "RCA100",
            "answer_key",
        ) or relative.parts[:2] == ("benchmarks", "LOCA-Bench")
        if not third_party and PRIVATE_PATH.search(text):
            errors.append(f"Machine-specific absolute path: {relative}")
        if not third_party and CJK.search(text):
            errors.append(f"Non-English owned text: {relative}")
        if SECRET.search(text):
            errors.append(f"Possible credential: {relative}")
        if path.suffix == ".py":
            try:
                tree = ast.parse(text, filename=str(relative))
                compile(tree, str(relative), "exec")
            except SyntaxError:
                errors.append(f"Invalid Python syntax: {relative}")
                continue
            for node in ast.walk(tree):
                imports = (
                    [item.name for item in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
                if not third_party and any(
                    set(name.lower().split(".")) & REMOVED_IMPORTS for name in imports
                ):
                    errors.append(f"Removed baseline import: {relative}:{node.lineno}")
    configs = list((root / "configs").glob("*/*.yaml"))
    if len(configs) != 16:
        errors.append(f"Expected 16 configuration files, found {len(configs)}")
    for required in (
        "LICENSE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "docs/development.md",
    ):
        if not (root / required).is_file():
            errors.append(f"Missing release document: {required}")
    return errors


def main():
    """Exit nonzero on violations; never treat data permission as a static check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    errors = audit(args.root)
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print(
        "Source package checks passed. Review third-party data permissions separately."
    )


if __name__ == "__main__":
    main()
