"""Project lint rules that ruff can't express.

Usage: uv run python scripts/lint_repo.py
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_PYTHON_LINES = 400
VAGUE_MODULE_NAMES = frozenset({"utils.py", "helpers.py", "misc.py", "common.py"})
# Absolute paths into a home or mount directory only work on one machine.
MACHINE_ROOTS = frozenset({"home", "Users", "srv", "mnt"})
WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:\\")
# One vocabulary: forge places sessions and holds time; it has no leases or budgets.
PREFERRED_WORDS = {"lease": "session", "rental": "session", "budget": "allowance"}
SKIPPED_DIRECTORIES = frozenset({".git", ".venv", "__pycache__", "node_modules"})


@dataclass(frozen=True)
class Violation:
    """One broken rule at one place in the repo."""

    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        """Format as path:line: message, the shape editors and terminals link to."""
        return f"{self.path}:{self.line}: {self.message}"


def main() -> int:
    """Check the whole repo and print every violation. Returns the process exit code."""
    root = Path(__file__).resolve().parent.parent
    violations = lint_repo(root)
    for violation in violations:
        print(violation)
    return 1 if violations else 0


def lint_repo(root: Path) -> list[Violation]:
    """Run every rule over the Python files under ``root``."""
    violations: list[Violation] = []
    for path in sorted(_tracked_files(root, "*.py")):
        violations += check_python_file(path.relative_to(root), path.read_text())
    return violations


def check_python_file(path: Path, source: str) -> list[Violation]:
    """Check one Python file for vague names, length, machine paths and off-vocabulary names."""
    violations: list[Violation] = []
    if path.name in VAGUE_MODULE_NAMES:
        violations.append(
            Violation(path, 1, "name the module for what it does, not utils/helpers/misc")
        )
    line_count = len(source.splitlines())
    if line_count > MAX_PYTHON_LINES:
        violations.append(
            Violation(path, 1, f"{line_count} lines; split it below {MAX_PYTHON_LINES}")
        )
    for node in ast.walk(ast.parse(source)):
        violations += _check_node(path, node)
    return violations


def _check_node(path: Path, node: ast.AST) -> list[Violation]:
    """Apply the per-node rules: no machine-specific paths, one vocabulary."""
    line = getattr(node, "lineno", 1)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if _machine_specific(node.value):
            return [Violation(path, line, "machine-specific path; take it from config or args")]
        return []
    name = _defined_name(node)
    for word, preferred in PREFERRED_WORDS.items():
        if name and word in name.lower():
            return [Violation(path, line, f"'{name}': say '{preferred}', not '{word}'")]
    return []


def _machine_specific(value: str) -> bool:
    """Whether a string is an absolute path under a home or mount root, or a Windows drive."""
    parts = PurePosixPath(value).parts
    # ("/", "home", "someone", ...): a bare "/home" names the root itself, not a machine.
    under_root = value.startswith("/") and len(parts[1:]) > 1 and parts[1] in MACHINE_ROOTS
    return under_root or WINDOWS_DRIVE.match(value) is not None


def _defined_name(node: ast.AST) -> str | None:
    """The name a node defines, if it defines one."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return node.name
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        return node.id
    if isinstance(node, ast.arg):
        return node.arg
    return None


def _tracked_files(root: Path, pattern: str) -> list[Path]:
    """Files under ``root`` matching ``pattern``, outside vendored and generated folders."""
    return [
        path
        for path in root.rglob(pattern)
        if not SKIPPED_DIRECTORIES.intersection(path.relative_to(root).parts)
    ]


if __name__ == "__main__":
    sys.exit(main())
