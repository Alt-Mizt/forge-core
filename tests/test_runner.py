"""An unchanged script survives SIGTERM at ordinary and epoch-boundary steps."""

import ast
from pathlib import Path

from resume_proof import prove_wrapper

ROOT = Path(__file__).resolve().parents[1]


def test_sigterm_resume_in_fresh_processes_matches_the_unwrapped_training_script(tmp_path):
    evidence = prove_wrapper(tmp_path)
    assert len(evidence) == 9
    for position in range(0, len(evidence), 3):
        elapsed = float(evidence[position].split("snapshot_seconds=")[1])
        assert elapsed < 60
        assert evidence[position + 1].split()[-1] == evidence[position + 2].split()[-1]


def test_the_tiny_example_never_imports_forge():
    tree = ast.parse((ROOT / "examples/tiny/train.py").read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    for node in imports:
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("forge")
        else:
            assert all(not alias.name.startswith("forge") for alias in node.names)
