"""Each project lint rule catches what it should and nothing else."""

from pathlib import Path

import pytest
from lint_repo import MAX_PYTHON_LINES, check_python_file

CLEAN_MODULE = '"""Places sessions."""\n\nsession_count = 3\n'


def test_a_clean_module_passes():
    assert check_python_file(Path("src/forge/core/placement.py"), CLEAN_MODULE) == []


def test_vague_module_names_are_rejected():
    [violation] = check_python_file(Path("src/forge/utils.py"), CLEAN_MODULE)
    assert "utils/helpers/misc" in violation.message


def test_long_files_are_rejected():
    source = "x = 1\n" * (MAX_PYTHON_LINES + 1)
    [violation] = check_python_file(Path("src/forge/core/long.py"), source)
    assert "split it" in violation.message


@pytest.mark.parametrize(
    "source",
    ['AUDIO = "/home/someone/checkpoints"\n', 'RUNS = "/mnt/data/runs"\n', "RUNS = 'D:\\\\runs'\n"],
)
def test_machine_specific_paths_are_rejected(source):
    [violation] = check_python_file(Path("src/forge/providers/modal.py"), source)
    assert "machine-specific path" in violation.message


@pytest.mark.parametrize("path", ["runs", "/workspace/job", "/etc/docker/daemon.json"])
def test_relative_and_system_paths_are_fine(path):
    assert check_python_file(Path("src/forge/providers/modal.py"), f"RUNS = {path!r}\n") == []


def test_off_vocabulary_names_point_to_the_preferred_word():
    source = "def load_leases(ledger):\n    return ledger\n"
    [violation] = check_python_file(Path("src/forge/ledger.py"), source)
    assert "say 'session'" in violation.message
