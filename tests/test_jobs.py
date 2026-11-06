"""Jobs pack the right files, refuse oversized folders, and agree on a torch version."""

import json
import zipfile

import pytest

from forge import jobs


def test_packing_leaves_out_caches_environments_and_old_outputs(tmp_path):
    (tmp_path / "train.py").write_text("print('hi')\n")
    for folder in (".git", ".venv", "__pycache__", "outputs", "data"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "file.txt").write_text("x")
    job = jobs.describe(tmp_path / "train.py", ["--epochs", "3"])
    jobs.pack(job, tmp_path.parent / "job.zip")
    names = zipfile.ZipFile(tmp_path.parent / "job.zip").namelist()
    assert sorted(names) == ["data/file.txt", "train.py"]


def test_an_oversized_job_folder_refuses_with_a_clear_message(tmp_path, monkeypatch):
    (tmp_path / "train.py").write_text("x" * 100)
    monkeypatch.setattr(jobs, "MAX_JOB_BYTES", 10)
    with pytest.raises(ValueError, match="Move data out"):
        jobs.pack(jobs.describe(tmp_path / "train.py", []), tmp_path.parent / "job.zip")


def test_requirements_beside_the_script_are_used_and_a_torch_pin_is_read(tmp_path):
    (tmp_path / "train.py").write_text("")
    (tmp_path / "requirements.txt").write_text("# deps\ntorch==2.8.0\ntqdm\n")
    job = jobs.describe(tmp_path / "train.py", [])
    assert job.requirements == ["torch==2.8.0", "tqdm"]
    assert job.torch_pin == "2.8.0"


def snapshot_with_torch(folder, version):
    folder.mkdir()
    document = {"state": {"mapping": [["step", 3], ["torch_version", version]]}}
    (folder / "snapshot.json").write_text(json.dumps(document))
    return folder


def test_a_snapshot_decides_the_next_sessions_torch(tmp_path):
    job = jobs.Job(tmp_path, "train.py", [], [])
    snapshot = snapshot_with_torch(tmp_path / "snapshot", "2.8.0+cu128")
    assert jobs.session_torch(job, snapshot, "2.9.0") == "2.8.0"
    assert jobs.session_torch(job, None, "2.9.0+cpu") == "2.9.0"


def test_a_pin_that_contradicts_the_snapshot_refuses_and_says_what_to_pin(tmp_path):
    job = jobs.Job(tmp_path, "train.py", [], ["torch==2.9.0"])
    snapshot = snapshot_with_torch(tmp_path / "snapshot", "2.8.0")
    with pytest.raises(ValueError, match=r"Pin torch==2\.8\.0"):
        jobs.session_torch(job, snapshot, "2.9.0")


def test_a_forge_toml_beside_the_script_declares_what_the_job_needs(tmp_path):
    (tmp_path / "train.py").write_text("")
    (tmp_path / "forge.toml").write_text('[job]\ngpu = "a100"\nmemory_gb = 24\nhours = 6\n')
    declaration = jobs.describe(tmp_path / "train.py", []).declaration
    assert declaration == jobs.Declaration(gpu="a100", memory_gb=24, hours=6.0)


def test_without_a_forge_toml_the_job_declares_nothing(tmp_path):
    (tmp_path / "train.py").write_text("")
    assert jobs.describe(tmp_path / "train.py", []).declaration == jobs.Declaration()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[job]\ngpus = 2\n", "unknown key 'gpus' in \\[job\\]"),
        ('[run]\ngpu = "T4"\n', "unknown table or key 'run'"),
        ('[job]\nmemory_gb = "24"\n', "memory_gb must be a whole number"),
        ("[job]\nhours = 0\n", "hours must be a number above 0"),
    ],
)
def test_a_forge_toml_with_an_unknown_key_or_a_wrong_value_refuses_naming_it(
    tmp_path, text, message
):
    (tmp_path / "train.py").write_text("")
    (tmp_path / "forge.toml").write_text(text)
    with pytest.raises(ValueError, match=message):
        jobs.describe(tmp_path / "train.py", [])
