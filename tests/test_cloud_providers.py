"""The Kaggle and Modal adapters, against fakes of their SDKs, so nothing leaves this machine."""

import ast
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import modal
import pytest

from forge.accounts import new_account
from forge.providers import State, kaggle_api, modal_api, notebook
from forge.providers import kaggle as kaggle_adapter


class FakeKaggle:
    """Just enough of the Kaggle SDK for the worker's calls."""

    def __init__(self, status="RUNNING", free_hours=30, private=True, exit_code=None):
        self.config_values = {"username": "someone"}
        self.status = status
        self.free = timedelta(hours=free_hours)
        self.private = private
        self.exit_code = exit_code
        self.saved = []

    def build_kaggle_client(self):
        fake = self

        class Context:
            def __enter__(self):
                kernels = SimpleNamespace(
                    get_accelerator_quota_statistics=fake.quota,
                    save_kernel=fake.save,
                    get_kernel=fake.get,
                )
                return SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=kernels))

            def __exit__(self, *_error):
                return None

        return Context()

    def quota(self, _request):
        quota = SimpleNamespace(
            is_pay_to_scale_enabled=False,
            total_time_allowed=self.free,
            time_used=timedelta(0),
            time_reserved=timedelta(0),
        )
        return SimpleNamespace(gpu_quota=quota)

    def save(self, request):
        self.saved.append(request)
        # Kaggle answers with a path, not the bare name.
        return SimpleNamespace(error="", ref=f"/code/{request.slug}")

    def get(self, request):
        ref = f"{request.user_name}/{request.kernel_slug}"
        metadata = SimpleNamespace(ref=ref, is_private=self.private, machine_shape="NvidiaTeslaT4")
        return SimpleNamespace(metadata=metadata)

    def kernels_status(self, _ref):
        return SimpleNamespace(status=SimpleNamespace(name=self.status))

    def kernels_output(self, _ref, folder, file_pattern, quiet):
        del file_pattern, quiet
        if self.exit_code is not None:
            out = Path(folder, "forge/out")
            out.mkdir(parents=True)
            (out / "exit_code").write_text(f"{self.exit_code}\n")


def kaggle_client(fake):
    client = object.__new__(kaggle_api.Client)
    client.api = fake
    return client


def test_a_kaggle_session_is_private_on_the_t4_machine_with_its_own_timeout():
    fake = FakeKaggle()
    request = {"handle": "forge-run-1", "code": "print(1)", "seconds": 600}
    assert kaggle_client(fake).invoke("start", request) == {"ref": "someone/forge-run-1"}
    saved = fake.saved[0]
    assert saved.is_private is True
    assert saved.machine_shape == "NvidiaTeslaT4"
    assert saved.session_timeout_seconds == 600


def test_kaggle_refuses_a_session_its_free_quota_cannot_cover():
    request = {"handle": "forge-run-1", "code": "", "seconds": 7200}
    with pytest.raises(ValueError, match="free quota"):
        kaggle_client(FakeKaggle(free_hours=1)).invoke("start", request)


def test_kaggle_refuses_a_notebook_it_cannot_confirm_as_private():
    request = {"handle": "forge-run-1", "code": "", "seconds": 600}
    with pytest.raises(ValueError, match="private"):
        kaggle_client(FakeKaggle(private=False)).invoke("start", request)


@pytest.mark.parametrize(
    ("status", "exit_code", "expected"),
    [
        ("RUNNING", None, {"state": "running"}),
        ("ERROR", None, {"state": "failed"}),
        ("COMPLETE", 143, {"exit_code": 143}),
        ("COMPLETE", 0, {"exit_code": 0}),
    ],
)
def test_kaggle_poll_reads_the_job_exit_status_once_the_notebook_ends(status, exit_code, expected):
    fake = FakeKaggle(status=status, exit_code=exit_code)
    assert kaggle_client(fake).invoke("poll", {"handle": "forge-run-1"}) == expected


def test_a_running_kaggle_notebook_is_never_reported_as_stopped():
    fake = FakeKaggle(status="RUNNING")
    assert kaggle_client(fake).invoke("stop", {"handle": "forge-run-1"}) == {"stopped": False}


def test_the_kaggle_adapter_turns_exit_codes_into_states(monkeypatch):
    replies = iter([{"exit_code": 143}, {"exit_code": 0}, {"exit_code": 1}, {"state": "running"}])
    monkeypatch.setattr(kaggle_adapter, "call", lambda *_args, **_kwargs: next(replies))
    client = kaggle_adapter.Client()
    account = new_account("kaggle")
    states = [client.poll(account, "handle") for _ in range(4)]
    assert states == [State.PAUSED, State.FINISHED, State.FAILED, State.RUNNING]


def test_the_notebook_is_valid_python_that_installs_the_sessions_torch():
    code = notebook.source(b"bundle", 600, "2.8.0", ["tqdm"])
    ast.parse(code)
    assert "'torch==2.8.0'" in code and "'tqdm'" in code


class FakeSandbox:
    def __init__(self, **options):
        self.options = options
        self.object_id = "sb-1"
        self.written = {}
        self.terminated = False
        self.filesystem = SimpleNamespace(write_bytes=self.write)

    def write(self, content, path):
        self.written[path] = content

    def terminate(self, wait):
        del wait
        self.terminated = True

    def poll(self):
        return 0 if self.terminated else None


def test_a_modal_session_has_a_gpu_a_timeout_gvisor_and_no_network(monkeypatch):
    created = []
    monkeypatch.setattr(modal.App, "lookup", lambda *_args, **_kwargs: SimpleNamespace(app_id="a"))
    image = SimpleNamespace(pip_install=lambda *packages: packages)
    monkeypatch.setattr(modal.Image, "debian_slim", lambda **_kwargs: image)

    def create(*command, **options):
        created.append(FakeSandbox(command=command, **options))
        return created[-1]

    monkeypatch.setattr(modal.Sandbox, "create", create)
    request = {
        "handle": "forge-run-1",
        "bundle": "YnVuZGxl",
        "seconds": 300,
        "torch": "2.8.0",
        "requirements": [],
    }
    assert modal_api.Client().invoke("start", request) == {"id": "sb-1"}
    options = created[0].options
    assert options["gpu"] == "T4" and options["runtime"] == "gvisor"
    assert options["block_network"] is True
    assert options["timeout"] == 300 + modal_api.COLLECT_SECONDS
    assert "torch==2.8.0" in options["image"]
    assert created[0].written["/forge.zip"] == b"bundle"


def test_modal_stop_confirms_every_tagged_sandbox_has_ended(monkeypatch):
    sandbox = FakeSandbox()
    monkeypatch.setattr(modal.App, "lookup", lambda *_args, **_kwargs: SimpleNamespace(app_id="a"))
    monkeypatch.setattr(modal.Sandbox, "list", lambda **_kwargs: [sandbox])
    assert modal_api.Client().invoke("stop", {"handle": "forge-run-1"}) == {"stopped": True}
    assert sandbox.terminated
