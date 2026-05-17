from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import pytest

from benchflow.paths import TrialPaths
from benchflow.sandbox.daytona import DaytonaEnvironment
from benchflow.task import EnvironmentConfig


class FakeResources:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeImage:
    @staticmethod
    def from_dockerfile(path):
        return {"dockerfile": Path(path)}


class FakeParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeProcess:
    def __init__(self):
        self.calls = []

    def exec(self, command, cwd=None, env=None, timeout=None):
        self.calls.append(
            {"command": command, "cwd": cwd, "env": env, "timeout": timeout}
        )
        return SimpleNamespace(exit_code=0, result="ok")


class FakeFs:
    def __init__(self):
        self.uploads = []
        self.files = {"/remote/readme.txt": b"hello"}

    def upload_file(self, src, dst):
        self.uploads.append((src, dst))
        self.files[dst] = Path(src).read_bytes()

    def download_file(self, src):
        return self.files.get(src)


class FakeSandbox:
    def __init__(self):
        self.process = FakeProcess()
        self.fs = FakeFs()
        self.deleted = False
        self.stopped = False

    def create_ssh_access(self):
        return SimpleNamespace(token="ssh-token")

    def delete(self, timeout=60):
        self.deleted = True

    def stop(self, timeout=60):
        self.stopped = True


class FakeDaytona:
    instances: ClassVar[list[FakeDaytona]] = []

    def __init__(self):
        self.created = []
        self.sandbox = FakeSandbox()
        FakeDaytona.instances.append(self)

    def create(self, params, timeout=60):
        self.created.append((params, timeout))
        return self.sandbox


def _make_env(tmp_path: Path, config: EnvironmentConfig | None = None):
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    return DaytonaEnvironment(
        environment_dir=env_dir,
        environment_name="task",
        session_id="Trial.One",
        trial_paths=trial_paths,
        task_env_config=config or EnvironmentConfig(),
    )


@pytest.mark.asyncio
async def test_daytona_environment_uses_daytona_sdk_without_harbor(tmp_path, monkeypatch):
    """Native Daytona creation should use Daytona SDK params directly."""
    import daytona

    FakeDaytona.instances.clear()
    monkeypatch.setattr(daytona, "Daytona", FakeDaytona)
    monkeypatch.setattr(daytona, "Image", FakeImage)
    monkeypatch.setattr(daytona, "Resources", FakeResources)
    monkeypatch.setattr(daytona, "CreateSandboxFromImageParams", FakeParams)

    env = _make_env(
        tmp_path,
        EnvironmentConfig(cpus=2, memory_mb=2048, storage_mb=4096, allow_internet=False),
    )
    with patch("benchflow._daytona_patches.apply"):
        await env.start(force_build=True)

    daytona_client = FakeDaytona.instances[0]
    params, timeout = daytona_client.created[0]
    assert timeout == 600.0
    assert params.kwargs["name"] == "trial-one"
    assert params.kwargs["image"] == {"dockerfile": tmp_path / "environment" / "Dockerfile"}
    assert params.kwargs["resources"].kwargs == {
        "cpu": 2,
        "memory": 2,
        "disk": 4,
        "gpu": None,
    }
    assert params.kwargs["network_block_all"] is True
    assert params.kwargs["ephemeral"] is True
    assert daytona_client.sandbox.process.calls[0]["command"].startswith("mkdir -p")


@pytest.mark.asyncio
async def test_daytona_environment_exec_wraps_non_root_user(tmp_path, monkeypatch):
    import daytona

    FakeDaytona.instances.clear()
    monkeypatch.setattr(daytona, "Daytona", FakeDaytona)
    monkeypatch.setattr(daytona, "Image", FakeImage)
    monkeypatch.setattr(daytona, "Resources", FakeResources)
    monkeypatch.setattr(daytona, "CreateSandboxFromImageParams", FakeParams)

    env = _make_env(tmp_path)
    with patch("benchflow._daytona_patches.apply"):
        await env.start()

    result = await env.exec("echo hi", user="agent", cwd="/app", env={"A": "B"})
    assert result.return_code == 0
    assert result.stdout == "ok"
    call = FakeDaytona.instances[0].sandbox.process.calls[-1]
    assert "su -s /bin/bash agent" in call["command"]
    assert call["cwd"] == "/app"
    assert call["env"] == {"A": "B"}


@pytest.mark.asyncio
async def test_daytona_environment_upload_download_and_stop(tmp_path, monkeypatch):
    import daytona

    FakeDaytona.instances.clear()
    monkeypatch.setattr(daytona, "Daytona", FakeDaytona)
    monkeypatch.setattr(daytona, "Image", FakeImage)
    monkeypatch.setattr(daytona, "Resources", FakeResources)
    monkeypatch.setattr(daytona, "CreateSandboxFromImageParams", FakeParams)

    env = _make_env(tmp_path)
    with patch("benchflow._daytona_patches.apply"):
        await env.start()

    local = tmp_path / "local.txt"
    local.write_text("local")
    await env.upload_file(local, "/remote/local.txt")
    out = tmp_path / "out.txt"
    await env.download_file("/remote/local.txt", out)
    assert out.read_text() == "local"

    await env.stop(delete=True)
    assert FakeDaytona.instances[0].sandbox.deleted is True
