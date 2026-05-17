"""Daytona sandbox support."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import tarfile
import tempfile
from math import ceil
from pathlib import Path
from typing import Any

from benchflow.env import resolve_env_vars
from benchflow.paths import EnvironmentPaths, TrialPaths
from benchflow.sandbox.protocol import ExecResult
from benchflow.task import EnvironmentConfig


def _resource_gib(value_mb: int) -> int:
    return max(1, ceil(value_mb / 1024))


def _sanitize_daytona_name(name: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name.lower())
    return clean.strip("-_") or "benchflow"


class _AsyncDaytonaSandboxFacade:
    """Async facade over the sync Daytona sandbox SDK object.

    BenchFlow's live ACP transport expects async Daytona methods because the
    earlier compatibility path used Harbor's async wrapper. The Daytona SDK API
    is sync, so this facade preserves the existing transport contract without
    bringing Harbor back into the execution path.
    """

    def __init__(self, sandbox: Any) -> None:
        self._sandbox = sandbox

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sandbox, name)

    async def create_ssh_access(self, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(
            self._sandbox.create_ssh_access, *args, **kwargs
        )


class DaytonaEnvironment:
    """BenchFlow-native Daytona environment.

    Uses Daytona's Python SDK directly. Docker remains the core local path;
    Daytona is an optional cloud adapter installed with ``benchflow[daytona]``.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
    ) -> None:
        self.environment_dir = Path(environment_dir)
        self.environment_name = environment_name
        self.session_id = session_id
        self.trial_paths = trial_paths
        self.task_env_config = task_env_config
        self.default_user: str | int | None = None
        self._daytona = None
        self._raw_sandbox = None
        self._sandbox: _AsyncDaytonaSandboxFacade | None = None
        self._environment_definition_path = self.environment_dir / "Dockerfile"

        if not self._environment_definition_path.exists() and not task_env_config.docker_image:
            raise FileNotFoundError(
                f"{self._environment_definition_path} not found. Daytona requires "
                "environment/Dockerfile or environment.docker_image."
            )

    @classmethod
    def preflight(cls) -> None:
        try:
            import daytona  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "Daytona sandbox support requires the optional 'daytona' extra. "
                "Install with: uv tool install 'benchflow[daytona]'"
            ) from e

    def _resources(self) -> Any:
        from daytona import Resources

        return Resources(
            cpu=self.task_env_config.cpus,
            memory=_resource_gib(self.task_env_config.memory_mb),
            disk=_resource_gib(self.task_env_config.storage_mb),
            gpu=self.task_env_config.gpus or None,
        )

    def _image(self, force_build: bool) -> Any:
        if self.task_env_config.docker_image and not force_build:
            return self.task_env_config.docker_image

        from daytona import Image

        return Image.from_dockerfile(self._environment_definition_path)

    def _create_params(self, force_build: bool) -> Any:
        from daytona import CreateSandboxFromImageParams

        env_vars = (
            resolve_env_vars(self.task_env_config.env)
            if self.task_env_config.env
            else None
        )
        return CreateSandboxFromImageParams(
            name=_sanitize_daytona_name(self.session_id),
            image=self._image(force_build),
            language="python",
            env_vars=env_vars,
            resources=self._resources(),
            labels={"benchflow": "true", "benchflow_session": self.session_id},
            network_block_all=not self.task_env_config.allow_internet,
            ephemeral=True,
        )

    async def start(self, force_build: bool = False) -> None:
        from daytona import Daytona

        from benchflow._daytona_patches import apply as _apply_daytona_patches

        _apply_daytona_patches()
        self._daytona = Daytona()
        params = self._create_params(force_build)
        self._raw_sandbox = await asyncio.to_thread(
            self._daytona.create,
            params,
            timeout=float(self.task_env_config.build_timeout_sec),
        )
        self._sandbox = _AsyncDaytonaSandboxFacade(self._raw_sandbox)
        await self.exec(
            "mkdir -p "
            f"{EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir} "
            f"{EnvironmentPaths.artifacts_dir} && "
            f"chmod 777 {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir}",
            user="root",
            timeout_sec=30,
        )

    async def stop(self, delete: bool = True) -> None:
        if not self._raw_sandbox:
            return
        with contextlib.suppress(Exception):
            if delete:
                await asyncio.to_thread(self._raw_sandbox.delete, timeout=60)
            else:
                await asyncio.to_thread(self._raw_sandbox.stop, timeout=60)
        self._raw_sandbox = None
        self._sandbox = None

    def _command_for_user(self, command: str, user: str | int | None) -> str:
        effective_user = self.default_user if user is None else user
        if effective_user is None or str(effective_user) == "root":
            return command
        return (
            f"su -s /bin/bash {shlex.quote(str(effective_user))} "
            f"-c {shlex.quote(command)}"
        )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if not self._raw_sandbox:
            raise RuntimeError("Daytona sandbox not started")
        wrapped = self._command_for_user(command, user)
        response = await asyncio.to_thread(
            self._raw_sandbox.process.exec,
            wrapped,
            cwd=cwd,
            env=env,
            timeout=timeout_sec,
        )
        return ExecResult(
            return_code=int(response.exit_code or 0),
            stdout=response.result or "",
            stderr="",
        )

    async def upload_file(
        self,
        source_path: Path | str | None = None,
        target_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not self._raw_sandbox:
            raise RuntimeError("Daytona sandbox not started")
        source_path = source_path if source_path is not None else kwargs["src"]
        target_path = target_path if target_path is not None else kwargs["dst"]
        await self.exec(f"mkdir -p {shlex.quote(str(Path(target_path).parent))}")
        await asyncio.to_thread(
            self._raw_sandbox.fs.upload_file, str(source_path), target_path
        )

    async def upload_dir(
        self,
        source_dir: Path | str | None = None,
        target_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        source_dir = source_dir if source_dir is not None else kwargs["src"]
        target_dir = target_dir if target_dir is not None else kwargs["dst"]
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            archive_path = Path(tmp.name)
        try:
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(source_dir, arcname=".")
            remote_archive = f"/tmp/benchflow-upload-{os.getpid()}-{archive_path.name}"
            await self.upload_file(archive_path, remote_archive)
            await self.exec(
                f"mkdir -p {shlex.quote(target_dir)} && "
                f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(target_dir)} && "
                f"rm -f {shlex.quote(remote_archive)}",
                user="root",
                timeout_sec=120,
            )
        finally:
            with contextlib.suppress(FileNotFoundError):
                archive_path.unlink()

    async def download_file(
        self,
        source_path: str | None = None,
        target_path: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        if not self._raw_sandbox:
            raise RuntimeError("Daytona sandbox not started")
        source_path = source_path if source_path is not None else kwargs["src"]
        target_path = target_path if target_path is not None else kwargs["dst"]
        data = await asyncio.to_thread(self._raw_sandbox.fs.download_file, source_path)
        if data is None:
            raise FileNotFoundError(source_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def read_file(self, path: str) -> bytes:
        result = await self.exec(f"cat {shlex.quote(path)}", timeout_sec=30)
        if result.return_code != 0:
            raise FileNotFoundError(
                f"read_file failed (rc={result.return_code}): {result.stderr or result.stdout}"
            )
        return result.stdout.encode()

    async def write_file(self, path: str, content: bytes) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name
        try:
            await self.upload_file(tmp_path, path)
        finally:
            os.unlink(tmp_path)


class DaytonaSandbox:
    """Adapts a Daytona-like environment object to the Sandbox protocol."""

    def __init__(
        self, inner, *, expose_ports: list[int] | None = None
    ) -> None:
        self._inner = inner
        self._expose_ports = expose_ports or []

    async def exec(
        self, cmd: str, *, user: str = "root", timeout_sec: int = 30
    ) -> ExecResult:
        result = await self._inner.exec(cmd, user=user, timeout_sec=timeout_sec)
        return ExecResult(
            return_code=result.return_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    async def read_file(self, path: str) -> bytes:
        result = await self._inner.exec(f"cat {shlex.quote(path)}", timeout_sec=30)
        if result.return_code != 0:
            raise FileNotFoundError(
                f"read_file failed (rc={result.return_code}): {result.stderr or ''}"
            )
        return (result.stdout or "").encode()

    async def write_file(self, path: str, content: bytes) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            try:
                await self._inner.upload_file(tmp.name, path)
            finally:
                os.unlink(tmp.name)

    async def upload_file(self, src: Path, dst: str) -> None:
        await self._inner.upload_file(src, dst)

    async def upload_dir(self, src: Path, dst: str) -> None:
        await self._inner.upload_dir(src, dst)

    async def download_file(self, src: str, dst: Path) -> None:
        await self._inner.download_file(src, dst)

    async def start(self) -> None:
        await self._inner.start(force_build=False)

    async def stop(self, *, delete: bool = True) -> None:
        await self._inner.stop(delete=delete)

    @property
    def host(self) -> str:
        return "localhost"

    @property
    def expose_ports(self) -> list[int]:
        return list(self._expose_ports)
