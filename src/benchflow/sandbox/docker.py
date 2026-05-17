"""BenchFlow-native Docker sandbox provider."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from benchflow.env import resolve_env_vars
from benchflow.paths import EnvironmentPaths, TrialPaths
from benchflow.sandbox.protocol import ExecResult
from benchflow.task import EnvironmentConfig

_COMPOSE_DIR = Path(__file__).parent
COMPOSE_BASE_PATH = _COMPOSE_DIR / "docker-compose-base.yaml"
COMPOSE_BUILD_PATH = _COMPOSE_DIR / "docker-compose-build.yaml"
COMPOSE_PREBUILT_PATH = _COMPOSE_DIR / "docker-compose-prebuilt.yaml"
COMPOSE_NO_NETWORK_PATH = _COMPOSE_DIR / "docker-compose-no-network.yaml"
_DIND_PATH_REWRITE: tuple[str, str] | None = None


def set_dind_path_rewrite(host_source: str, container_dest: str) -> None:
    """Translate host bind paths when host Docker is reached from inside Docker."""
    global _DIND_PATH_REWRITE
    _DIND_PATH_REWRITE = (host_source, container_dest)


def _sanitize_docker_image_name(name: str) -> str:
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9._-]", "-", name)


def _sanitize_docker_compose_project_name(name: str) -> str:
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9_-]", "-", name)


@dataclass
class DockerEnvironmentEnvVars:
    main_image_name: str
    context_dir: str
    host_verifier_logs_path: str
    host_agent_logs_path: str
    host_artifacts_path: str
    env_verifier_logs_path: str
    env_agent_logs_path: str
    env_artifacts_path: str
    prebuilt_image_name: str | None = None
    cpus: int = 1
    memory: str = "1G"

    def to_env_dict(self, include_os_env: bool = True) -> dict[str, str]:
        env_dict = dict(os.environ) if include_os_env else {}
        fields = {
            "MAIN_IMAGE_NAME": self.main_image_name,
            "CONTEXT_DIR": self.context_dir,
            "HOST_VERIFIER_LOGS_PATH": self.host_verifier_logs_path,
            "HOST_AGENT_LOGS_PATH": self.host_agent_logs_path,
            "HOST_ARTIFACTS_PATH": self.host_artifacts_path,
            "ENV_VERIFIER_LOGS_PATH": self.env_verifier_logs_path,
            "ENV_AGENT_LOGS_PATH": self.env_agent_logs_path,
            "ENV_ARTIFACTS_PATH": self.env_artifacts_path,
            "PREBUILT_IMAGE_NAME": self.prebuilt_image_name,
            "CPUS": str(self.cpus),
            "MEMORY": self.memory,
        }
        env_dict.update({k: v for k, v in fields.items() if v is not None})
        if _DIND_PATH_REWRITE:
            host_source, container_dest = _DIND_PATH_REWRITE
            for key in (
                "HOST_VERIFIER_LOGS_PATH",
                "HOST_AGENT_LOGS_PATH",
                "HOST_ARTIFACTS_PATH",
            ):
                val = env_dict.get(key, "")
                if val.startswith(container_dest):
                    env_dict[key] = host_source + val[len(container_dest) :]
        return env_dict


class DockerEnvironment:
    """Run BenchFlow tasks in local Docker Compose."""

    _image_build_locks: ClassVar[dict[str, asyncio.Lock]] = {}

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *,
        keep_containers: bool = False,
        persistent_env: dict[str, str] | None = None,
    ) -> None:
        self.environment_dir = Path(environment_dir)
        self.environment_name = environment_name
        self.session_id = session_id
        self.trial_paths = trial_paths
        self.task_env_config = task_env_config
        self.default_user: str | int | None = None
        self._keep_containers = keep_containers
        self._persistent_env = persistent_env or {}
        self._use_prebuilt = False

        self._validate_definition()
        self._validate_resource_config()

        self._env_vars = DockerEnvironmentEnvVars(
            main_image_name=_sanitize_docker_image_name(f"bf__{environment_name}"),
            context_dir=str(self.environment_dir.resolve().absolute()),
            host_verifier_logs_path=str(trial_paths.verifier_dir.resolve().absolute()),
            host_agent_logs_path=str(trial_paths.agent_dir.resolve().absolute()),
            host_artifacts_path=str(trial_paths.artifacts_dir.resolve().absolute()),
            env_verifier_logs_path=str(EnvironmentPaths.verifier_dir),
            env_agent_logs_path=str(EnvironmentPaths.agent_dir),
            env_artifacts_path=str(EnvironmentPaths.artifacts_dir),
            prebuilt_image_name=task_env_config.docker_image,
            cpus=task_env_config.cpus,
            memory=f"{task_env_config.memory_mb}M",
        )
        self._compose_task_env: dict[str, str] = {}
        if task_env_config.env and self._uses_compose:
            self._compose_task_env = resolve_env_vars(task_env_config.env)

    @classmethod
    def preflight(cls) -> None:
        if not shutil.which("docker"):
            raise SystemExit(
                "Docker is not installed or not on PATH. Install Docker and try again."
            )
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise SystemExit("Docker daemon is not running.") from e

    @property
    def _dockerfile_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    def _environment_docker_compose_path(self) -> Path:
        return self.environment_dir / "docker-compose.yaml"

    @property
    def _uses_compose(self) -> bool:
        return self._environment_docker_compose_path.exists()

    @property
    def _docker_compose_paths(self) -> list[Path]:
        build_or_prebuilt = (
            COMPOSE_PREBUILT_PATH if self._use_prebuilt else COMPOSE_BUILD_PATH
        )
        if self._environment_docker_compose_path.exists():
            paths = [
                COMPOSE_BASE_PATH,
                build_or_prebuilt,
                self._environment_docker_compose_path,
            ]
        else:
            paths = [COMPOSE_BASE_PATH, build_or_prebuilt]
        if not self.task_env_config.allow_internet:
            paths.append(COMPOSE_NO_NETWORK_PATH)
        return paths

    @property
    def is_mounted(self) -> bool:
        return True

    @property
    def host(self) -> str:
        return "localhost"

    @property
    def expose_ports(self) -> list[int]:
        return []

    def _validate_definition(self) -> None:
        if (
            not self._dockerfile_path.exists()
            and not self._environment_docker_compose_path.exists()
        ):
            raise FileNotFoundError(
                f"{self._dockerfile_path} and {self._environment_docker_compose_path} "
                "not found. Ensure the task has environment/Dockerfile or docker-compose.yaml."
            )

    def _validate_resource_config(self) -> None:
        if self.task_env_config.gpus > 0:
            raise RuntimeError("Docker sandbox GPU allocation is not configured yet.")

    def _resolve_user(self, user: str | int | None) -> str | int | None:
        return user if user is not None else self.default_user

    def _merge_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        if not self._persistent_env and not env:
            return None
        merged = {**self._persistent_env}
        if env:
            merged.update(env)
        return merged or None

    async def _run_docker_compose_command(
        self, command: list[str], check: bool = True, timeout_sec: int | None = None
    ) -> ExecResult:
        full_command = [
            "docker",
            "compose",
            "--project-name",
            _sanitize_docker_compose_project_name(self.session_id),
            "--project-directory",
            str(self.environment_dir.resolve().absolute()),
        ]
        for path in self._docker_compose_paths:
            full_command.extend(["-f", str(path.resolve().absolute())])
        full_command.extend(command)

        env = self._env_vars.to_env_dict(include_os_env=True)
        if self._compose_task_env:
            env.update(self._compose_task_env)
        if self._persistent_env:
            env.update(self._persistent_env)

        process = await asyncio.create_subprocess_exec(
            *full_command,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            if timeout_sec:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_sec
                )
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except TimeoutError as e:
            process.terminate()
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=5
                )
            except TimeoutError:
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds") from e

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
        result = ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=process.returncode or 0,
        )
        if check and result.return_code != 0:
            raise RuntimeError(
                f"Docker compose command failed for {self.environment_name}. "
                f"Command: {' '.join(full_command)}. Return code: {result.return_code}. "
                f"Output: {result.stdout or result.stderr}"
            )
        return result

    async def start(self, force_build: bool = False) -> None:
        self._use_prebuilt = bool(not force_build and self.task_env_config.docker_image)
        if not self._use_prebuilt:
            lock = self._image_build_locks.setdefault(
                self.environment_name, asyncio.Lock()
            )
            async with lock:
                await self._run_docker_compose_command(["build"])
        with contextlib.suppress(RuntimeError):
            await self._run_docker_compose_command(["down", "--remove-orphans"])
        await self._run_docker_compose_command(["up", "--detach", "--wait"])
        await self.exec(
            f"chmod 777 {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir}"
        )

    async def stop(self, delete: bool = True) -> None:
        with contextlib.suppress(Exception):
            await self._chown_to_host_user(str(EnvironmentPaths.logs_dir), recursive=True)
        if self._keep_containers:
            with contextlib.suppress(Exception):
                await self._run_docker_compose_command(["stop"])
            return
        if delete:
            with contextlib.suppress(Exception):
                await self._run_docker_compose_command(
                    ["down", "--rmi", "all", "--volumes", "--remove-orphans"]
                )
        else:
            with contextlib.suppress(Exception):
                await self._run_docker_compose_command(["down"])

    async def upload_file(
        self,
        source_path: Path | str | None = None,
        target_path: str | None = None,
        **kwargs,
    ) -> None:
        source_path = source_path if source_path is not None else kwargs["src"]
        target_path = target_path if target_path is not None else kwargs["dst"]
        await self._run_docker_compose_command(
            ["cp", str(source_path), f"main:{target_path}"],
            check=True,
        )

    async def upload_dir(
        self,
        source_dir: Path | str | None = None,
        target_dir: str | None = None,
        **kwargs,
    ) -> None:
        source_dir = source_dir if source_dir is not None else kwargs["src"]
        target_dir = target_dir if target_dir is not None else kwargs["dst"]
        await self._run_docker_compose_command(
            ["cp", f"{source_dir}/.", f"main:{target_dir}"],
            check=True,
        )
        if sys.platform == "win32":
            await self._run_docker_compose_command(
                [
                    "exec",
                    "main",
                    "bash",
                    "-c",
                    f"find {target_dir} -type f \\( -name '*.sh' -o -name '*.py' \\) "
                    "-exec sed -i 's/\\r$//' {} \\;",
                ],
                check=False,
            )

    async def _chown_to_host_user(self, path: str, recursive: bool = False) -> None:
        if not hasattr(os, "getuid"):
            return
        flag = "-R " if recursive else ""
        await self.exec(
            f"chown {flag}{os.getuid()}:{os.getgid()} {shlex.quote(path)}",
            user="root",
        )

    async def download_file(
        self,
        source_path: str | None = None,
        target_path: Path | str | None = None,
        **kwargs,
    ) -> None:
        source_path = source_path if source_path is not None else kwargs["src"]
        target_path = target_path if target_path is not None else kwargs["dst"]
        await self._chown_to_host_user(source_path)
        await self._run_docker_compose_command(
            ["cp", f"main:{source_path}", str(target_path)],
            check=True,
        )

    async def download_dir(
        self,
        source_dir: str | None = None,
        target_dir: Path | str | None = None,
        **kwargs,
    ) -> None:
        source_dir = source_dir if source_dir is not None else kwargs["src"]
        target_dir = target_dir if target_dir is not None else kwargs["dst"]
        await self._chown_to_host_user(source_dir, recursive=True)
        await self._run_docker_compose_command(
            ["cp", f"main:{source_dir}/.", str(target_dir)],
            check=True,
        )

    async def read_file(self, path: str) -> bytes:
        result = await self.exec(f"cat {shlex.quote(path)}", timeout_sec=30)
        if result.return_code != 0:
            raise FileNotFoundError(
                f"read_file failed (rc={result.return_code}): {result.stderr or result.stdout}"
            )
        return result.stdout.encode()

    async def write_file(self, path: str, content: bytes) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name
        try:
            await self.upload_file(tmp_path, path)
        finally:
            os.unlink(tmp_path)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        user = self._resolve_user(user)
        env = self._merge_env(env)
        exec_command = ["exec"]
        if cwd:
            exec_command.extend(["-w", cwd])
        if env:
            for key, value in env.items():
                exec_command.extend(["-e", f"{key}={value}"])
        if user is not None:
            exec_command.extend(["-u", str(user)])
        exec_command.append("main")
        exec_command.extend(["bash", "-c", command])
        return await self._run_docker_compose_command(
            exec_command, check=False, timeout_sec=timeout_sec
        )


class DockerSandbox:
    """Generic adapter for already-created sandbox-like objects.

    Kept for callers that wrap custom Docker environments against the Sandbox
    protocol. Core rollout code uses :class:`DockerEnvironment` directly.
    """

    def __init__(self, inner, *, expose_ports: list[int] | None = None) -> None:
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
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name
        try:
            await self._inner.upload_file(tmp_path, path)
        finally:
            os.unlink(tmp_path)

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
