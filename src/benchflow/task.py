"""BenchFlow-native task model.

This module intentionally covers the task/config surface BenchFlow needs at
runtime without importing any external benchmark framework.  Compatibility
loaders may adapt other task formats into these types, but core rollout code
should use this module directly.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

_CANARY_LINE_RE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)
_ORG_NAME_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$"


def strip_canary(text: str) -> str:
    """Strip leading data-provenance canary comments from task instructions."""
    lines = text.split("\n")
    idx = 0
    while idx < len(lines) and _CANARY_LINE_RE.match(lines[idx].strip()):
        idx += 1
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    return "\n".join(lines[idx:])


class TaskPaths:
    """Filesystem layout for a BenchFlow task directory."""

    CONFIG_FILENAME = "task.toml"

    def __init__(self, task_dir: Path | str):
        self.task_dir = Path(task_dir).resolve()

    @property
    def instruction_path(self) -> Path:
        return self.task_dir / "instruction.md"

    @property
    def readme_path(self) -> Path:
        return self.task_dir / "README.md"

    @property
    def gitignore_path(self) -> Path:
        return self.task_dir / ".gitignore"

    @property
    def config_path(self) -> Path:
        return self.task_dir / self.CONFIG_FILENAME

    @property
    def environment_dir(self) -> Path:
        return self.task_dir / "environment"

    @property
    def solution_dir(self) -> Path:
        return self.task_dir / "solution"

    @property
    def solve_path(self) -> Path:
        return self.solution_dir / "solve.sh"

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / "tests"

    @property
    def test_path(self) -> Path:
        return self.tests_dir / "test.sh"

    def is_valid(self, disable_verification: bool = False) -> bool:
        return (
            self.config_path.exists()
            and self.environment_dir.exists()
            and self.instruction_path.exists()
            and (disable_verification or self.test_path.exists())
        )


class Author(BaseModel):
    """Task author metadata."""

    name: str
    email: str | None = None


class PackageInfo(BaseModel):
    """Optional package metadata from ``[task]``."""

    name: str
    description: str = ""
    authors: list[Author] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, value: str) -> str:
        if not re.match(_ORG_NAME_PATTERN, value) or ".." in value:
            raise ValueError(
                "Package name must be in 'org/name' format with alphanumeric "
                f"characters, hyphens, underscores, and dots. Got: {value}"
            )
        return value

    @property
    def org(self) -> str:
        return self.name.split("/")[0]

    @property
    def short_name(self) -> str:
        return self.name.split("/")[1]


class VerifierConfig(BaseModel):
    timeout_sec: float = 600.0
    env: dict[str, str] = Field(default_factory=dict)
    user: str | int | None = None
    pytest_plugins: list[str] = Field(default_factory=list)


class SolutionConfig(BaseModel):
    env: dict[str, str] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    timeout_sec: float | None = None
    user: str | int | None = None


class MCPServerConfig(BaseModel):
    name: str
    transport: str = "sse"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_transport_fields(self) -> MCPServerConfig:
        if self.transport in ("sse", "streamable-http") and not self.url:
            raise ValueError(f"'url' is required for transport {self.transport!r}")
        if self.transport == "stdio" and not self.command:
            raise ValueError("'command' is required for transport 'stdio'")
        return self


class EnvironmentConfig(BaseModel):
    build_timeout_sec: float = 600.0
    docker_image: str | None = None
    cpus: int = 1
    memory_mb: int = 2048
    storage_mb: int = 10240
    gpus: int = 0
    gpu_types: list[str] | None = None
    allow_internet: bool = True
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    skills_dir: str | None = None

    memory: str | None = Field(default=None, exclude=True)
    storage: str | None = Field(default=None, exclude=True)

    @staticmethod
    def _parse_size_to_mb(size_str: str) -> int:
        normalized = size_str.strip().upper()
        if normalized.endswith("G"):
            return int(float(normalized[:-1]) * 1024)
        if normalized.endswith("M"):
            return int(float(normalized[:-1]))
        if normalized.endswith("K"):
            return int(float(normalized[:-1]) / 1024)
        raise ValueError(f"Invalid size format: {size_str}")

    @model_validator(mode="after")
    def handle_deprecated_fields(self) -> EnvironmentConfig:
        if self.memory is not None:
            warnings.warn(
                "The 'memory' field is deprecated. Use 'memory_mb' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.memory_mb = self._parse_size_to_mb(self.memory)
            self.memory = None
        if self.storage is not None:
            warnings.warn(
                "The 'storage' field is deprecated. Use 'storage_mb' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.storage_mb = self._parse_size_to_mb(self.storage)
            self.storage = None
        return self


class TaskConfig(BaseModel):
    schema_version: str = "1.1"
    task: PackageInfo | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    solution: SolutionConfig = Field(default_factory=SolutionConfig)
    source: str | None = None

    @model_validator(mode="before")
    @classmethod
    def handle_version_rename(cls, data: Any) -> Any:
        if isinstance(data, dict) and "version" in data:
            data.setdefault("schema_version", data.pop("version"))
        return data

    @classmethod
    def model_validate_toml(cls, toml_data: str) -> TaskConfig:
        return cls.model_validate(tomllib.loads(toml_data))


class Task:
    """A BenchFlow task directory and its parsed runtime config."""

    def __init__(self, task_dir: Path | str):
        self._task_dir = Path(task_dir).resolve()
        self.paths = TaskPaths(self._task_dir)
        self.instruction = strip_canary(self.paths.instruction_path.read_text())
        self.config = TaskConfig.model_validate_toml(self.paths.config_path.read_text())
        self.name = self.config.task.name if self.config.task else self.paths.task_dir.name

    @property
    def checksum(self) -> str:
        """Generate a deterministic hash over task file names and contents."""
        digest = hashlib.sha256()
        for path in sorted(p for p in self._task_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(self._task_dir).as_posix()
            digest.update(rel.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    @property
    def task_dir(self) -> Path:
        return self._task_dir
