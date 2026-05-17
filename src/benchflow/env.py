"""Environment variable helpers for BenchFlow task configs."""

from __future__ import annotations

import os
import re

_TEMPLATE_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def resolve_env_vars(env_dict: dict[str, str]) -> dict[str, str]:
    """Resolve ``${VAR}`` and ``${VAR:-default}`` values from the host env."""
    resolved: dict[str, str] = {}
    for key, value in env_dict.items():
        match = _TEMPLATE_PATTERN.fullmatch(value)
        if not match:
            resolved[key] = value
            continue
        var_name = match.group(1)
        default = match.group(2)
        if var_name in os.environ:
            resolved[key] = os.environ[var_name]
        elif default is not None:
            resolved[key] = default
        else:
            raise ValueError(
                f"Environment variable {var_name!r} not found in host environment"
            )
    return resolved


def get_required_host_vars(env_dict: dict[str, str]) -> list[tuple[str, str | None]]:
    """Return host env references used by templated env values."""
    result: list[tuple[str, str | None]] = []
    for value in env_dict.values():
        match = _TEMPLATE_PATTERN.fullmatch(value)
        if match:
            result.append((match.group(1), match.group(2)))
    return result
