"""acpx-backed agent execution.

``acpx`` is a headless ACP client. Unlike ``connect_acp()``, BenchFlow does
not open an ACP stdio transport directly here. It invokes the acpx CLI inside
the sandbox, asks for raw ACP NDJSON, and normalizes those updates into the
same trajectory shape used by native ACP sessions.
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path
from typing import Any

from benchflow._trajectory import _capture_session_trajectory
from benchflow.acp.session import ACPSession
from benchflow.agents.providers import strip_provider_prefix

logger = logging.getLogger(__name__)

_ACPX_BIN = "/opt/benchflow/bin/acpx"
_ACPX_PROMPT_DIR = "/tmp/benchflow-acpx"
_DIAG_TRUNCATE = 2000


def _safe_log_stem(agent: str, prompt_idx: int) -> str:
    clean = "".join(c if c.isalnum() else "_" for c in agent).strip("_")
    return f"{clean or 'agent'}_acpx_{prompt_idx}"


def _build_acpx_command(
    *,
    agent_launch: str,
    prompt_path: str,
    model: str | None,
    timeout: int,
    agent_cwd: str,
) -> str:
    """Build the shell command that invokes acpx for one prompt."""
    parts = [
        _ACPX_BIN,
        "--format",
        "json",
        "--json-strict",
        "--cwd",
        agent_cwd,
        "--approve-all",
        "--timeout",
        str(timeout),
    ]
    if model:
        parts.extend(["--model", strip_provider_prefix(model)])
    parts.extend(["--agent", agent_launch, "exec", "--file", prompt_path])
    return " ".join(shlex.quote(part) for part in parts)


def _parse_acpx_ndjson(
    stdout: str,
    *,
    session: ACPSession,
) -> tuple[list[dict[str, Any]], str | None]:
    """Parse acpx raw ACP JSON-RPC NDJSON into trajectory events."""
    agent_name: str | None = None
    for lineno, line in enumerate(stdout.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping non-JSON acpx stdout line %s", lineno)
            continue
        if not isinstance(message, dict):
            continue

        result = message.get("result")
        if isinstance(result, dict):
            session.record_usage(result)
            info = result.get("agentInfo")
            if isinstance(info, dict) and isinstance(info.get("name"), str):
                agent_name = info["name"]

        if message.get("method") != "session/update":
            continue
        params = message.get("params")
        if isinstance(params, dict):
            session.record_usage(params)
            update = params.get("update")
            session.handle_update(update if isinstance(update, dict) else params)

    session.mark_prompt_end()
    return _capture_session_trajectory(session), agent_name


async def execute_prompts_with_acpx(
    env: Any,
    *,
    agent: str,
    agent_launch: str,
    model: str | None,
    prompts: list[str],
    agent_env: dict[str, str],
    sandbox_user: str | None,
    timeout: int,
    trial_dir: Path,
    agent_cwd: str,
) -> tuple[list[dict[str, Any]], int, str | None]:
    """Run prompts through acpx and return (trajectory, n_tool_calls, agent_name)."""
    session = ACPSession("acpx")
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)

    effective_timeout = max(1, int(timeout or 0))
    await env.exec(
        f"mkdir -p {shlex.quote(_ACPX_PROMPT_DIR)} /logs/agent",
        user="root",
        timeout_sec=10,
    )

    agent_name: str | None = None
    trajectory: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts, start=1):
        prompt_path = f"{_ACPX_PROMPT_DIR}/prompt-{idx}.md"
        await env.write_file(prompt_path, prompt.encode())
        if sandbox_user:
            q_user = shlex.quote(sandbox_user)
            q_path = shlex.quote(prompt_path)
            chown = await env.exec(
                f"chown {q_user}:{q_user} {q_path}",
                user="root",
                timeout_sec=10,
            )
            if chown.return_code != 0:
                raise RuntimeError(
                    f"Failed to prepare acpx prompt file {prompt_path}: "
                    f"{chown.stderr or chown.stdout}"
                )

        command = _build_acpx_command(
            agent_launch=agent_launch,
            prompt_path=prompt_path,
            model=model,
            timeout=effective_timeout,
            agent_cwd=agent_cwd,
        )
        log_stem = _safe_log_stem(agent, idx)
        (agent_dir / f"{log_stem}.cmd.txt").write_text(command + "\n")

        logger.info("acpx prompt %s/%s via %s", idx, len(prompts), agent)
        session.record_user_prompt(prompt)
        result = await env.exec(
            command,
            cwd=agent_cwd,
            env=agent_env,
            user=sandbox_user,
            timeout_sec=effective_timeout + 30,
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        (agent_dir / f"{log_stem}.ndjson").write_text(stdout)
        if stderr:
            (agent_dir / f"{log_stem}.stderr.txt").write_text(stderr)

        parsed, parsed_agent_name = _parse_acpx_ndjson(stdout, session=session)
        trajectory = parsed
        agent_name = parsed_agent_name or agent_name

        if result.return_code == 3:
            raise TimeoutError(
                f"acpx agent timed out after {effective_timeout}s "
                f"(log: {agent_dir / f'{log_stem}.ndjson'})"
            )
        if result.return_code != 0:
            details = (stderr or stdout).strip()
            if len(details) > _DIAG_TRUNCATE:
                details = details[:_DIAG_TRUNCATE] + "..."
            raise RuntimeError(
                f"acpx agent failed for {agent} (rc={result.return_code}; "
                f"log: {agent_dir / f'{log_stem}.ndjson'}): {details}"
            )

    return trajectory, len(session.tool_calls), agent_name
