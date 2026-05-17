"""Parsers for agent trace formats.

Supported formats:

* **Claude Code JSONL** — ``~/.claude/projects/<encoded-dir>/<session>.jsonl``
* **opentraces JSONL** — ``TraceRecord`` schema (v0.1–v0.3)

Both are normalized into :class:`~benchflow.traces.models.ParsedTrace`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from benchflow.traces.models import GitContext, ParsedTrace, ToolCall, TraceStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Claude Code JSONL parser
# ---------------------------------------------------------------------------


def _extract_content_text(content: object) -> str:
    """Extract plain text from Claude Code message content.

    Content can be a string, a list of content blocks, or missing.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_dict = cast(dict[str, Any], block)
                if block_dict.get("type") == "text":
                    parts.append(str(block_dict.get("text", "")))
                elif block_dict.get("type") == "tool_use":
                    pass  # handled separately
                elif block_dict.get("type") == "tool_result":
                    parts.append(str(block_dict.get("content", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _extract_tool_calls(content: object) -> list[ToolCall]:
    """Extract tool_use blocks from Claude Code assistant message content."""
    calls: list[ToolCall] = []
    if not isinstance(content, list):
        return calls
    for block in content:
        if not isinstance(block, dict):
            continue
        block_dict = cast(dict[str, Any], block)
        if block_dict.get("type") == "tool_use":
            input_value = block_dict.get("input", {})
            calls.append(
                ToolCall(
                    name=str(block_dict.get("name", "unknown")),
                    input=input_value if isinstance(input_value, dict) else {},
                )
            )
    return calls


def parse_claude_code_session(
    path: Path,
    *,
    session_id: str | None = None,
) -> ParsedTrace:
    """Parse a Claude Code JSONL session file into a ``ParsedTrace``.

    Assumes the file contains entries for a single session. For files
    with multiple sessions interleaved, use :func:`parse_claude_code_file`.

    Args:
        path: Path to a ``.jsonl`` file from ``~/.claude/projects/``.
        session_id: Override session ID (default: derived from filename).

    Returns:
        Normalized trace ready for task generation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {path}")

    entries: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            logger.debug("Skipping malformed JSONL line in %s", path)
            continue

    if not entries:
        raise ValueError(f"No valid entries in {path}")

    return _parse_claude_entries(entries, session_id=session_id, source_path=path)


def _parse_claude_entries(
    entries: list[dict[str, object]],
    *,
    session_id: str | None = None,
    source_path: Path | None = None,
) -> ParsedTrace:
    """Parse a list of Claude Code JSONL entries into a ``ParsedTrace``.

    Shared implementation used by both :func:`parse_claude_code_session`
    (single file) and :func:`parse_claude_code_file` (multi-session split).
    """
    sid = session_id or str(entries[0].get("sessionId", "unknown"))
    source_name = source_path.name if source_path else sid
    cwd = None
    git_branch = None
    started_at = None
    ended_at = None
    model = None
    steps: list[TraceStep] = []

    for entry in entries:
        entry_type = entry.get("type")
        timestamp = entry.get("timestamp")
        ts_str = str(timestamp) if timestamp else None

        # Track time bounds
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if started_at is None or dt < started_at:
                    started_at = dt
                if ended_at is None or dt > ended_at:
                    ended_at = dt
            except ValueError:
                pass

        # Extract context from first entry
        if cwd is None and "cwd" in entry:
            cwd = str(entry["cwd"])
        if git_branch is None and entry.get("gitBranch"):
            git_branch = str(entry["gitBranch"])

        if entry_type == "user":
            msg = entry.get("message", {})
            if isinstance(msg, dict):
                msg_dict = cast(dict[str, Any], msg)
                content = _extract_content_text(msg_dict.get("content", ""))
                role = str(msg_dict.get("role", "user"))
            else:
                content = str(msg) if msg else ""
                role = "user"
            if content.strip():
                steps.append(TraceStep(role=role, content=content, timestamp=ts_str))

        elif entry_type == "assistant":
            msg = entry.get("message", {})
            if isinstance(msg, dict):
                msg_dict = cast(dict[str, Any], msg)
                raw_content = msg_dict.get("content", "")
                content = _extract_content_text(raw_content)
                tool_calls = _extract_tool_calls(raw_content)
                # Try to get model
                if not model and msg_dict.get("model"):
                    model = str(msg_dict["model"])
            else:
                content = str(msg) if msg else ""
                tool_calls = []
            steps.append(
                TraceStep(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                    timestamp=ts_str,
                )
            )

    # Infer outcome from last assistant message
    outcome = "unknown"
    for step in reversed(steps):
        if step.role == "assistant":
            lower = step.content.lower()
            if any(
                w in lower
                for w in (
                    "complete",
                    "done",
                    "finished",
                    "success",
                    "fixed",
                    "created",
                    "built",
                    "updated",
                    "refactored",
                    "implemented",
                    "added",
                )
            ):
                outcome = "success"
            elif any(w in lower for w in ("error", "failed", "cannot")):
                outcome = "failure"
            break

    total_input = 0
    total_output = 0
    for entry in entries:
        if entry.get("type") == "assistant":
            msg = entry.get("message", {})
            if isinstance(msg, dict):
                msg_dict = cast(dict[str, Any], msg)
                usage = msg_dict.get("usage", {})
                if isinstance(usage, dict):
                    usage_dict = cast(dict[str, Any], usage)
                    total_input += int(usage_dict.get("input_tokens", 0))
                    total_output += int(usage_dict.get("output_tokens", 0))

    deterministic_id = hashlib.sha256(f"{sid}:{source_name}".encode()).hexdigest()[:16]

    return ParsedTrace(
        trace_id=deterministic_id,
        session_id=sid,
        agent_name="claude-code",
        model=model,
        started_at=started_at,
        ended_at=ended_at,
        steps=steps,
        git=GitContext(branch=git_branch),
        cwd=cwd,
        outcome=outcome,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )


# ---------------------------------------------------------------------------
# opentraces JSONL parser
# ---------------------------------------------------------------------------


def parse_claude_code_file(path: Path) -> list[ParsedTrace]:
    """Parse a Claude Code JSONL file that may contain multiple sessions.

    Groups entries by ``sessionId`` and parses each group as a separate
    trace. Falls back to treating the whole file as one session if no
    ``sessionId`` field is present.

    Args:
        path: Path to a ``.jsonl`` file containing Claude Code entries.

    Returns:
        List of parsed traces, one per unique session.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {path}")

    entries: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            logger.debug("Skipping malformed JSONL line in %s", path)
            continue

    if not entries:
        raise ValueError(f"No valid entries in {path}")

    # Group entries by sessionId
    session_groups: dict[str, list[dict[str, object]]] = {}
    no_session_id = []
    for entry in entries:
        sid = entry.get("sessionId")
        if isinstance(sid, str) and sid:
            session_groups.setdefault(sid, []).append(entry)
        else:
            no_session_id.append(entry)

    # If no sessionId found at all, treat as single session
    if not session_groups:
        return [parse_claude_code_session(path)]

    # Append ungrouped entries to the first session (context lines, etc.)
    if no_session_id and session_groups:
        first_key = next(iter(session_groups))
        session_groups[first_key] = no_session_id + session_groups[first_key]

    traces: list[ParsedTrace] = []
    for sid, group_entries in session_groups.items():
        traces.append(
            _parse_claude_entries(group_entries, session_id=sid, source_path=path)
        )
    return traces


def parse_opentraces_record(
    record: dict[str, object],
) -> ParsedTrace:
    """Parse a single opentraces ``TraceRecord`` dict into a ``ParsedTrace``.

    The opentraces schema (v0.1–v0.3) uses a TAO-loop step model with
    ``thought``, ``action`` (tool_call), and ``observation`` fields.

    Args:
        record: A dict parsed from one JSONL line of an opentraces dataset.

    Returns:
        Normalized trace ready for task generation.
    """
    trace_id = str(record.get("trace_id", uuid.uuid4()))
    session_id = str(record.get("session_id", trace_id))

    # Agent info
    agent_info = record.get("agent", {})
    if isinstance(agent_info, dict):
        agent_dict = cast(dict[str, Any], agent_info)
        agent_name = str(agent_dict.get("name", "unknown"))
        agent_version = agent_dict.get("version")
    else:
        agent_name = str(agent_info) if agent_info else "unknown"
        agent_version = None

    # Environment info
    env_info = record.get("environment", {})
    cwd = None
    git_branch = None
    git_repo = None
    if isinstance(env_info, dict):
        env_dict = cast(dict[str, Any], env_info)
        cwd = str(env_dict.get("cwd", "")) or None
        vcs = env_dict.get("vcs", {})
        if isinstance(vcs, dict):
            vcs_dict = cast(dict[str, Any], vcs)
            git_branch = str(vcs_dict.get("branch", "")) or None
            git_repo = str(vcs_dict.get("remote", "")) or None

    # Task info
    task_info = record.get("task", {})
    tags: list[str] = []
    if isinstance(task_info, dict):
        task_dict = cast(dict[str, Any], task_info)
        task_tags = task_dict.get("tags")
        if isinstance(task_tags, list):
            tags = [str(t) for t in task_tags]

    # Timestamps
    started_at = _parse_iso(record.get("timestamp_start"))
    ended_at = _parse_iso(record.get("timestamp_end"))

    # Steps (TAO loop)
    raw_steps = record.get("steps", [])
    steps: list[TraceStep] = []
    if isinstance(raw_steps, list):
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                continue
            step_dict = cast(dict[str, Any], raw_step)

            # Thought → user-like prompt or assistant reasoning
            thought = step_dict.get("thought")
            if thought and isinstance(thought, str):
                steps.append(TraceStep(role="assistant", content=thought))

            # Action → tool call
            action = step_dict.get("action")
            tool_calls: list[ToolCall] = []
            if isinstance(action, dict):
                action_dict = cast(dict[str, Any], action)
                tc_data = action_dict.get("tool_call", action_dict)
                if isinstance(tc_data, dict):
                    tc_dict = cast(dict[str, Any], tc_data)
                    input_value = tc_dict.get("input", tc_dict.get("arguments", {}))
                    tool_calls.append(
                        ToolCall(
                            name=str(tc_dict.get("name", tc_dict.get("tool", "unknown"))),
                            input=input_value if isinstance(input_value, dict) else {},
                        )
                    )

            # Observation → tool output
            observation = step_dict.get("observation")
            obs_text = ""
            if isinstance(observation, dict):
                observation_dict = cast(dict[str, Any], observation)
                obs_text = str(
                    observation_dict.get("content", observation_dict.get("output", ""))
                )
            elif isinstance(observation, str):
                obs_text = observation

            if tool_calls or obs_text:
                steps.append(
                    TraceStep(
                        role="assistant",
                        content=obs_text,
                        tool_calls=tool_calls,
                    )
                )

    # Outcome
    outcome_info = record.get("outcome", {})
    outcome = "unknown"
    if isinstance(outcome_info, dict):
        outcome_dict = cast(dict[str, Any], outcome_info)
        status = str(outcome_dict.get("status", ""))
        if status in ("success", "completed"):
            outcome = "success"
        elif status in ("failure", "error", "failed"):
            outcome = "failure"

    # Metrics
    metrics_info = record.get("metrics", {})
    total_input = 0
    total_output = 0
    if isinstance(metrics_info, dict):
        metrics_dict = cast(dict[str, Any], metrics_info)
        tokens = metrics_dict.get("tokens", {})
        if isinstance(tokens, dict):
            tokens_dict = cast(dict[str, Any], tokens)
            total_input = int(tokens_dict.get("input", 0))
            total_output = int(tokens_dict.get("output", 0))

    metadata: dict[str, object] = {}
    if agent_version:
        metadata["agent_version"] = agent_version
    schema_version = record.get("schema_version")
    if schema_version:
        metadata["opentraces_schema_version"] = str(schema_version)

    return ParsedTrace(
        trace_id=trace_id,
        session_id=session_id,
        agent_name=agent_name,
        model=None,
        started_at=started_at,
        ended_at=ended_at,
        steps=steps,
        git=GitContext(repo=git_repo, branch=git_branch),
        cwd=cwd,
        outcome=outcome,
        tags=tags,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        metadata=metadata,
    )


def parse_opentraces_file(path: Path) -> list[ParsedTrace]:
    """Parse all records from an opentraces JSONL file.

    Args:
        path: Path to a ``.jsonl`` file in opentraces format.

    Returns:
        List of parsed traces.
    """
    path = Path(path)
    traces: list[ParsedTrace] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Skipping malformed line in %s", path)
            continue
        if isinstance(record, dict):
            traces.append(parse_opentraces_record(record))
    return traces


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_iso(value: object) -> datetime | None:
    """Parse an ISO-8601 string, returning None on failure."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
