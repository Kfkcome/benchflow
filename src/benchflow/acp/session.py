"""ACP session lifecycle management."""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

from .types import (
    AgentCapabilities,
    AgentInfo,
    StopReason,
    ToolCallStatus,
)

logger = logging.getLogger(__name__)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _int_field(source: dict[str, Any], *names: str) -> int:
    for name in names:
        value = _number(source.get(name))
        if value is not None:
            return int(value)
    return 0


def _cost_field(source: dict[str, Any]) -> float | None:
    for name in ("total_cost_usd", "cost_usd", "estimated_cost_usd"):
        value = _number(source.get(name))
        if value is not None:
            return value
    cost = source.get("cost")
    return _number(cost)


def _usage_payload(payload: dict[str, Any]) -> tuple[dict[str, int], float | None]:
    usage = payload.get("usage") or payload.get("tokenUsage") or payload.get(
        "token_usage"
    )
    source = usage if isinstance(usage, dict) else payload
    normalized = {
        "input_tokens": _int_field(
            source,
            "input_tokens",
            "inputTokens",
            "prompt_tokens",
            "promptTokens",
        ),
        "output_tokens": _int_field(
            source,
            "output_tokens",
            "outputTokens",
            "completion_tokens",
            "completionTokens",
        ),
        "cache_read_tokens": _int_field(
            source,
            "cache_read_tokens",
            "cacheReadTokens",
            "cache_read_input_tokens",
            "cacheRead",
        ),
        "cache_write_tokens": _int_field(
            source,
            "cache_write_tokens",
            "cacheWriteTokens",
            "cache_creation_input_tokens",
            "cacheWrite",
        ),
        "reasoning_tokens": _int_field(
            source,
            "reasoning_tokens",
            "reasoningTokens",
        ),
    }
    total = _int_field(source, "total_tokens", "totalTokens")
    if not total:
        total = sum(normalized.values())
    normalized["total_tokens"] = total
    normalized = {key: value for key, value in normalized.items() if value}
    cost = _cost_field(source)
    if cost is None:
        cost = _cost_field(payload)
    return normalized, cost


class ToolCallRecord:
    """Record of a single tool call within a session.

    Tracks identity (tool_call_id, title, kind), lifecycle status, captured
    content blocks, and wall-clock timing.
    """

    def __init__(self, tool_call_id: str, title: str, kind: str):
        self.tool_call_id = tool_call_id
        self.title = title
        self.kind = kind
        self.status = ToolCallStatus.PENDING
        self.content: list[dict] = []
        self.started_at = datetime.now()
        self.finished_at: datetime | None = None

    def update_status(
        self, status: ToolCallStatus, content: list[dict] | None = None
    ) -> None:
        self.status = status
        if content:
            self.content.extend(content)
        if status in (
            ToolCallStatus.COMPLETED,
            ToolCallStatus.FAILED,
            ToolCallStatus.CANCELLED,
        ):
            self.finished_at = datetime.now()


class ACPSession:
    """Tracks mutable state for one ACP session.

    Accumulates streaming chunks (message_chunks, thought_chunks) and
    tool-call records as session/update notifications arrive.  Use
    ``full_message`` / ``full_thought`` to read the assembled text.

    The ``events`` list records every significant event in chronological
    order (user prompts, tool calls, message/thought boundaries) so that
    ``_capture_session_trajectory`` can produce a faithful interleaved
    trajectory instead of a flat blob.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.agent_info: AgentInfo | None = None
        self.agent_capabilities: AgentCapabilities | None = None
        self.message_chunks: list[str] = []
        self.thought_chunks: list[str] = []
        self.tool_calls: list[ToolCallRecord] = []
        self._tool_call_map: dict[str, ToolCallRecord] = {}
        self.stop_reason: StopReason | None = None
        self.created_at = datetime.now()
        self.events: list[dict] = []
        self._pending_text: list[dict] = []
        self._events_active: bool = False
        self.usage_events: list[dict] = []
        self.token_usage: defaultdict[str, int] = defaultdict(int)
        self.total_cost_usd: float = 0.0

    def record_user_prompt(self, text: str) -> None:
        """Record a user prompt. Call before sending each ACP prompt."""
        self._events_active = True
        self._flush_agent_text()
        self.events.append({"type": "user_message", "text": text})

    def mark_prompt_end(self) -> None:
        """Flush pending agent text after a prompt completes."""
        self._flush_agent_text()

    def _flush_agent_text(self) -> None:
        """Flush pending text events, merging consecutive same-type chunks."""
        if not self._pending_text:
            return
        current = self._pending_text[0].copy()
        for event in self._pending_text[1:]:
            if event["type"] == current["type"]:
                current["text"] += event["text"]
            else:
                self.events.append(current)
                current = event.copy()
        self.events.append(current)
        self._pending_text.clear()

    def handle_update(self, update: dict) -> None:
        """Process a session/update notification."""
        self._events_active = True
        update_type = update.get("sessionUpdate")

        if update_type == "tool_call":
            self._flush_agent_text()
            record = ToolCallRecord(
                tool_call_id=update.get("toolCallId", ""),
                title=update.get("title", ""),
                kind=update.get("kind", "other"),
            )
            self.tool_calls.append(record)
            self._tool_call_map[record.tool_call_id] = record
            self.events.append({"type": "tool_call", "record": record})

        elif update_type == "tool_call_update":
            tc_id = update.get("toolCallId", "")
            record = self._tool_call_map.get(tc_id)
            if not record:
                self._flush_agent_text()
                record = ToolCallRecord(
                    tool_call_id=tc_id,
                    title=update.get("title", ""),
                    kind=update.get("kind", "tool"),
                )
                self.tool_calls.append(record)
                self._tool_call_map[tc_id] = record
                self.events.append({"type": "tool_call", "record": record})
            try:
                status = ToolCallStatus(update.get("status", "in_progress"))
            except ValueError:
                logger.warning(f"Unknown tool call status: {update.get('status')}")
                status = ToolCallStatus.IN_PROGRESS
            record.update_status(status, update.get("content"))

        elif update_type == "agent_message_chunk":
            content = update.get("content", {})
            if content.get("type") == "text":
                text = content.get("text", "")
                self.message_chunks.append(text)
                self._pending_text.append({"type": "agent_message", "text": text})

        elif update_type == "text_update":
            # Used by openclaw shim — full text (not chunked)
            text = update.get("text", "")
            if text:
                self.message_chunks.append(text)
                self._pending_text.append({"type": "agent_message", "text": text})

        elif update_type == "agent_thought":
            # Used by openclaw shim — full thought (not chunked)
            text = update.get("text", "")
            if text:
                self.thought_chunks.append(text)
                self._pending_text.append({"type": "agent_thought", "text": text})

        elif update_type == "agent_thought_chunk":
            content = update.get("content", {})
            if content.get("type") == "text":
                text = content.get("text", "")
                self.thought_chunks.append(text)
                self._pending_text.append({"type": "agent_thought", "text": text})

        self.record_usage(update)

    def record_usage(self, payload: dict[str, Any] | None) -> None:
        """Record token/cost usage from an ACP result or update payload."""
        if not isinstance(payload, dict):
            return
        usage, cost = _usage_payload(payload)
        if not usage and cost is None:
            return

        self._events_active = True
        self._flush_agent_text()
        event: dict[str, Any] = {"type": "usage", "usage": usage}
        if cost is not None:
            event["total_cost_usd"] = cost
        self.events.append(event)
        self.usage_events.append(event.copy())
        for key, value in usage.items():
            self.token_usage[key] += value
        if cost is not None:
            self.total_cost_usd += cost

    def usage_summary(self) -> dict[str, int]:
        """Aggregate token usage observed in this session."""
        return dict(self.token_usage)

    @property
    def full_message(self) -> str:
        """Concatenated agent message text from all received chunks."""
        return "".join(self.message_chunks)

    @property
    def full_thought(self) -> str:
        """Concatenated agent thought/reasoning text from all received chunks."""
        return "".join(self.thought_chunks)
