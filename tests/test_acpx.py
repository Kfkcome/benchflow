from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow._acpx_run import execute_prompts_with_acpx
from benchflow.rollout import Trial, TrialConfig
from benchflow.sandbox.protocol import ExecResult


class FakeEnv:
    def __init__(self, stdout: str = "", return_code: int = 0, stderr: str = ""):
        self.stdout = stdout
        self.return_code = return_code
        self.stderr = stderr
        self.commands: list[dict] = []
        self.files: dict[str, bytes] = {}

    async def exec(self, command: str, **kwargs) -> ExecResult:
        self.commands.append({"command": command, "kwargs": kwargs})
        if "/opt/benchflow/bin/acpx" in command:
            return ExecResult(
                return_code=self.return_code,
                stdout=self.stdout,
                stderr=self.stderr,
            )
        return ExecResult(return_code=0, stdout="", stderr="")

    async def write_file(self, path: str, content: bytes) -> None:
        self.files[path] = content


def _line(obj: dict) -> str:
    return json.dumps(obj) + "\n"


@pytest.mark.asyncio
async def test_execute_prompts_with_acpx_parses_raw_acp_ndjson(tmp_path: Path):
    stdout = "".join(
        [
            _line(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "protocolVersion": 0,
                        "agentInfo": {"name": "Codex", "version": "test"},
                    },
                }
            ),
            _line(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "tc_1",
                        "title": "ls",
                        "kind": "bash",
                    },
                }
            ),
            _line(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": "tc_1",
                        "status": "completed",
                        "content": [{"type": "text", "text": "ok"}],
                    },
                }
            ),
            _line(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "session-1",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "done"},
                        },
                    },
                }
            ),
            _line(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "result": {
                        "stopReason": "end_turn",
                        "usage": {"inputTokens": 8, "outputTokens": 2},
                        "total_cost_usd": 0.001,
                    },
                }
            ),
        ]
    )
    env = FakeEnv(stdout=stdout)

    trajectory, n_tools, agent_name = await execute_prompts_with_acpx(
        env,
        agent="codex-acpx",
        agent_launch="/opt/benchflow/bin/codex-acp -c tools.web_search=false",
        model="zai/glm-5",
        prompts=["solve it"],
        agent_env={"OPENAI_API_KEY": "test"},
        sandbox_user="agent",
        timeout=120,
        trial_dir=tmp_path,
        agent_cwd="/app",
    )

    assert env.files["/tmp/benchflow-acpx/prompt-1.md"] == b"solve it"
    acpx_cmd = next(
        item["command"]
        for item in env.commands
        if "/opt/benchflow/bin/acpx" in item["command"]
    )
    assert "--format json --json-strict" in acpx_cmd
    assert "--model glm-5" in acpx_cmd
    assert "--agent '/opt/benchflow/bin/codex-acp -c tools.web_search=false'" in acpx_cmd
    assert "exec --file /tmp/benchflow-acpx/prompt-1.md" in acpx_cmd
    assert n_tools == 1
    assert agent_name == "Codex"
    assert trajectory[0] == {"type": "user_message", "text": "solve it"}
    assert trajectory[1]["type"] == "tool_call"
    assert trajectory[1]["status"] == "completed"
    assert trajectory[2] == {"type": "agent_message", "text": "done"}
    assert trajectory[3] == {
        "type": "usage",
        "usage": {
            "input_tokens": 8,
            "output_tokens": 2,
            "total_tokens": 10,
        },
        "total_cost_usd": 0.001,
    }
    assert (tmp_path / "agent" / "codex_acpx_acpx_1.ndjson").read_text() == stdout


@pytest.mark.asyncio
async def test_execute_prompts_with_acpx_maps_timeout_exit_code(tmp_path: Path):
    env = FakeEnv(return_code=3)

    with pytest.raises(TimeoutError, match="acpx agent timed out"):
        await execute_prompts_with_acpx(
            env,
            agent="codex-acpx",
            agent_launch="/opt/benchflow/bin/codex-acp",
            model=None,
            prompts=["solve it"],
            agent_env={},
            sandbox_user=None,
            timeout=5,
            trial_dir=tmp_path,
            agent_cwd="/app",
        )


@pytest.mark.asyncio
async def test_rollout_acpx_skips_connect_acp_and_executes_cli(tmp_path: Path):
    cfg = TrialConfig(
        task_path=tmp_path / "task",
        agent="codex-acpx",
        model="gpt-5.4-mini/low",
    )
    trial = Trial.__new__(Trial)
    trial._config = cfg
    trial._env = MagicMock()
    trial._trial_dir = tmp_path
    trial._timing = {}
    trial._agent_cwd = "/app"
    trial._agent_launch = "/opt/benchflow/bin/codex-acp"
    trial._agent_env = {"OPENAI_API_KEY": "test"}
    trial._provider_runtime = None
    trial._phase = "installed"
    trial._resolved_prompts = ["default prompt"]
    trial._timeout = 60
    trial._trajectory = []
    trial._n_tool_calls = 0

    with (
        patch(
            "benchflow.rollout.ensure_bedrock_proxy_runtime",
            new=AsyncMock(return_value=(trial._agent_env, None)),
        ),
        patch("benchflow.rollout.connect_acp", new=AsyncMock()) as connect_acp,
        patch(
            "benchflow.rollout.execute_prompts_with_acpx",
            new=AsyncMock(
                return_value=(
                    [{"type": "agent_message", "text": "ok"}],
                    2,
                    "Codex",
                )
            ),
        ) as exec_acpx,
    ):
        await trial.connect()
        await trial.execute(["prompt"])

    connect_acp.assert_not_awaited()
    exec_acpx.assert_awaited_once()
    assert exec_acpx.await_args.kwargs["agent"] == "codex-acpx"
    assert exec_acpx.await_args.kwargs["prompts"] == ["prompt"]
    assert trial._agent_name == "Codex"
    assert trial._trajectory_source == "acpx"
    assert trial._n_tool_calls == 2
