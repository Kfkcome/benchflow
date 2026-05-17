"""Tests for AgentConfig + ProviderConfig registry shape:
env_mapping (BENCHFLOW_PROVIDER_* → agent-native vars) and credential_files.

Negative invariants ("agent X should NOT have feature Y configured") live in
test_registry_invariants.py — search there for the consolidated tripwire.
"""

from benchflow._agent_env import resolve_provider_env
from benchflow.agents.providers import PROVIDERS
from benchflow.agents.registry import (
    AGENT_ALIASES,
    AGENT_INSTALLERS,
    AGENT_LAUNCH,
    AGENTS,
    CUSTOM_AGENT_REGISTRY_ENV,
    list_agents,
    load_agent_registry,
    resolve_agent,
)


def _remove_agent(name: str, *aliases: str) -> None:
    AGENTS.pop(name, None)
    AGENT_INSTALLERS.pop(name, None)
    AGENT_LAUNCH.pop(name, None)
    for alias in aliases:
        AGENT_ALIASES.pop(alias, None)


class TestEnvMappingField:
    """env_mapping exists on AgentConfig and is populated for known agents."""

    def test_claude_agent_has_mapping(self):
        cfg = AGENTS["claude-agent-acp"]
        assert "BENCHFLOW_PROVIDER_BASE_URL" in cfg.env_mapping
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "ANTHROPIC_BASE_URL"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "ANTHROPIC_AUTH_TOKEN"

    def test_pi_acp_no_static_mapping(self):
        """pi-acp is multi-protocol — launch wrapper handles env translation."""
        cfg = AGENTS["pi-acp"]
        assert cfg.env_mapping == {}

    def test_codex_acp_has_mapping(self):
        cfg = AGENTS["codex-acp"]
        assert cfg.api_protocol == "openai-responses"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "OPENAI_BASE_URL"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "OPENAI_API_KEY"
        assert "openai_base_url=$OPENAI_BASE_URL" in cfg.launch_cmd

    def test_codex_acpx_wraps_codex_acp(self):
        cfg = AGENTS["codex-acpx"]
        assert cfg.protocol == "acpx"
        assert (
            "npm install -g --prefix /opt/benchflow/js-agents acpx@latest"
            in cfg.install_cmd
        )
        assert "@zed-industries/codex-acp@latest" in cfg.install_cmd
        assert cfg.launch_cmd == "/opt/benchflow/bin/codex-acp-launch"
        assert "openai_base_url=$OPENAI_BASE_URL" in cfg.install_cmd
        assert "exec /opt/benchflow/bin/codex-acp" in cfg.install_cmd
        assert (
            cfg.credential_files[0].path == AGENTS["codex-acp"].credential_files[0].path
        )

    def test_gemini_has_mapping(self):
        cfg = AGENTS["gemini"]
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "GEMINI_API_BASE_URL"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "GOOGLE_API_KEY"

    def test_openhands_has_mapping(self):
        cfg = AGENTS["openhands"]
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_BASE_URL"] == "LLM_BASE_URL"
        assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "LLM_API_KEY"
        # OpenHands model is normalized in _normalize_openhands_model().
        assert "BENCHFLOW_PROVIDER_MODEL" not in cfg.env_mapping

    def test_openhands_normalizes_model(self):
        env = {}
        resolve_provider_env(
            agent="openhands",
            model="zai/glm-5",
            agent_env=env,
        )

        assert env["LLM_MODEL"] == "glm-5"


class TestOpenHandsConfig:
    def test_openhands_uses_agentskills_paths(self):
        cfg = AGENTS["openhands"]
        assert "$HOME/.agents/skills" in cfg.skill_paths
        assert "$WORKSPACE/.agents/skills" in cfg.skill_paths

    def test_openhands_install_cmd_forces_github_main(self):
        cfg = AGENTS["openhands"]
        assert "apt-get install -y -qq curl ca-certificates git" in cfg.install_cmd
        assert (
            "uv tool install --force --refresh "
            "--from 'git+https://github.com/OpenHands/OpenHands-CLI.git@main' "
            "openhands --python 3.12" in cfg.install_cmd
        )
        assert "command -v git" in cfg.install_cmd
        assert "install.openhands.dev/install.sh" not in cfg.install_cmd

    def test_openhands_skips_acp_set_model(self):
        cfg = AGENTS["openhands"]
        assert cfg.supports_acp_set_model is False


class TestHarveyLabConfig:
    def test_harvey_lab_uses_venv_for_python_deps(self):
        cfg = AGENTS["harvey-lab-harness"]
        assert "python3 -m venv /opt/benchflow/harvey-lab-venv" in cfg.install_cmd
        assert (
            "/opt/benchflow/harvey-lab-venv/bin/python -m pip install"
            in cfg.install_cmd
        )
        assert cfg.launch_cmd.startswith(
            "HARVEY_LABS_ROOT=/opt/harvey-labs "
            "/opt/benchflow/harvey-lab-venv/bin/python "
        )


class TestAgentCredentialFiles:
    def test_codex_has_auth_json(self):
        cfg = AGENTS["codex-acp"]
        assert len(cfg.credential_files) == 1
        cf = cfg.credential_files[0]
        assert cf.env_source == "OPENAI_API_KEY"
        assert ".codex/auth.json" in cf.path
        assert "{home}" in cf.path
        assert "{value}" in cf.template


class TestProviderCredentialFiles:
    def test_vertex_providers_have_adc(self):
        for name in ("google-vertex", "anthropic-vertex"):
            cfg = PROVIDERS[name]
            assert len(cfg.credential_files) == 1, (
                f"{name} should have 1 credential_file"
            )
            cf = cfg.credential_files[0]
            assert cf["env_source"] == "GOOGLE_APPLICATION_CREDENTIALS_JSON"
            assert "gcloud" in cf["path"]
            assert "GOOGLE_APPLICATION_CREDENTIALS" in cf.get("post_env", {})

    def test_zai_no_credential_files(self):
        cfg = PROVIDERS["zai"]
        assert cfg.credential_files == []


class TestCustomAgentRegistry:
    def test_load_agent_registry_from_yaml(self, tmp_path):
        registry = tmp_path / "agents.yaml"
        registry.write_text("""
agents:
  custom-acp:
    description: Custom ACP agent
    install_cmd: "true"
    launch_cmd: "custom-agent --acp"
    protocol: acp
    requires_env: [CUSTOM_API_KEY]
    skill_paths: ["$HOME/.custom/skills"]
    env_mapping:
      BENCHFLOW_PROVIDER_API_KEY: CUSTOM_API_KEY
    aliases: [custom]
aliases:
  custom-latest: custom-acp
""")

        try:
            loaded = load_agent_registry(registry)
            cfg = resolve_agent("custom")

            assert [agent.name for agent in loaded] == ["custom-acp"]
            assert cfg.name == "custom-acp"
            assert cfg.launch_cmd == "custom-agent --acp"
            assert cfg.requires_env == ["CUSTOM_API_KEY"]
            assert cfg.env_mapping["BENCHFLOW_PROVIDER_API_KEY"] == "CUSTOM_API_KEY"
            assert resolve_agent("custom-latest").name == "custom-acp"
        finally:
            _remove_agent("custom-acp", "custom", "custom-latest")

    def test_resolve_agent_loads_registry_from_env(self, tmp_path, monkeypatch):
        registry = tmp_path / "env-agents.yaml"
        registry.write_text("""
agents:
  env-agent:
    install_cmd: "echo install"
    launch_cmd: "env-agent acp"
    protocol: acpx
""")
        monkeypatch.setenv(CUSTOM_AGENT_REGISTRY_ENV, str(registry))

        try:
            cfg = resolve_agent("env-agent")

            assert cfg.protocol == "acpx"
            assert "env-agent" in {agent.name for agent in list_agents()}
        finally:
            _remove_agent("env-agent")
