---
name: agent-creator
description: Create or update BenchFlow managed agents. Use this when adding a new agent adapter, ACP/acpx launch path, provider mapping, subscription-auth credential file, install command, or managed-agent registry entry in BenchFlow.
---

# BenchFlow Agent Creator

Use this skill to add a BenchFlow-managed agent cleanly, without smuggling
one-off install logic into the rollout path.

## Workflow

1. Inspect the existing registry shape:
   - `src/benchflow/agents/registry.py`
   - `src/benchflow/agents/providers.py`
   - `tests/test_agent_registry.py`
   - `tests/test_registry_invariants.py`
   - `tests/test_agent_spec.py`
2. Decide the protocol:
   - `acp`: the agent speaks ACP directly over stdio.
   - `acpx`: BenchFlow launches `acpx` as the headless ACP client and the agent
     command is passed to `--agent`.
   - Use an adapter shim only when the upstream tool has no native ACP server.
3. Add or update one `AgentConfig` entry. Keep install, launch, env mapping, and
   credential detection in the registry entry rather than scattering conditionals
   through rollout/runtime code.
4. For provider keys, prefer generic BenchFlow inputs:
   - `BENCHFLOW_PROVIDER_BASE_URL`
   - `BENCHFLOW_PROVIDER_API_KEY`
   - `BENCHFLOW_PROVIDER_MODEL`
   Map those into agent-native variables through `env_mapping` or a launch
   wrapper.
5. For subscription auth, add a `CredentialFile` entry instead of requiring API
   keys when a host login file exists.
6. For shims, install them with `_install_python_script()` or
   `_install_shell_script()` in the registry. Keep the shim deterministic and
   protocol-focused.

## Required Checks

Run the narrow checks first:

```bash
uv run python -m pytest tests/test_agent_registry.py tests/test_registry_invariants.py tests/test_agent_spec.py -q
uv run ruff check src/benchflow/agents tests/test_agent_registry.py tests/test_registry_invariants.py tests/test_agent_spec.py
uv run ty check src/
```

If the new agent touches runtime transport, also run the relevant ACP/process
tests:

```bash
uv run python -m pytest tests/test_acp.py tests/test_acpx.py tests/test_process.py -q
```

For docs-facing changes, update `docs/reference/cli.md`, `docs/getting-started.md`,
and `docs/integration-tests.md` when the agent is user-visible.

## Review Checklist

- Agent name has a stable alias only when users already know that spelling.
- Install command is idempotent and can run per task.
- Secrets are passed via env vars or credential files, never shell-expanded into
  process arguments.
- Network-restricted tasks install the agent before hard network policy applies.
- Default model is available in the target agent.
- Tests pin the registry contract, not just the happy-path string literal.
