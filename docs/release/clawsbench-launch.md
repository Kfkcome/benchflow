# ClawsBench Launch Draft

## Positioning

ClawsBench is the demo-friendly benchmark story for BenchFlow: realistic
agent tasks, managed agents, and verifiable results running through the same
Rollout/Scene stack that powers larger suite evaluations.

## Selling Points

- Run multiple ACP-compatible agents through one BenchFlow command surface.
- Use sandboxed tasks with verifier outputs, trajectories, token usage, and
  reward summaries captured per trial.
- Compare native ACP runs with acpx-backed Codex runs without changing the task
  format.
- Keep task and docs workflows local-first: task sources can come from GitHub,
  HuggingFace datasets, or local directories.
- Ship demos that are inspectable after the fact: `result.json`,
  `summary.json`, and `trajectory/acp_trajectory.jsonl` are the source of truth.

## Demo Video Steps

1. Start from a clean checkout and show the install:

   ```bash
   uv sync --extra dev --extra daytona --extra hf --locked
   uv run bench agent list
   ```

2. Show the task source options:

   ```bash
   uv run bench eval create --help | rg 'source-repo|source-hf|tasks-dir'
   ```

3. Run a small local or remote subset with a managed ACP agent:

   ```bash
   GEMINI_API_KEY=... uv run bench eval create \
     --source-repo benchflow-ai/skillsbench \
     --source-path tasks/edit-pdf \
     --agent gemini \
     --model gemini-3.1-flash-lite-preview \
     --sandbox docker \
     --jobs-dir jobs/clawsbench-demo
   ```

4. Run the same task through Codex via acpx:

   ```bash
   OPENAI_API_KEY=... uv run bench eval create \
     --source-repo benchflow-ai/skillsbench \
     --source-path tasks/edit-pdf \
     --agent codex-acpx \
     --model gpt-5.4-mini/low \
     --sandbox docker \
     --jobs-dir jobs/clawsbench-demo
   ```

5. Inspect the artifacts:

   ```bash
   find jobs/clawsbench-demo -name result.json -o -name summary.json
   jq '.task_name, .agent, .rewards, .token_usage, .trajectory_source' \
     "$(find jobs/clawsbench-demo -name result.json | head -n 1)"
   ```

6. Close on the value proposition:

   BenchFlow is no longer a Harbor wrapper. It owns rollouts, sandboxes, agents,
   adapters, rewards, trajectories, and docs as first-class release surfaces.

## Post Copy

BenchFlow v0.4 turns our benchmark runtime into a native stack: Rollouts and
Scenes for agent execution, managed ACP/acpx agents, Docker/Daytona/Modal
adapters, rubric rewards, and source-aware task loading from GitHub or
HuggingFace.

The ClawsBench demo shows the practical loop: pick a benchmark task, run it
through multiple agents, then inspect verifier rewards, token usage, and full
ACP trajectories from the job folder. Same task format, different agents, one
auditable result trail.

Try the demo path:

```bash
uv sync --extra dev --extra daytona --extra hf --locked
uv run bench agent list
uv run bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks/edit-pdf \
  --agent codex-acpx \
  --model gpt-5.4-mini/low \
  --sandbox docker
```

This is the direction for BenchFlow: more complete than a compatibility layer,
small enough to reason about, and strict enough that benchmark failures stay
separable from harness, auth, sandbox, and verifier failures.
