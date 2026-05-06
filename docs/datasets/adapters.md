# Benchmark Adapters

A benchmark adapter is a small, self-contained directory under
`benchmarks/<name>/` that converts an upstream benchmark into the BenchFlow
task format and ships the parity evidence backing the conversion. It is the
mechanism by which third-party benchmarks become first-class BenchFlow
benchmarks without requiring upstream changes.

## Directory layout

```
benchmarks/<name>/
├── benchflow.py             # adapter CLI: upstream → BenchFlow task dirs
├── benchmark.yaml           # benchmark identity, source, verification, parity
├── adapter_metadata.json    # adapter / parity provenance
├── parity_test.py           # structural | side-by-side | agent-runs
├── parity_experiment.json   # live parity results (mean ± SEM)
└── README.md
```

The generated task corpus (the actual `task.toml` / `instruction.md` / …
files for every task) is **not** committed in this directory. It lives in
the corpus repo (`benchflow-ai/benchmarks`) and HuggingFace mirror
(`benchflow/benchmarks`), each under `datasets/<name>/`.

## `benchflow.py` — adapter CLI

The adapter is a single entry-point script (or package with a `__main__`
shim that reuses `benchflow.py` as its entry name). Required flags:

| Flag | Purpose |
|---|---|
| `--output-dir DIR`             | Where to write generated task directories. |
| `--limit N`                    | Process at most N tasks. |
| `--task-ids ID,ID`             | Generate exactly the listed task IDs. |
| `--split {full,parity}`        | `full` = every task; `parity` = the fixed parity subset. |
| `--overwrite`                  | Replace existing task directories. |

Each generated directory must contain at minimum:

| File | Purpose |
|---|---|
| `task.toml`                          | Required `[task].name`, `[agent].timeout_sec`, plus metadata. |
| `instruction.md`                     | First prompt sent to the agent. |
| `environment/Dockerfile`             | Sandbox image. |
| `tests/test.sh`                      | Verifier entry point. Must write `0.0`–`1.0` to `/logs/verifier/reward.txt`. |
| `solution/solve.sh` *(if oracle)*    | Reference solution; omit when upstream ships none. |

Task `name` lives under `[task]` (e.g. `harvey-lab/<task-id>`). Sanitize
upstream IDs before using them: lowercase, replace runs of non-alphanumerics
with single hyphens, strip leading/trailing separators.

## `benchmark.yaml` — descriptor

A small declarative YAML that names the benchmark, points at the upstream
source (with a pinned commit), describes the task counts and verification
method, and tracks parity evidence. See
[`benchmarks/harvey-lab/benchmark.yaml`](../../benchmarks/harvey-lab/benchmark.yaml)
for a worked example. Job configs (model, agent, environment) live in
separate YAMLs next to it — `benchmark.yaml` describes what the benchmark
*is*, not how to run it.

## `parity_test.py` — three modes

Adapters claim a parity result; `parity_test.py` is how that claim is made
reproducible. Three modes, increasing in cost and substance.

### 1. structural

No API calls. Verifies every generated task is well-formed: required files
exist, `task.toml` parses, criteria counts match the upstream rubric,
`instruction.md` preserves the upstream instruction prefix, `tests/test.sh`
is executable. Run on the full set on every change.

### 2. side-by-side (judge prompt equivalence)

Per task, pick a fixed prefix of N criteria and build a synthetic
deliverable. Run the **upstream judge prompt** and the **adapter judge
prompt** through the **same** judge model on the **same** synthetic
deliverable. Compare verdicts. This isolates "did the conversion preserve
the judge prompt semantics" from any agent variance.

### 3. agent-runs (substantive parity)

The only mode that produces a defensible `parity_experiment.json`. Run the
**same agent + same model** on each task on both sides; score the produced
deliverables with the **same judge** under both prompts. Repeat 3+ times per
side. Report:

- `mean ± sample SEM` for the all-pass reward and per-criterion pass rate
- match: `max(side_A) >= min(side_B) AND max(side_B) >= min(side_A)`

Sample SEM = `stdev / sqrt(n)` (use the *sample* standard deviation, not
the population). Refuse to claim parity unless both sides used the same
agent + model.

The execution order is non-negotiable: **sanity check 5–10 tasks → one
full run on both sides → three runs on both sides**. Never report parity
from a single run per side.

## `parity_experiment.json` — results

Schema (per task entry):

```jsonc
{
  "task_id": "corporate-ma/analyze-cim-deal-teaser/scenario-01",
  "n_criteria": 39,
  "metrics": [
    {
      "metric": "all-pass-reward",
      "side_A_original_runs": [0.0, 0.0, 0.0],
      "side_B_adapter_runs":  [0.0, 0.0, 0.0],
      "side_A_summary": "0.000 ± 0.000",
      "side_B_summary": "0.000 ± 0.000",
      "match": true
    },
    {
      "metric": "criterion-pass-rate",
      "side_A_original_runs": [0.18, 0.21, 0.20],
      "side_B_adapter_runs":  [0.19, 0.21, 0.20],
      "side_A_summary": "0.197 ± 0.009",
      "side_B_summary": "0.200 ± 0.006",
      "match": true
    }
  ]
}
```

## Naming, discovery, and registration

- Adapter directory: `benchmarks/<name>/` (one level, kebab-case).
- Task `name` field: `<name>/<sanitized-task-id>` — stable and unique.
- Generated corpus path on the mirror: `datasets/<name>/<sanitized-task-id>/`.
- Register the benchmark in `src/benchflow/task_download.py` so users can
  `ensure_tasks("<name>")`.

## Honesty rules

- Do not call something parity-validated if you only ran one side.
- Do not call something parity-validated if you used different agents or
  different models on the two sides. The whole point is "same agent + same
  model on both sides".
- Record judge model, agent model, temperature, and run count in
  `parity_experiment.json`. They are part of the claim.
- A side-by-side prompt-equivalence pass is necessary but not sufficient —
  it does not say anything about score distributions on real agent output.
