# Harvey LAB — BenchFlow Adapter

[Harvey LAB](https://github.com/harveyai/harvey-labs) (Legal Agent Benchmark)
in BenchFlow format: 1,251 realistic legal tasks across 24 practice areas.

## What this directory is

```
benchmarks/harvey-lab/
├── benchflow.py             # adapter CLI: harvey-labs → BenchFlow tasks
├── benchmark.yaml           # benchmark identity, source, verification, parity
├── adapter_metadata.json    # adapter / parity provenance
├── parity_test.py           # structural | side-by-side | agent-runs
├── parity_experiment.json   # live parity results (mean ± SEM)
└── README.md
```

The **generated** task corpus is hosted in a separate repo / dataset:

| Surface | Location |
|---|---|
| GitHub mirror | `benchflow-ai/benchmarks` (`datasets/harvey-lab/`) |
| HuggingFace mirror | `benchflow/benchmarks` (`datasets/harvey-lab/`) |

The generated tasks are not committed here; this directory hosts only the
adapter and parity evidence.

## Mapping

| Harvey LAB | BenchFlow |
|---|---|
| `tasks/<area>/<slug>[/scenario-N]/task.json` | sanitized task directory |
| `task.json::title` + `instructions` | `instruction.md` |
| `task.json::deliverables` | listed in `instruction.md`, mirrored in `rubric.json` |
| `task.json::criteria` (rubric) | `environment/rubric.json` |
| `documents/` | `environment/documents/` (baked into the image) |
| Anthropic LLM judge with 4-var prompt | `tests/evaluate.py` Gemini judge with the same 4 vars |
| `score = 1.0 if all-pass else 0.0` | identical: `reward.txt` is 1.0 only when every criterion passes |
| Per-criterion verdict trace | `evaluation_details.json` (alongside reward.txt) |

Harvey LAB ships no oracle solutions, so generated tasks omit `solution/`.

## Generate

```bash
# Full benchmark (1,251 tasks)
uv run python benchmarks/harvey-lab/benchflow.py \
    --output-dir /tmp/harvey-lab-benchflow \
    --harvey-root .ref/harvey-lab \
    --overwrite

# Parity subset (5 tasks across 5 practice areas)
uv run python benchmarks/harvey-lab/benchflow.py \
    --output-dir /tmp/harvey-lab-parity \
    --harvey-root .ref/harvey-lab \
    --split parity --overwrite

# Specific tasks
uv run python benchmarks/harvey-lab/benchflow.py \
    --output-dir /tmp/harvey-lab-pick \
    --harvey-root .ref/harvey-lab \
    --task-ids "corporate-ma/analyze-cim-deal-teaser/scenario-01,real-estate/draft-construction-contract"
```

CLI flags follow the BenchFlow adapter convention: `--output-dir`,
`--harvey-root`, `--limit`, `--task-ids`, `--split {full,parity}`,
`--overwrite`.

## Parity protocol

Three modes, in order. **Only `agent-runs` produces a substantive parity
claim**; the others are cheap preconditions.

### 1. structural — every task is well-formed (no API calls)

```bash
uv run python benchmarks/harvey-lab/parity_test.py --mode structural --split full
```

Validates every generated task has `task.toml` (with `[task].name`),
`instruction.md` (preserving upstream instructions), `environment/Dockerfile`,
`environment/rubric.json` (criteria count matches upstream), executable
`tests/test.sh`, and `tests/evaluate.py`.

### 2. side-by-side — original judge prompt vs. adapter judge prompt

```bash
GEMINI_API_KEY=… uv run --with google-genai \
    python benchmarks/harvey-lab/parity_test.py \
    --mode side-by-side --limit-criteria 5
```

For each parity-subset task, picks the first N criteria, builds a synthetic
deliverable, and asks the **same** Gemini judge to verdict against the
**original** (str.format-style) and the **adapter** (`string.Template`)
prompts. Verifies prompt equivalence in isolation from agent variance.

### 3. agent-runs — the substantive parity claim

```bash
GEMINI_API_KEY=… uv run --with google-genai --with pdfplumber --with openpyxl --with pandas \
    python benchmarks/harvey-lab/parity_test.py \
    --mode agent-runs --runs 3 --split parity
```

Runs the **same agent + model** (`gemini-3.1-flash-lite-preview`) on the
**same parsed documents** for each task, then scores the produced
deliverables with the **same judge model** under both prompts. Reports per
task:

- `mean ± sample SEM` for `all-pass-reward` and `criterion-pass-rate`
- the harbor-style match: `max(A) >= min(B) AND max(B) >= min(A)`

Sample SEM = `stdev / sqrt(n)`. The protocol is symmetric: both sides see
identical agent output; the only difference is the judge prompt template, so
agreement isolates "is the conversion faithful" from agent stochasticity.

> **Honesty constraint.** This adapter intentionally does not claim full
> upstream-vs-BenchFlow score-distribution parity until matched runs exist
> on the upstream `harvey-labs.harness.run` (Podman-sandboxed) and on
> BenchFlow's own runtime. The agent-runs mode here covers the
> *conversion-faithfulness* half of that claim with a controlled experiment.

## Running

A Job config and a runner are provided once a model + sandbox is available
in your environment:

```bash
# YAML-driven
uv run python -c "import asyncio; from benchflow.job import Job; \
    asyncio.run(Job.from_yaml('benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml').run())"
```

## Stats

- 24 practice areas
- 1,251 tasks
- 4 work types: analyze (490), draft (444), review (293), research (24)
- Criteria per task: min 23, max 194, median ~60
