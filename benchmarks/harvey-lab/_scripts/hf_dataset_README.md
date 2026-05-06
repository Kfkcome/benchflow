---
license: apache-2.0
tags:
  - benchflow
  - benchmarks
  - parity-experiments
  - agent-evaluation
pretty_name: BenchFlow Benchmarks
---

# BenchFlow Benchmarks

Parity-experiment artifacts for benchmarks adapted to the
[BenchFlow](https://github.com/benchflow-ai/benchflow) task format.

This dataset is the public mirror of the parity evidence backing each
adapted benchmark. The generated task corpora themselves
(`task.toml` / `instruction.md` / `environment/` / `tests/` for every
task) live in [`benchflow-ai/benchmarks`](https://github.com/benchflow-ai/benchmarks)
under `datasets/<name>/`.

## Layout

```
benchmarks/
└── <name>/
    ├── README.md                                   # adapter overview
    ├── benchmark.yaml                              # benchmark descriptor
    ├── adapter_metadata.json                       # adapter / parity provenance
    ├── benchflow_parity/parity_experiment.json     # raw parity record from parity_test.py
    └── results_collection/parity_summary.json      # flattened per-task summary
```

## Available Benchmarks

| Benchmark | Tasks | Verification | Parity protocol | Source |
|---|---|---|---|---|
| [harvey-lab](benchmarks/harvey-lab/) | 1,251 | LLM-as-judge (Gemini 3.1 Flash Lite, all-pass) | structural ✅, agent-runs (subset) ✅ | [harveyai/harvey-labs](https://github.com/harveyai/harvey-labs) |

## Parity protocol (summary)

Every adapter ships a `parity_experiment.json` produced by its
`parity_test.py`. Two of the three modes are non-substantive preconditions:

- **structural** — every generated task is well-formed (no API calls)
- **side-by-side** — original judge prompt vs. adapter judge prompt agree on
  synthetic deliverables

The substantive mode is **agent-runs**: the **same agent + model** is run on
both sides, deliverables are scored with the **same judge**, mean ± sample
SEM is reported, and the harbor-style match
`max(A) >= min(B) AND max(B) >= min(A)` is checked. Refer to each
benchmark's `parity_summary.json` for the live numbers.

## How to add a benchmark

Run your adapter's parity_test, then upload exactly the artifacts above.
See the BenchFlow adapter convention at
[`docs/datasets/adapters.md`](https://github.com/benchflow-ai/benchflow/blob/main/docs/datasets/adapters.md).

## Links

- [Adapter code](https://github.com/benchflow-ai/benchflow/tree/main/benchmarks/)
- [Generated task corpora](https://github.com/benchflow-ai/benchmarks)
