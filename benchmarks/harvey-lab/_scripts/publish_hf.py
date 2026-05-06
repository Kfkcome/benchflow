"""Publish Harvey LAB parity artifacts to the Hugging Face dataset mirror.

Reads HUGGINGFACE_TOKEN from the environment. Uploads:

    benchmarks/harvey-lab/README.md
    benchmarks/harvey-lab/benchmark.yaml
    benchmarks/harvey-lab/adapter_metadata.json
    benchmarks/harvey-lab/benchflow_parity/parity_experiment.json
    benchmarks/harvey-lab/results_collection/parity_summary.json

Run:
    HUGGINGFACE_TOKEN=hf_... uv run --with huggingface_hub python \
        benchmarks/harvey-lab/_scripts/publish_hf.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "benchflow/benchmarks"
ADAPTER_DIR = Path(__file__).resolve().parent.parent  # benchmarks/harvey-lab/


def summarize_parity(parity: dict) -> dict:
    """Compute a flat summary suitable for results_collection/parity_summary.json."""
    summary = {
        "benchmark": "harvey-lab",
        "mode": parity.get("mode"),
        "agent_model": parity.get("agent_model"),
        "judge_model": parity.get("judge_model"),
        "runs_per_side": parity.get("runs_per_side"),
        "tasks": [],
    }
    for t in parity.get("tasks", []):
        per_metric: dict[str, dict] = {}
        for m in t.get("metrics", []):
            per_metric[m["metric"]] = {
                "side_A": m["side_A_summary"],
                "side_B": m["side_B_summary"],
                "match": m["match"],
                "side_A_runs": m["side_A_original_runs"],
                "side_B_runs": m["side_B_adapter_runs"],
            }
        summary["tasks"].append({
            "task_id": t["task_id"],
            "n_criteria": t["n_criteria"],
            "metrics": per_metric,
        })
    if (s := parity.get("summary")):
        summary["overall"] = s
    return summary


def main() -> int:
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HUGGINGFACE_TOKEN", file=sys.stderr)
        return 1

    parity_file = ADAPTER_DIR / "parity_experiment.json"
    if not parity_file.exists():
        print(f"ERROR: missing {parity_file}", file=sys.stderr)
        return 1

    parity = json.loads(parity_file.read_text())
    summary = summarize_parity(parity)

    api = HfApi(token=token)
    api.create_repo(REPO, repo_type="dataset", exist_ok=True, token=token)

    uploads: list[tuple[Path | str, str]] = [
        (ADAPTER_DIR / "_scripts" / "hf_dataset_README.md", "README.md"),
        (ADAPTER_DIR / "README.md",          "benchmarks/harvey-lab/README.md"),
        (ADAPTER_DIR / "benchmark.yaml",     "benchmarks/harvey-lab/benchmark.yaml"),
        (ADAPTER_DIR / "adapter_metadata.json", "benchmarks/harvey-lab/adapter_metadata.json"),
        (parity_file,                         "benchmarks/harvey-lab/benchflow_parity/parity_experiment.json"),
    ]

    summary_path = ADAPTER_DIR / "_scripts" / "parity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    uploads.append((summary_path, "benchmarks/harvey-lab/results_collection/parity_summary.json"))

    for src, dest in uploads:
        if not Path(src).exists():
            print(f"  skip (missing): {src}")
            continue
        print(f"  upload {src} -> {dest}")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dest,
            repo_id=REPO,
            repo_type="dataset",
            commit_message="harvey-lab: refresh adapter parity artifacts",
        )

    print(f"\nView at https://huggingface.co/datasets/{REPO}/tree/main/benchmarks/harvey-lab")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
