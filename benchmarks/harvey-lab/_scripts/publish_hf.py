"""Publish Harvey LAB parity artifacts to the Hugging Face dataset mirror.

Each adapter owns exactly its own subdirectory under
`benchmarks/<adapter-name>/`. Top-level files (the dataset README) are
shared and require human review — they are uploaded only when
`--touch-shared` is passed.

By default the upload opens a Hugging Face PR (Discussion) rather than
pushing to `main`, so changes go through review even when tokens are
shared. Use `--direct` for trusted automation.

Run:
    HUGGINGFACE_TOKEN=hf_... uv run --with huggingface_hub python \\
        benchmarks/harvey-lab/_scripts/publish_hf.py
    # or to push directly to main (skip PR):
    HUGGINGFACE_TOKEN=hf_... uv run --with huggingface_hub python \\
        benchmarks/harvey-lab/_scripts/publish_hf.py --direct
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "benchflow/benchmarks"
ADAPTER_NAME = "harvey-lab"
ADAPTER_DIR = Path(__file__).resolve().parent.parent  # benchmarks/harvey-lab/
NAMESPACE = f"benchmarks/{ADAPTER_NAME}"  # this adapter's owned subtree


def summarize_parity(parity: dict) -> dict:
    """Flatten the parity record for results_collection/parity_summary.json."""
    summary = {
        "benchmark": ADAPTER_NAME,
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


def build_operations(*, touch_shared: bool) -> list[CommitOperationAdd]:
    """Files this adapter is allowed to add/update.

    By convention an adapter owns only its own subdirectory under
    `benchmarks/<adapter-name>/`. The top-level dataset README is shared
    across adapters and is only re-uploaded when explicitly requested.
    """
    parity_file = ADAPTER_DIR / "parity_experiment.json"
    if not parity_file.exists():
        raise SystemExit(f"missing parity record: {parity_file}")

    summary_path = ADAPTER_DIR / "_scripts" / "parity_summary.json"
    summary_path.write_text(json.dumps(
        summarize_parity(json.loads(parity_file.read_text())),
        indent=2,
    ) + "\n")

    files: list[tuple[Path, str]] = [
        # ── owned subtree: benchmarks/<adapter-name>/ ──────────────
        (ADAPTER_DIR / "README.md",             f"{NAMESPACE}/README.md"),
        (ADAPTER_DIR / "benchmark.yaml",        f"{NAMESPACE}/benchmark.yaml"),
        (ADAPTER_DIR / "adapter_metadata.json", f"{NAMESPACE}/adapter_metadata.json"),
        (parity_file,                            f"{NAMESPACE}/benchflow_parity/parity_experiment.json"),
        (summary_path,                           f"{NAMESPACE}/results_collection/parity_summary.json"),
    ]
    if touch_shared:
        files.insert(0, (ADAPTER_DIR / "_scripts" / "hf_dataset_README.md", "README.md"))

    ops: list[CommitOperationAdd] = []
    for src, dest in files:
        if not src.exists():
            print(f"  skip (missing): {src}", file=sys.stderr)
            continue
        if not (dest == "README.md" or dest.startswith(f"{NAMESPACE}/")):
            raise SystemExit(
                f"refusing to write outside owned subtree: {dest}"
            )
        ops.append(CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(src)))
        print(f"  + {dest}  ({src.name})")
    return ops


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direct", action="store_true",
                   help="Push directly to main instead of opening a PR.")
    p.add_argument("--touch-shared", action="store_true",
                   help="Also overwrite the top-level dataset README. Off by default.")
    args = p.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HUGGINGFACE_TOKEN", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    api.create_repo(REPO, repo_type="dataset", exist_ok=True, token=token)

    ops = build_operations(touch_shared=args.touch_shared)
    if not ops:
        print("nothing to upload")
        return 0

    title = f"{ADAPTER_NAME}: refresh adapter parity artifacts"
    description = (
        f"Adapter owns `{NAMESPACE}/`. Sourced from "
        "github.com/benchflow-ai/benchflow tree feat/dataset-adapters. "
        "Same agent + same judge model on both sides for the agent-runs "
        "parity check."
    )
    result = api.create_commit(
        repo_id=REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=title,
        commit_description=description,
        create_pr=not args.direct,
    )
    if args.direct:
        print(f"\npushed to main: {getattr(result, 'commit_url', result)}")
    else:
        print(f"\nopened PR: {getattr(result, 'pr_url', result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
