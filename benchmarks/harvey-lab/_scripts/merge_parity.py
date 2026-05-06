"""Merge multiple parity runs into a single parity_experiment.json.

Each input file is the output of one `parity_test.py --mode agent-runs`
invocation. Tasks are deduplicated by task_id; if the same task appears in
multiple inputs the entry with more runs wins.

Usage:
    python _scripts/merge_parity.py out.json in1.json in2.json ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1

    out_path = Path(sys.argv[1])
    inputs = [Path(p) for p in sys.argv[2:]]

    merged: dict = {}
    by_task: dict[str, dict] = {}

    for inp in inputs:
        data = json.loads(inp.read_text())
        # Latest non-empty value per top-level field.
        for k, v in data.items():
            if k == "tasks":
                continue
            if v not in (None, "", [], {}):
                merged[k] = v
        for t in data.get("tasks", []):
            tid = t["task_id"]
            existing = by_task.get(tid)
            if not existing or len(t.get("runs", [])) > len(existing.get("runs", [])):
                by_task[tid] = t

    merged["tasks"] = list(by_task.values())
    merged["task_count"] = len(merged["tasks"])

    # Recompute summary
    per_metric: dict[str, dict[str, int]] = {}
    for t in merged["tasks"]:
        for m in t.get("metrics", []):
            slot = per_metric.setdefault(m["metric"], {"matched": 0, "total": 0})
            slot["total"] += 1
            slot["matched"] += int(m["match"])
    merged["summary"] = per_metric
    merged["merged_from"] = [str(p) for p in inputs]

    out_path.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"merged {len(inputs)} inputs → {out_path} ({len(merged['tasks'])} tasks)")
    for metric, slot in per_metric.items():
        print(f"  {metric}: {slot['matched']}/{slot['total']} tasks matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
