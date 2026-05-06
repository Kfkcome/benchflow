"""Adapter-aware runner for Harvey LAB.

Discovers (or downloads) the upstream tasks, converts them via benchflow.py,
then drives BenchFlow's Job to execute the agent loop.

Usage:
    uv run python benchmarks/harvey-lab/run_harvey_lab.py
    uv run python benchmarks/harvey-lab/run_harvey_lab.py path/to/job.yaml
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

from benchflow.job import Job
from benchflow.task_download import ensure_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("harvey-lab.run")

SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER = SCRIPT_DIR / "benchflow.py"


def _repo_root() -> Path:
    here = Path.cwd()
    while here != here.parent:
        if (here / ".git").is_dir():
            return here
        here = here.parent
    return Path.cwd()


def ensure_corpus() -> Path:
    """Download upstream tasks (if needed) and adapt them once."""
    raw_root = ensure_tasks("harvey-lab")
    repo = _repo_root()
    out = repo / ".ref" / "harvey-lab-benchflow"
    if out.exists() and any(out.iterdir()):
        log.info("adapter output exists at %s", out)
        return out
    log.info("adapting harvey-lab tasks → %s", out)
    r = subprocess.run(
        [sys.executable, str(ADAPTER),
         "--output-dir", str(out),
         "--harvey-root", str(raw_root.parent if raw_root.name == "tasks" else raw_root),
         "--overwrite"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        log.error("adapter failed: %s", r.stderr)
        raise RuntimeError("adapter failed")
    return out


async def main() -> None:
    cfg = sys.argv[1] if len(sys.argv) > 1 else str(SCRIPT_DIR / "harvey-lab-gemini-flash-lite.yaml")
    ensure_corpus()
    job = Job.from_yaml(cfg)
    result = await job.run()
    print(f"\nScore: {result.passed}/{result.total} ({result.score:.1%})")


if __name__ == "__main__":
    asyncio.run(main())
