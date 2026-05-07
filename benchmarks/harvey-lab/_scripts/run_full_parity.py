"""End-to-end source-vs-adapter parity for Harvey LAB.

Drives the upstream `harvey-labs.harness.run` and the BenchFlow
`Job.from_yaml(...)` runner with the SAME agent + SAME model on the SAME
task subset, then scores both sides' deliverables with the SAME judge
model and writes mean ± sample SEM per task to parity_experiment.json.

Both sides:
    agent_model  = gemini-3.1-flash-lite-preview
    judge_model  = gemini-3.1-flash-lite-preview
    judge_temp   = 0.0

Usage:
    HUGGINGFACE_TOKEN=... GEMINI_API_KEY=... DAYTONA_API_KEY=... \\
    uv run --with google-genai --with pdfplumber --with openpyxl --with pandas \\
        python benchmarks/harvey-lab/_scripts/run_full_parity.py \\
        --runs 3 --tasks parity \\
        [--skip-upstream]   # only run BenchFlow side
        [--skip-benchflow]  # only run upstream side
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import shutil
import statistics
import string
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = SCRIPT_DIR.parent
REPO_ROOT = ADAPTER_DIR.parent.parent
HARVEY_REPO = Path("/tmp/harvey-labs")  # cloned earlier
HARVEY_REF = REPO_ROOT / ".ref" / "harvey-lab"
BENCHFLOW_TASKS = REPO_ROOT / ".ref" / "harvey-lab-benchflow-parity"
RUNS_DIR = REPO_ROOT / ".ref" / "harvey-lab-parity-runs"

JUDGE_MODEL = "gemini-3.1-flash-lite-preview"
AGENT_MODEL = "gemini-3.1-flash-lite-preview"
PARITY_SUBSET = (
    "corporate-ma/analyze-cim-deal-teaser/scenario-01",
    "insurance/compare-reinsurance-treaty-against-underlying-policy",
    "real-estate/draft-construction-contract",
    "intellectual-property/review-enterprise-saas-agreement",
    "employment-labor/draft-workplace-policy-memorandum",
)


def sanitize(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", raw.lower().strip()).strip("-")


# ── Shared judge ──────────────────────────────────────────────────────

JUDGE_PROMPT = string.Template("""\
You are evaluating a legal AI agent's work product against one rubric criterion.

## Task
$task_description

## Agent Output
$agent_output

## Criterion
**$criterion_title**

$match_criteria

## Instructions
- PASS if the output satisfies the criterion as described.
- FAIL otherwise.
- Reason on substance, not wording.

Reply with JSON only:

```json
{"verdict": "pass" | "fail", "reasoning": "Brief explanation"}
```
""")


def call_gemini(prompt: str, *, api_key: str, retries: int = 3) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    last: Exception | None = None
    for i in range(retries):
        try:
            r = client.models.generate_content(
                model=JUDGE_MODEL, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            return r.text or ""
        except Exception as exc:
            last = exc
            if i < retries - 1:
                time.sleep(2 ** i)
    raise RuntimeError(f"gemini failed: {last}")


def parse_verdict(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1])
                    except json.JSONDecodeError:
                        break
        break
    return {"verdict": "fail", "reasoning": f"unparseable: {text[:200]!r}"}


# ── File reading (same logic as the verifier) ─────────────────────────

def read_file_text(path: Path) -> str:
    s = path.suffix.lower()
    try:
        if s == ".docx":
            r = subprocess.run(
                ["pandoc", str(path), "-t", "markdown", "--wrap=none", "--track-changes=accept"],
                capture_output=True, text=True, timeout=30,
            )
            return r.stdout if r.returncode == 0 else f"(pandoc error: {r.stderr[:200]})"
        if s == ".xlsx":
            import pandas as pd
            sheets = pd.read_excel(path, sheet_name=None)
            return "\n".join(f"=== {n} ===\n{df.to_string(index=False)}" for n, df in sheets.items())
        if s == ".pdf":
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"(error: {exc})"


def collect_deliverables(out_dir: Path) -> dict[str, str]:
    keep_exts = {".docx", ".xlsx", ".pdf", ".md", ".txt", ".json", ".csv"}
    skip_dirs = {"documents", "node_modules", "__pycache__", ".git", ".venv"}
    skip_names = {"rubric.json", "package-lock.json"}
    out: dict[str, str] = {}
    if not out_dir.exists():
        return out
    for f in sorted(out_dir.rglob("*")):
        if not f.is_file() or f.name.startswith("."):
            continue
        rel = f.relative_to(out_dir)
        if any(p in skip_dirs for p in rel.parts):
            continue
        if f.name in skip_names or f.suffix.lower() not in keep_exts:
            continue
        out[str(rel)] = read_file_text(f)
    return out


def score_against_rubric(*, criteria: list[dict], title: str,
                          deliverables: dict[str, str], api_key: str) -> dict:
    blob = "\n\n".join(f"--- {n} ---\n{c[:15000]}" for n, c in deliverables.items()) \
           if deliverables else "(no deliverables produced)"
    results = []
    for c in criteria:
        prompt = JUDGE_PROMPT.safe_substitute(
            task_description=title, agent_output=blob,
            criterion_title=c["title"], match_criteria=c["match_criteria"],
        )
        try:
            v = parse_verdict(call_gemini(prompt, api_key=api_key))
            verdict = str(v.get("verdict", "fail")).lower()
        except Exception as exc:
            verdict = "fail"
            v = {"reasoning": f"judge error: {exc}"}
        results.append({"id": c["id"], "title": c["title"],
                        "verdict": verdict, "reasoning": str(v.get("reasoning", ""))[:300]})
    n = len(results)
    n_pass = sum(1 for r in results if r["verdict"] == "pass")
    return {"n": n, "n_pass": n_pass,
            "criterion_pass_rate": (n_pass / n) if n else 0.0,
            "all_pass": n > 0 and n_pass == n,
            "reward": 1.0 if (n > 0 and n_pass == n) else 0.0,
            "results": results}


# ── Upstream side: harvey-labs.harness.run ────────────────────────────

def run_upstream(task_id: str, run_idx: int) -> Path:
    """Invoke the upstream harness once. Returns the run output dir."""
    run_id = f"parity-{run_idx}-{datetime.now().strftime('%H%M%S')}"
    cmd = [
        sys.executable, "-m", "harness.run",
        "--task", task_id,
        "--model", f"google/{AGENT_MODEL}",
        "--temperature", "0.0",
        "--run-id", f"{task_id}/{run_id}",
        "--max-turns", "60",
    ]
    print(f"  upstream: {' '.join(cmd[-6:])}")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    r = subprocess.run(cmd, cwd=str(HARVEY_REPO), env=env, capture_output=True, text=True,
                       timeout=2400)
    if r.returncode != 0:
        print(f"    upstream failed: {r.stderr[-400:]}", file=sys.stderr)
    out_dir = HARVEY_REPO / "results" / task_id / run_id / "output"
    return out_dir


# ── BenchFlow side: Job.from_yaml(...).run() ──────────────────────────

async def run_benchflow_one_task(task_dir: Path, run_idx: int, jobs_root: Path) -> Path | None:
    """Run BenchFlow Job for a single task dir; return that task's agent output dir."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from benchflow.job import Job, JobConfig

    jobs_dir = jobs_root / f"run-{run_idx}"
    cfg = JobConfig(
        agent="gemini",
        model=AGENT_MODEL,
        environment="daytona",
        concurrency=1,
        sandbox_user=None,
        agent_env={"GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", "")},
    )
    job = Job(tasks_dir=task_dir.parent, jobs_dir=jobs_dir, config=cfg)
    # Restrict to just this task
    job._config.exclude_tasks = {p.name for p in task_dir.parent.iterdir()
                                  if p.is_dir() and p.name != task_dir.name}
    try:
        await job.run()
    except Exception as exc:
        print(f"    benchflow run failed: {exc}", file=sys.stderr)
        return None

    # Find the run's workspace output. Job writes under jobs_dir.
    candidates = list(jobs_dir.rglob("workspace/output"))
    return candidates[0] if candidates else None


# ── Stats ──────────────────────────────────────────────────────────────

def sample_sem(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return statistics.stdev(xs) / math.sqrt(len(xs))


def fmt(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.3f} (n=1)"
    return f"{statistics.fmean(xs):.3f} ± {sample_sem(xs):.3f}"


def matches(a: list[float], b: list[float]) -> bool:
    if not a or not b:
        return False
    return max(a) >= min(b) and max(b) >= min(a)


# ── Orchestrator ──────────────────────────────────────────────────────

async def main_async(args) -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        return 1

    if not args.skip_benchflow and not os.environ.get("DAYTONA_API_KEY"):
        print("ERROR: DAYTONA_API_KEY not set (required for BenchFlow side)", file=sys.stderr)
        return 1

    if not args.skip_upstream and shutil.which("podman") is None:
        print("ERROR: podman not installed (required for upstream harvey-labs harness)",
              file=sys.stderr)
        return 1

    task_ids = list(PARITY_SUBSET) if args.tasks == "parity" else args.tasks.split(",")

    runs_root = RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    runs_root.mkdir(parents=True, exist_ok=True)
    record: dict = {
        "experiment": "source-vs-adapter-agent-parity",
        "benchmark": "harvey-lab",
        "agent_model": AGENT_MODEL,
        "judge_model": JUDGE_MODEL,
        "runs_per_side": args.runs,
        "matching_criterion": "max(A) >= min(B) AND max(B) >= min(A)",
        "started_at": datetime.now(UTC).isoformat(),
        "tasks": [],
    }

    for tid in task_ids:
        src = HARVEY_REF / "tasks" / Path(*tid.split("/")) / "task.json"
        if not src.exists():
            print(f"  skip {tid}: not found")
            continue
        cfg = json.loads(src.read_text())
        title = cfg.get("title", tid)
        criteria = cfg.get("criteria") or []
        bf_task_dir = BENCHFLOW_TASKS / sanitize(tid)

        side_a_rewards: list[float] = []
        side_b_rewards: list[float] = []
        side_a_pass: list[float] = []
        side_b_pass: list[float] = []
        run_records: list[dict] = []

        for ri in range(1, args.runs + 1):
            print(f"\n=== {tid} run {ri}/{args.runs} ===")
            run_record = {"run": ri}

            # Side A: upstream
            if not args.skip_upstream:
                t0 = time.time()
                out_a = run_upstream(tid, ri)
                run_record["upstream_seconds"] = time.time() - t0
                deliv_a = collect_deliverables(out_a)
                print(f"  upstream produced {len(deliv_a)} file(s)")
                a = score_against_rubric(criteria=criteria, title=title,
                                          deliverables=deliv_a, api_key=api_key)
                print(f"  side A reward={a['reward']:.1f} pass_rate={a['criterion_pass_rate']:.1%}")
                side_a_rewards.append(a["reward"])
                side_a_pass.append(a["criterion_pass_rate"])
                run_record["side_A_upstream"] = {k: a[k] for k in
                    ("n", "n_pass", "criterion_pass_rate", "all_pass", "reward")}

            # Side B: benchflow
            if not args.skip_benchflow:
                t0 = time.time()
                out_b = await run_benchflow_one_task(bf_task_dir, ri, runs_root)
                run_record["benchflow_seconds"] = time.time() - t0
                deliv_b = collect_deliverables(out_b) if out_b else {}
                print(f"  benchflow produced {len(deliv_b)} file(s)")
                b = score_against_rubric(criteria=criteria, title=title,
                                          deliverables=deliv_b, api_key=api_key)
                print(f"  side B reward={b['reward']:.1f} pass_rate={b['criterion_pass_rate']:.1%}")
                side_b_rewards.append(b["reward"])
                side_b_pass.append(b["criterion_pass_rate"])
                run_record["side_B_benchflow"] = {k: b[k] for k in
                    ("n", "n_pass", "criterion_pass_rate", "all_pass", "reward")}

            run_records.append(run_record)

        record["tasks"].append({
            "task_id": tid, "n_criteria": len(criteria), "runs": run_records,
            "metrics": [
                {
                    "metric": "all-pass-reward",
                    "side_A_upstream_runs": side_a_rewards,
                    "side_B_benchflow_runs": side_b_rewards,
                    "side_A_summary": fmt(side_a_rewards),
                    "side_B_summary": fmt(side_b_rewards),
                    "match": matches(side_a_rewards, side_b_rewards),
                },
                {
                    "metric": "criterion-pass-rate",
                    "side_A_upstream_runs": side_a_pass,
                    "side_B_benchflow_runs": side_b_pass,
                    "side_A_summary": fmt(side_a_pass),
                    "side_B_summary": fmt(side_b_pass),
                    "match": matches(side_a_pass, side_b_pass),
                },
            ],
        })

        # Save after every task so we don't lose progress.
        (ADAPTER_DIR / "parity_experiment.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )

    # Roll-up.
    per_metric: dict = {}
    for t in record["tasks"]:
        for m in t["metrics"]:
            slot = per_metric.setdefault(m["metric"], {"matched": 0, "total": 0})
            slot["total"] += 1
            slot["matched"] += int(m["match"])
    record["summary"] = per_metric
    record["finished_at"] = datetime.now(UTC).isoformat()
    (ADAPTER_DIR / "parity_experiment.json").write_text(json.dumps(record, indent=2) + "\n")
    for k, v in per_metric.items():
        print(f"\n{k}: {v['matched']}/{v['total']} tasks matched")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--tasks", default="parity",
                   help="'parity' (the 5-task subset) or comma-separated task IDs")
    p.add_argument("--skip-upstream", action="store_true",
                   help="Skip side A (upstream harvey-labs harness)")
    p.add_argument("--skip-benchflow", action="store_true",
                   help="Skip side B (BenchFlow Daytona Job)")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
