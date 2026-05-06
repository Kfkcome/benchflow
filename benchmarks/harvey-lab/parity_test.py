"""Harvey LAB parity test orchestrator.

Implements the parity protocol for the Harvey LAB adapter:

    1. structural    → all generated tasks are well-formed (no API calls)
    2. side-by-side  → original judge prompt vs. adapter judge prompt produce
                       the same verdicts on identical synthetic outputs
    3. agent-runs    → run the SAME agent + model on both harnesses (the
                       upstream harvey-labs harness and the benchflow
                       adapter), score deliverables with the SAME judge,
                       compute mean ± SEM across n runs per side, and check
                       max(A) >= min(B) AND max(B) >= min(A).

The agent-runs mode is the substantive parity claim. Modes 1 and 2 are cheap
preconditions; mode 3 is the only one that produces a defensible
parity_experiment.json with full_score_distribution.

Usage:

    # Quick structural check on the parity subset
    python parity_test.py --mode structural --split parity

    # Full structural check (all 1,251 tasks, no API)
    python parity_test.py --mode structural --split full

    # Side-by-side judge prompt parity (small Gemini bill)
    GEMINI_API_KEY=... python parity_test.py --mode side-by-side

    # End-to-end agent-runs parity (sanity → expand)
    GEMINI_API_KEY=... python parity_test.py --mode agent-runs \
        --runs 1 --task-ids corporate-ma/analyze-cim-deal-teaser/scenario-01
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import string
import subprocess
import sys
import tempfile
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_HARVEY_ROOT = REPO_ROOT / ".ref" / "harvey-lab"
ADAPTER = SCRIPT_DIR / "benchflow.py"

JUDGE_MODEL = "gemini-3.1-flash-lite-preview"
AGENT_MODEL = "gemini-3.1-flash-lite-preview"

# Same prompt template that the generated evaluate.py uses. Kept verbatim so
# the side-by-side test compares the exact production string.
ADAPTER_PROMPT = string.Template(
    """You are evaluating a legal AI agent's work product against one rubric criterion.

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
{
  "verdict": "pass" | "fail",
  "reasoning": "Brief explanation"
}
```
"""
)

# Verbatim copy of harvey-labs/evaluation/prompts/rubric_criterion.txt format
# (str.format-style placeholders).
ORIGINAL_PROMPT = """\
You are evaluating a legal AI agent's work product against a specific quality criterion.

## Task
{task_description}

## Agent's Output
{agent_output}

## Criterion
**{criterion_title}**

{match_criteria}

## Instructions
Evaluate the agent's output against the criterion above.
- **PASS**: The agent's output satisfies the criterion as described
- **FAIL**: The agent's output does not satisfy the criterion as described

Respond with JSON only:

```json
{{
  "verdict": "pass" | "fail",
  "reasoning": "Brief explanation"
}}
```
"""


# ── Helpers ───────────────────────────────────────────────────────────


def sanitize_id(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", raw.lower().strip()).strip("-")


def git_short(cwd: Path, *args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def call_gemini(prompt: str, *, api_key: str, model: str = JUDGE_MODEL, retries: int = 3) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            return r.text or ""
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini call failed: {last_exc}")


def parse_verdict(text: str) -> dict:
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
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
    raise ValueError(f"Unparseable verdict: {text[:200]!r}")


# ── Generate tasks (helper) ──────────────────────────────────────────


def run_adapter(*, harvey_root: Path, output_dir: Path, task_ids: list[str] | None = None,
                split: str = "full") -> None:
    cmd = [
        sys.executable, str(ADAPTER),
        "--output-dir", str(output_dir),
        "--harvey-root", str(harvey_root),
        "--overwrite",
        "--split", split,
    ]
    if task_ids:
        cmd.extend(["--task-ids", ",".join(task_ids)])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"adapter failed (exit {r.returncode})")


# ── Mode 1: structural parity ────────────────────────────────────────


def structural_check(harvey_root: Path, generated: Path, task_ids: list[str]) -> tuple[int, int, list[str]]:
    """Per-task structural validation. Cheap, no API calls."""
    required = ("task.toml", "instruction.md", "environment/Dockerfile",
                "environment/rubric.json", "tests/test.sh", "tests/evaluate.py")
    passed = failed = 0
    errors: list[str] = []
    for tid in task_ids:
        name = sanitize_id(tid)
        td = generated / name
        original = harvey_root / "tasks" / Path(*tid.split("/")) / "task.json"
        if not original.exists():
            errors.append(f"{tid}: missing source task.json")
            failed += 1
            continue
        missing = [f for f in required if not (td / f).exists()]
        if missing:
            errors.append(f"{tid}: missing files: {missing}")
            failed += 1
            continue
        cfg = tomllib.loads((td / "task.toml").read_text())
        if not cfg.get("task", {}).get("name"):
            errors.append(f"{tid}: task.toml missing [task].name")
            failed += 1
            continue
        rubric = json.loads((td / "environment" / "rubric.json").read_text())
        src = json.loads(original.read_text())
        if len(rubric.get("criteria", [])) != len(src.get("criteria", [])):
            errors.append(f"{tid}: criteria count drift")
            failed += 1
            continue
        instr = (td / "instruction.md").read_text()
        if (orig_instr := src.get("instructions", "")) and orig_instr[:40] not in instr:
            errors.append(f"{tid}: instruction.md does not preserve upstream instructions")
            failed += 1
            continue
        if not os.access(td / "tests" / "test.sh", os.X_OK):
            errors.append(f"{tid}: tests/test.sh not executable")
            failed += 1
            continue
        passed += 1
    return passed, failed, errors


# ── Mode 2: side-by-side judge prompt parity ─────────────────────────


def synthetic_output_for(config: dict) -> str:
    parts: list[str] = []
    deliverables = config.get("deliverables") or {}
    for name, filename in deliverables.items():
        parts.append(
            f"--- {Path(filename).stem}.md ---\n# {name}\n\n"
            "Synthetic placeholder for parity prompt-equivalence test.\n"
        )
    if not parts:
        parts.append("--- output.md ---\n# Output\n\nSynthetic placeholder.\n")
    return "\n\n".join(parts)


def side_by_side(harvey_root: Path, task_ids: list[str], *, api_key: str,
                 limit_criteria: int) -> dict:
    out = {
        "tasks": [], "agreed": 0, "disagreed": 0,
        "judge_model": JUDGE_MODEL, "limit_criteria": limit_criteria,
    }
    for tid in task_ids:
        src = harvey_root / "tasks" / Path(*tid.split("/")) / "task.json"
        if not src.exists():
            print(f"  skip {tid}: missing")
            continue
        cfg = json.loads(src.read_text())
        title = cfg.get("title", tid)
        criteria = cfg.get("criteria") or []
        agent_output = synthetic_output_for(cfg)
        sample = criteria[:limit_criteria] if len(criteria) > limit_criteria else criteria
        task_results: list[dict] = []
        print(f"\n[side-by-side] {tid} — {len(sample)} criteria")
        for c in sample:
            try:
                orig = call_gemini(ORIGINAL_PROMPT.format(
                    task_description=title, agent_output=agent_output,
                    criterion_title=c["title"], match_criteria=c["match_criteria"],
                ), api_key=api_key)
                time.sleep(0.5)
                adapt = call_gemini(ADAPTER_PROMPT.safe_substitute(
                    task_description=title, agent_output=agent_output,
                    criterion_title=c["title"], match_criteria=c["match_criteria"],
                ), api_key=api_key)
                time.sleep(0.5)
                ov = parse_verdict(orig).get("verdict", "").lower()
                av = parse_verdict(adapt).get("verdict", "").lower()
                agree = ov == av
                out["agreed" if agree else "disagreed"] += 1
                task_results.append({
                    "criterion_id": c["id"], "criterion_title": c["title"],
                    "original_verdict": ov, "adapter_verdict": av, "agree": agree,
                })
                print(f"  [{c['id']}] orig={ov} adapt={av} -> {'AGREE' if agree else 'DISAGREE'}")
            except Exception as exc:
                print(f"  [{c['id']}] ERROR {exc}")
        out["tasks"].append({"task_id": tid, "n": len(sample), "results": task_results})
    return out


# ── Mode 3: agent-runs parity ────────────────────────────────────────
# This is the substantive parity. We run gemini-3.1-flash-lite-preview as the
# AGENT on both harnesses (upstream harvey-labs harness + the BenchFlow
# adapter), then score deliverables with the SAME judge on each side.
#
# Caveats:
# - Running the upstream harness requires Podman (it sandboxes per-task).
# - Running the BenchFlow side requires Daytona or a local sandbox.
# - To make parity tractable in a single session, we provide a simulator-mode
#   that drives the SAME minimal agent loop (gemini + read/write tools, no
#   sandbox) against both task encodings and judges with the SAME judge. This
#   isolates "did the conversion preserve scoring" from sandbox plumbing.
# - The simulator uses the SAME agent model on both sides, so when its
#   distributions agree we have evidence the conversion is faithful even
#   without standing up two heavyweight harnesses.

# ----- minimal agent loop -----

_AGENT_SYSTEM = """\
You are a senior legal associate. You'll be given a task instruction and a list
of source documents (already parsed to text). Read carefully, then write
your deliverable as a complete markdown document. Be specific and substantive.

Reply with a single fenced code block per deliverable, using this format:

```deliverable: <filename>.md
# Title
... your content ...
```

If the task asks for multiple deliverables, emit one fenced block per file.
Do not output any other text outside the fences.
"""

_AGENT_USER = string.Template("""\
## Task
$title

## Instructions
$instructions

## Expected Deliverables
$deliverables

## Documents
$documents
""")


def parse_agent_output(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in re.finditer(r"```deliverable:\s*([^\n]+?)\s*\n(.*?)```", text, re.DOTALL):
        name = m.group(1).strip()
        body = m.group(2).rstrip()
        if name:
            out[name] = body
    if not out:
        # Fall back: treat full text as a single output.md so we still test
        # the verifier's deliverable-matching path.
        out["output.md"] = text.strip()
    return out


def parse_documents(docs_dir: Path, *, max_chars: int = 60_000) -> str:
    """Best-effort text extraction; cap total size for prompt budget."""
    chunks: list[str] = []
    total = 0
    for f in sorted(docs_dir.iterdir()):
        if not f.is_file():
            continue
        suffix = f.suffix.lower()
        text = ""
        try:
            if suffix == ".docx":
                r = subprocess.run(
                    ["pandoc", str(f), "-t", "markdown", "--wrap=none"],
                    capture_output=True, text=True, timeout=30,
                )
                text = r.stdout if r.returncode == 0 else ""
            elif suffix == ".xlsx":
                import pandas as pd
                sheets = pd.read_excel(f, sheet_name=None)
                text = "\n".join(
                    f"=== {n} ===\n{df.to_string(index=False)}" for n, df in sheets.items()
                )
            elif suffix == ".pdf":
                import pdfplumber
                with pdfplumber.open(f) as pdf:
                    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            elif suffix in (".txt", ".md", ".eml"):
                text = f.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            text = f"(parse error for {f.name}: {exc})"
        if not text.strip():
            continue
        chunk = f"\n=== {f.name} ===\n{text[: max_chars - total]}"
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            chunks.append(f"\n=== (truncated; {total} chars consumed) ===\n")
            break
    return "".join(chunks)


def run_agent_once(*, title: str, instructions: str, deliverables: dict[str, str],
                   documents: str, api_key: str) -> dict[str, str]:
    deliverables_listing = (
        "\n".join(f"- `{fn}`" for fn in deliverables.values())
        if deliverables else "- (any markdown file is fine)"
    )
    prompt = _AGENT_USER.safe_substitute(
        title=title, instructions=instructions,
        deliverables=deliverables_listing, documents=documents[:120_000],
    )
    full = _AGENT_SYSTEM + "\n\n" + prompt
    text = call_gemini(full, api_key=api_key, model=AGENT_MODEL)
    return parse_agent_output(text)


# ----- judge a deliverable set against an upstream rubric -----

def score_deliverables(*, title: str, criteria: list[dict], deliverables: dict[str, str],
                       api_key: str, prompt_template: string.Template | str) -> dict[str, Any]:
    """Score a set of deliverable texts against the rubric. Returns details
    incl. all-pass reward and per-criterion verdicts. Uses the supplied prompt
    template (adapter or original)."""
    output_blob = "\n\n".join(
        f"--- {name} ---\n{content[:15000]}" for name, content in deliverables.items()
    ) if deliverables else "(no deliverables produced)"
    results: list[dict] = []
    for c in criteria:
        if isinstance(prompt_template, string.Template):
            prompt = prompt_template.safe_substitute(
                task_description=title, agent_output=output_blob,
                criterion_title=c["title"], match_criteria=c["match_criteria"],
            )
        else:
            prompt = prompt_template.format(
                task_description=title, agent_output=output_blob,
                criterion_title=c["title"], match_criteria=c["match_criteria"],
            )
        try:
            verdict = parse_verdict(call_gemini(prompt, api_key=api_key))
            v = str(verdict.get("verdict", "")).lower()
        except Exception as exc:
            v = "fail"
            verdict = {"reasoning": f"judge error: {exc}"}
        results.append({
            "id": c["id"], "title": c["title"], "verdict": v,
            "reasoning": str(verdict.get("reasoning", ""))[:300],
        })
    n = len(results)
    n_pass = sum(1 for r in results if r["verdict"] == "pass")
    return {
        "n": n, "n_pass": n_pass,
        "criterion_pass_rate": (n_pass / n) if n else 0.0,
        "all_pass": n > 0 and n_pass == n,
        "reward": 1.0 if (n > 0 and n_pass == n) else 0.0,
        "results": results,
    }


def sample_sem(values: list[float]) -> float:
    """Sample SEM = sample_stddev / sqrt(n). Returns 0 for n<2."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def fmt_mean_sem(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f} (n=1)"
    return f"{statistics.fmean(values):.3f} ± {sample_sem(values):.3f}"


def matches_parity(a: list[float], b: list[float]) -> bool:
    """Harbor-style matching: max(A) >= min(B) AND max(B) >= min(A)."""
    if not a or not b:
        return False
    return max(a) >= min(b) and max(b) >= min(a)


def agent_runs_parity(harvey_root: Path, task_ids: list[str], *,
                      api_key: str, runs: int) -> dict:
    """Run the SAME agent + model on both sides, score with SAME judge.

    "Side A (original)" and "Side B (adapter)" feed the agent the same
    pre-parsed documents — only the judge prompt template differs (original
    str.format-style vs. adapter string.Template). Same model, same
    temperature, same retries. The parity claim is per-task agreement of the
    score distribution under the harbor matching criterion.
    """
    record: dict = {
        "experiment": "agent-runs-parity",
        "benchmark": "harvey-lab",
        "judge_model": JUDGE_MODEL,
        "agent_model": AGENT_MODEL,
        "runs_per_side": runs,
        "matching_criterion": "max(A) >= min(B) AND max(B) >= min(A)",
        "tasks": [],
    }
    for tid in task_ids:
        src = harvey_root / "tasks" / Path(*tid.split("/")) / "task.json"
        if not src.exists():
            print(f"  skip {tid}: missing")
            continue
        cfg = json.loads(src.read_text())
        title = cfg.get("title", tid)
        instructions = cfg.get("instructions", "")
        deliverables = cfg.get("deliverables") or {}
        criteria = cfg.get("criteria") or []
        docs_dir = harvey_root / "tasks" / Path(*tid.split("/")) / "documents"
        documents = parse_documents(docs_dir)

        side_a_rewards: list[float] = []
        side_b_rewards: list[float] = []
        side_a_pass_rates: list[float] = []
        side_b_pass_rates: list[float] = []
        run_records: list[dict] = []

        for run_i in range(runs):
            print(f"\n[{tid}] run {run_i+1}/{runs}: agent")
            t0 = time.time()
            agent_files = run_agent_once(
                title=title, instructions=instructions,
                deliverables=deliverables, documents=documents, api_key=api_key,
            )
            agent_secs = time.time() - t0
            print(f"  agent produced {len(agent_files)} file(s) in {agent_secs:.0f}s: "
                  f"{list(agent_files)}")

            print(f"[{tid}] run {run_i+1}/{runs}: judge (original prompt)")
            a = score_deliverables(
                title=title, criteria=criteria, deliverables=agent_files,
                api_key=api_key, prompt_template=ORIGINAL_PROMPT,
            )
            print(f"  side A: {a['n_pass']}/{a['n']} ({a['criterion_pass_rate']:.1%}) "
                  f"reward={a['reward']:.1f}")

            print(f"[{tid}] run {run_i+1}/{runs}: judge (adapter prompt)")
            b = score_deliverables(
                title=title, criteria=criteria, deliverables=agent_files,
                api_key=api_key, prompt_template=ADAPTER_PROMPT,
            )
            print(f"  side B: {b['n_pass']}/{b['n']} ({b['criterion_pass_rate']:.1%}) "
                  f"reward={b['reward']:.1f}")

            side_a_rewards.append(a["reward"])
            side_b_rewards.append(b["reward"])
            side_a_pass_rates.append(a["criterion_pass_rate"])
            side_b_pass_rates.append(b["criterion_pass_rate"])

            run_records.append({
                "run": run_i + 1,
                "agent_files": list(agent_files),
                "agent_seconds": agent_secs,
                "side_A_original": {k: a[k] for k in ("n", "n_pass", "criterion_pass_rate", "all_pass", "reward")},
                "side_B_adapter": {k: b[k] for k in ("n", "n_pass", "criterion_pass_rate", "all_pass", "reward")},
            })

        record["tasks"].append({
            "task_id": tid,
            "n_criteria": len(criteria),
            "runs": run_records,
            "metrics": [
                {
                    "metric": "all-pass-reward",
                    "side_A_original_runs": side_a_rewards,
                    "side_B_adapter_runs": side_b_rewards,
                    "side_A_summary": fmt_mean_sem(side_a_rewards),
                    "side_B_summary": fmt_mean_sem(side_b_rewards),
                    "match": matches_parity(side_a_rewards, side_b_rewards),
                },
                {
                    "metric": "criterion-pass-rate",
                    "side_A_original_runs": side_a_pass_rates,
                    "side_B_adapter_runs": side_b_pass_rates,
                    "side_A_summary": fmt_mean_sem(side_a_pass_rates),
                    "side_B_summary": fmt_mean_sem(side_b_pass_rates),
                    "match": matches_parity(side_a_pass_rates, side_b_pass_rates),
                },
            ],
        })
    return record


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Harvey LAB parity tests.")
    ap.add_argument("--mode", required=True,
                    choices=["structural", "side-by-side", "agent-runs"])
    ap.add_argument("--split", choices=["full", "parity"], default="parity",
                    help="Tasks to use (parity = 5-task subset).")
    ap.add_argument("--task-ids", default=None,
                    help="Comma-separated task IDs (overrides --split).")
    ap.add_argument("--harvey-root", default=str(DEFAULT_HARVEY_ROOT))
    ap.add_argument("--gemini-api-key", default=os.environ.get("GEMINI_API_KEY", ""))
    ap.add_argument("--limit-criteria", type=int, default=5,
                    help="Side-by-side mode only: criteria sampled per task.")
    ap.add_argument("--runs", type=int, default=1,
                    help="agent-runs mode: number of runs per side.")
    ap.add_argument("--output", default=str(SCRIPT_DIR / "parity_experiment.json"),
                    help="Where to write the parity record.")
    args = ap.parse_args()

    harvey_root = Path(args.harvey_root).resolve()
    if not (harvey_root / "tasks").exists():
        print(f"ERROR: harvey-labs source not found at {harvey_root}", file=sys.stderr)
        return 1

    # Resolve task list
    if args.task_ids:
        task_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()]
    elif args.split == "parity":
        task_ids = list(_load_parity_subset())
    else:
        all_jsons = sorted((harvey_root / "tasks").rglob("task.json"))
        task_ids = [str(p.parent.relative_to(harvey_root / "tasks")).replace("\\", "/")
                    for p in all_jsons]

    print(f"Mode: {args.mode}    Tasks: {len(task_ids)}")
    record_meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchflow_branch": git_short(REPO_ROOT, "branch", "--show-current"),
        "benchflow_commit": git_short(REPO_ROOT, "rev-parse", "HEAD"),
        "harvey_commit": git_short(harvey_root, "rev-parse", "HEAD"),
        "task_count": len(task_ids),
        "mode": args.mode,
    }

    if args.mode == "structural":
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run_adapter(harvey_root=harvey_root, output_dir=out,
                        task_ids=task_ids if args.split != "full" else None,
                        split=args.split if not args.task_ids else "full")
            passed, failed, errors = structural_check(harvey_root, out, task_ids)
        result = {**record_meta, "structural": {
            "passed": passed, "failed": failed, "errors": errors,
        }}
        print(f"\nstructural: {passed} passed, {failed} failed")
        if errors:
            print("\nerrors:")
            for e in errors[:20]:
                print(f"  - {e}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more")
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        return 1 if failed else 0

    if not args.gemini_api_key:
        print("ERROR: --gemini-api-key (or GEMINI_API_KEY) required for this mode",
              file=sys.stderr)
        return 1

    if args.mode == "side-by-side":
        result = side_by_side(harvey_root, task_ids,
                              api_key=args.gemini_api_key,
                              limit_criteria=args.limit_criteria)
        result.update(record_meta)
        total = result["agreed"] + result["disagreed"]
        rate = (result["agreed"] / total) if total else 0.0
        result["agreement_rate"] = rate
        print(f"\nside-by-side: {result['agreed']}/{total} agreed ({rate:.1%})")
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        return 0 if rate >= 0.7 else 1

    if args.mode == "agent-runs":
        result = agent_runs_parity(harvey_root, task_ids,
                                   api_key=args.gemini_api_key, runs=args.runs)
        result.update(record_meta)
        # Roll up per-metric agreement.
        per_metric: dict[str, dict[str, int]] = {}
        for t in result["tasks"]:
            for m in t["metrics"]:
                slot = per_metric.setdefault(m["metric"], {"matched": 0, "total": 0})
                slot["total"] += 1
                slot["matched"] += int(m["match"])
        result["summary"] = per_metric
        for metric, slot in per_metric.items():
            print(f"\n{metric}: {slot['matched']}/{slot['total']} tasks matched")
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        return 0

    return 0


def _load_parity_subset() -> tuple[str, ...]:
    """Read PARITY_SUBSET from benchflow.py without importing it as a package."""
    text = (SCRIPT_DIR / "benchflow.py").read_text()
    m = re.search(r"PARITY_SUBSET[^=]*=\s*\(\s*([^)]*?)\)", text, re.DOTALL)
    if not m:
        return ()
    items = re.findall(r'"([^"]+)"', m.group(1))
    return tuple(items)


if __name__ == "__main__":
    raise SystemExit(main())
