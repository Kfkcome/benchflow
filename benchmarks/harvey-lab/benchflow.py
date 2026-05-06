"""Harvey LAB → BenchFlow benchmark adapter.

Converts the upstream Harvey LAB benchmark (https://github.com/harveyai/harvey-labs)
into the BenchFlow task format. Each Harvey LAB task ships a `task.json`
describing instructions, deliverables, and a rubric of pass/fail criteria
graded by an LLM judge. This adapter emits, per task:

    <task-id>/
        task.toml                 # registry name, timeouts, resources
        instruction.md            # agent prompt
        environment/
            Dockerfile            # python:3.13-slim with parsing libs
            documents/            # input docs (.docx/.xlsx/.pdf/.eml/...)
            rubric.json           # rubric criteria for the verifier
        tests/
            test.sh               # verifier entry point
            evaluate.py           # LLM-as-judge using Gemini

CLI:
    benchflow.py --output-dir DIR --harvey-root PATH
                 [--limit N] [--task-ids ID,ID] [--split full|parity]
                 [--overwrite]

Notes:
- Harvey LAB does not ship oracle solutions, so no `solution/` is emitted.
- The verifier scores 1.0 only when every rubric criterion passes (all-pass);
  the per-criterion pass rate is recorded in `evaluation_details.json` as a
  diagnostic alongside the binary reward.
- Document parsing inside the verifier uses pandoc / pdfplumber / openpyxl,
  installed in the task image so test.sh can run offline once built.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import textwrap
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────

# 5-task cross-practice subset for cheap, reproducible parity runs.
PARITY_SUBSET: tuple[str, ...] = (
    "corporate-ma/analyze-cim-deal-teaser/scenario-01",
    "insurance/compare-reinsurance-treaty-against-underlying-policy",
    "real-estate/draft-construction-contract",
    "intellectual-property/review-enterprise-saas-agreement",
    "employment-labor/draft-workplace-policy-memorandum",
)

# Difficulty thresholds (criteria count). Tuned against Harvey LAB's empirical
# distribution: 25th percentile ≈ 35, 75th percentile ≈ 80.
_EASY_MAX = 35
_HARD_MIN = 80

# Files copied verbatim into environment/documents/. Anything else is skipped.
_DOC_EXTS = {".docx", ".xlsx", ".pptx", ".pdf", ".txt", ".md", ".eml", ".csv"}

# Verifier reads these inside the container.
_DELIVERABLE_EXTS = {".docx", ".xlsx", ".pptx", ".pdf", ".md", ".txt", ".json", ".csv"}


def sanitize_task_id(raw: str) -> str:
    """Lowercase, replace non-alphanumerics with hyphens, collapse runs."""
    sanitized = re.sub(r"[^a-z0-9]+", "-", raw.lower().strip())
    return sanitized.strip("-")


def infer_difficulty(criteria_count: int) -> str:
    if criteria_count <= _EASY_MAX:
        return "easy"
    if criteria_count >= _HARD_MIN:
        return "hard"
    return "medium"


# ── File generators ──────────────────────────────────────────────────

def render_task_toml(*, task_name: str, work_type: str, tags: list[str], criteria_count: int) -> str:
    difficulty = infer_difficulty(criteria_count)
    # Generous bounds — harvey-labs uses 1800s timeouts and large rubrics
    # mean many judge calls.
    agent_timeout = max(1800, criteria_count * 30)
    verifier_timeout = max(300, criteria_count * 15)
    tag_list = ", ".join(f'"{t}"' for t in tags)
    return textwrap.dedent(f"""\
        version = "1.0"

        [task]
        name = "{task_name}"

        [metadata]
        author_name = "Harvey AI"
        author_email = "labs@harvey.ai"
        difficulty = "{difficulty}"
        category = "legal-{work_type}"
        tags = [{tag_list}]
        upstream = "https://github.com/harveyai/harvey-labs"

        [agent]
        timeout_sec = {agent_timeout}

        [verifier]
        timeout_sec = {verifier_timeout}

        [verifier.env]
        GEMINI_API_KEY = "${{GEMINI_API_KEY}}"

        [environment]
        build_timeout_sec = 900
        cpus = 1
        memory_mb = 4096
        storage_mb = 20480
    """)


def render_instruction_md(*, title: str, instructions: str, deliverables: dict[str, str]) -> str:
    parts = [f"# {title}", "", instructions.rstrip(), ""]
    if deliverables:
        parts.extend(["## Expected Deliverables", ""])
        for filename in deliverables.values():
            parts.append(f"- `{filename}`")
        parts.append("")
    parts.extend([
        "## Workspace",
        "",
        "- Source documents are in `documents/` (read-only).",
        "- Write deliverables to `/app/output/` using the exact filenames listed above.",
        "- Use pandoc / pdfplumber / openpyxl (already installed) to read .docx, .pdf, .xlsx files.",
        "",
    ])
    return "\n".join(parts)


def render_dockerfile() -> str:
    return textwrap.dedent("""\
        FROM python:3.13-slim

        RUN apt-get update -qq \\
            && apt-get install -y -qq pandoc curl ca-certificates \\
            && rm -rf /var/lib/apt/lists/*

        RUN pip install --no-cache-dir \\
            pdfplumber==0.11.4 \\
            openpyxl==3.1.5 \\
            python-docx==1.1.2 \\
            python-pptx==1.0.2 \\
            markitdown==0.0.1a3 \\
            pandas==2.2.3 \\
            google-genai==0.8.0

        WORKDIR /app

        COPY documents/ /app/documents/
        COPY rubric.json /app/rubric.json

        RUN mkdir -p /app/output /logs/verifier /logs/agent /logs/artifacts
    """)


def render_test_sh() -> str:
    return textwrap.dedent("""\
        #!/bin/bash
        # BenchFlow verifier — invokes the Gemini-judged rubric scorer
        # and writes a single float to /logs/verifier/reward.txt.
        set -e

        python3 /tests/evaluate.py \\
            --rubric /app/rubric.json \\
            --output-dir /app/output \\
            --fallback-output-dir /app \\
            --reward-file /logs/verifier/reward.txt \\
            --details-file /logs/verifier/evaluation_details.json
    """)


# ── Verifier (LLM-as-judge) ─────────────────────────────────────────
#
# Emitted as a string into each generated task. Kept as a single self-contained
# script so the container does not need to import benchflow at runtime.

_EVALUATE_PY = '''\
"""Harvey LAB rubric judge.

Reads rubric.json (criteria + title), collects the agent's deliverable files,
and grades each criterion via Gemini. Writes:

    <reward-file>           single float, 1.0 only when every criterion passes
    <details-file>          per-criterion verdicts and reasoning
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pdfplumber

JUDGE_MODEL = (
    os.environ.get("BENCHFLOW_JUDGE_MODEL")
    or os.environ.get("GEMINI_MODEL")
    or "gemini-3.1-flash-lite-preview"
)

SKIP_DIR_PARTS = {"documents", "node_modules", ".npm", "__pycache__", ".git", ".venv", "venv"}
SKIP_EXTS = {".lock", ".map"}
SKIP_NAMES = {"package-lock.json", "rubric.json"}
DELIVERABLE_EXTS = {".docx", ".xlsx", ".pptx", ".pdf", ".md", ".txt", ".json", ".csv"}

VERDICT_PROMPT = string.Template(
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


def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".docx":
            res = subprocess.run(
                ["pandoc", str(path), "-t", "markdown", "--wrap=none", "--track-changes=accept"],
                capture_output=True, text=True, timeout=30,
            )
            return res.stdout if res.returncode == 0 else f"(pandoc error: {res.stderr[:200]})"
        if suffix == ".xlsx":
            sheets = pd.read_excel(path, sheet_name=None)
            chunks = []
            for name, df in sheets.items():
                chunks.append(f"=== Sheet: {name} ===")
                chunks.append(df.to_string(index=False))
            return "\\n".join(chunks)
        if suffix == ".pptx":
            from markitdown import MarkItDown
            return MarkItDown().convert(str(path)).text_content
        if suffix == ".pdf":
            chunks = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    if (text := page.extract_text()):
                        chunks.append(text)
                    for table in page.extract_tables():
                        for row in table:
                            chunks.append("\\t".join(c or "" for c in row))
                        chunks.append("")
            return "\\n".join(chunks)
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"(error reading {path.name}: {exc})"


def call_gemini(prompt: str, *, retries: int = 3) -> str:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) must be set in the verifier env.")
    client = genai.Client(api_key=api_key)
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            return resp.text or ""
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Gemini call failed after {retries} attempts: {exc}")
    raise RuntimeError("unreachable")


def parse_verdict(text: str) -> dict:
    fence = re.search(r"```(?:json)?\\s*\\n?(.*?)\\n?```", text, re.DOTALL)
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
    raise ValueError(f"No JSON verdict in response: {text[:200]!r}")


def stem_words(name: str) -> set[str]:
    stem = Path(name).stem.lower().replace("-", " ").replace("_", " ")
    return {w for w in stem.split() if w}


def pick_relevant(expected: list[str], texts: dict[str, str]) -> dict[str, str]:
    """Map a criterion's expected deliverables to actual files written."""
    if not expected:
        return texts
    relevant: dict[str, str] = {}
    used: set[str] = set()
    actual = list(texts)
    for exp in expected:
        exp_path = Path(exp)
        exp_name = exp_path.name.lower()
        exp_stem = exp_path.stem.lower()
        exp_ext = exp_path.suffix.lower()
        # Exact basename match.
        for a in actual:
            if a in used:
                continue
            ap = Path(a)
            if a.lower() == exp.lower() or ap.name.lower() == exp_name:
                relevant[a] = texts[a]; used.add(a); break
        else:
            # Same stem (e.g., generated .md stand-in for binary deliverable).
            for a in actual:
                if a in used:
                    continue
                if Path(a).stem.lower() == exp_stem:
                    relevant[a] = texts[a]; used.add(a); break
            else:
                # Fall back to extension match if there is exactly one candidate.
                ext_matches = [
                    a for a in actual
                    if a not in used and Path(a).suffix.lower() == exp_ext
                ]
                if len(ext_matches) == 1:
                    a = ext_matches[0]
                    relevant[a] = texts[a]; used.add(a)
                    continue
                # Word-overlap heuristic.
                exp_words = stem_words(exp)
                best, score = None, 0
                for a in ext_matches:
                    s = len(exp_words & stem_words(a))
                    if s > score:
                        best, score = a, s
                if best and score > 0:
                    relevant[best] = texts[best]; used.add(best)
    return relevant


def collect_deliverables(dirs: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.rglob("*")):
            if not f.is_file() or f.name.startswith("."):
                continue
            rel = f.relative_to(d)
            if any(part in SKIP_DIR_PARTS for part in rel.parts):
                continue
            if f.name in SKIP_NAMES or f.suffix.lower() in SKIP_EXTS:
                continue
            if f.suffix.lower() not in DELIVERABLE_EXTS:
                continue
            key = str(rel)
            out.setdefault(key, read_text(f))
    return out


def judge(criterion: dict, title: str, texts: dict[str, str]) -> dict:
    relevant = pick_relevant(criterion.get("deliverables", []), texts)
    if not relevant:
        return {
            "id": criterion["id"], "title": criterion["title"],
            "verdict": "fail", "reasoning": "No matching deliverable files found.",
        }
    output_blob = "\\n\\n".join(
        f"--- {name} ---\\n{content[:15000]}" for name, content in relevant.items()
    )
    try:
        prompt = VERDICT_PROMPT.safe_substitute(
            task_description=title,
            agent_output=output_blob,
            criterion_title=criterion["title"],
            match_criteria=criterion["match_criteria"],
        )
        result = parse_verdict(call_gemini(prompt))
        return {
            "id": criterion["id"], "title": criterion["title"],
            "verdict": str(result.get("verdict", "fail")).lower(),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as exc:
        return {
            "id": criterion["id"], "title": criterion["title"],
            "verdict": "fail", "reasoning": f"Judge error: {exc}",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fallback-output-dir", default=None)
    ap.add_argument("--reward-file", required=True)
    ap.add_argument("--details-file", default=None)
    args = ap.parse_args()

    rubric = json.loads(Path(args.rubric).read_text())
    title = rubric.get("title", "Legal task")
    criteria = rubric.get("criteria", [])
    reward_path = Path(args.reward_file)
    reward_path.parent.mkdir(parents=True, exist_ok=True)

    if not criteria:
        reward_path.write_text("0")
        return 0

    dirs = [Path(args.output_dir)]
    if args.fallback_output_dir:
        fb = Path(args.fallback_output_dir)
        if fb != dirs[0]:
            dirs.append(fb)
    texts = collect_deliverables(dirs)

    if not texts:
        print("No deliverable files found.", file=sys.stderr)
        reward_path.write_text("0")
        return 0

    print(f"Found {len(texts)} deliverable(s); judging {len(criteria)} criteria...")
    results = []
    for i, c in enumerate(criteria, 1):
        print(f"  [{i}/{len(criteria)}] {c['id']}: {c['title'][:60]}", flush=True)
        r = judge(c, title, texts)
        results.append(r)
        print(f"    -> {r['verdict'].upper()}: {r['reasoning'][:80]}")

    n_pass = sum(1 for r in results if r["verdict"] == "pass")
    n_total = len(results)
    all_pass = n_total > 0 and n_pass == n_total
    reward = 1.0 if all_pass else 0.0
    pass_rate = n_pass / n_total if n_total else 0.0

    print(f"\\nCriteria passed: {n_pass}/{n_total} ({pass_rate:.1%})")
    print(f"Reward: {reward:.1f} ({'all pass' if all_pass else 'not all pass'})")

    reward_path.write_text(str(reward))
    if args.details_file:
        details = {
            "score": reward, "all_pass": all_pass,
            "criterion_pass_rate": pass_rate,
            "n_passed": n_pass, "n_total": n_total,
            "judge_model": JUDGE_MODEL,
            "results": results,
        }
        Path(args.details_file).write_text(json.dumps(details, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def render_evaluate_py() -> str:
    return _EVALUATE_PY


# ── Discovery and generation ─────────────────────────────────────────

def discover_tasks(harvey_root: Path) -> list[dict]:
    """Walk <harvey_root>/tasks and yield task descriptors."""
    tasks_root = harvey_root / "tasks"
    if not tasks_root.is_dir():
        raise FileNotFoundError(f"No tasks directory at {tasks_root}")
    out: list[dict] = []
    for task_json in sorted(tasks_root.rglob("task.json")):
        rel = task_json.parent.relative_to(tasks_root)
        task_id = str(rel).replace("\\", "/")
        config = json.loads(task_json.read_text(encoding="utf-8"))
        docs_dir = task_json.parent / "documents"
        if not docs_dir.is_dir():
            print(f"  skip {task_id}: no documents/", file=sys.stderr)
            continue
        out.append({
            "task_id": task_id,
            "task_json": task_json,
            "docs_dir": docs_dir,
            "config": config,
        })
    return out


def copy_documents(src: Path, dst: Path) -> int:
    """Copy supported documents into dst/. Returns count."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.rglob("*"):
        if not f.is_file() or f.name.startswith("."):
            continue
        if f.suffix.lower() not in _DOC_EXTS:
            continue
        rel = f.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        n += 1
    return n


def generate_task(info: dict, out_root: Path, *, overwrite: bool) -> Path | None:
    task_id = info["task_id"]
    config = info["config"]
    name = sanitize_task_id(task_id)
    task_dir = out_root / name
    if task_dir.exists():
        if not overwrite:
            return task_dir
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    title = config.get("title", task_id)
    work_type = config.get("work_type", "analyze")
    tags = list(config.get("tags", []))
    instructions = config.get("instructions", "")
    if not instructions:
        instr_md = info["task_json"].parent / "instructions.md"
        if instr_md.exists():
            instructions = instr_md.read_text(encoding="utf-8")
    deliverables = dict(config.get("deliverables") or {})
    criteria = list(config.get("criteria") or [])

    # Stable registry name preserves the upstream hierarchy.
    registry_name = f"harvey-lab/{task_id}"

    (task_dir / "task.toml").write_text(
        render_task_toml(
            task_name=registry_name,
            work_type=work_type,
            tags=tags,
            criteria_count=len(criteria),
        )
    )
    (task_dir / "instruction.md").write_text(
        render_instruction_md(title=title, instructions=instructions, deliverables=deliverables)
    )

    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(render_dockerfile())
    copy_documents(info["docs_dir"], env_dir / "documents")
    (env_dir / "rubric.json").write_text(json.dumps(
        {"title": title, "deliverables": deliverables, "criteria": criteria},
        indent=2,
    ))

    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    test_sh = tests_dir / "test.sh"
    test_sh.write_text(render_test_sh())
    test_sh.chmod(0o755)
    (tests_dir / "evaluate.py").write_text(render_evaluate_py())

    return task_dir


# ── CLI ──────────────────────────────────────────────────────────────

def _default_harvey_root() -> Path:
    """Prefer the local mirror at .ref/harvey-lab; fall back to ../harvey-labs."""
    here = Path(__file__).resolve()
    repo = here.parent.parent.parent
    ref = repo / ".ref" / "harvey-lab"
    if (ref / "tasks").is_dir():
        return ref
    sibling = repo.parent / "harvey-labs"
    return sibling


def main() -> int:
    p = argparse.ArgumentParser(description="Adapt Harvey LAB to the BenchFlow task format.")
    p.add_argument("--output-dir", required=True, help="Directory to write generated tasks into.")
    p.add_argument("--harvey-root", default=str(_default_harvey_root()),
                   help="Path to the harvey-labs source tree (must contain tasks/).")
    p.add_argument("--limit", type=int, default=None, help="Process at most N tasks.")
    p.add_argument("--task-ids", default=None,
                   help="Comma-separated task IDs to include (e.g. 'corporate-ma/x,real-estate/y').")
    p.add_argument("--split", choices=["full", "parity"], default="full",
                   help="Generate the full benchmark or the fixed parity subset.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace task directories that already exist.")
    args = p.parse_args()

    harvey_root = Path(args.harvey_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering tasks in {harvey_root}/tasks/ ...")
    tasks = discover_tasks(harvey_root)
    print(f"  found {len(tasks)} tasks")

    if args.task_ids:
        wanted = {tid.strip() for tid in args.task_ids.split(",") if tid.strip()}
        tasks = [t for t in tasks if t["task_id"] in wanted]
        print(f"  filtered to {len(tasks)} by --task-ids")
    elif args.split == "parity":
        wanted = set(PARITY_SUBSET)
        tasks = [t for t in tasks if t["task_id"] in wanted]
        print(f"  filtered to {len(tasks)} parity-subset tasks")

    if args.limit:
        tasks = tasks[: args.limit]
        print(f"  limited to {len(tasks)}")

    generated = errors = 0
    for info in tasks:
        try:
            generate_task(info, output_dir, overwrite=args.overwrite)
            generated += 1
        except Exception as exc:
            print(f"  ERROR {info['task_id']}: {exc}", file=sys.stderr)
            errors += 1

    print(f"\nGenerated {generated} tasks ({errors} errors) in {output_dir}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
