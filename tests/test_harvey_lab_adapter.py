"""Smoke tests for the Harvey LAB adapter.

These tests verify the converter's structural output without making network
calls. Live judge-prompt and agent-runs parity tests live alongside the
adapter (see benchmarks/harvey-lab/parity_test.py).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ADAPTER_PATH = (
    Path(__file__).resolve().parent.parent
    / "benchmarks" / "harvey-lab" / "benchflow.py"
)


def _load_adapter():
    spec = importlib.util.spec_from_file_location("harvey_adapter", ADAPTER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_fake_harvey_root(tmp: Path, *, criteria_n: int = 4) -> Path:
    """Build a tiny Harvey-LAB-shaped tree with one task and one document."""
    root = tmp / "harvey-fake"
    task_dir = root / "tasks" / "demo-area" / "tiny-task"
    docs = task_dir / "documents"
    docs.mkdir(parents=True)
    (docs / "memo.txt").write_text("Sample input document.\n")
    (task_dir / "task.json").write_text(json.dumps({
        "title": "Tiny task",
        "instructions": "Read the memo and write a one-line summary.",
        "work_type": "analyze",
        "tags": ["demo", "tiny"],
        "deliverables": {"summary.md": "summary.md"},
        "criteria": [
            {
                "id": f"C-{i:03d}",
                "title": f"criterion {i}",
                "deliverables": ["summary.md"],
                "match_criteria": "PASS if the summary mentions the memo.",
            }
            for i in range(1, criteria_n + 1)
        ],
    }))
    return root


def test_sanitize_task_id():
    mod = _load_adapter()
    assert mod.sanitize_task_id("Corporate-M&A/My Task") == "corporate-m-a-my-task"
    assert mod.sanitize_task_id("a   b___c") == "a-b-c"
    assert mod.sanitize_task_id("--leading--") == "leading"


def test_infer_difficulty():
    mod = _load_adapter()
    assert mod.infer_difficulty(10) == "easy"
    assert mod.infer_difficulty(50) == "medium"
    assert mod.infer_difficulty(100) == "hard"


def test_generate_task_creates_required_layout(tmp_path: Path):
    mod = _load_adapter()
    src_root = _make_fake_harvey_root(tmp_path, criteria_n=3)
    out = tmp_path / "out"
    out.mkdir()
    tasks = mod.discover_tasks(src_root)
    assert len(tasks) == 1
    td = mod.generate_task(tasks[0], out, overwrite=False)
    assert td is not None
    assert (td / "task.toml").exists()
    assert (td / "instruction.md").exists()
    assert (td / "environment" / "Dockerfile").exists()
    assert (td / "environment" / "rubric.json").exists()
    assert (td / "tests" / "test.sh").exists()
    assert (td / "tests" / "evaluate.py").exists()
    # Documents are copied
    assert (td / "environment" / "documents" / "memo.txt").exists()


def test_generated_task_toml_is_valid_and_named(tmp_path: Path):
    mod = _load_adapter()
    src_root = _make_fake_harvey_root(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    [info] = mod.discover_tasks(src_root)
    td = mod.generate_task(info, out, overwrite=False)
    cfg = tomllib.loads((td / "task.toml").read_text())
    assert cfg["task"]["name"] == "harvey-lab/demo-area-tiny-task"
    assert cfg["agent"]["timeout_sec"] >= 1800
    assert cfg["verifier"]["env"]["GEMINI_API_KEY"] == "${GEMINI_API_KEY}"


def test_rubric_json_preserves_criteria_count(tmp_path: Path):
    mod = _load_adapter()
    src_root = _make_fake_harvey_root(tmp_path, criteria_n=7)
    out = tmp_path / "out"
    out.mkdir()
    [info] = mod.discover_tasks(src_root)
    td = mod.generate_task(info, out, overwrite=False)
    rubric = json.loads((td / "environment" / "rubric.json").read_text())
    assert len(rubric["criteria"]) == 7
    assert rubric["title"] == "Tiny task"


def test_test_sh_is_executable(tmp_path: Path):
    mod = _load_adapter()
    src_root = _make_fake_harvey_root(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    [info] = mod.discover_tasks(src_root)
    td = mod.generate_task(info, out, overwrite=False)
    import os
    assert os.access(td / "tests" / "test.sh", os.X_OK)


def test_overwrite_flag(tmp_path: Path):
    mod = _load_adapter()
    src_root = _make_fake_harvey_root(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    [info] = mod.discover_tasks(src_root)
    td = mod.generate_task(info, out, overwrite=False)
    marker = td / "marker.tmp"
    marker.write_text("first run")
    # Second call without overwrite returns same dir without modifying.
    td2 = mod.generate_task(info, out, overwrite=False)
    assert td2 == td and marker.exists()
    # With overwrite, the dir is replaced.
    td3 = mod.generate_task(info, out, overwrite=True)
    assert td3 == td and not marker.exists()


def test_cli_runs_against_real_subset(tmp_path: Path):
    """Smoke test that the CLI itself runs and emits the parity subset
    when the pinned upstream snapshot is locally available.
    """
    repo_root = ADAPTER_PATH.parent.parent.parent
    harvey_root = repo_root / ".ref" / "harvey-lab"
    if not (harvey_root / "tasks").is_dir():
        pytest.skip("no local .ref/harvey-lab; skipping CLI smoke")
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(ADAPTER_PATH),
         "--output-dir", str(out),
         "--harvey-root", str(harvey_root),
         "--split", "parity",
         "--overwrite"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    generated = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert len(generated) == 5
