#!/usr/bin/env python3
"""Export BenchFlow markdown docs to website MDX files."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocPage:
    source: str
    target: str
    title: str
    description: str


DOC_PAGES: tuple[DocPage, ...] = (
    DocPage(
        "getting-started.md",
        "getting-started.mdx",
        "Getting started",
        "Install BenchFlow and run your first agent against a verifiable task.",
    ),
    DocPage(
        "concepts.md",
        "concepts.mdx",
        "Concepts",
        "Understand BenchFlow rollouts, scenes, sandboxes, agents, and rewards.",
    ),
    DocPage(
        "task-authoring.md",
        "task-authoring.mdx",
        "Task authoring",
        "Create BenchFlow tasks with instructions, environments, and verifiers.",
    ),
    DocPage(
        "skill-eval.md",
        "skill-eval.mdx",
        "Skill eval",
        "Evaluate reusable agent skills with BenchFlow.",
    ),
    DocPage(
        "progressive-disclosure.md",
        "progressive-disclosure.mdx",
        "Progressive disclosure",
        "Run multi-round tasks that reveal context over time.",
    ),
    DocPage(
        "sandbox-hardening.md",
        "sandbox-hardening.mdx",
        "Sandbox hardening",
        "Understand BenchFlow's sandbox isolation and verifier protections.",
    ),
    DocPage(
        "use-cases.md",
        "use-cases.mdx",
        "Use cases",
        "Patterns for single-agent, multi-agent, simulated-user, and BYOS runs.",
    ),
    DocPage(
        "running-benchmarks.md",
        "running-benchmarks.mdx",
        "Running benchmarks",
        "Run adapted benchmark suites and inspect BenchFlow results.",
    ),
    DocPage(
        "reference/cli.md",
        "reference/cli.mdx",
        "CLI reference",
        "BenchFlow command-line flags, commands, and YAML config reference.",
    ),
    DocPage(
        "reference/python-api.md",
        "reference/python-api.mdx",
        "Python API",
        "Programmatic BenchFlow rollout, scene, and result APIs.",
    ),
)


LOCAL_MARKDOWN_LINK = re.compile(
    r"(?P<prefix>\]\()(?P<target>(?!https?://|mailto:|#)[^)#]+?)\.md(?P<anchor>#[^)]+)?(?P<suffix>\))"
)


def _yaml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _convert_local_links(markdown: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return (
            match.group("prefix")
            + match.group("target")
            + (match.group("anchor") or "")
            + match.group("suffix")
        )

    return LOCAL_MARKDOWN_LINK.sub(replace, markdown)


def render_mdx(markdown: str, page: DocPage) -> str:
    frontmatter = (
        "---\n"
        f"title: {_yaml_string(page.title)}\n"
        f"description: {_yaml_string(page.description)}\n"
        "---\n\n"
    )
    return frontmatter + _convert_local_links(markdown).lstrip()


def export_docs(
    *,
    benchflow_root: Path,
    website_root: Path,
    write: bool,
) -> list[Path]:
    docs_root = benchflow_root / "docs"
    website_docs_root = website_root / "src" / "content" / "docs" / "benchflow"
    changed: list[Path] = []

    if not docs_root.exists():
        raise FileNotFoundError(f"BenchFlow docs directory not found: {docs_root}")
    if not website_docs_root.exists():
        raise FileNotFoundError(
            f"Website BenchFlow docs directory not found: {website_docs_root}"
        )

    for page in DOC_PAGES:
        source = docs_root / page.source
        target = website_docs_root / page.target
        if not source.exists():
            raise FileNotFoundError(f"Source doc not found: {source}")

        rendered = render_mdx(source.read_text(encoding="utf-8"), page)
        existing = target.read_text(encoding="utf-8") if target.exists() else None
        if existing == rendered:
            continue

        changed.append(target)
        if write:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")

    return changed


def _default_website_root(benchflow_root: Path) -> Path:
    return benchflow_root.parent / "www.benchflow.ai"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export benchflow/docs Markdown files to website MDX."
    )
    parser.add_argument(
        "--benchflow-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="BenchFlow repository root.",
    )
    parser.add_argument(
        "--website-root",
        type=Path,
        default=None,
        help="Website repository root. Defaults to ../www.benchflow.ai.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated website MDX differs from benchflow/docs.",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Write generated MDX files into the website repo.",
    )
    args = parser.parse_args(argv)

    benchflow_root = args.benchflow_root.resolve()
    website_root = (
        args.website_root.resolve()
        if args.website_root
        else _default_website_root(benchflow_root).resolve()
    )
    changed = export_docs(
        benchflow_root=benchflow_root,
        website_root=website_root,
        write=args.write,
    )

    if not changed:
        print("BenchFlow website docs are in sync.")
        return 0

    for path in changed:
        rel = path.relative_to(website_root)
        action = "updated" if args.write else "out of sync"
        print(f"{action}: {rel}")
    return 0 if args.write else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
