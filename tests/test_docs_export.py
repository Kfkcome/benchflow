"""Tests for the BenchFlow docs-to-website export tool."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "export_docs_to_website.py"
)
SPEC = importlib.util.spec_from_file_location("export_docs_to_website", SCRIPT_PATH)
assert SPEC and SPEC.loader
export_docs_to_website = importlib.util.module_from_spec(SPEC)
sys.modules["export_docs_to_website"] = export_docs_to_website
SPEC.loader.exec_module(export_docs_to_website)


def _seed_docs(root: Path) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True)
    for page in export_docs_to_website.DOC_PAGES:
        source = docs / page.source
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            f"# {page.title}\n\nSee [Getting started](./getting-started.md#install).\n",
            encoding="utf-8",
        )


def _seed_website(root: Path) -> None:
    (root / "src" / "content" / "docs" / "benchflow").mkdir(parents=True)


def test_render_mdx_adds_frontmatter_and_converts_local_links():
    page = export_docs_to_website.DocPage(
        "source.md",
        "source.mdx",
        "Source",
        "A useful source page.",
    )

    rendered = export_docs_to_website.render_mdx(
        "See [local](./other.md#anchor) and [web](https://example.com/a.md).\n",
        page,
    )

    assert rendered.startswith(
        '---\ntitle: "Source"\ndescription: "A useful source page."\n---\n\n'
    )
    assert "[local](./other#anchor)" in rendered
    assert "[web](https://example.com/a.md)" in rendered


def test_export_docs_check_reports_drift(tmp_path):
    benchflow_root = tmp_path / "benchflow"
    website_root = tmp_path / "www.benchflow.ai"
    _seed_docs(benchflow_root)
    _seed_website(website_root)

    changed = export_docs_to_website.export_docs(
        benchflow_root=benchflow_root,
        website_root=website_root,
        write=False,
    )

    assert len(changed) == len(export_docs_to_website.DOC_PAGES)
    assert not changed[0].exists()


def test_export_docs_write_then_check_is_clean(tmp_path):
    benchflow_root = tmp_path / "benchflow"
    website_root = tmp_path / "www.benchflow.ai"
    _seed_docs(benchflow_root)
    _seed_website(website_root)

    written = export_docs_to_website.export_docs(
        benchflow_root=benchflow_root,
        website_root=website_root,
        write=True,
    )
    changed = export_docs_to_website.export_docs(
        benchflow_root=benchflow_root,
        website_root=website_root,
        write=False,
    )

    assert len(written) == len(export_docs_to_website.DOC_PAGES)
    assert changed == []
    assert (
        website_root
        / "src"
        / "content"
        / "docs"
        / "benchflow"
        / "reference"
        / "cli.mdx"
    ).exists()
