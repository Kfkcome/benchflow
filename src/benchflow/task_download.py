"""Resolve and cache benchmark task datasets.

Datasets are referenced with two fields (inspired by Vercel's project config):

    source:
      repo: org/repo          # GitHub repository (org/repo)
      hf: org/dataset         # HuggingFace dataset repo (alternative to repo)
      path: sub/dir           # optional subpath within the source
      ref: main               # optional branch/tag/revision

The repo is cloned once into ``.cache/datasets/org/repo/`` and reused on
subsequent calls.
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Source:
    """A benchmark dataset source — identifies a repo and optional subpath."""

    repo: str
    path: str | None = None
    ref: str | None = None

    def resolve(self) -> Path:
        """Clone the repo (if needed) and return the local filesystem path."""
        return resolve_source(self.repo, self.path, self.ref)


def _repo_root() -> Path:
    """Find the repo root via .git directory."""
    d = Path.cwd()
    while d != d.parent:
        if (d / ".git").exists():
            return d
        d = d.parent
    return Path.cwd()


def _cache_dir() -> Path:
    """Return the local cache directory for cloned dataset repos."""
    return _repo_root() / ".cache" / "datasets"


def _clean_subpath(path: str) -> str:
    """Normalize a source subpath and reject paths that can escape the cache."""
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Invalid source path: {path!r}")
    return rel.as_posix().strip("/")


def _is_hf_ref(repo: str) -> bool:
    return repo.startswith(("hf://", "huggingface://"))


def _normalize_hf_repo_id(repo_id: str) -> str:
    """Return an HF dataset repo id from supported CLI/YAML spellings."""
    for prefix in ("hf://", "huggingface://"):
        if repo_id.startswith(prefix):
            repo_id = repo_id[len(prefix) :]
            break
    repo_id = repo_id.removeprefix("datasets/")
    parts = repo_id.split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Invalid HuggingFace dataset repo: {repo_id!r}. "
            "Expected 'org/dataset' (or 'hf://org/dataset')."
        )
    return repo_id


def _hf_cache_dir(repo_id: str) -> Path:
    org, dataset = repo_id.split("/", 1)
    return _cache_dir() / "huggingface" / org / dataset


def _snapshot_download_hf(
    *,
    repo_id: str,
    local_dir: Path,
    revision: str | None,
    allow_patterns: list[str] | None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace task sources require the optional 'hf' extra. "
            "Install with: uv tool install 'benchflow[hf]' or pip install 'benchflow[hf]'"
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir=str(local_dir),
    )
    return Path(downloaded)


def _clone_repo(org: str, repo: str, ref: str | None = None) -> Path:
    """Clone a GitHub repo into the cache if not already present.

    If the cache exists but is on a different ref, fetches and checks out
    the requested ref.

    Returns the path to the cloned repo root.
    """
    cache = _cache_dir() / org / repo
    if cache.exists() and (cache / ".git").exists():
        if ref:
            # Ensure the cached clone is on the requested ref.
            current = subprocess.run(
                ["git", "-C", str(cache), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            if current != ref:
                logger.info("Switching %s/%s from %s to %s", org, repo, current, ref)
                subprocess.run(
                    ["git", "-C", str(cache), "fetch", "--depth", "1", "origin", ref],
                    check=True,
                )
                subprocess.run(
                    ["git", "-C", str(cache), "checkout", ref],
                    check=True,
                )
        return cache

    url = f"https://github.com/{org}/{repo}.git"
    logger.info("Cloning %s/%s from %s ...", org, repo, url)
    cache.parent.mkdir(parents=True, exist_ok=True)
    clone_tmp = cache.parent / f"_{repo}_clone"

    try:
        if clone_tmp.exists():
            shutil.rmtree(clone_tmp)
        cmd = ["git", "clone", "--depth", "1"]
        if ref:
            cmd.extend(["--branch", ref])
        cmd.extend([url, str(clone_tmp)])
        subprocess.run(cmd, check=True)
        if cache.exists():
            shutil.rmtree(cache)
        clone_tmp.rename(cache)
    finally:
        if clone_tmp.exists():
            shutil.rmtree(clone_tmp, ignore_errors=True)

    return cache


def resolve_source(repo: str, path: str | None = None, ref: str | None = None) -> Path:
    """Resolve a dataset source to a local filesystem path.

    Args:
        repo: GitHub repository as ``org/repo`` (e.g. ``benchflow-ai/benchmarks``)
            or HuggingFace dataset repo with ``hf://``/``huggingface://`` prefix.
        path: Optional subpath within the source (e.g. ``terminal-bench-2``).
        ref: Optional branch, tag, or revision (e.g. ``main``, ``v2.0``).

    Returns:
        Path to the resolved directory on the local filesystem.
    """
    if _is_hf_ref(repo):
        return resolve_hf_source(repo, path=path, ref=ref)

    parts = repo.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid repo format: {repo!r}. Expected 'org/repo' (e.g. 'benchflow-ai/benchmarks')."
        )
    org, repo_name = parts
    root = _clone_repo(org, repo_name, ref)

    if path:
        target = root / path
        if not target.exists():
            raise FileNotFoundError(
                f"Path {path!r} not found in {org}/{repo_name}. "
                f"Available: {[p.name for p in root.iterdir() if p.is_dir() and p.name != '.git']}"
            )
        return target
    return root


def resolve_hf_source(
    repo_id: str, path: str | None = None, ref: str | None = None
) -> Path:
    """Resolve a HuggingFace dataset repository to a local filesystem path."""
    normalized_repo = _normalize_hf_repo_id(repo_id)
    normalized_path = _clean_subpath(path) if path else None
    allow_patterns = [f"{normalized_path}/**"] if normalized_path else None
    root = _snapshot_download_hf(
        repo_id=normalized_repo,
        local_dir=_hf_cache_dir(normalized_repo),
        revision=ref,
        allow_patterns=allow_patterns,
    )

    if normalized_path:
        target = root / normalized_path
        if not target.exists():
            raise FileNotFoundError(
                f"Path {normalized_path!r} not found in HuggingFace dataset "
                f"{normalized_repo}."
            )
        return target
    return root


# ---------------------------------------------------------------------------
# Benchmark aliases
# ---------------------------------------------------------------------------

# Aliases for ensure_tasks("shortname") callers.
# Format: (org/repo, ref, subpath)
TASK_ALIASES: dict[str, tuple[str, str | None, str | None]] = {
    "skillsbench": ("benchflow-ai/skillsbench", "main", "tasks"),
    "programbench": ("benchflow-ai/benchmarks", "main", "datasets/programbench/tasks"),
    "harvey-lab": ("benchflow-ai/benchmarks", "main", "datasets/harvey-lab/tasks"),
}

# Old dict shape kept for imports that reference TASK_REPOS.
TASK_REPOS = {
    name: {
        "repo": f"https://github.com/{org_repo}.git",
        **({"ref": branch} if branch else {}),
        **({"subdir": subdir} if subdir else {}),
    }
    for name, (org_repo, branch, subdir) in TASK_ALIASES.items()
}


def ensure_tasks(benchmark: str) -> Path:
    """Clone task repo if not present. Supports aliases and org/repo strings."""
    if benchmark in TASK_ALIASES:
        org_repo, ref, path = TASK_ALIASES[benchmark]
        return resolve_source(org_repo, path, ref)
    return resolve_source(benchmark)
