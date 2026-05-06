#!/bin/bash
# Generate the harvey-lab corpus and push it to benchflow-ai/benchmarks
# under datasets/harvey-lab/. Requires GITHUB_TOKEN or `gh` already
# authenticated against the org.
#
# Usage:
#   bash benchmarks/harvey-lab/_scripts/publish_corpus.sh           # full
#   bash benchmarks/harvey-lab/_scripts/publish_corpus.sh parity    # subset

set -euo pipefail

SPLIT="${1:-full}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
HARVEY_ROOT="$REPO_ROOT/.ref/harvey-lab"
CORPUS_OUT="/tmp/harvey-lab-corpus-$SPLIT"
WORK_DIR="/tmp/benchflow-benchmarks-corpus"

echo "[1/4] Generating tasks (split=$SPLIT)"
rm -rf "$CORPUS_OUT"
uv run python "$REPO_ROOT/benchmarks/harvey-lab/benchflow.py" \
    --output-dir "$CORPUS_OUT" \
    --harvey-root "$HARVEY_ROOT" \
    --split "$SPLIT" \
    --overwrite

count=$(find "$CORPUS_OUT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
echo "  generated $count tasks"

echo "[2/4] Cloning benchflow-ai/benchmarks (or refreshing)"
if [ -d "$WORK_DIR/.git" ]; then
    git -C "$WORK_DIR" fetch origin
    git -C "$WORK_DIR" reset --hard origin/main 2>/dev/null \
        || git -C "$WORK_DIR" checkout main
else
    rm -rf "$WORK_DIR"
    gh repo clone benchflow-ai/benchmarks "$WORK_DIR"
fi

echo "[3/4] Replacing datasets/harvey-lab/"
mkdir -p "$WORK_DIR/datasets"
rm -rf "$WORK_DIR/datasets/harvey-lab"
cp -R "$CORPUS_OUT" "$WORK_DIR/datasets/harvey-lab"

echo "[4/4] Committing and pushing"
cd "$WORK_DIR"
git add datasets/harvey-lab
if git diff --cached --quiet; then
    echo "  no changes to push"
    exit 0
fi
adapter_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)
git commit -m "harvey-lab: refresh corpus (split=$SPLIT, $count tasks, adapter ${adapter_commit:0:7})"
git push origin main
echo "  pushed."
