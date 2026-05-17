"""CLI tests for remote task source selection."""

from typer.testing import CliRunner

from benchflow.cli.main import app


def test_eval_create_help_lists_huggingface_source():
    """HuggingFace task-source support is visible on the primary eval command."""
    result = CliRunner().invoke(app, ["eval", "create", "--help"])

    assert result.exit_code == 0
    assert "--source-hf" in result.output


def test_eval_create_rejects_multiple_remote_sources():
    """Users must choose either GitHub or HuggingFace as the remote source."""
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--source-repo",
            "benchflow-ai/skillsbench",
            "--source-hf",
            "benchflow/skillsbench",
        ],
    )

    assert result.exit_code == 1
    assert "Use only one of --source-repo or --source-hf" in result.output
