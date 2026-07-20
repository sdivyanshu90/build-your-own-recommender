"""CLI command-surface and failure-contract tests."""

from typer.testing import CliRunner

from recommender.cli import app

runner = CliRunner()
COMMANDS = (
    "generate-data",
    "validate-data",
    "prepare-data",
    "preprocess",
    "train",
    "evaluate",
    "export-item-embeddings",
    "build-index",
    "validate-index",
    "serve",
    "batch-recommend",
    "inspect-artifact",
    "inspect-artifacts",
    "smoke-test",
    "run-pipeline",
)


def test_all_documented_commands_have_help() -> None:
    root = runner.invoke(app, ["--help"])
    assert root.exit_code == 0
    for command in COMMANDS:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, f"{command}: {result.output}"


def test_missing_config_has_nonzero_actionable_failure() -> None:
    result = runner.invoke(app, ["train", "--config", "does-not-exist.yaml"])
    assert result.exit_code != 0
    assert "Invalid value for '--config'" in result.output
    assert "does-not-exist.yaml" in result.output
