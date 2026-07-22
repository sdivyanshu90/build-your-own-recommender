"""Unified lifecycle command-line interface."""

import json
import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
import uvicorn

from recommender.artifacts.manifest import ArtifactManifest
from recommender.batch.job import batch_recommend as run_batch
from recommender.config import AppConfig, load_config
from recommender.data.preparation import prepare_data
from recommender.data.synthetic import generate_synthetic
from recommender.data.validation import validate_raw
from recommender.embeddings.export import export_item_embeddings
from recommender.evaluation.evaluator import evaluate_model
from recommender.exceptions import RecommenderError
from recommender.features.processor import fit_and_transform_features
from recommender.indexing.build import build_index, load_index
from recommender.observability.logging import configure_logging
from recommender.retrieval.runtime import RecommendationRuntime
from recommender.serving.app import create_app
from recommender.training.trainer import train_model
from recommender.utils.io import ensure_within

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Two-tower recommender lifecycle CLI.",
)
ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config", "-c", exists=True, dir_okay=False, help="Validated YAML configuration."
    ),
]


def _config(path: Path) -> AppConfig:
    config = load_config(path)
    configure_logging(config.environment == "production")
    return config


def _print_path(path: Path) -> None:
    typer.echo(json.dumps({"status": "ok", "path": str(path)}, sort_keys=True))


@app.command("generate-data")
def generate_data(config_path: ConfigOption = Path("configs/demo.yaml")) -> None:
    """Generate deterministic synthetic user, item, and interaction Parquet files."""
    config = _config(config_path)
    paths = generate_synthetic(config)
    typer.echo(
        json.dumps(
            {"status": "ok", "files": {key: str(value) for key, value in paths.items()}},
            sort_keys=True,
        )
    )


@app.command("validate-data")
def validate_data(
    config_path: ConfigOption = Path("configs/demo.yaml"),
    strict: Annotated[bool, typer.Option(help="Fail when error-severity findings exist.")] = False,
) -> None:
    """Validate raw schemas and values and write a data-quality report."""
    config = _config(config_path)
    report = config.paths.report_dir / "raw-data-quality.json"
    findings = validate_raw(config.paths.data_dir / "raw", report, fail_on_error=strict)
    typer.echo(
        json.dumps(
            {"status": "reported", "findings": len(findings), "report": str(report)}, sort_keys=True
        )
    )


def _prepare(config_path: Path) -> None:
    config = _config(config_path)
    dataset = prepare_data(config)
    features, transformed = fit_and_transform_features(config)
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "dataset": str(dataset),
                "features": str(features),
                "transformed": str(transformed),
            },
            sort_keys=True,
        )
    )


@app.command("prepare-data")
def prepare_data_command(config_path: ConfigOption = Path("configs/demo.yaml")) -> None:
    """Clean, label, time-split, fit features, and persist transformed datasets."""
    _prepare(config_path)


@app.command("preprocess")
def preprocess(config_path: ConfigOption = Path("configs/demo.yaml")) -> None:
    """Alias for prepare-data."""
    _prepare(config_path)


@app.command()
def train(config_path: ConfigOption = Path("configs/demo.yaml")) -> None:
    """Train the two-tower model and publish best/final checkpoints."""
    _print_path(train_model(_config(config_path)))


@app.command("export-item-embeddings")
def export_embeddings(config_path: ConfigOption = Path("configs/demo.yaml")) -> None:
    """Encode all eligible items with the trained item tower."""
    _print_path(export_item_embeddings(_config(config_path)))


@app.command()
def evaluate(config_path: ConfigOption = Path("configs/demo.yaml")) -> None:
    """Run leakage-aware exact evaluation and baseline comparison."""
    config = _config(config_path)
    embedding_path = config.paths.artifact_dir / "embeddings" / "embeddings-v001"
    if not embedding_path.exists():
        export_item_embeddings(config)
    _print_path(evaluate_model(config))


@app.command("build-index")
def build_index_command(config_path: ConfigOption = Path("configs/demo.yaml")) -> None:
    """Build and validate a versioned exact or FAISS index."""
    config = _config(config_path)
    embedding_path = config.paths.artifact_dir / "embeddings" / "embeddings-v001"
    if not embedding_path.exists():
        export_item_embeddings(config)
    _print_path(build_index(config))


@app.command("validate-index")
def validate_index(config_path: ConfigOption = Path("configs/demo.yaml")) -> None:
    """Reload an index and verify query dimensions and self-retrieval."""
    config = _config(config_path)
    index = load_index(config.paths.artifact_dir / "indexes" / "index-v001", "model-v001")
    embeddings = np.load(
        config.paths.artifact_dir / "embeddings" / "embeddings-v001" / "item_embeddings.npy",
        allow_pickle=False,
    )
    _, ids = index.search(embeddings[: min(20, len(embeddings))], 1)
    success = bool(np.all(ids[:, 0] == index.item_ids[: len(ids)]))
    if not success:
        raise typer.Exit(2)
    typer.echo(json.dumps({"status": "ok", "self_retrieval": success}))


@app.command("batch-recommend")
def batch_recommend_command(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    config_path: ConfigOption = Path("configs/demo.yaml"),
) -> None:
    """Generate restartable, versioned recommendations from Parquet users."""
    _print_path(run_batch(_config(config_path), input_path))


@app.command("inspect-artifact")
def inspect_artifact(
    path: Annotated[
        Path, typer.Argument(help="Artifact directory beneath configured artifact root.")
    ],
    config_path: ConfigOption = Path("configs/demo.yaml"),
) -> None:
    """Verify and print an artifact manifest without loading executable objects."""
    config = _config(config_path)
    safe = ensure_within(path, config.paths.artifact_dir)
    typer.echo(ArtifactManifest.load(safe).model_dump_json(indent=2))


@app.command("inspect-artifacts")
def inspect_artifacts(
    path: Annotated[Path, typer.Argument()], config_path: ConfigOption = Path("configs/demo.yaml")
) -> None:
    """Alias for inspect-artifact."""
    inspect_artifact(path, config_path)


@app.command()
def serve(config_path: ConfigOption = Path("configs/demo.yaml")) -> None:
    """Start the FastAPI service after compatible artifacts load."""
    config = _config(config_path)
    uvicorn.run(
        create_app(config), host=config.serving.host, port=config.serving.port, log_config=None
    )


@app.command("smoke-test")
def smoke_test(config_path: ConfigOption = Path("configs/demo.yaml")) -> None:
    """Load serving artifacts and execute known- and unknown-user recommendations."""
    config = _config(config_path)
    runtime = RecommendationRuntime.load(config)
    users = pd.read_parquet(
        config.paths.artifact_dir / "datasets" / "dataset-v001" / "users.parquet"
    )
    known = runtime.recommend(str(users.iloc[0]["user_id"]), config.serving.default_k)
    unknown = runtime.recommend("unknown-smoke-user", config.serving.default_k)
    if not known.candidates or not unknown.candidates or unknown.fallback_reason != "unknown_user":
        raise typer.Exit(2)
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "known_results": len(known.candidates),
                "unknown_results": len(unknown.candidates),
            }
        )
    )


@app.command("run-pipeline")
def run_pipeline(config_path: ConfigOption = Path("configs/demo.yaml")) -> None:
    """Run the complete local data-to-serving smoke pipeline."""
    config = _config(config_path)
    generate_synthetic(config)
    validate_raw(config.paths.data_dir / "raw", config.paths.report_dir / "raw-data-quality.json")
    prepare_data(config)
    fit_and_transform_features(config)
    train_model(config)
    export_item_embeddings(config)
    evaluate_model(config)
    build_index(config)
    runtime = RecommendationRuntime.load(config)
    result = runtime.recommend("unknown-demo-user", config.serving.default_k)
    if not result.candidates:
        raise typer.Exit(2)
    typer.echo(
        json.dumps(
            {"status": "ok", "pipeline": "complete", "fallback_results": len(result.candidates)}
        )
    )


def main() -> None:
    try:
        app()
    except RecommenderError as error:
        logging.getLogger(__name__).error(
            "command_failed", extra={"error_code": error.__class__.__name__, "detail": str(error)}
        )
        raise typer.Exit(2) from error
