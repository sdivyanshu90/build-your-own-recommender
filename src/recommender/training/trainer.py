"""Deterministic native-PyTorch trainer with versioned checkpoints."""

import logging
import os
import platform
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from recommender.artifacts.manifest import ArtifactManifest
from recommender.config.models import AppConfig
from recommender.exceptions import ConfigurationError
from recommender.features.processor import FeatureProcessor
from recommender.models.losses import InBatchSoftmaxLoss
from recommender.models.two_tower import TwoTowerModel
from recommender.training.datasets import InteractionDataset, move_batch
from recommender.utils.io import atomic_write_json
from recommender.utils.seeds import seed_everything

LOGGER = logging.getLogger(__name__)


def _epoch(
    model: TwoTowerModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    loss_function: InBatchSoftmaxLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    clip_norm: float,
    use_amp: bool,
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_examples = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            users, items = model(batch)
            loss = loss_function(users, items, batch["item_identity"])
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        count = len(batch["item_identity"])
        total_loss += float(loss.detach()) * count
        total_examples += count
    return total_loss / max(total_examples, 1)


def load_model(
    model_dir: Path,
    processor: FeatureProcessor,
    config: AppConfig,
    device: torch.device | None = None,
) -> TwoTowerModel:
    manifest = ArtifactManifest.load(model_dir)
    manifest.require_dependency("features", processor.version)
    model = TwoTowerModel(processor.vocabulary_sizes(), config.model)
    state = torch.load(model_dir / "best.pt", map_location=device or "cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device or torch.device("cpu"))
    model.eval()
    return model


def train_model(
    config: AppConfig,
    dataset_version: str = "dataset-v001",
    feature_version: str = "features-v001",
    model_version: str = "model-v001",
) -> Path:
    if config.training.negative_strategy != "in_batch":
        raise ConfigurationError(
            "The native trainer currently supports negative_strategy='in_batch' only. "
            "Uniform, popularity, and hard samplers are available as components, but require "
            "a sampled-softmax training loop."
        )
    seed_everything(config.seed, config.training.deterministic)
    dataset_dir = config.paths.artifact_dir / "datasets" / dataset_version / "transformed"
    feature_dir = config.paths.artifact_dir / "feature-pipelines" / feature_version
    processor = FeatureProcessor.load(feature_dir)
    train_frame = pd.read_parquet(dataset_dir / "train.parquet")
    validation_frame = pd.read_parquet(dataset_dir / "validation.parquet")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        InteractionDataset(train_frame),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        generator=generator,
    )
    training_example_count = int(train_frame["label"].eq(1).sum())
    validation_loader = DataLoader(
        InteractionDataset(validation_frame), batch_size=config.training.batch_size, shuffle=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerModel(processor.vocabulary_sizes(), config.model).to(device)
    resumed_from: str | None = None
    if config.training.resume_checkpoint is not None:
        checkpoint = config.training.resume_checkpoint.resolve()
        if not checkpoint.is_file() or checkpoint.suffix != ".pt":
            raise ConfigurationError(
                f"resume_checkpoint must be an existing .pt weights file: {checkpoint}"
            )
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        resumed_from = str(checkpoint)
    loss_function = InBatchSoftmaxLoss(
        config.model.temperature, config.training.label_smoothing, config.training.symmetric_loss
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler: Any
    if config.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.epochs
        )
    elif config.training.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
    else:
        scheduler = None
    model_dir = config.paths.artifact_dir / "models" / model_version
    model_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    use_amp = config.training.mixed_precision and device.type == "cuda"
    mlflow_module: Any | None = None
    if config.tracking.enabled:
        import mlflow

        if config.tracking.uri.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            tracking_path = Path(unquote(urlparse(config.tracking.uri).path))
            tracking_path.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(config.tracking.uri)
        mlflow.set_experiment(config.tracking.experiment)
        mlflow.start_run(run_name=model_version)
        mlflow.log_params(
            {
                "model_version": model_version,
                "dataset_version": dataset_version,
                "feature_version": feature_version,
                "seed": config.seed,
                "embedding_dim": config.model.embedding_dim,
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate,
                "similarity": config.model.similarity,
            }
        )
        mlflow_module = mlflow
    try:
        for epoch in range(config.training.epochs):
            started = time.perf_counter()
            train_loss = _epoch(
                model,
                train_loader,
                loss_function,
                device,
                optimizer,
                config.training.gradient_clip_norm,
                use_amp,
            )
            with torch.inference_mode():
                validation_loss = _epoch(
                    model,
                    validation_loader,
                    loss_function,
                    device,
                    None,
                    config.training.gradient_clip_norm,
                    use_amp,
                )
            elapsed = time.perf_counter() - started
            record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "duration_seconds": elapsed,
                "examples_per_second": training_example_count / max(elapsed, 1e-9),
            }
            history.append(record)
            LOGGER.info("epoch_complete", extra=record)
            if mlflow_module is not None:
                mlflow_module.log_metrics(
                    {key: float(value) for key, value in record.items() if key != "epoch"},
                    step=epoch + 1,
                )
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                stale_epochs = 0
                torch.save(model.state_dict(), model_dir / "best.pt")
            else:
                stale_epochs += 1
            if scheduler is not None:
                scheduler.step(
                    validation_loss
                ) if config.training.scheduler == "plateau" else scheduler.step()
            if stale_epochs >= config.training.patience:
                break
    except KeyboardInterrupt:
        torch.save(model.state_dict(), model_dir / "interrupted.pt")
        if mlflow_module is not None:
            mlflow_module.end_run(status="KILLED")
        raise
    torch.save(model.state_dict(), model_dir / "final.pt")
    metadata = {
        "history": history,
        "best_validation_loss": best_loss,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "vocabulary_sizes": processor.vocabulary_sizes(),
        "resumed_from": resumed_from,
    }
    atomic_write_json(model_dir / "training.json", metadata)
    (model_dir / "MODEL_CARD.md").write_text(
        "# Model card\n\n"
        f"Version: `{model_version}`  \nFeatures: `{feature_version}`  \n"
        f"Objective: in-batch softmax retrieval  \nSimilarity: `{config.model.similarity}`\n\n"
        "Intended use: candidate retrieval. A downstream ranker and online "
        "experiment are required.\n"
        "Known limitations: synthetic training data, exposure bias, and limited "
        "cold-start identity signal.\n",
        encoding="utf-8",
    )
    manifest = ArtifactManifest.create(
        "model",
        model_version,
        config.model_dump(mode="json"),
        processor.vocabulary_sizes(),
        dependencies={"dataset": dataset_version, "features": feature_version},
        metadata=metadata,
    )
    manifest.write(model_dir)
    if mlflow_module is not None:
        mlflow_module.log_artifact(str(model_dir / "training.json"), artifact_path="model-metadata")
        mlflow_module.log_artifact(str(model_dir / "MODEL_CARD.md"), artifact_path="model-metadata")
        mlflow_module.end_run(status="FINISHED")
    return model_dir
