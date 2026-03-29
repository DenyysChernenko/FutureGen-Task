import logging
from pathlib import Path
from typing import Any

import yaml  # type: ignore

from .config_models import (
    AudioStreamerConfig,
    ClassifierConfig,
    PipelineConfig,
    ProjectConfig,
    TranscriptionConfig,
)

logger = logging.getLogger(__name__)


def load_yaml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading config from: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_pipeline_config(config_path: Path) -> PipelineConfig:
    raw_config = load_yaml(config_path)

    project_config = ProjectConfig(**raw_config.get("project", {}))

    audio_streamer_config = AudioStreamerConfig(**raw_config.get("audio_streamer", {}))

    transcription_raw = raw_config.get("transcription", {})
    transcription_raw["output_dir"] = project_config.output_dir
    transcription_config = TranscriptionConfig(**transcription_raw)

    classifier_config = ClassifierConfig(**raw_config.get("classifier", {}))

    config = PipelineConfig(
        project=project_config,
        audio_streamer=audio_streamer_config,
        transcription=transcription_config,
        classifier=classifier_config,
    )

    logger.info(f"Output directory: {config.project.output_dir}")
    return config
