import asyncio
import json
import logging
from pathlib import Path

import aiofiles  # type: ignore

from .audio_streamer import AudioStreamer
from .classifier import ClassificationResult, ClassifierEngine
from .config import PipelineConfig
from .transcription import TranscriptionEngine, TranscriptionState

logger = logging.getLogger(__name__)


class AudioPipeline:

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._setup_output_directories()

        self.streamer = AudioStreamer(config=self.config.audio_streamer)
        self.engine = TranscriptionEngine(config=self.config.transcription)
        self.classifier = ClassifierEngine(
            model_name=self.config.classifier.model_name,
            class_descriptions=self.config.classifier.class_descriptions,
        )

    def _setup_output_directories(self):
        self.config.project.output_dir.mkdir(parents=True, exist_ok=True)

    async def process(
        self,
        audio_file_path: str,
        session_id: str,
        start_timestamp: float | None = None,
    ) -> None:
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        logger.info(f"Starting streaming pipeline for: {audio_file_path}")
        logger.info(f"Session ID: {session_id}")

        transcripts = []
        state = TranscriptionState()
        try:
            async for chunk in self.streamer.stream(
                audio_file_path, session_id, start_timestamp
            ):

                transcript, state = await self.engine.transcribe(chunk, state)
                transcripts.append(transcript)

                print(json.dumps(transcript.model_dump(), indent=2))

                classification = await self.classifier.classify(transcripts, session_id)
                await self._save_classification(classification)
                logger.debug(f"Updated classification after chunk {len(transcripts)}")

            logger.info(f"Processing complete. Total chunks: {len(transcripts)}")
            logger.info(
                f"Final classification: {classification.classification} (confidence: {classification.confidence:.2f})"
            )

        except asyncio.CancelledError:
            logger.info("Shutting down gracefully...")
            raise

    async def _save_classification(self, classification: ClassificationResult) -> None:
        classification_file = self.config.project.output_dir / "classification.json"

        classification_json = json.dumps(classification.model_dump(), indent=2)

        async with aiofiles.open(classification_file, "w") as f:
            await f.write(classification_json)

        logger.debug(f"Saved classification update: {classification_file}")
