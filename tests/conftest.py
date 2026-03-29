import asyncio
import base64
import io
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from src.audio_streamer.models import AudioChunk
from src.config.config_models import (
    AudioStreamerConfig,
    ClassifierConfig,
    PipelineConfig,
    ProjectConfig,
    TranscriptionConfig,
)
from src.transcription.models import TranscriptionRecord, Word


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_audio_data() -> np.ndarray:
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


@pytest.fixture
def test_audio_file(tmp_path: Path, sample_audio_data: np.ndarray) -> Path:
    audio_file = tmp_path / "test_audio.wav"
    sf.write(audio_file, sample_audio_data, 16000)
    return audio_file


@pytest.fixture
def test_audio_file_5s(tmp_path: Path) -> Path:
    sample_rate = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)

    audio_file = tmp_path / "test_5s.wav"
    sf.write(audio_file, audio, sample_rate)
    return audio_file


@pytest.fixture
def sample_chunk() -> AudioChunk:
    audio = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))

    buffer = io.BytesIO()
    sf.write(buffer, audio, 16000, format="WAV")
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return AudioChunk(
        session_id="test_session",
        timestamp_start=0.0,
        timestamp_end=1.0,
        sample_rate=16000,
        duration_ms=1000,
        audio_data=audio_base64,
    )


@pytest.fixture
def test_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def audio_config() -> AudioStreamerConfig:
    return AudioStreamerConfig(
        chunk_duration=2.5,
        overlap=0.5,
        target_sample_rate=16000,
    )


@pytest.fixture
def transcription_config(test_output_dir: Path) -> TranscriptionConfig:
    return TranscriptionConfig(
        model_name="base",
        output_dir=test_output_dir,
        context_duration=0.5,
    )


@pytest.fixture
def pipeline_config(test_output_dir: Path) -> PipelineConfig:
    return PipelineConfig(
        project=ProjectConfig(output_dir=test_output_dir, data_dir=test_output_dir),
        audio_streamer=AudioStreamerConfig(
            chunk_duration=2.5, overlap=0.5, target_sample_rate=16000
        ),
        transcription=TranscriptionConfig(
            model_name="base",
            output_dir=test_output_dir,
            context_duration=0.5,
        ),
        classifier=ClassifierConfig(model_name="all-MiniLM-L6-v2"),
    )


@pytest.fixture
def mock_whisper_model():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "Hello this is a test of the audio processing system",
        "segments": [
            {
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.95},
                    {"word": "this", "start": 0.5, "end": 0.7, "probability": 0.94},
                    {"word": "is", "start": 0.7, "end": 0.85, "probability": 0.96},
                    {"word": "a", "start": 0.85, "end": 0.95, "probability": 0.97},
                    {"word": "test", "start": 0.95, "end": 1.3, "probability": 0.95},
                ],
                "avg_logprob": -0.3,
                "no_speech_prob": 0.05,
            }
        ],
        "language": "en",
    }
    return mock_model


@pytest.fixture
def mock_sentence_transformer():
    mock_model = MagicMock()

    def mock_encode(text):
        if isinstance(text, list):
            return np.array([[0.1, 0.2], [0.8, 0.9]])
        else:
            if any(word in text.lower() for word in ["personal", "private", "love"]):
                return np.array([0.15, 0.25])
            else:
                return np.array([0.75, 0.85])

    mock_model.encode.side_effect = mock_encode
    return mock_model


@pytest.fixture
def mock_transcription_records():
    return [
        TranscriptionRecord(
            chunk_id="chunk_1",
            session_id="test_session",
            speaker_id="SPEAKER_01",
            text="Hello world",
            language="en",
            timestamp_start=0.0,
            timestamp_end=2.5,
            confidence=0.95,
            words=[
                Word(word="Hello", start=0.0, end=0.5),
                Word(word="world", start=0.6, end=1.0),
            ],
        ),
        TranscriptionRecord(
            chunk_id="chunk_2",
            session_id="test_session",
            speaker_id="SPEAKER_01",
            text="This is a test",
            language="en",
            timestamp_start=2.0,
            timestamp_end=4.5,
            confidence=0.92,
            words=[
                Word(word="This", start=2.0, end=2.3),
                Word(word="is", start=2.3, end=2.5),
                Word(word="a", start=2.5, end=2.6),
                Word(word="test", start=2.7, end=3.0),
            ],
        ),
    ]


@pytest.fixture
def create_config_file(tmp_path: Path):
    """Helper fixture to create config files for CLI tests"""

    def _create_config(pipeline_config: PipelineConfig) -> Path:
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(f"""
project:
  output_dir: "{pipeline_config.project.output_dir}"
  data_dir: "{pipeline_config.project.data_dir}"

audio_streamer:
  chunk_duration: {pipeline_config.audio_streamer.chunk_duration}
  overlap: {pipeline_config.audio_streamer.overlap}
  target_sample_rate: {pipeline_config.audio_streamer.target_sample_rate}

transcription:
  model_name: "{pipeline_config.transcription.model_name}"
  context_duration: {pipeline_config.transcription.context_duration}

classifier:
  model_name: "{pipeline_config.classifier.model_name}"
  class_descriptions:
    private: "{pipeline_config.classifier.class_descriptions['private']}"
    topic_based: "{pipeline_config.classifier.class_descriptions['topic_based']}"
""")
        return config_file

    return _create_config
