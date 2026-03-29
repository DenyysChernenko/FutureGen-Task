import json
from unittest.mock import patch

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from scripts.main import app
from src.config import load_pipeline_config


def test_pipeline_end_to_end(
    pipeline_config,
    test_audio_file_5s,
    mock_whisper_model,
    mock_sentence_transformer,
    create_config_file,
):
    runner = CliRunner()
    config_file = create_config_file(pipeline_config)

    with patch("whisper.load_model", return_value=mock_whisper_model), patch(
        "src.classifier.engine.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ):
        result = runner.invoke(
            app,
            [
                str(test_audio_file_5s),
                "--session-id",
                "integration_test",
                "--config",
                str(config_file),
            ],
        )

    assert result.exit_code == 0, f"CLI failed with output: {result.output}"

    output_dir = pipeline_config.project.output_dir

    transcription_file = output_dir / "transcription.jsonl"
    assert transcription_file.exists()

    content = transcription_file.read_text()
    lines = content.strip().split("\n")
    assert len(lines) >= 1

    for line in lines:
        record = json.loads(line)
        assert "chunk_id" in record
        assert "session_id" in record
        assert record["session_id"] == "integration_test"
        assert "text" in record
        assert "confidence" in record

    classification_file = output_dir / "classification.json"
    assert classification_file.exists()

    classification = json.loads(classification_file.read_text())
    assert "session_id" in classification
    assert classification["session_id"] == "integration_test"
    assert "classification" in classification
    assert classification["classification"] in ["private", "topic_based"]
    assert "confidence" in classification
    assert 0.0 <= classification["confidence"] <= 1.0
    assert "topics" in classification
    assert "sentiment" in classification
    assert "participants_count" in classification
    assert "total_duration_s" in classification


def test_pipeline_streaming_chain(
    pipeline_config,
    test_audio_file_5s,
    mock_whisper_model,
    mock_sentence_transformer,
    create_config_file,
):
    runner = CliRunner()
    config_file = create_config_file(pipeline_config)
    transcription_calls = []

    async def track_transcribe(self, chunk, state=None):
        transcription_calls.append(chunk.chunk_id)
        from src.transcription.models import TranscriptionRecord, TranscriptionState

        record = TranscriptionRecord(
            chunk_id=chunk.chunk_id,
            session_id=chunk.session_id,
            text="Test text",
            timestamp_start=chunk.timestamp_start,
            timestamp_end=chunk.timestamp_end,
            confidence=0.9,
        )
        new_state = TranscriptionState()
        return record, new_state

    with patch("whisper.load_model", return_value=mock_whisper_model), patch(
        "src.classifier.engine.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ), patch(
        "src.transcription.engine.TranscriptionEngine.transcribe", new=track_transcribe
    ):
        result = runner.invoke(
            app,
            [
                str(test_audio_file_5s),
                "--session-id",
                "streaming_test",
                "--config",
                str(config_file),
            ],
        )

    assert result.exit_code == 0, f"CLI failed with output: {result.output}"
    assert len(transcription_calls) >= 2


def test_pipeline_with_real_config(tmp_path, test_audio_file_5s):
    config_content = f"""
project:
  output_dir: "{tmp_path / 'output'}"
  data_dir: "{tmp_path / 'data'}"

audio_streamer:
  chunk_duration: 2.5
  overlap: 0.5
  target_sample_rate: 16000

transcription:
  model_name: "base"
  context_duration: 0.5

classifier:
  model_name: "all-MiniLM-L6-v2"
  class_descriptions:
    private: "Personal conversation"
    topic_based: "Business discussion"
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)

    config = load_pipeline_config(config_file)

    assert config.project.output_dir == tmp_path / "output"
    assert config.audio_streamer.chunk_duration == 2.5
    assert config.audio_streamer.overlap == 0.5
    assert config.transcription.model_name == "base"
    assert config.classifier.model_name == "all-MiniLM-L6-v2"
    assert "private" in config.classifier.class_descriptions
    assert "topic_based" in config.classifier.class_descriptions


def test_pipeline_multiple_chunks_context(
    pipeline_config,
    tmp_path,
    mock_whisper_model,
    mock_sentence_transformer,
    create_config_file,
):
    runner = CliRunner()
    config_file = create_config_file(pipeline_config)

    sample_rate = 16000
    duration = 8.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)

    audio_file = tmp_path / "test_8s.wav"
    sf.write(audio_file, audio, sample_rate)

    with patch("whisper.load_model", return_value=mock_whisper_model), patch(
        "src.classifier.engine.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ):
        result = runner.invoke(
            app,
            [
                str(audio_file),
                "--session-id",
                "context_test",
                "--config",
                str(config_file),
            ],
        )

    assert result.exit_code == 0, f"CLI failed with output: {result.output}"

    transcription_file = pipeline_config.project.output_dir / "transcription.jsonl"
    lines = transcription_file.read_text().strip().split("\n")

    assert len(lines) >= 3

    records = [json.loads(line) for line in lines]

    for i in range(len(records) - 1):
        gap = records[i + 1]["timestamp_start"] - records[i]["timestamp_start"]
        assert gap < 2.5


def test_pipeline_file_not_found(tmp_path):
    runner = CliRunner()
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(f"""
project:
  output_dir: "{tmp_path / 'output'}"
  data_dir: "{tmp_path / 'data'}"

audio_streamer:
  chunk_duration: 2.5
  overlap: 0.5
  target_sample_rate: 16000

transcription:
  model_name: "base"
  context_duration: 0.5

classifier:
  model_name: "all-MiniLM-L6-v2"
  class_descriptions:
    private: "Personal conversation"
    topic_based: "Business discussion"
""")

    result = runner.invoke(
        app,
        [
            "nonexistent_file.wav",
            "--session-id",
            "error_test",
            "--config",
            str(config_file),
        ],
    )

    assert result.exit_code == 2
    assert "does not exist" in result.output.lower()


def test_cli_with_valid_inputs(
    pipeline_config,
    test_audio_file_5s,
    mock_whisper_model,
    mock_sentence_transformer,
    create_config_file,
):
    runner = CliRunner()
    config_file = create_config_file(pipeline_config)

    with patch("whisper.load_model", return_value=mock_whisper_model), patch(
        "src.classifier.engine.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ):
        result = runner.invoke(
            app,
            [
                str(test_audio_file_5s),
                "--config",
                str(config_file),
            ],
        )

    assert result.exit_code == 0, f"CLI failed with output: {result.output}"

    output_dir = pipeline_config.project.output_dir
    assert (output_dir / "transcription.jsonl").exists()
    assert (output_dir / "classification.json").exists()
