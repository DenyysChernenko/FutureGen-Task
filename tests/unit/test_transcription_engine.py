from unittest.mock import patch

import numpy as np
import pytest

from src.audio_streamer.models import AudioChunk
from src.transcription.engine import TranscriptionEngine
from src.transcription.models import (
    TranscriptionRecord,
    TranscriptionState,
    WhisperResult,
    WhisperSegment,
    WhisperWord,
)


@pytest.fixture
def engine(transcription_config, mock_whisper_model):
    with patch("whisper.load_model", return_value=mock_whisper_model):
        return TranscriptionEngine(config=transcription_config)


def test_decode_audio(engine: TranscriptionEngine, sample_chunk):
    audio_array = engine._decode_audio(sample_chunk.audio_data)

    assert isinstance(audio_array, np.ndarray)
    assert len(audio_array) > 0
    assert audio_array.dtype == np.float32


@pytest.mark.parametrize(
    ("has_previous_audio", "expected_offset", "expected_length"),
    [
        pytest.param(
            False,
            0.0,
            16000,
            id="first_chunk_no_context",
        ),
        pytest.param(
            True,
            0.5,
            24000,
            id="with_context_0_5s_added",
        ),
    ],
)
def test_apply_sliding_window(
    engine: TranscriptionEngine,
    sample_chunk,
    has_previous_audio,
    expected_offset,
    expected_length,
):
    current_audio = np.zeros(16000)
    previous_audio = np.ones(16000) if has_previous_audio else None

    audio_with_context, context_offset = engine._apply_sliding_window(
        current_audio, sample_chunk, previous_audio
    )

    assert context_offset == expected_offset
    assert len(audio_with_context) == expected_length

    if has_previous_audio:
        np.testing.assert_array_equal(audio_with_context[:8000], np.ones(8000))
        np.testing.assert_array_equal(audio_with_context[8000:], np.zeros(16000))
    else:
        np.testing.assert_array_equal(audio_with_context, current_audio)


def test_extract_words(engine: TranscriptionEngine):
    whisper_result = WhisperResult(
        text="Hello world",
        segments=[
            WhisperSegment(
                words=[
                    WhisperWord(word="Hello", start=0.0, end=0.5, probability=0.95),
                    WhisperWord(word="world", start=0.6, end=1.0, probability=0.93),
                ]
            )
        ],
        language="en",
    )

    chunk_offset = 2.0

    words = engine._extract_words(whisper_result, chunk_offset)

    assert len(words) == 2
    assert words[0].word == "Hello"
    assert words[0].start == 2.0
    assert words[0].end == 2.5
    assert words[1].word == "world"
    assert words[1].start == 2.6


@pytest.mark.parametrize(
    ("text", "segments", "expected_confidence_range"),
    [
        pytest.param(
            "Hello world",
            [WhisperSegment(words=[], avg_logprob=-0.3, no_speech_prob=0.1)],
            (0.5, 1.0),
            id="high_confidence_with_segments",
        ),
        pytest.param(
            "",
            [],
            (0.0, 0.0),
            id="zero_confidence_empty_text",
        ),
        pytest.param(
            "Low confidence",
            [WhisperSegment(words=[], avg_logprob=-1.5, no_speech_prob=0.8)],
            (0.0, 0.5),
            id="low_confidence_poor_audio",
        ),
    ],
)
def test_estimate_confidence(
    engine: TranscriptionEngine, text, segments, expected_confidence_range
):
    whisper_result = WhisperResult(text=text, segments=segments, language="en")
    confidence = engine._estimate_confidence(whisper_result, text)

    assert 0.0 <= confidence <= 1.0
    min_conf, max_conf = expected_confidence_range
    assert min_conf <= confidence <= max_conf


@pytest.mark.parametrize(
    ("prev_speaker", "prev_end_time", "current_start_time", "expected_speaker"),
    [
        pytest.param(
            None,
            None,
            0.0,
            "SPEAKER_01",
            id="first_chunk_no_previous",
        ),
        pytest.param(
            "SPEAKER_01",
            2.5,
            2.7,
            "SPEAKER_01",
            id="same_speaker_short_gap_0_2s",
        ),
        pytest.param(
            "SPEAKER_01",
            2.5,
            3.4,
            "SPEAKER_01",
            id="same_speaker_gap_0_9s",
        ),
        pytest.param(
            "SPEAKER_01",
            2.5,
            4.0,
            "SPEAKER_02",
            id="speaker_change_gap_1_5s",
        ),
        pytest.param(
            "SPEAKER_02",
            5.0,
            6.5,
            "SPEAKER_01",
            id="speaker_change_back_to_speaker_01",
        ),
    ],
)
def test_detect_speaker(
    engine: TranscriptionEngine,
    prev_speaker,
    prev_end_time,
    current_start_time,
    expected_speaker,
):
    previous_record = None
    if prev_speaker is not None:
        previous_record = TranscriptionRecord(
            chunk_id="prev",
            session_id="test",
            speaker_id=prev_speaker,
            text="Previous",
            timestamp_start=0.0,
            timestamp_end=prev_end_time,
            confidence=0.9,
        )

    current_chunk = AudioChunk(
        session_id="test",
        timestamp_start=current_start_time,
        timestamp_end=current_start_time + 2.5,
        sample_rate=16000,
        duration_ms=2500,
        audio_data="base64",
    )

    speaker_id = engine._detect_speaker(current_chunk, previous_record)
    assert speaker_id == expected_speaker


@pytest.mark.asyncio
async def test_transcribe_generates_record(engine: TranscriptionEngine, sample_chunk):
    record, state = await engine.transcribe(sample_chunk)

    assert record.chunk_id == sample_chunk.chunk_id
    assert record.session_id == sample_chunk.session_id
    assert len(record.text) > 0
    assert record.language == "en"
    assert 0.0 <= record.confidence <= 1.0
    assert record.timestamp_start == sample_chunk.timestamp_start
    assert record.timestamp_end == sample_chunk.timestamp_end
    assert isinstance(state, TranscriptionState)


@pytest.mark.asyncio
async def test_transcribe_filters_context_words(
    engine: TranscriptionEngine, sample_chunk
):
    previous_audio = np.ones(16000)
    state = TranscriptionState(previous_audio=previous_audio)

    engine.model.transcribe.return_value = {
        "text": "Context word current word",
        "segments": [
            {
                "words": [
                    {"word": "Context", "start": 0.1, "end": 0.4, "probability": 0.9},
                    {"word": "word", "start": 0.9, "end": 1.2, "probability": 0.9},
                    {"word": "current", "start": 1.2, "end": 1.5, "probability": 0.95},
                    {"word": "word", "start": 1.6, "end": 1.9, "probability": 0.93},
                ],
                "avg_logprob": -0.3,
                "no_speech_prob": 0.1,
            }
        ],
        "language": "en",
    }

    record, new_state = await engine.transcribe(sample_chunk, state)

    assert len(record.words) >= 1
    for word in record.words:
        assert word.start >= 0.0


@pytest.mark.asyncio
async def test_transcribe_saves_to_jsonl(
    engine: TranscriptionEngine, sample_chunk, test_output_dir
):
    record, state = await engine.transcribe(sample_chunk)

    jsonl_file = test_output_dir / "transcription.jsonl"
    assert jsonl_file.exists()

    content = jsonl_file.read_text()
    assert sample_chunk.session_id in content
    assert "text" in content
    assert "confidence" in content
