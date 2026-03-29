import base64

import numpy as np
import pytest
import soundfile as sf

from src.audio_streamer import AudioStreamer
from src.audio_streamer.models import AudioChunk
from src.config.config_models import AudioStreamerConfig


@pytest.fixture
def streamer(audio_config):
    return AudioStreamer(config=audio_config)


@pytest.mark.parametrize(
    ("chunk_duration", "overlap", "target_sample_rate", "should_fail"),
    [
        pytest.param(2.5, 3.0, 16000, True, id="overlap_exceeds_chunk_duration"),
        pytest.param(2.5, 2.5, 16000, True, id="overlap_equals_chunk_duration"),
        pytest.param(2.5, 0.5, 16000, False, id="valid_config"),
        pytest.param(3.0, 1.0, 16000, False, id="valid_config_different_values"),
    ],
)
def test_config_validation(chunk_duration, overlap, target_sample_rate, should_fail):
    if should_fail:
        with pytest.raises(ValueError, match="must be less than chunk_duration"):
            AudioStreamerConfig(
                chunk_duration=chunk_duration,
                overlap=overlap,
                target_sample_rate=target_sample_rate,
            )
    else:
        config = AudioStreamerConfig(
            chunk_duration=chunk_duration,
            overlap=overlap,
            target_sample_rate=target_sample_rate,
        )
        assert config.chunk_duration == chunk_duration
        assert config.overlap == overlap


def test_load_audio(streamer: AudioStreamer, test_audio_file):
    audio_data, sample_rate = streamer._load_audio(str(test_audio_file))

    assert isinstance(audio_data, np.ndarray)
    assert sample_rate == 16000
    assert len(audio_data) > 0


def test_load_audio_not_found(streamer: AudioStreamer):
    with pytest.raises(
        (FileNotFoundError, ValueError),
        match="(Audio file not found|Failed to load audio file)",
    ):
        streamer._load_audio("nonexistent.wav")


@pytest.mark.parametrize(
    ("input_audio", "expected_output"),
    [
        pytest.param(
            np.array([[0.5, 0.3], [0.7, 0.2], [0.1, 0.9]]),
            np.array([0.4, 0.45, 0.5]),
            id="stereo_to_mono",
        ),
        pytest.param(
            np.array([0.5, 0.7, 0.1]),
            np.array([0.5, 0.7, 0.1]),
            id="already_mono",
        ),
    ],
)
def test_convert_to_mono(streamer: AudioStreamer, input_audio, expected_output):
    result = streamer._convert_to_mono(input_audio)
    assert result.ndim == 1
    np.testing.assert_array_almost_equal(result, expected_output)


@pytest.mark.parametrize(
    ("orig_rate", "target_rate", "audio_duration", "expected_samples"),
    [
        pytest.param(16000, 16000, 1.0, 16000, id="same_rate_no_resampling"),
        pytest.param(44100, 16000, 1.0, 16000, id="downsample_44k_to_16k"),
        pytest.param(8000, 16000, 1.0, 16000, id="upsample_8k_to_16k"),
    ],
)
def test_resample_audio(
    streamer: AudioStreamer, orig_rate, target_rate, audio_duration, expected_samples
):
    audio = np.ones(int(orig_rate * audio_duration))
    result = streamer._resample_audio(audio, orig_rate, target_rate)

    assert len(result) == expected_samples


def test_extract_chunks(streamer: AudioStreamer):
    sample_rate = 16000
    audio = np.ones(10 * sample_rate)
    chunks = streamer._extract_chunks(audio, sample_rate)
    assert len(chunks) == 5

    chunk_audio, start_time, end_time = chunks[0]
    assert start_time == 0.0
    assert end_time == 2.5
    assert len(chunk_audio) == int(2.5 * sample_rate)

    chunk_audio, start_time, end_time = chunks[1]
    assert start_time == 2.0
    assert end_time == 4.5


def test_audio_to_base64(streamer: AudioStreamer, sample_audio_data):
    base64_str = streamer._audio_to_base64(sample_audio_data, 16000)
    assert isinstance(base64_str, str)
    assert len(base64_str) > 0
    audio_bytes = base64.b64decode(base64_str)
    assert len(audio_bytes) > 0


@pytest.mark.asyncio
async def test_stream_generates_chunks(streamer: AudioStreamer, test_audio_file):
    chunks = []

    async for chunk in streamer.stream(str(test_audio_file), "test_session"):
        chunks.append(chunk)

    assert len(chunks) >= 1

    chunk = chunks[0]
    assert isinstance(chunk, AudioChunk)
    assert chunk.session_id == "test_session"
    assert chunk.sample_rate == 16000
    assert chunk.timestamp_start >= 0.0
    assert chunk.timestamp_end > chunk.timestamp_start
    assert len(chunk.audio_data) > 0


@pytest.mark.asyncio
async def test_stream_chunk_overlap(streamer: AudioStreamer, tmp_path):
    sample_rate = 16000
    audio = np.ones(5 * sample_rate)
    audio_file = tmp_path / "test.wav"
    sf.write(audio_file, audio, sample_rate)

    chunks = []
    async for chunk in streamer.stream(str(audio_file), "test_session"):
        chunks.append(chunk)

    assert len(chunks) >= 2

    if len(chunks) >= 2:
        assert chunks[1].timestamp_start < chunks[0].timestamp_end


@pytest.mark.asyncio
async def test_stream_empty_file_raises_error(streamer: AudioStreamer, tmp_path):
    empty_file = tmp_path / "empty.wav"
    sf.write(empty_file, np.array([]), 16000)

    with pytest.raises(ValueError, match="Audio file is empty"):
        async for _ in streamer.stream(str(empty_file), "test_session"):
            pass


@pytest.mark.parametrize(
    ("audio_generator", "duration", "description"),
    [
        pytest.param(
            lambda sr, dur: np.zeros(int(sr * dur)),
            5.0,
            "complete_silence",
            id="silent_audio_all_zeros",
        ),
        pytest.param(
            lambda sr, dur: np.random.uniform(-0.001, 0.001, int(sr * dur)),
            5.0,
            "very_quiet_noise",
            id="very_low_amplitude_noise",
        ),
        pytest.param(
            lambda sr, dur: np.concatenate(
                [
                    np.sin(2 * np.pi * 440 * np.linspace(0, 2, int(sr * 2))),
                    np.zeros(int(sr * 2)),
                    np.sin(2 * np.pi * 440 * np.linspace(0, 1, int(sr * 1))),
                ]
            ),
            5.0,
            "speech_with_silence_gaps",
            id="mixed_audio_with_silence_gaps",
        ),
    ],
)
@pytest.mark.asyncio
async def test_stream_edge_case_audio(
    streamer: AudioStreamer, tmp_path, audio_generator, duration, description
):
    sample_rate = 16000
    audio = audio_generator(sample_rate, duration).astype(np.float32)

    audio_file = tmp_path / f"{description}.wav"
    sf.write(audio_file, audio, sample_rate)

    chunks = []
    async for chunk in streamer.stream(str(audio_file), "test_session"):
        chunks.append(chunk)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert isinstance(chunk, AudioChunk)
        assert len(chunk.audio_data) > 0
        assert chunk.sample_rate == 16000


@pytest.mark.asyncio
async def test_stream_noisy_audio(streamer: AudioStreamer, tmp_path):
    sample_rate = 16000
    duration = 3.0

    noise = np.random.uniform(-1.0, 1.0, int(sample_rate * duration))
    noise = np.clip(noise * 2, -0.99, 0.99)

    noisy_file = tmp_path / "noisy.wav"
    sf.write(noisy_file, noise.astype(np.float32), sample_rate)

    chunks = []
    async for chunk in streamer.stream(str(noisy_file), "test_session"):
        chunks.append(chunk)

    assert len(chunks) >= 1
    assert all(chunk.sample_rate == 16000 for chunk in chunks)
