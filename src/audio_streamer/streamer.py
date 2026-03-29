import asyncio
import base64
import io
import logging
import time
from typing import AsyncIterator

import numpy as np
import soundfile as sf
from scipy import signal

from ..config.config_models import AudioStreamerConfig
from .models import AudioChunk

logger = logging.getLogger(__name__)


class AudioStreamer:

    def __init__(
        self,
        config: AudioStreamerConfig,
    ):
        self.chunk_duration = config.chunk_duration
        self.overlap = config.overlap
        self.target_sample_rate = config.target_sample_rate
        self.hop_duration = config.chunk_duration - config.overlap

        logger.debug(
            f"AudioStreamer initialized: chunk_duration={self.chunk_duration}s, "
            f"overlap={self.overlap}s, target_sample_rate={self.target_sample_rate}Hz"
        )

    def _load_audio(self, audio_file_path: str) -> tuple[np.ndarray, int]:
        try:
            audio_data, sample_rate = sf.read(audio_file_path)

            if len(audio_data) == 0:
                raise ValueError(f"Audio file is empty: {audio_file_path}")

            return audio_data, sample_rate
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}") from e
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {e!r}") from e

    def _convert_to_mono(self, audio_data: np.ndarray) -> np.ndarray:
        if audio_data.ndim == 1:
            return audio_data
        elif audio_data.ndim == 2:
            return audio_data.mean(axis=1)
        else:
            raise ValueError(
                f"Unexpected audio dimensions: {audio_data.ndim}D. "
                f"Expected 1D (mono) or 2D (stereo)"
            )

    def _resample_audio(
        self, audio_data: np.ndarray, orig_sample_rate: int, target_sample_rate: int
    ) -> np.ndarray:
        if orig_sample_rate == target_sample_rate:
            return audio_data

        num_samples = int(len(audio_data) * target_sample_rate / orig_sample_rate)
        resampled = signal.resample(audio_data, num_samples)

        return resampled

    def _extract_chunks(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> list[tuple[np.ndarray, float, float]]:
        chunk_size = int(self.chunk_duration * sample_rate)
        hop_size = int(self.hop_duration * sample_rate)

        chunks = []
        start_sample = 0

        while start_sample < len(audio_data):
            end_sample = min(start_sample + chunk_size, len(audio_data))
            chunk_audio = audio_data[start_sample:end_sample]

            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate

            chunks.append((chunk_audio, start_time, end_time))

            start_sample += hop_size

        return chunks

    def _audio_to_base64(self, audio_samples: np.ndarray, sample_rate: int) -> str:

        buffer = io.BytesIO()
        try:
            sf.write(buffer, audio_samples, sample_rate, format="WAV")
            buffer.seek(0)
            audio_bytes = buffer.read()
            return base64.b64encode(audio_bytes).decode("utf-8")
        finally:
            buffer.close()

    async def stream(
        self,
        audio_file_path: str,
        session_id: str,
        start_timestamp: float | None = None,
    ) -> AsyncIterator[AudioChunk]:
        if start_timestamp is None:
            start_timestamp = time.time()

        audio_data, original_sample_rate = await asyncio.to_thread(
            self._load_audio, audio_file_path
        )

        audio_data = await asyncio.to_thread(self._convert_to_mono, audio_data)

        audio_data = await asyncio.to_thread(
            self._resample_audio,
            audio_data,
            original_sample_rate,
            self.target_sample_rate,
        )

        chunks = await asyncio.to_thread(
            self._extract_chunks, audio_data, self.target_sample_rate
        )

        for chunk_audio, start_time, end_time in chunks:
            audio_base64 = await asyncio.to_thread(
                self._audio_to_base64, chunk_audio, self.target_sample_rate
            )

            duration_ms = int((end_time - start_time) * 1000)

            chunk = AudioChunk(
                session_id=session_id,
                timestamp_start=start_timestamp + start_time,
                timestamp_end=start_timestamp + end_time,
                sample_rate=self.target_sample_rate,
                duration_ms=duration_ms,
                audio_data=audio_base64,
            )

            yield chunk
