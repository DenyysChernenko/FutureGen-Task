import asyncio
import base64
import io
import logging

import aiofiles  # type: ignore
import numpy as np
import soundfile as sf
import whisper

from ..audio_streamer import AudioChunk
from ..config.config_models import TranscriptionConfig
from .models import TranscriptionRecord, TranscriptionState, WhisperResult, Word

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
MAX_RETRY_DELAY = 10.0  # seconds


class TranscriptionEngine:

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.context_duration = config.context_duration

        logger.info(f"Loading Whisper model: {config.model_name}")
        logger.info(f"Context window: {self.context_duration}s")
        self.model = whisper.load_model(config.model_name)
        logger.info(f"Whisper model '{config.model_name}' loaded successfully")

    async def transcribe(
        self, chunk: AudioChunk, state: TranscriptionState | None = None
    ) -> tuple[TranscriptionRecord, TranscriptionState]:
        if state is None:
            state = TranscriptionState()

        audio_array = await asyncio.to_thread(self._decode_audio, chunk.audio_data)

        audio_with_context, context_offset = await asyncio.to_thread(
            self._apply_sliding_window, audio_array, chunk, state.previous_audio
        )

        result_dict = await self._transcribe_with_retry(
            audio_with_context,
            chunk.chunk_id,
            chunk.timestamp_start,
            chunk.timestamp_end,
        )

        result = WhisperResult(**result_dict)

        text = result.text.strip()
        language = result.language

        timestamp_base = chunk.timestamp_start - context_offset
        words = self._extract_words(result, timestamp_base)

        words = [w for w in words if w.start >= chunk.timestamp_start]

        text = " ".join([w.word for w in words]) if words else text

        confidence = self._estimate_confidence(result, text)

        speaker_id = self._detect_speaker(chunk, state.previous_record)

        record = TranscriptionRecord(
            chunk_id=chunk.chunk_id,
            session_id=chunk.session_id,
            speaker_id=speaker_id,
            text=text,
            language=language,
            timestamp_start=chunk.timestamp_start,
            timestamp_end=chunk.timestamp_end,
            confidence=confidence,
            words=words,
        )

        await self._append_to_jsonl(record)

        new_state = TranscriptionState(
            previous_audio=audio_array, previous_record=record
        )

        return record, new_state

    def _decode_audio(self, base64_audio: str) -> np.ndarray:
        audio_bytes = base64.b64decode(base64_audio)

        buffer = io.BytesIO(audio_bytes)
        try:
            audio_data, _ = sf.read(buffer, dtype="float32")
            return audio_data
        finally:
            buffer.close()

    async def _transcribe_with_retry(
        self,
        audio_with_context: np.ndarray,
        chunk_id: str,
        timestamp_start: float,
        timestamp_end: float,
    ) -> dict:
        last_exception = None

        for attempt in range(MAX_RETRIES):
            try:
                result_dict = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.model.transcribe,
                        audio_with_context,
                        word_timestamps=True,
                    ),
                    timeout=60.0,
                )
                if attempt > 0:
                    logger.info(
                        f"Transcription succeeded on attempt {attempt + 1} "
                        f"for chunk {chunk_id}"
                    )
                return result_dict

            except asyncio.TimeoutError as e:
                last_exception = e
                retry_delay = min(INITIAL_RETRY_DELAY * (2**attempt), MAX_RETRY_DELAY)

                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Transcription timeout for chunk {chunk_id} "
                        f"({timestamp_start:.2f}s - {timestamp_end:.2f}s). "
                        f"Attempt {attempt + 1}/{MAX_RETRIES}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"Transcription timeout for chunk {chunk_id} "
                        f"after {MAX_RETRIES} attempts"
                    )

        raise TimeoutError(
            f"Whisper transcription exceeded 60s timeout for chunk {chunk_id} "
            f"after {MAX_RETRIES} retries"
        ) from last_exception

    def _apply_sliding_window(
        self,
        audio_array: np.ndarray,
        chunk: AudioChunk,
        previous_audio: np.ndarray | None,
    ) -> tuple[np.ndarray, float]:
        if previous_audio is not None:
            context_samples = int(self.context_duration * chunk.sample_rate)
            context = previous_audio[-context_samples:]

            audio_with_context = np.concatenate([context, audio_array])
            context_offset = self.context_duration

            logger.debug(
                f"Added {self.context_duration}s context "
                f"({len(context)} samples) to chunk {chunk.chunk_id}"
            )
        else:
            audio_with_context = audio_array
            context_offset = 0.0
            logger.debug(f"No context available for first chunk {chunk.chunk_id}")

        return audio_with_context, context_offset

    def _extract_words(
        self, whisper_result: WhisperResult, chunk_offset: float
    ) -> list[Word]:
        words = []

        for segment in whisper_result.segments:
            for word_info in segment.words:
                words.append(
                    Word(
                        word=word_info.word.strip(),
                        start=word_info.start + chunk_offset,
                        end=word_info.end + chunk_offset,
                    )
                )

        return words

    def _estimate_confidence(self, whisper_result: WhisperResult, text: str) -> float:
        if not whisper_result.segments or not text.strip():
            return 0.0

        avg_logprob = float(
            np.mean([seg.avg_logprob for seg in whisper_result.segments])
        )
        logprob_confidence = min(1.0, max(0.0, 1.0 + avg_logprob))

        avg_no_speech = float(
            np.mean([seg.no_speech_prob for seg in whisper_result.segments])
        )
        speech_confidence = 1.0 - avg_no_speech

        confidence = (logprob_confidence + speech_confidence) / 2.0

        return round(confidence, 2)

    def _detect_speaker(
        self, current_chunk: AudioChunk, previous_record: TranscriptionRecord | None
    ) -> str:
        if previous_record is None:
            return "SPEAKER_01"

        silence_gap = current_chunk.timestamp_start - previous_record.timestamp_end

        if silence_gap > 1.0:
            if previous_record.speaker_id == "SPEAKER_01":
                return "SPEAKER_02"
            else:
                return "SPEAKER_01"

        return previous_record.speaker_id or "SPEAKER_01"

    async def _append_to_jsonl(self, record: TranscriptionRecord):
        output_file = self.output_dir / "transcription.jsonl"

        json_line = record.model_dump_json() + "\n"

        async with aiofiles.open(output_file, mode="a") as f:
            await f.write(json_line)
