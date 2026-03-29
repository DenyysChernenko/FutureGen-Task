from dataclasses import dataclass
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class WhisperWord(BaseModel):
    word: str
    start: float
    end: float
    probability: float = 1.0

    model_config = ConfigDict(extra="ignore")


class WhisperSegment(BaseModel):
    words: list[WhisperWord] = Field(default_factory=list)
    avg_logprob: float = -0.5
    no_speech_prob: float = 0.0

    model_config = ConfigDict(extra="ignore")


class WhisperResult(BaseModel):
    text: str
    segments: list[WhisperSegment] = Field(default_factory=list)
    language: str = "en"

    model_config = ConfigDict(extra="ignore")


class Word(BaseModel):
    word: str = Field(..., description="The word text")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")


class TranscriptionRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    chunk_id: str = Field(..., description="Reference to source audio chunk")
    session_id: str = Field(..., description="Session identifier")
    speaker_id: str | None = Field(default=None, description="Speaker identifier")
    text: str = Field(..., description="Transcribed text")
    language: str = Field(default="en", description="Detected language code")
    timestamp_start: float = Field(..., description="Start time in seconds")
    timestamp_end: float = Field(..., description="End time in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    words: list[Word] = Field(default_factory=list, description="Word-level timestamps")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "record_id": "rec_001",
                "chunk_id": "550e8400-e29b-41d4-a716-446655440001",
                "session_id": "meeting_20260328",
                "speaker_id": "SPEAKER_01",
                "text": "Good morning everyone",
                "language": "en",
                "timestamp_start": 0.0,
                "timestamp_end": 2.3,
                "confidence": 0.94,
                "words": [
                    {"word": "Good", "start": 0.0, "end": 0.3},
                    {"word": "morning", "start": 0.3, "end": 0.7},
                    {"word": "everyone", "start": 0.8, "end": 1.3},
                ],
            }
        },
    )


@dataclass
class TranscriptionState:
    previous_audio: np.ndarray | None = None
    previous_record: TranscriptionRecord | None = None
