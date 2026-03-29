from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AudioChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = Field(..., description="Unique identifier for the audio session")
    timestamp_start: float = Field(
        ..., description="Start time as Unix timestamp in seconds"
    )
    timestamp_end: float = Field(
        ..., description="End time as Unix timestamp in seconds"
    )
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    duration_ms: int = Field(..., description="Duration in milliseconds")
    audio_data: str = Field(..., description="Base64 encoded PCM audio data")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "chunk_id": "550e8400-e29b-41d4-a716-446655440001",
                "session_id": "meeting_20260328",
                "timestamp_start": 1700000000.000,
                "timestamp_end": 1700000002.500,
                "sample_rate": 16000,
                "duration_ms": 2500,
                "audio_data": "UklGRiQAAABXQVZFZm10...",
            }
        },
    )
