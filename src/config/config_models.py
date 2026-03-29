from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class AudioStreamerConfig(BaseModel):
    chunk_duration: float = Field(
        gt=0, default=2.5, description="Chunk duration in seconds"
    )
    overlap: float = Field(
        ge=0, default=0.5, description="Overlap between chunks in seconds"
    )
    target_sample_rate: int = Field(
        gt=0, default=16000, description="Target sample rate in Hz"
    )

    @field_validator("overlap")
    @classmethod
    def validate_overlap(cls, v, info):
        chunk_duration = info.data.get("chunk_duration")
        if chunk_duration and v >= chunk_duration:
            raise ValueError(
                f"Overlap ({v}s) must be less than chunk_duration ({chunk_duration}s)"
            )
        return v


class TranscriptionConfig(BaseModel):
    model_name: str = Field(
        default="base",
        description="Whisper model: tiny, base, small, medium, large",
    )
    output_dir: Path = Field(
        default=Path("output"), description="Output directory for transcription results"
    )
    context_duration: float = Field(
        default=0.5,
        gt=0,
        description="Sliding window context duration in seconds (from previous chunk)",
    )


class ClassifierConfig(BaseModel):
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for semantic classification",
    )
    class_descriptions: dict[str, str] = Field(
        default={
            "private": (
                "Personal, confidential, and intimate conversation with emotional tone, "
                "personal information like names, addresses, phone numbers, "
                "private family matters, medical consultations, or sensitive topics"
            ),
            "topic_based": (
                "Professional, structured, business-oriented discussion about specific topics, "
                "agenda items, technical subjects, meetings, lectures, presentations, "
                "customer support, or work-related matters"
            ),
        },
        description="Class descriptions for semantic classification",
    )


class ProjectConfig(BaseModel):
    output_dir: Path = Field(
        default=Path("output"), description="Output directory for results"
    )
    data_dir: Path = Field(default=Path("data"), description="Input data directory")


class PipelineConfig(BaseModel):
    project: ProjectConfig = Field(
        default_factory=ProjectConfig, description="Project configuration"
    )
    audio_streamer: AudioStreamerConfig = Field(
        default_factory=AudioStreamerConfig, description="AudioStreamer configuration"
    )
    transcription: TranscriptionConfig = Field(
        default_factory=TranscriptionConfig, description="Transcription configuration"
    )
    classifier: ClassifierConfig = Field(
        default_factory=ClassifierConfig, description="Classifier configuration"
    )

    model_config = {"extra": "forbid"}
