from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class ConversationType(str, Enum):
    PRIVATE = "private"
    TOPIC_BASED = "topic_based"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ClassificationResult(BaseModel):
    classification_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    classification: ConversationType
    confidence: float = Field(ge=0.0, le=1.0)
    topics: list[str] = Field(default_factory=list, description="Top-3 topics")
    privacy_signals: list[str] = Field(
        default_factory=list, description="Detected privacy indicators"
    )
    dominant_topic: str | None = Field(default=None, description="Main topic")
    sentiment: Sentiment
    participants_count: int
    total_duration_s: float
    model_used: str = "whisper-base + keyword-classifier"

    model_config = {"extra": "forbid"}
