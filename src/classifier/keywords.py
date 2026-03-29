from pydantic import BaseModel, Field


class TopicKeywords(BaseModel):
    technology: list[str] = Field(
        default=[
            "software",
            "code",
            "app",
            "system",
            "database",
            "api",
            "programming",
        ]
    )
    business: list[str] = Field(
        default=[
            "meeting",
            "agenda",
            "project",
            "deadline",
            "budget",
            "report",
            "sales",
        ]
    )
    education: list[str] = Field(
        default=["lecture", "course", "study", "exam", "homework", "class"]
    )
    support: list[str] = Field(
        default=["issue", "problem", "fix", "help", "support", "ticket"]
    )
    health: list[str] = Field(
        default=["doctor", "medical", "health", "treatment", "symptoms"]
    )
    finance: list[str] = Field(
        default=["money", "payment", "cost", "price", "budget", "invoice"]
    )

    def to_dict(self) -> dict[str, list[str]]:
        return self.model_dump()


class SentimentKeywords(BaseModel):
    positive: list[str] = Field(
        default=[
            "good",
            "great",
            "excellent",
            "happy",
            "love",
            "perfect",
            "wonderful",
            "amazing",
        ]
    )
    negative: list[str] = Field(
        default=[
            "bad",
            "terrible",
            "hate",
            "sad",
            "angry",
            "awful",
            "horrible",
            "disappointed",
        ]
    )


class PrivacyKeywords(BaseModel):
    personal: list[str] = Field(default=["name", "address", "phone", "email"])
    emotional: list[str] = Field(default=["love", "hate", "worry", "fear", "trust"])
    confidential: list[str] = Field(
        default=["secret", "private", "confidential", "personal"]
    )

    def to_dict(self) -> dict[str, list[str]]:
        return self.model_dump()


class ClassifierKeywords(BaseModel):
    topics: TopicKeywords = Field(default_factory=TopicKeywords)
    sentiment: SentimentKeywords = Field(default_factory=SentimentKeywords)
    privacy: PrivacyKeywords = Field(default_factory=PrivacyKeywords)
