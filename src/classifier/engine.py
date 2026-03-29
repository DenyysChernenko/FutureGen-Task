import asyncio
import logging
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..transcription import TranscriptionRecord
from .keywords import ClassifierKeywords
from .models import ClassificationResult, ConversationType, Sentiment

logger = logging.getLogger(__name__)


class ClassifierEngine:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        class_descriptions: dict[str, str] | None = None,
        keywords: ClassifierKeywords | None = None,
    ):
        self.model_name = model_name
        self.keywords = keywords or ClassifierKeywords()
        self.class_descriptions = class_descriptions or {
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
        }

        logger.info(f"Loading sentence-transformers model: {model_name}")

        self.model = SentenceTransformer(model_name)

        self.class_embeddings = self.model.encode(
            list(self.class_descriptions.values())
        )

        logger.info(f"Model '{model_name}' loaded successfully")

    async def classify(
        self, transcripts: list[TranscriptionRecord], session_id: str
    ) -> ClassificationResult:
        if not transcripts:
            logger.warning(f"No transcripts provided for session {session_id}")
            return self._empty_classification(session_id)

        full_text = " ".join([t.text for t in transcripts if t.text.strip()])

        if not full_text.strip():
            logger.warning(f"Empty text for session {session_id}")
            return self._empty_classification(session_id)

        classification, confidence = await self._classify_type(full_text)

        topics = self._extract_topics(full_text.lower())
        dominant_topic = topics[0] if topics else None

        privacy_signals = self._detect_privacy_signals(full_text.lower())
        sentiment = self._analyze_sentiment(full_text.lower())

        participants_count = self._count_participants(transcripts)

        total_duration = (
            transcripts[-1].timestamp_end - transcripts[0].timestamp_start
            if transcripts
            else 0.0
        )

        result = ClassificationResult(
            session_id=session_id,
            classification=classification,
            confidence=confidence,
            topics=topics[:3],
            privacy_signals=privacy_signals,
            dominant_topic=dominant_topic,
            sentiment=sentiment,
            participants_count=participants_count,
            total_duration_s=round(total_duration, 2),
            model_used=f"whisper-base + {self.model_name}",
        )

        logger.info(
            f"Classification: {classification} "
            f"(confidence: {confidence:.2f}, topics: {topics[:3]})"
        )

        return result

    async def _classify_type(self, text: str) -> tuple[ConversationType, float]:
        conversation_embedding = await asyncio.to_thread(self.model.encode, text)

        similarities = cosine_similarity(
            [conversation_embedding], self.class_embeddings
        )[0]

        best_idx = int(np.argmax(similarities))
        confidence = float(similarities[best_idx])

        classification = (
            ConversationType.PRIVATE if best_idx == 0 else ConversationType.TOPIC_BASED
        )

        logger.debug(
            f"Similarity scores - Private: {similarities[0]:.3f}, "
            f"Topic-based: {similarities[1]:.3f}"
        )

        return classification, round(confidence, 2)

    def _extract_topics(self, text: str) -> list[str]:
        topic_scores: Counter[str] = Counter()

        for topic, keywords in self.keywords.topics.to_dict().items():
            score = sum(text.count(keyword) for keyword in keywords)
            if score > 0:
                topic_scores[topic] = score

        return [topic for topic, _ in topic_scores.most_common()]

    def _detect_privacy_signals(self, text: str) -> list[str]:
        signals = []

        for category, keywords in self.keywords.privacy.to_dict().items():
            for keyword in keywords:
                if keyword in text:
                    signals.append(f"{category}: {keyword}")
                    break

        return signals

    def _analyze_sentiment(self, text: str) -> Sentiment:
        positive_count = sum(
            text.count(word) for word in self.keywords.sentiment.positive
        )
        negative_count = sum(
            text.count(word) for word in self.keywords.sentiment.negative
        )

        if positive_count > negative_count:
            return Sentiment.POSITIVE
        elif negative_count > positive_count:
            return Sentiment.NEGATIVE
        else:
            return Sentiment.NEUTRAL

    def _count_participants(self, transcripts: list[TranscriptionRecord]) -> int:
        speakers = set(t.speaker_id for t in transcripts if t.speaker_id)
        return len(speakers) if speakers else 1

    def _empty_classification(self, session_id: str) -> ClassificationResult:
        return ClassificationResult(
            session_id=session_id,
            classification=ConversationType.TOPIC_BASED,
            confidence=0.0,
            sentiment=Sentiment.NEUTRAL,
            participants_count=0,
            total_duration_s=0.0,
            model_used=f"whisper-base + {self.model_name}",
        )
