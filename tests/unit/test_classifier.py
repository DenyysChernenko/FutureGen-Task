from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.classifier.engine import ClassifierEngine
from src.classifier.keywords import ClassifierKeywords
from src.classifier.models import ClassificationResult, ConversationType, Sentiment
from src.transcription.models import TranscriptionRecord


@pytest.fixture
def classifier(mock_sentence_transformer):
    with patch(
        "src.classifier.engine.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ):
        return ClassifierEngine(model_name="all-MiniLM-L6-v2")


@pytest.mark.parametrize(
    ("text", "expected_topics", "min_count"),
    [
        pytest.param(
            "we need to fix the software bug in our app and update the database",
            ["technology"],
            1,
            id="single_topic_technology",
        ),
        pytest.param(
            "random text with no keywords",
            [],
            0,
            id="no_matching_topics",
        ),
        pytest.param(
            "meeting agenda includes software development and budget discussion",
            ["business", "technology"],
            1,
            id="multiple_topics_mixed",
        ),
    ],
)
def test_extract_topics(classifier: ClassifierEngine, text, expected_topics, min_count):
    topics = classifier._extract_topics(text.lower())

    assert len(topics) >= min_count

    if expected_topics:
        assert any(topic in topics for topic in expected_topics)


@pytest.mark.parametrize(
    ("text", "expected_keywords", "min_count"),
    [
        pytest.param(
            "my phone number is 123-456-7890 and I love this",
            ["phone", "love"],
            1,
            id="detects_personal_and_emotional",
        ),
        pytest.param(
            "this is a normal business conversation",
            [],
            0,
            id="no_privacy_signals",
        ),
        pytest.param(
            "keep this secret and private please",
            ["secret", "private"],
            1,
            id="detects_confidential_keywords",
        ),
    ],
)
def test_detect_privacy_signals(
    classifier: ClassifierEngine, text, expected_keywords, min_count
):
    signals = classifier._detect_privacy_signals(text.lower())

    assert len(signals) >= min_count

    if expected_keywords:
        assert any(keyword in " ".join(signals) for keyword in expected_keywords)


@pytest.mark.parametrize(
    ("text", "expected_sentiment"),
    [
        pytest.param(
            "this is great excellent amazing wonderful",
            Sentiment.POSITIVE,
            id="positive_sentiment",
        ),
        pytest.param(
            "this is terrible awful horrible bad",
            Sentiment.NEGATIVE,
            id="negative_sentiment",
        ),
        pytest.param(
            "this is a normal conversation",
            Sentiment.NEUTRAL,
            id="neutral_sentiment",
        ),
        pytest.param(
            "happy wonderful good great love",
            Sentiment.POSITIVE,
            id="multiple_positive_words",
        ),
    ],
)
def test_analyze_sentiment(classifier: ClassifierEngine, text, expected_sentiment):
    sentiment = classifier._analyze_sentiment(text.lower())
    assert sentiment == expected_sentiment


def test_count_participants(classifier: ClassifierEngine):
    transcripts = [
        TranscriptionRecord(
            chunk_id="1",
            session_id="test",
            speaker_id="SPEAKER_01",
            text="Hello",
            timestamp_start=0.0,
            timestamp_end=1.0,
            confidence=0.9,
        ),
        TranscriptionRecord(
            chunk_id="2",
            session_id="test",
            speaker_id="SPEAKER_02",
            text="Hi",
            timestamp_start=1.5,
            timestamp_end=2.0,
            confidence=0.9,
        ),
        TranscriptionRecord(
            chunk_id="3",
            session_id="test",
            speaker_id="SPEAKER_01",
            text="How are you",
            timestamp_start=2.5,
            timestamp_end=3.5,
            confidence=0.9,
        ),
    ]

    count = classifier._count_participants(transcripts)
    assert count == 2


def test_count_participants_single(classifier: ClassifierEngine):
    transcripts = [
        TranscriptionRecord(
            chunk_id="1",
            session_id="test",
            speaker_id="SPEAKER_01",
            text="Hello world",
            timestamp_start=0.0,
            timestamp_end=1.0,
            confidence=0.9,
        ),
    ]
    count = classifier._count_participants(transcripts)
    assert count == 1


@pytest.mark.parametrize(
    ("text", "expected_classification"),
    [
        pytest.param(
            "let's discuss the meeting agenda and project timeline",
            ConversationType.TOPIC_BASED,
            id="business_meeting_topic_based",
        ),
        pytest.param(
            "I love you and I'm worried about our personal relationship",
            ConversationType.PRIVATE,
            id="personal_emotional_private",
        ),
        pytest.param(
            "technical discussion about software architecture",
            ConversationType.TOPIC_BASED,
            id="technical_topic_based",
        ),
    ],
)
@pytest.mark.asyncio
async def test_classify_type(
    classifier: ClassifierEngine, text, expected_classification
):
    classification, confidence = await classifier._classify_type(text)
    assert classification == expected_classification
    assert 0.0 <= confidence <= 1.0


@pytest.mark.asyncio
async def test_classify_full(classifier: ClassifierEngine, mock_transcription_records):
    result = await classifier.classify(
        mock_transcription_records, session_id="test_session"
    )

    assert isinstance(result, ClassificationResult)
    assert result.session_id == "test_session"
    assert result.classification in [
        ConversationType.PRIVATE,
        ConversationType.TOPIC_BASED,
    ]
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.topics, list)
    assert isinstance(result.privacy_signals, list)
    assert result.sentiment in [
        Sentiment.POSITIVE,
        Sentiment.NEUTRAL,
        Sentiment.NEGATIVE,
    ]
    assert result.participants_count >= 1
    assert result.total_duration_s > 0.0
    assert "whisper" in result.model_used.lower()


@pytest.mark.asyncio
async def test_classify_empty_transcripts(classifier: ClassifierEngine):
    result = await classifier.classify([], session_id="test_session")

    assert result.classification == ConversationType.TOPIC_BASED
    assert result.confidence == 0.0
    assert result.topics == []
    assert result.privacy_signals == []
    assert result.participants_count == 0
    assert result.total_duration_s == 0.0


@pytest.mark.asyncio
async def test_classify_calculates_duration(
    classifier: ClassifierEngine, mock_transcription_records
):
    result = await classifier.classify(
        mock_transcription_records, session_id="test_session"
    )
    assert result.total_duration_s == 4.5


def test_custom_keywords():
    custom_keywords = ClassifierKeywords()
    custom_keywords.topics.technology = ["custom", "keyword"]

    with patch("src.classifier.engine.SentenceTransformer"):
        classifier = ClassifierEngine(
            model_name="all-MiniLM-L6-v2", keywords=custom_keywords
        )

    text = "this is about custom keyword"
    topics = classifier._extract_topics(text.lower())
    assert "technology" in topics


def test_custom_class_descriptions():
    custom_descriptions = {
        "private": "Custom private description",
        "topic_based": "Custom topic description",
    }

    with patch("src.classifier.engine.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.8, 0.9]])
        mock_st.return_value = mock_model

        classifier = ClassifierEngine(
            model_name="all-MiniLM-L6-v2", class_descriptions=custom_descriptions
        )

    assert classifier.class_descriptions == custom_descriptions
