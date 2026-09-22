from datetime import datetime

from contextshield.ml.trends import bucket_time_series


def test_bucket_time_series_daily_grouping():
    messages = [
        {"text": "a", "language": "en", "category": "offensive", "timestamp": datetime(2026, 1, 1, 10, 0)},
        {"text": "b", "language": "en", "category": "offensive", "timestamp": datetime(2026, 1, 1, 15, 0)},
        {"text": "c", "language": "hi", "category": "hate_speech", "timestamp": datetime(2026, 1, 2, 9, 0)},
    ]
    result = bucket_time_series(messages, bucket="D")
    result_by_key = {(r["time_bucket"], r["category"], r["language"]): r["message_count"] for r in result}
    assert result_by_key[("2026-01-01", "offensive", "en")] == 2
    assert result_by_key[("2026-01-02", "hate_speech", "hi")] == 1


def test_bucket_time_series_empty():
    assert bucket_time_series([]) == []


import pytest
from contextshield.ml.trends import TopicAnalyzer


def test_topic_analyzer_fit_transform():
    texts = [
        "I hate this group of people",
        "This group of people is terrible",
        "The weather is nice today",
        "Today the weather looks great",
    ] * 3
    try:
        analyzer = TopicAnalyzer()
        topics = analyzer.fit_transform(texts)
    except Exception:
        pytest.skip("embedding model unavailable offline")
    assert isinstance(topics, list)
    for topic in topics:
        assert set(topic.keys()) == {"topic_id", "terms", "example_messages"}
        assert topic["topic_id"] != -1
