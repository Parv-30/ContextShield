# backend/contextshield/services/trend_service.py
from contextshield.db.models import Message, MessageLabel, TrendSnapshot
from contextshield.ml.trends import bucket_time_series, TopicAnalyzer

MIN_MESSAGES_FOR_TOPICS = 10


def recompute_trends(session) -> int:
    rows = (
        session.query(Message, MessageLabel)
        .join(MessageLabel, MessageLabel.message_id == Message.id)
        .filter(MessageLabel.category != "none")
        .filter(MessageLabel.probability >= 0.5)
        .all()
    )

    harmful_messages = [
        {
            "text": message.text,
            "language": message.language,
            "category": label.category,
            "timestamp": message.created_at,
        }
        for message, label in rows
    ]

    session.query(TrendSnapshot).delete()

    if not harmful_messages:
        session.commit()
        return 0

    time_series = bucket_time_series(harmful_messages)

    topic_by_bucket_key = {}
    if len(harmful_messages) >= MIN_MESSAGES_FOR_TOPICS:
        texts = [m["text"] for m in harmful_messages]
        try:
            analyzer = TopicAnalyzer()
            topics = analyzer.fit_transform(texts)
            if topics:
                topic_by_bucket_key["_top_topic"] = topics[0]
        except Exception:
            pass

    created = 0
    for entry in time_series:
        top_topic = topic_by_bucket_key.get("_top_topic")
        session.add(
            TrendSnapshot(
                time_bucket=entry["time_bucket"],
                category=entry["category"],
                language=entry["language"],
                topic_id=top_topic["topic_id"] if top_topic else None,
                topic_terms=",".join(top_topic["terms"][:5]) if top_topic else None,
                message_count=entry["message_count"],
            )
        )
        created += 1

    session.commit()
    return created
