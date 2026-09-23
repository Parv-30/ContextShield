from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from contextshield.db.models import Base, Conversation, Message, MessageLabel, EscalationSnapshot, TrendSnapshot


def make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_create_conversation_with_messages_and_labels():
    session = make_session()
    convo = Conversation(source="upload", language="en")
    session.add(convo)
    session.flush()

    msg = Message(conversation_id=convo.id, seq_index=0, text="hello", language="en")
    session.add(msg)
    session.flush()

    label = MessageLabel(message_id=msg.id, category="offensive", probability=0.8, toxicity_score=0.8)
    session.add(label)
    session.commit()

    fetched = session.query(Conversation).first()
    assert fetched.language == "en"
    assert len(fetched.messages) == 1
    assert fetched.messages[0].labels[0].category == "offensive"


def test_escalation_snapshot_and_trend_snapshot():
    session = make_session()
    convo = Conversation(source="seed", language="hi")
    session.add(convo)
    session.flush()

    snapshot = EscalationSnapshot(
        conversation_id=convo.id, seq_index=2, windowed_mean=0.5, slope=0.1,
        escalation_score=0.4, status="elevated",
    )
    session.add(snapshot)

    trend = TrendSnapshot(
        time_bucket="2026-01-01", category="hate_speech", language="hi",
        topic_id=3, topic_terms="term1,term2", message_count=5,
    )
    session.add(trend)
    session.commit()

    assert session.query(EscalationSnapshot).count() == 1
    assert session.query(TrendSnapshot).first().message_count == 5
