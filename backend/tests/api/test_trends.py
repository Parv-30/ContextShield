import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from contextshield.db.models import Base, TrendSnapshot
from contextshield.api.main import app
from contextshield.api import deps


@pytest.fixture()
def client_and_session():
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db_session():
        session = TestSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[deps.get_db_session] = override_get_db_session
    client = TestClient(app)
    session = TestSession()
    yield client, session
    app.dependency_overrides.clear()


def test_get_trends_filters_by_category_and_language(client_and_session):
    client, session = client_and_session
    session.add_all([
        TrendSnapshot(time_bucket="2026-01-01", category="offensive", language="en", topic_id=1, topic_terms="a,b", message_count=3),
        TrendSnapshot(time_bucket="2026-01-02", category="hate_speech", language="hi", topic_id=2, topic_terms="c,d", message_count=5),
    ])
    session.commit()

    resp = client.get("/trends", params={"category": "offensive"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["category"] == "offensive"

    resp2 = client.get("/trends", params={"language": "hi"})
    assert len(resp2.json()) == 1
    assert resp2.json()[0]["language"] == "hi"


def test_get_trends_date_range(client_and_session):
    client, session = client_and_session
    session.add_all([
        TrendSnapshot(time_bucket="2026-01-01", category="offensive", language="en", topic_id=None, topic_terms=None, message_count=1),
        TrendSnapshot(time_bucket="2026-02-01", category="offensive", language="en", topic_id=None, topic_terms=None, message_count=2),
    ])
    session.commit()

    resp = client.get("/trends", params={"start": "2026-01-15", "end": "2026-02-28"})
    assert len(resp.json()) == 1
    assert resp.json()[0]["time_bucket"] == "2026-02-01"


def test_recompute_creates_snapshots_from_harmful_messages(client_and_session):
    client, session = client_and_session
    from contextshield.db.models import Conversation, Message, MessageLabel
    from datetime import datetime, timezone

    convo = Conversation(source="seed", language="en")
    session.add(convo)
    session.flush()
    for i in range(12):
        msg = Message(conversation_id=convo.id, seq_index=i, text=f"bad message {i}", language="en", created_at=datetime.now(timezone.utc))
        session.add(msg)
        session.flush()
        session.add(MessageLabel(message_id=msg.id, category="offensive", probability=0.9, toxicity_score=0.9))
    session.commit()

    resp = client.post("/trends/recompute")
    assert resp.status_code == 200
    assert resp.json()["snapshots_created"] >= 1

    trends_resp = client.get("/trends")
    assert len(trends_resp.json()) >= 1
    # New code path no longer stamps a misleading global topic onto trend rows.
    for row in trends_resp.json():
        assert row["topic_id"] is None
        assert row["topic_terms"] is None

    topics_resp = client.get("/trends/topics")
    assert topics_resp.status_code == 200
    for topic in topics_resp.json():
        assert "topic_id" in topic
        assert "topic_terms" in topic
        assert "topic_examples" in topic
