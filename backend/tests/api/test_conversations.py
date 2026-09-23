import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from contextshield.db.models import Base
from contextshield.api.main import app
from contextshield.api import deps


class FakeClassifier:
    def predict_batch(self, texts):
        results = []
        for i, text in enumerate(texts):
            score = min(0.1 * (i + 1), 1.0)
            probs = {"hate_speech": 0.0, "offensive": score, "threat": 0.0, "harassment": 0.0, "none": 1 - score}
            results.append({"probabilities": probs, "toxicity_score": score})
        return results


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    def override_get_db_session():
        session = TestSession()
        try:
            yield session
        finally:
            session.close()

    def override_get_classifier():
        return FakeClassifier()

    def override_get_thresholds():
        return (0.2, 0.5)

    app.dependency_overrides[deps.get_db_session] = override_get_db_session
    app.dependency_overrides[deps.get_classifier] = override_get_classifier
    app.dependency_overrides[deps.get_thresholds] = override_get_thresholds

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_create_and_get_conversation(client):
    payload = {
        "messages": [{"text": "hello"}, {"text": "world"}, {"text": "bad stuff"}],
        "source": "upload",
        "language": "en",
    }
    resp = client.post("/conversations", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body["language"] == "en"
    assert len(body["messages"]) == 3
    assert len(body["messages"][0]["labels"]) == 5

    convo_id = body["id"]
    resp2 = client.get(f"/conversations/{convo_id}")
    assert resp2.status_code == 200
    assert resp2.json()["id"] == convo_id


def test_get_conversation_not_found(client):
    resp = client.get("/conversations/99999")
    assert resp.status_code == 404


def test_list_conversations(client):
    client.post("/conversations", json={"messages": [{"text": "a"}], "source": "upload", "language": "en"})
    client.post("/conversations", json={"messages": [{"text": "b"}], "source": "upload", "language": "hi"})
    resp = client.get("/conversations")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_create_conversation_rejects_empty_messages(client):
    resp = client.post("/conversations", json={"messages": [], "source": "upload", "language": "en"})
    assert resp.status_code == 422


def test_create_conversation_rejects_unsupported_language(client):
    resp = client.post("/conversations", json={"messages": [{"text": "hi"}], "source": "upload", "language": "fr"})
    assert resp.status_code == 422
