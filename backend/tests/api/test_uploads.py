import io
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
        return [
            {
                "probabilities": {"hate_speech": 0.0, "offensive": 0.1, "threat": 0.0, "harassment": 0.0, "none": 0.9},
                "toxicity_score": 0.1,
            }
            for _ in texts
        ]


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

    app.dependency_overrides[deps.get_db_session] = override_get_db_session
    app.dependency_overrides[deps.get_classifier] = lambda: FakeClassifier()
    app.dependency_overrides[deps.get_thresholds] = lambda: (0.2, 0.5)

    yield TestClient(app)
    app.dependency_overrides.clear()


def test_upload_conversation_from_file(client):
    file_content = b"hello there\nhow are you\nsomething offensive maybe\n"
    resp = client.post(
        "/conversations/upload",
        files={"file": ("convo.txt", io.BytesIO(file_content), "text/plain")},
        data={"language": "en"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert len(body["messages"]) == 3
    assert body["source"] == "upload"


def test_upload_rejects_empty_file(client):
    resp = client.post(
        "/conversations/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        data={"language": "en"},
    )
    assert resp.status_code == 422
