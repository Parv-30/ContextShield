from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from contextshield.db.models import Base
from contextshield.services.conversation_service import ConversationService


class FakeClassifier:
    def predict_batch(self, texts):
        results = []
        for i, text in enumerate(texts):
            score = min(0.1 * (i + 1), 1.0)
            probs = {"hate_speech": 0.0, "offensive": score, "threat": 0.0, "harassment": 0.0, "none": 1 - score}
            results.append({"probabilities": probs, "toxicity_score": score})
        return results


def make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_create_conversation_persists_messages_and_labels():
    session = make_session()
    service = ConversationService(session, FakeClassifier(), tau_low=0.2, tau_high=0.5)

    messages = [{"text": "msg1", "author": None}, {"text": "msg2", "author": "u1"}, {"text": "msg3", "author": None}]
    convo = service.create_conversation(messages, source="upload", language="en")

    assert convo.id is not None
    assert len(convo.messages) == 3
    assert convo.messages[0].seq_index == 0
    assert len(convo.messages[0].labels) == 5  # one MessageLabel row per LABELS entry


def test_create_conversation_computes_escalation_snapshots():
    session = make_session()
    service = ConversationService(session, FakeClassifier(), tau_low=0.2, tau_high=0.25)

    messages = [{"text": f"msg{i}", "author": None} for i in range(5)]
    convo = service.create_conversation(messages, source="upload", language="en")

    assert len(convo.escalation_snapshots) == 3  # w=3 default -> len(messages) - 2
    assert convo.escalation_snapshots[-1].status in {"stable", "elevated", "escalating"}


def test_get_and_list_conversations():
    session = make_session()
    service = ConversationService(session, FakeClassifier(), tau_low=0.2, tau_high=0.5)
    convo = service.create_conversation([{"text": "hi", "author": None}] * 3, source="seed", language="ta")

    fetched = service.get_conversation(convo.id)
    assert fetched is not None
    assert fetched.language == "ta"

    all_convos = service.list_conversations()
    assert len(all_convos) == 1

    assert service.get_conversation(9999) is None
