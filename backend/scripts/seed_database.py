# backend/scripts/seed_database.py
from datetime import datetime, timedelta, timezone
from pathlib import Path

from contextshield.db.session import get_default_engine, get_session_factory, init_db
from contextshield.services.conversation_service import ConversationService
from contextshield.services.trend_service import recompute_trends
from contextshield.ml.classifier import Classifier
from contextshield.ml.train import load_jsonl

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "classifier"
CHUNK_SIZE = 6


def load_thresholds():
    import json

    thresholds_path = MODEL_DIR / "thresholds.json"
    if not thresholds_path.exists():
        return (0.3, 0.7)
    with open(thresholds_path, encoding="utf-8") as f:
        data = json.load(f)
    return (data["tau_low"], data["tau_high"])


def chunk_by_language(examples: list[dict], size: int) -> list[list[dict]]:
    by_language: dict[str, list[dict]] = {}
    for ex in examples:
        by_language.setdefault(ex["language"], []).append(ex)

    chunks = []
    for lang_examples in by_language.values():
        for i in range(0, len(lang_examples), size):
            chunk = lang_examples[i : i + size]
            if chunk:
                chunks.append(chunk)
    return chunks


def main():
    engine = get_default_engine()
    init_db(engine)
    session = get_session_factory(engine)()

    classifier = Classifier(MODEL_DIR)
    tau_low, tau_high = load_thresholds()
    service = ConversationService(session, classifier, tau_low=tau_low, tau_high=tau_high)

    all_examples = load_jsonl(DATA_DIR / "train.jsonl") + load_jsonl(DATA_DIR / "val.jsonl") + load_jsonl(DATA_DIR / "test.jsonl")
    chunks = chunk_by_language(all_examples, CHUNK_SIZE)

    base_time = datetime.now(timezone.utc) - timedelta(days=90)
    for i, chunk in enumerate(chunks):
        messages = [{"text": ex["text"], "author": None} for ex in chunk]
        language = chunk[0]["language"]
        convo = service.create_conversation(messages, source="seed", language=language)

        convo_time = base_time + timedelta(days=i)
        for message in convo.messages:
            message.created_at = convo_time
        session.commit()

    count = recompute_trends(session)
    print(f"Seeded {len(chunks)} conversations. Created {count} trend snapshots.")


if __name__ == "__main__":
    main()
