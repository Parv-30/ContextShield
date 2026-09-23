import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends

from contextshield.db.session import get_default_engine, get_session_factory
from contextshield.ml.classifier import Classifier
from contextshield.services.conversation_service import ConversationService

_engine = get_default_engine()
_SessionFactory = get_session_factory(_engine)


def get_db_session():
    session = _SessionFactory()
    try:
        yield session
    finally:
        session.close()


def _resolve_model_dir() -> Path:
    return Path(
        os.environ.get(
            "CONTEXTSHIELD_MODEL_DIR",
            str(Path(__file__).resolve().parent.parent.parent / "models" / "classifier"),
        )
    )


@lru_cache(maxsize=1)
def _load_classifier() -> Classifier:
    return Classifier(str(_resolve_model_dir()))


def get_classifier() -> Classifier:
    return _load_classifier()


def get_thresholds() -> tuple[float, float]:
    model_dir = _resolve_model_dir()
    thresholds_path = model_dir / "thresholds.json"
    if not thresholds_path.exists():
        return (0.3, 0.7)
    with open(thresholds_path, encoding="utf-8") as f:
        data = json.load(f)
    return (data["tau_low"], data["tau_high"])


def get_conversation_service(
    session=Depends(get_db_session),
    classifier=Depends(get_classifier),
    thresholds=Depends(get_thresholds),
) -> ConversationService:
    tau_low, tau_high = thresholds
    return ConversationService(session, classifier, tau_low=tau_low, tau_high=tau_high)
