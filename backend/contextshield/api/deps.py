import json
import os
from functools import lru_cache
from pathlib import Path

from contextshield.db.session import get_default_engine, get_session_factory
from contextshield.ml.classifier import Classifier

_engine = get_default_engine()
_SessionFactory = get_session_factory(_engine)


def get_db_session():
    session = _SessionFactory()
    try:
        yield session
    finally:
        session.close()


@lru_cache(maxsize=1)
def _load_classifier() -> Classifier:
    model_dir = os.environ.get(
        "CONTEXTSHIELD_MODEL_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "models" / "classifier"),
    )
    return Classifier(model_dir)


def get_classifier() -> Classifier:
    return _load_classifier()


def get_thresholds() -> tuple[float, float]:
    model_dir = os.environ.get(
        "CONTEXTSHIELD_MODEL_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "models" / "classifier"),
    )
    thresholds_path = Path(model_dir) / "thresholds.json"
    if not thresholds_path.exists():
        return (0.3, 0.7)
    with open(thresholds_path, encoding="utf-8") as f:
        data = json.load(f)
    return (data["tau_low"], data["tau_high"])
