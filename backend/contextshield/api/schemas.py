from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    author: str | None = None


class ConversationCreateIn(BaseModel):
    messages: list[MessageIn] = Field(min_length=1)
    source: Literal["upload", "seed"] = "upload"
    language: Literal["en", "hi", "ta"]


class MessageLabelOut(BaseModel):
    category: str
    probability: float
    toxicity_score: float

    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    id: int
    seq_index: int
    text: str
    labels: list[MessageLabelOut]

    model_config = {"from_attributes": True}


class EscalationSnapshotOut(BaseModel):
    seq_index: int
    windowed_mean: float
    slope: float
    escalation_score: float
    status: str

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: int
    source: str
    language: str
    messages: list[MessageOut]
    escalation_snapshots: list[EscalationSnapshotOut]

    model_config = {"from_attributes": True}


class TrendSnapshotOut(BaseModel):
    time_bucket: str
    category: str
    language: str
    topic_id: int | None
    topic_terms: str | None
    message_count: int

    model_config = {"from_attributes": True}


class TopicSnapshotOut(BaseModel):
    id: int
    topic_id: int
    topic_terms: str
    topic_examples: str
    created_at: datetime

    model_config = {"from_attributes": True}
