from datetime import datetime, timezone

from sqlalchemy import ForeignKey, String, Float, Integer, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow():
    return datetime.now(timezone.utc)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(16))
    language: Mapped[str] = mapped_column(String(8))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", order_by="Message.seq_index")
    escalation_snapshots: Mapped[list["EscalationSnapshot"]] = relationship(back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    seq_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(String(2000))
    language: Mapped[str] = mapped_column(String(8))
    author: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
    labels: Mapped[list["MessageLabel"]] = relationship(back_populates="message")


class MessageLabel(Base):
    __tablename__ = "message_labels"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"))
    category: Mapped[str] = mapped_column(String(32))
    probability: Mapped[float] = mapped_column(Float)
    toxicity_score: Mapped[float] = mapped_column(Float)

    message: Mapped["Message"] = relationship(back_populates="labels")


class EscalationSnapshot(Base):
    __tablename__ = "escalation_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    seq_index: Mapped[int] = mapped_column(Integer)
    windowed_mean: Mapped[float] = mapped_column(Float)
    slope: Mapped[float] = mapped_column(Float)
    escalation_score: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16))

    conversation: Mapped["Conversation"] = relationship(back_populates="escalation_snapshots")


class TrendSnapshot(Base):
    __tablename__ = "trend_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    time_bucket: Mapped[str] = mapped_column(String(16))
    category: Mapped[str] = mapped_column(String(32))
    language: Mapped[str] = mapped_column(String(8))
    # Kept for backward compatibility with existing rows/tests. New code no longer
    # populates these with a misleading global topic value - see TopicSnapshot instead.
    topic_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    topic_terms: Mapped[str | None] = mapped_column(String(500), nullable=True)
    message_count: Mapped[int] = mapped_column(Integer)


class TopicSnapshot(Base):
    __tablename__ = "topic_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(Integer)
    topic_terms: Mapped[str] = mapped_column(String(500))
    topic_examples: Mapped[str] = mapped_column(String(4000))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
