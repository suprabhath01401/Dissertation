import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer,
    String, Text, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


def now_utc() -> datetime:
    """Return the current time as a timezone-aware UTC datetime (used as a column default)."""
    return datetime.now(timezone.utc)


class Session(Base):
    """A chat session/conversation thread belonging to a user."""
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String(500))
    summary: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, onupdate=now_utc)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    messages: Mapped[list["Message"]] = relationship(back_populates="session", cascade="all, delete-orphan")
    constraints: Mapped[list["Constraint"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    """A single user or assistant turn within a session, including routing/source metadata."""
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)          # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model_used: Mapped[str | None] = mapped_column(String(100))
    route_path: Mapped[str | None] = mapped_column(String(50))             # local | cluster | temporal
    sources: Mapped[dict | None] = mapped_column(JSONB)
    temporal_facts: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    session: Mapped["Session"] = relationship(back_populates="messages")


class Constraint(Base):
    """A user-specified rule or preference scoped to a session (optionally permanent) that shapes responses."""
    __tablename__ = "constraints"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    constraint_type: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_permanent: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    session: Mapped["Session"] = relationship(back_populates="constraints")


class Document(Base):
    """Metadata for an ingested source file, tracking its chunk count and corresponding Qdrant point IDs."""
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(500), unique=True, nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)   # legal | app_docs
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    qdrant_ids: Mapped[list | None] = mapped_column(JSONB)


class EvaluationResult(Base):
    """Per-sample evaluation score row for a single metric within an evaluation run."""
    __tablename__ = "evaluation_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True)
    dataset: Mapped[str] = mapped_column(String(100), nullable=False)
    config: Mapped[str] = mapped_column(String(100), nullable=False)
    metric: Mapped[str] = mapped_column(String(100), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    notes: Mapped[str | None] = mapped_column(Text)


class EvaluationRun(Base):
    """One row per (run, dataset, config, metric) — the final aggregated score for
    a single 'Run Evaluation' execution, as opposed to EvaluationResult which holds
    one row per individual sample."""
    __tablename__ = "evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    dataset: Mapped[str] = mapped_column(String(100), nullable=False)
    config: Mapped[str] = mapped_column(String(100), nullable=False)
    metric: Mapped[str] = mapped_column(String(100), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    notes: Mapped[str | None] = mapped_column(Text)
