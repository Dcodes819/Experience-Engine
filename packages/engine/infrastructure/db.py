"""Database models and session factory.

Embeddings are stored as JSON arrays for exact baseline search. PostgreSQL deployments
can migrate this column to pgvector without changing the application contract.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class TaskRun(Base):
    __tablename__ = "task_runs"
    __table_args__ = (UniqueConstraint("agent_id", "idempotency_key", name="uq_task_run_agent_idempotency"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id: Mapped[str] = mapped_column(String(200), index=True)
    task_text: Mapped[str] = mapped_column(Text)
    context_json: Mapped[dict] = mapped_column(JSON, default=dict)
    idempotency_key: Mapped[str] = mapped_column(String(200), index=True)
    success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    outcome_code: Mapped[str | None] = mapped_column(String(200), nullable=True)
    final_output_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    finalized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    events: Mapped[list["RunEvent"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    experiences: Mapped[list["Experience"]] = relationship(back_populates="source_run")


class RunEvent(Base):
    __tablename__ = "run_events"
    __table_args__ = (UniqueConstraint("run_id", "sequence", name="uq_run_event_sequence"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("task_runs.id"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(100))
    tool_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    input_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(200), nullable=True)

    run: Mapped[TaskRun] = relationship(back_populates="events")


class Experience(Base):
    __tablename__ = "experiences"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_run_id: Mapped[str] = mapped_column(ForeignKey("task_runs.id"), index=True, unique=True)
    situation: Mapped[str] = mapped_column(Text)
    problem: Mapped[str] = mapped_column(Text)
    resolution: Mapped[str] = mapped_column(Text)
    lesson: Mapped[str] = mapped_column(Text)
    recommended_action: Mapped[str] = mapped_column(Text)
    caveats: Mapped[str] = mapped_column(Text)
    scope_json: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), default="candidate", index=True)
    retrieval_text: Mapped[str] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(String(200), default="hashing-v1")
    embedding: Mapped[list[float]] = mapped_column(JSON)
    embedding_dimension: Mapped[int] = mapped_column(Integer, default=256, server_default="256")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, onupdate=now_utc)
    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    source_run: Mapped[TaskRun] = relationship(back_populates="experiences")
    evidence: Mapped[list["ExperienceEvidence"]] = relationship(back_populates="experience", cascade="all, delete-orphan")
    feedback: Mapped[list["ExperienceFeedback"]] = relationship(back_populates="experience", cascade="all, delete-orphan")


class ExperienceEvidence(Base):
    __tablename__ = "experience_evidence"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experience_id: Mapped[str] = mapped_column(ForeignKey("experiences.id"), index=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("run_events.id"), index=True)
    evidence_type: Mapped[str] = mapped_column(String(100))
    note: Mapped[str] = mapped_column(Text)

    experience: Mapped[Experience] = relationship(back_populates="evidence")


class ExperienceFeedback(Base):
    __tablename__ = "experience_feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experience_id: Mapped[str] = mapped_column(ForeignKey("experiences.id"), index=True)
    retrieval_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    helpful: Mapped[bool] = mapped_column(Boolean)
    note: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    experience: Mapped[Experience] = relationship(back_populates="feedback")


class ExperienceRelation(Base):
    __tablename__ = "experience_relations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    from_experience_id: Mapped[str] = mapped_column(ForeignKey("experiences.id"), index=True)
    to_experience_id: Mapped[str | None] = mapped_column(ForeignKey("experiences.id"), nullable=True)
    relation_type: Mapped[str] = mapped_column(String(100))
    rationale: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("task_runs.id"), nullable=True)
    query_text: Mapped[str] = mapped_column(Text)
    filters_json: Mapped[dict] = mapped_column(JSON, default=dict)
    result_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    score_breakdown_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


def database_url() -> str:
    return os.getenv("EXPERIENCE_ENGINE_DATABASE_URL", "sqlite:///./experience_engine.db")


def make_engine(url: str | None = None):
    resolved = url or database_url()
    engine = create_engine(resolved, connect_args={"check_same_thread": False} if resolved.startswith("sqlite") else {})
    if resolved.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def enable_foreign_keys(connection, _record) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return engine


def make_session_factory(url: str | None = None):
    return sessionmaker(bind=make_engine(url), autoflush=False, expire_on_commit=False)


def initialize_database(url: str | None = None) -> None:
    Base.metadata.create_all(make_engine(url))


SessionLocal = make_session_factory()


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
