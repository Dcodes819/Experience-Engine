"""Shared request, response, and lifecycle contracts for Experience Engine."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExperienceStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    ARCHIVED = "archived"


class LifecycleAction(StrEnum):
    APPROVE = "approve"
    INVALIDATE = "invalidate"
    SUPERSEDE = "supersede"
    ARCHIVE = "archive"


class RunCreate(BaseModel):
    task_text: str = Field(min_length=1, max_length=10_000)
    agent_id: str = Field(min_length=1, max_length=200)
    context: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=1, max_length=200)


class RunEventCreate(BaseModel):
    sequence: int = Field(ge=1)
    event_type: str = Field(min_length=1, max_length=100)
    tool_name: str | None = Field(default=None, max_length=200)
    input_summary: str | None = Field(default=None, max_length=4_000)
    output_summary: str | None = Field(default=None, max_length=4_000)
    error_code: str | None = Field(default=None, max_length=200)


class RunFinalize(BaseModel):
    success: bool
    final_output_summary: str = Field(min_length=1, max_length=8_000)
    outcome_code: str = Field(min_length=1, max_length=200)


class ExperienceSearch(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    domain: str | None = None
    task_type: str | None = None
    environment_fingerprint: str | None = None
    preconditions: str | None = None
    limit: int = Field(default=3, ge=1, le=20)
    run_id: str | None = None

    def requested_scope(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "domain": self.domain,
                "task_type": self.task_type,
                "environment_fingerprint": self.environment_fingerprint,
                "preconditions": self.preconditions,
            }.items()
            if value
        }


class FeedbackCreate(BaseModel):
    retrieval_id: str | None = None
    helpful: bool
    note: str = Field(min_length=1, max_length=2_000)


class LifecycleRequest(BaseModel):
    action: LifecycleAction
    rationale: str = Field(min_length=1, max_length=2_000)
    replacement_id: str | None = None


class RankComponents(BaseModel):
    final_score: float
    semantic_similarity: float
    scope_compatibility: float
    confidence: float
    helpfulness: float
    recency: float


class Applicability(BaseModel):
    matched_context_fields: list[str]
    unmatched_context_fields: list[str]


class RunView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    task_text: str
    agent_id: str
    context: dict[str, Any]
    success: bool | None
    outcome_code: str | None
    final_output_summary: str | None
    created_at: datetime
    finalized_at: datetime | None


class RunEventView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    sequence: int
    event_type: str
    tool_name: str | None
    input_summary: str | None
    output_summary: str | None
    error_code: str | None


class ExperienceView(BaseModel):
    id: str
    source_run_id: str
    situation: str
    problem: str
    resolution: str
    lesson: str
    recommended_action: str
    caveats: str
    scope: dict[str, Any]
    confidence: float
    status: ExperienceStatus
    evidence_event_ids: list[str]
    created_at: datetime
    updated_at: datetime
    rank: RankComponents | None = None
    applicability: Applicability | None = None


class ExperienceDetail(ExperienceView):
    feedback_summary: dict[str, int]
    relations: list[dict[str, Any]]


class SearchResponse(BaseModel):
    retrieval_id: str
    results: list[ExperienceView]


class HealthView(BaseModel):
    version: str
    database: str
    embedding_provider: str
