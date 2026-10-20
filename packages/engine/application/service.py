"""Application commands shared by the REST, SDK, MCP, and demo adapters."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from packages.engine.application.embeddings import EmbeddingProvider, configured_embedding_provider, cosine_similarity
from packages.engine.application.redaction import redact_text, redact_value
from packages.engine.domain.schemas import (
    Applicability,
    ExperienceDetail,
    ExperienceSearch,
    ExperienceStatus,
    ExperienceView,
    FeedbackCreate,
    LifecycleAction,
    LifecycleRequest,
    RankComponents,
    RunCreate,
    RunEventCreate,
    RunEventView,
    RunFinalize,
    RunView,
    SearchResponse,
)
from packages.engine.infrastructure.db import (
    Experience,
    ExperienceEvidence,
    ExperienceFeedback,
    ExperienceRelation,
    RetrievalLog,
    RunEvent,
    TaskRun,
    now_utc,
)


class NotFoundError(ValueError):
    pass


class ConflictError(ValueError):
    pass


class ValidationPolicyError(ValueError):
    pass


class ExperienceService:
    """The only layer that implements capture, validation, ranking, and lifecycle rules."""

    def __init__(self, session: Session, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.session = session
        self.embedding_provider = embedding_provider or configured_embedding_provider()

    def create_run(self, request: RunCreate) -> RunView:
        existing = self.session.scalar(
            select(TaskRun).where(
                TaskRun.agent_id == request.agent_id,
                TaskRun.idempotency_key == request.idempotency_key,
            )
        )
        if existing:
            if existing.task_text != redact_text(request.task_text) or existing.context_json != redact_value(request.context):
                raise ConflictError("This idempotency key is already associated with a different run request.")
            return self._run_view(existing)
        run = TaskRun(
            agent_id=request.agent_id,
            task_text=redact_text(request.task_text),
            context_json=redact_value(request.context),
            idempotency_key=request.idempotency_key,
        )
        self.session.add(run)
        self.session.commit()
        return self._run_view(run)

    def append_event(self, run_id: str, request: RunEventCreate) -> RunEventView:
        run = self._get_run(run_id)
        if run.finalized_at:
            raise ConflictError("A finalized run cannot receive additional events.")
        existing = self.session.scalar(
            select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.sequence == request.sequence)
        )
        if existing:
            raise ConflictError(f"Event sequence {request.sequence} already exists for this run.")
        latest_sequence = self.session.scalar(
            select(RunEvent.sequence)
            .where(RunEvent.run_id == run_id)
            .order_by(RunEvent.sequence.desc())
            .limit(1)
        )
        expected_sequence = (latest_sequence or 0) + 1
        if request.sequence != expected_sequence:
            raise ValidationPolicyError(f"Event sequence must be {expected_sequence} for this run.")
        payload = request.model_dump()
        for field in ("input_summary", "output_summary"):
            if payload[field] is not None:
                payload[field] = redact_text(payload[field])
        event = RunEvent(run_id=run_id, **payload)
        self.session.add(event)
        self.session.commit()
        return RunEventView.model_validate(event)

    def finalize_run(self, run_id: str, request: RunFinalize) -> ExperienceView:
        run = self._get_run(run_id)
        existing = self.session.scalar(
            select(Experience)
            .options(selectinload(Experience.evidence), selectinload(Experience.feedback))
            .where(Experience.source_run_id == run_id)
        )
        if existing:
            if (
                run.success != request.success
                or run.outcome_code != request.outcome_code
                or run.final_output_summary != redact_text(request.final_output_summary)
            ):
                raise ConflictError("This run was already finalized with a different outcome.")
            return self._experience_view(existing)
        events = list(
            self.session.scalars(select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.sequence))
        )
        if not events:
            raise ValidationPolicyError("A run needs at least one evidence event before it can be finalized.")

        run.success = request.success
        run.outcome_code = request.outcome_code
        run.final_output_summary = redact_text(request.final_output_summary)
        run.finalized_at = now_utc()
        experience = self._extract_and_validate(run, events)
        self.session.add(experience)
        self.session.commit()
        self.session.refresh(experience)
        return self._experience_view(experience)

    def search(self, request: ExperienceSearch) -> SearchResponse:
        requested_scope = request.requested_scope()
        query_embedding = self.embedding_provider.embed(request.query)
        experiences = list(
            self.session.scalars(
                select(Experience)
                .options(selectinload(Experience.evidence), selectinload(Experience.feedback))
                .where(Experience.status == ExperienceStatus.ACTIVE.value)
            )
        )
        incompatible_models = [
            experience
            for experience in experiences
            if experience.embedding_model != self.embedding_provider.name
            or experience.embedding_dimension != len(query_embedding)
            or len(experience.embedding or []) != experience.embedding_dimension
        ]
        if incompatible_models:
            raise ValidationPolicyError(
                "Active experiences use a different embedding model or dimension; re-embed them before searching."
            )
        ranked: list[tuple[Experience, RankComponents, Applicability]] = []
        for experience in experiences:
            compatible, applicability = self._scope_compatibility(experience.scope_json or {}, requested_scope)
            if not compatible:
                continue
            helpful = sum(1 for item in experience.feedback if item.helpful)
            harmful = sum(1 for item in experience.feedback if not item.helpful)
            helpfulness = (helpful + 1) / (helpful + harmful + 2)
            created_at = self._as_utc(experience.created_at)
            age_days = max(0.0, (now_utc() - created_at).total_seconds() / 86_400)
            recency = math.exp(-age_days / 90)
            semantic = cosine_similarity(query_embedding, experience.embedding or [])
            scope_score = 1.0 if not requested_scope else len(applicability.matched_context_fields) / len(requested_scope)
            components = RankComponents(
                final_score=round(
                    0.45 * semantic
                    + 0.20 * scope_score
                    + 0.15 * experience.confidence
                    + 0.10 * helpfulness
                    + 0.10 * recency,
                    6,
                ),
                semantic_similarity=round(semantic, 6),
                scope_compatibility=round(scope_score, 6),
                confidence=round(experience.confidence, 6),
                helpfulness=round(helpfulness, 6),
                recency=round(recency, 6),
            )
            ranked.append((experience, components, applicability))
        ranked.sort(key=lambda candidate: candidate[1].final_score, reverse=True)
        top = ranked[: request.limit]
        log = RetrievalLog(
            run_id=request.run_id,
            query_text=request.query,
            filters_json=requested_scope,
            result_ids=[experience.id for experience, _, _ in top],
            score_breakdown_json={experience.id: components.model_dump() for experience, components, _ in top},
        )
        self.session.add(log)
        self.session.commit()
        return SearchResponse(
            retrieval_id=log.id,
            results=[self._experience_view(experience, rank, applicability) for experience, rank, applicability in top],
        )

    def get_experience(self, experience_id: str) -> ExperienceDetail:
        experience = self.session.scalar(
            select(Experience)
            .options(selectinload(Experience.evidence), selectinload(Experience.feedback))
            .where(Experience.id == experience_id)
        )
        if not experience:
            raise NotFoundError("Experience not found.")
        feedback = Counter("helpful" if item.helpful else "harmful" for item in experience.feedback)
        relations = list(
            self.session.scalars(
                select(ExperienceRelation).where(
                    (ExperienceRelation.from_experience_id == experience_id)
                    | (ExperienceRelation.to_experience_id == experience_id)
                )
            )
        )
        return ExperienceDetail(
            **self._experience_view(experience).model_dump(),
            feedback_summary={"helpful": feedback["helpful"], "harmful": feedback["harmful"]},
            relations=[
                {
                    "id": relation.id,
                    "from_experience_id": relation.from_experience_id,
                    "to_experience_id": relation.to_experience_id,
                    "relation_type": relation.relation_type,
                    "rationale": relation.rationale,
                }
                for relation in relations
            ],
        )

    def submit_feedback(self, experience_id: str, request: FeedbackCreate) -> ExperienceView:
        experience = self._get_experience(experience_id)
        if experience.status not in {ExperienceStatus.ACTIVE.value, ExperienceStatus.SUPERSEDED.value}:
            raise ValidationPolicyError("Feedback is accepted only for active or superseded experiences.")
        if request.retrieval_id:
            retrieval = self.session.get(RetrievalLog, request.retrieval_id)
            if not retrieval or experience_id not in (retrieval.result_ids or []):
                raise ValidationPolicyError("Feedback retrieval_id must identify a search result containing this experience.")
        self.session.add(
            ExperienceFeedback(
                experience_id=experience_id,
                retrieval_id=request.retrieval_id,
                helpful=request.helpful,
                note=redact_text(request.note),
            )
        )
        self.session.commit()
        self.session.refresh(experience)
        return self._experience_view(experience)

    def apply_lifecycle(self, experience_id: str, request: LifecycleRequest) -> ExperienceView:
        experience = self._get_experience(experience_id)
        if request.replacement_id and request.action is not LifecycleAction.SUPERSEDE:
            raise ValidationPolicyError("replacement_id is supported only for the supersede action.")
        if request.action is LifecycleAction.APPROVE:
            if experience.status != ExperienceStatus.CANDIDATE.value:
                raise ValidationPolicyError("Only a candidate can be approved.")
            experience.status = ExperienceStatus.ACTIVE.value
            experience.last_verified_at = now_utc()
        elif request.action is LifecycleAction.INVALIDATE:
            if experience.status in {ExperienceStatus.INVALIDATED.value, ExperienceStatus.ARCHIVED.value}:
                raise ValidationPolicyError("An invalidated or archived experience cannot be invalidated again.")
            experience.status = ExperienceStatus.INVALIDATED.value
        elif request.action is LifecycleAction.ARCHIVE:
            if experience.status == ExperienceStatus.ARCHIVED.value:
                raise ValidationPolicyError("An archived experience cannot be changed.")
            experience.status = ExperienceStatus.ARCHIVED.value
        elif request.action is LifecycleAction.SUPERSEDE:
            if experience.status not in {ExperienceStatus.CANDIDATE.value, ExperienceStatus.ACTIVE.value}:
                raise ValidationPolicyError("Only a candidate or active experience can be superseded.")
            if not request.replacement_id:
                raise ValidationPolicyError("Superseding an experience requires a replacement_id.")
            if request.replacement_id == experience_id:
                raise ValidationPolicyError("An experience cannot replace itself.")
            replacement = self._get_experience(request.replacement_id)
            if replacement.status != ExperienceStatus.ACTIVE.value:
                raise ValidationPolicyError("The replacement experience must be active.")
            experience.status = ExperienceStatus.SUPERSEDED.value
        else:  # pragma: no cover - enum protects this branch
            raise ValidationPolicyError("Unsupported lifecycle action.")
        self.session.add(
            ExperienceRelation(
                from_experience_id=experience_id,
                to_experience_id=request.replacement_id,
                relation_type=request.action.value,
                rationale=redact_text(request.rationale),
            )
        )
        self.session.commit()
        self.session.refresh(experience)
        return self._experience_view(experience)

    def _extract_and_validate(self, run: TaskRun, events: list[RunEvent]) -> Experience:
        """Deterministic extraction path; an LLM adapter can produce this same schema later."""
        scope = {
            key: str(value)
            for key, value in (run.context_json or {}).items()
            if key in {"domain", "task_type", "environment_fingerprint", "preconditions"} and value is not None
        }
        scope.setdefault("domain", "general")
        last_event = events[-1]
        error_event = next((event for event in reversed(events) if event.error_code), None)
        event_summary = error_event.output_summary if error_event else last_event.output_summary
        problem = (
            f"{error_event.error_code}: {event_summary or 'tool step failed'}"
            if error_event
            else f"Outcome {run.outcome_code}: {run.final_output_summary}"
        )
        situation = f"{scope.get('domain', 'general')} task: {run.task_text}"
        if run.success:
            resolution = run.final_output_summary or "The run completed successfully."
            lesson = f"For similar tasks, reuse the successful approach that produced outcome {run.outcome_code}."
            recommended_action = "Check the matched preconditions, then follow the verified sequence recorded in the source run."
            confidence = 0.85 if len(events) > 1 else 0.75
            status = ExperienceStatus.ACTIVE.value
        else:
            resolution = "The run did not succeed; use this as a review candidate rather than automatic advice."
            lesson = f"Avoid repeating the failure mode {run.outcome_code}; investigate the recorded evidence before retrying."
            recommended_action = "Review the failed event and correct its precondition before attempting the task again."
            confidence = 0.35
            status = ExperienceStatus.CANDIDATE.value
        caveats = "Advisory only. Apply this lesson only when the recorded scope matches the current task."
        retrieval_text = " ".join([situation, problem, resolution, lesson, recommended_action, " ".join(scope.values())])
        embedding = self.embedding_provider.embed(retrieval_text)
        experience = Experience(
            source_run_id=run.id,
            situation=situation,
            problem=problem,
            resolution=resolution,
            lesson=lesson,
            recommended_action=recommended_action,
            caveats=caveats,
            scope_json=scope,
            confidence=confidence,
            status=status,
            retrieval_text=retrieval_text,
            embedding_model=self.embedding_provider.name,
            embedding=embedding,
            embedding_dimension=len(embedding),
            last_verified_at=now_utc() if status == ExperienceStatus.ACTIVE.value else None,
        )
        self.session.flush()
        experience.evidence.append(
            ExperienceEvidence(event_id=last_event.id, evidence_type="final_event", note="Final event supporting extracted lesson.")
        )
        return experience

    @staticmethod
    def _scope_compatibility(scope: dict[str, Any], requested: dict[str, str]) -> tuple[bool, Applicability]:
        matched = [key for key, value in requested.items() if str(scope.get(key, "")) == str(value)]
        unmatched = [key for key in requested if key not in matched]
        # A requested field is a hard compatibility constraint when the record declares it.
        incompatible = [key for key in unmatched if key in scope and scope[key] not in {"", "general", "unknown"}]
        return not incompatible, Applicability(matched_context_fields=matched, unmatched_context_fields=unmatched)

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    def _get_run(self, run_id: str) -> TaskRun:
        run = self.session.get(TaskRun, run_id)
        if not run:
            raise NotFoundError("Task run not found.")
        return run

    def _get_experience(self, experience_id: str) -> Experience:
        experience = self.session.scalar(
            select(Experience)
            .options(selectinload(Experience.evidence), selectinload(Experience.feedback))
            .where(Experience.id == experience_id)
        )
        if not experience:
            raise NotFoundError("Experience not found.")
        return experience

    @staticmethod
    def _run_view(run: TaskRun) -> RunView:
        return RunView(
            id=run.id,
            task_text=run.task_text,
            agent_id=run.agent_id,
            context=run.context_json or {},
            success=run.success,
            outcome_code=run.outcome_code,
            final_output_summary=run.final_output_summary,
            created_at=run.created_at,
            finalized_at=run.finalized_at,
        )

    @staticmethod
    def _experience_view(
        experience: Experience,
        rank: RankComponents | None = None,
        applicability: Applicability | None = None,
    ) -> ExperienceView:
        return ExperienceView(
            id=experience.id,
            source_run_id=experience.source_run_id,
            situation=experience.situation,
            problem=experience.problem,
            resolution=experience.resolution,
            lesson=experience.lesson,
            recommended_action=experience.recommended_action,
            caveats=experience.caveats,
            scope=experience.scope_json or {},
            confidence=experience.confidence,
            status=ExperienceStatus(experience.status),
            evidence_event_ids=[evidence.event_id for evidence in experience.evidence],
            created_at=experience.created_at,
            updated_at=experience.updated_at,
            rank=rank,
            applicability=applicability,
        )
