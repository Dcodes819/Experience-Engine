from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from packages.engine.application.redaction import redact_text
from packages.engine.application.service import ExperienceService
from packages.engine.domain.schemas import (
    ExperienceSearch,
    ExperienceStatus,
    FeedbackCreate,
    LifecycleAction,
    LifecycleRequest,
    RunCreate,
    RunEventCreate,
    RunFinalize,
)
from packages.engine.infrastructure.db import Base, Experience


def make_service() -> ExperienceService:
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return ExperienceService(sessionmaker(bind=engine, expire_on_commit=False)())


def verified_experience(service: ExperienceService, run_key: str = "verified-run"):
    run = service.create_run(
        RunCreate(
            task_text="Configure a container service to use the database hostname",
            agent_id="test-agent",
            context={
                "domain": "container-configuration",
                "task_type": "database-connectivity",
                "environment_fingerprint": "docker-compose-v2",
            },
            idempotency_key=run_key,
        )
    )
    service.append_event(
        run.id,
        RunEventCreate(
            sequence=1,
            event_type="tool",
            tool_name="docker-compose",
            input_summary="Started application container",
            output_summary="Database host changed from localhost to db and connection succeeded",
        ),
    )
    return service.finalize_run(
        run.id,
        RunFinalize(success=True, final_output_summary="Use db as the Compose service hostname.", outcome_code="connected"),
    )


def test_successful_evidence_backed_run_becomes_active_and_retrievable():
    service = make_service()
    experience = verified_experience(service)

    assert experience.status is ExperienceStatus.ACTIVE
    assert len(experience.evidence_event_ids) == 1

    search = service.search(
        ExperienceSearch(
            query="container cannot reach postgres database",
            domain="container-configuration",
            task_type="database-connectivity",
            environment_fingerprint="docker-compose-v2",
        )
    )
    assert [item.id for item in search.results] == [experience.id]
    assert search.results[0].rank is not None
    assert search.results[0].rank.final_score > 0


def test_scope_mismatch_and_invalidated_records_are_withheld():
    service = make_service()
    experience = verified_experience(service)

    mismatch = service.search(
        ExperienceSearch(query="connect to database", environment_fingerprint="local-windows")
    )
    assert mismatch.results == []

    service.apply_lifecycle(
        experience.id,
        LifecycleRequest(action=LifecycleAction.INVALIDATE, rationale="The image now uses a different network mode."),
    )
    after_invalidation = service.search(ExperienceSearch(query="database hostname"))
    assert after_invalidation.results == []


def test_failed_run_is_candidate_until_reviewed_then_feedback_is_audited():
    service = make_service()
    run = service.create_run(
        RunCreate(
            task_text="Start the service",
            agent_id="test-agent",
            context={"domain": "container-configuration"},
            idempotency_key="failed-run",
        )
    )
    service.append_event(
        run.id,
        RunEventCreate(sequence=1, event_type="tool", tool_name="docker", error_code="PORT_IN_USE", output_summary="8080 is occupied"),
    )
    candidate = service.finalize_run(
        run.id,
        RunFinalize(success=False, final_output_summary="Service did not start", outcome_code="PORT_IN_USE"),
    )
    assert candidate.status is ExperienceStatus.CANDIDATE
    assert service.search(ExperienceSearch(query="port is occupied")).results == []

    active = service.apply_lifecycle(
        candidate.id,
        LifecycleRequest(action=LifecycleAction.APPROVE, rationale="Reviewer confirmed this diagnostic lesson."),
    )
    service.submit_feedback(active.id, FeedbackCreate(helpful=True, note="Prevented a repeat check."))
    detail = service.get_experience(active.id)
    assert detail.feedback_summary == {"helpful": 1, "harmful": 0}


def test_event_order_and_finalized_outcome_are_immutable():
    service = make_service()
    run = service.create_run(
        RunCreate(task_text="Check database connectivity", agent_id="agent", idempotency_key="ordered-run")
    )
    conflicting_retry = RunCreate(
        task_text="Different task", agent_id="agent", idempotency_key="ordered-run"
    )
    try:
        service.create_run(conflicting_retry)
        assert False, "an idempotency key cannot be reused for a different payload"
    except ValueError as error:
        assert "different run request" in str(error)

    try:
        service.append_event(run.id, RunEventCreate(sequence=2, event_type="tool"))
        assert False, "non-contiguous event sequence should be rejected"
    except ValueError as error:
        assert "sequence must be 1" in str(error)

    service.append_event(run.id, RunEventCreate(sequence=1, event_type="tool"))
    request = RunFinalize(success=True, final_output_summary="Connected", outcome_code="connected")
    service.finalize_run(run.id, request)
    try:
        service.finalize_run(
            run.id,
            RunFinalize(success=False, final_output_summary="Failed", outcome_code="failed"),
        )
        assert False, "a finalized outcome must be immutable"
    except ValueError as error:
        assert "different outcome" in str(error)


def test_obvious_secrets_are_redacted_before_storage():
    assert redact_text("password=hunter2 Bearer abc.def https://user:secret@example.test") == (
        "password=[REDACTED] Bearer [REDACTED] https://user:[REDACTED]@example.test"
    )

    service = make_service()
    run = service.create_run(
        RunCreate(
            task_text="Use api_key=supersecret to connect",
            agent_id="agent",
            context={"api_key": "supersecret", "domain": "container-configuration"},
            idempotency_key="redacted-run",
        )
    )
    assert "supersecret" not in run.task_text
    assert run.context["api_key"] == "[REDACTED]"


def test_feedback_retrieval_must_contain_the_experience():
    service = make_service()
    experience = verified_experience(service)
    try:
        service.submit_feedback(
            experience.id,
            FeedbackCreate(retrieval_id="not-a-search", helpful=True, note="Useful"),
        )
        assert False, "unknown retrieval provenance must be rejected"
    except ValueError as error:
        assert "retrieval_id" in str(error)


def test_superseding_requires_a_different_active_replacement():
    service = make_service()
    original = verified_experience(service)
    try:
        service.apply_lifecycle(
            original.id,
            LifecycleRequest(action=LifecycleAction.SUPERSEDE, rationale="Replace it."),
        )
        assert False, "superseding requires an explicit active replacement"
    except ValueError as error:
        assert "replacement_id" in str(error)

    replacement = verified_experience(service, run_key="replacement-run")
    try:
        service.apply_lifecycle(
            original.id,
            LifecycleRequest(
                action=LifecycleAction.SUPERSEDE,
                rationale="Use the newer verified record.",
                replacement_id=original.id,
            ),
        )
        assert False, "an experience cannot replace itself"
    except ValueError as error:
        assert "itself" in str(error)

    superseded = service.apply_lifecycle(
        original.id,
        LifecycleRequest(
            action=LifecycleAction.SUPERSEDE,
            rationale="Use the newer verified record.",
            replacement_id=replacement.id,
        ),
    )
    assert superseded.status is ExperienceStatus.SUPERSEDED


def test_search_rejects_experiences_from_another_embedding_space():
    service = make_service()
    experience = verified_experience(service)
    stored = service.session.get(Experience, experience.id)
    stored.embedding_model = "different-model"
    stored.embedding_dimension = 256
    service.session.commit()

    try:
        service.search(ExperienceSearch(query="container database hostname"))
        assert False, "incompatible embedding models must not be compared"
    except ValueError as error:
        assert "different embedding model" in str(error)
