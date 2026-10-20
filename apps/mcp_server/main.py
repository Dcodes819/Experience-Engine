"""Thin MCP adapter over the shared application service; it owns no separate logic or schema."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from packages.engine.application.service import ExperienceService
from packages.engine.domain.schemas import (
    ExperienceSearch,
    LifecycleRequest,
    FeedbackCreate,
    RunCreate,
    RunEventCreate,
    RunFinalize,
)
from packages.engine.infrastructure.db import SessionLocal, initialize_database

initialize_database()
server = MCPServer(
    name="experience-engine",
    description="Retrieve validated, scoped lessons and record evidence-backed agent runs.",
)


def _as_json(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json")


@server.tool(description="Find active, compatible experience lessons before beginning a task. Results are advisory and include provenance and rank explanations.")
def search_experiences(
    query: str,
    domain: str | None = None,
    task_type: str | None = None,
    environment_fingerprint: str | None = None,
    limit: int = 3,
) -> dict[str, Any]:
    with SessionLocal() as session:
        response = ExperienceService(session).search(
            ExperienceSearch(
                query=query,
                domain=domain,
                task_type=task_type,
                environment_fingerprint=environment_fingerprint,
                limit=limit,
            )
        )
        return _as_json(response)


@server.tool(description="Create a task run that will later supply evidence for deterministic experience extraction. This does not write permanent agent memory.")
def record_task_run(task_text: str, agent_id: str, context: dict[str, Any], idempotency_key: str) -> dict[str, Any]:
    with SessionLocal() as session:
        return _as_json(
            ExperienceService(session).create_run(
                RunCreate(task_text=task_text, agent_id=agent_id, context=context, idempotency_key=idempotency_key)
            )
        )


@server.tool(description="Attach a normalized, redacted event to an unfinished task run as evidence.")
def append_task_event(
    run_id: str,
    sequence: int,
    event_type: str,
    tool_name: str | None = None,
    input_summary: str | None = None,
    output_summary: str | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    with SessionLocal() as session:
        return _as_json(
            ExperienceService(session).append_event(
                run_id,
                RunEventCreate(
                    sequence=sequence,
                    event_type=event_type,
                    tool_name=tool_name,
                    input_summary=input_summary,
                    output_summary=output_summary,
                    error_code=error_code,
                ),
            )
        )


@server.tool(description="Finalize a recorded task run. The server extracts a candidate and applies evidence and outcome validation rules.")
def finalize_task_run(run_id: str, success: bool, final_output_summary: str, outcome_code: str) -> dict[str, Any]:
    with SessionLocal() as session:
        return _as_json(
            ExperienceService(session).finalize_run(
                run_id,
                RunFinalize(success=success, final_output_summary=final_output_summary, outcome_code=outcome_code),
            )
        )


@server.tool(description="Record whether a retrieved lesson helped or harmed the later task. This creates an auditable learning signal.")
def submit_experience_feedback(experience_id: str, helpful: bool, note: str, retrieval_id: str | None = None) -> dict[str, Any]:
    with SessionLocal() as session:
        return _as_json(
            ExperienceService(session).submit_feedback(
                experience_id,
                FeedbackCreate(helpful=helpful, note=note, retrieval_id=retrieval_id),
            )
        )


@server.tool(description="Fetch an experience's source run, evidence links, lifecycle relations, and feedback summary for transparency.")
def get_experience(experience_id: str) -> dict[str, Any]:
    with SessionLocal() as session:
        return _as_json(ExperienceService(session).get_experience(experience_id))


@server.tool(description="Apply a reviewed lifecycle transition. Superseding requires a different active replacement; history is retained.")
def apply_experience_lifecycle(
    experience_id: str,
    action: str,
    rationale: str,
    replacement_id: str | None = None,
) -> dict[str, Any]:
    with SessionLocal() as session:
        request = LifecycleRequest(
            action=action,
            rationale=rationale,
            replacement_id=replacement_id,
        )
        return _as_json(ExperienceService(session).apply_lifecycle(experience_id, request))


if __name__ == "__main__":
    server.run(transport="stdio")
