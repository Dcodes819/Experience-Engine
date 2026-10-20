from __future__ import annotations

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from apps.api.main import app, get_session
from packages.engine.infrastructure.db import Base


@pytest.mark.anyio
async def test_rest_capture_finalize_and_search_share_the_core_contract():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    def override_session():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = override_session
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            run = await client.post(
                "/v1/task-runs",
                json={
                    "task_text": "Set the container database host",
                    "agent_id": "contract-agent",
                    "context": {"domain": "container-configuration", "environment_fingerprint": "compose"},
                    "idempotency_key": "contract-run",
                },
            )
            assert run.status_code == 201
            run_id = run.json()["id"]
            event = await client.post(
                f"/v1/task-runs/{run_id}/events",
                json={
                    "sequence": 1,
                    "event_type": "tool",
                    "tool_name": "compose",
                    "output_summary": "Configured host db",
                },
            )
            assert event.status_code == 201
            finalized = await client.post(
                f"/v1/task-runs/{run_id}/finalize",
                json={"success": True, "final_output_summary": "db is reachable", "outcome_code": "connected"},
            )
            assert finalized.status_code == 200
            response = await client.get(
                "/v1/experiences/search",
                params={"query": "container database host", "domain": "container-configuration", "environment_fingerprint": "compose"},
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["results"][0]["id"] == finalized.json()["id"]
            assert payload["results"][0]["rank"]["final_score"] > 0
            detail = await client.get(f"/v1/experiences/{finalized.json()['id']}")
            assert detail.status_code == 200
            feedback = await client.post(
                f"/v1/experiences/{finalized.json()['id']}/feedback",
                json={
                    "retrieval_id": payload["retrieval_id"],
                    "helpful": True,
                    "note": "The scoped advice helped.",
                },
            )
            assert feedback.status_code == 200
            assert (await client.get("/health")).json()["database"] == "ready"
    finally:
        app.dependency_overrides.clear()
