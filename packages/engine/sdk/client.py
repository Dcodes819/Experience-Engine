"""Small synchronous SDK for agents and command-line clients."""

from __future__ import annotations

from typing import Any

import httpx

from packages.engine.domain.schemas import (
    ExperienceDetail,
    ExperienceView,
    FeedbackCreate,
    LifecycleRequest,
    RunCreate,
    RunEventCreate,
    RunEventView,
    RunFinalize,
    RunView,
    SearchResponse,
)


class ExperienceClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ExperienceClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def create_run(self, request: RunCreate) -> RunView:
        return RunView.model_validate(self._post("/v1/task-runs", request.model_dump()))

    def append_event(self, run_id: str, request: RunEventCreate) -> RunEventView:
        return RunEventView.model_validate(self._post(f"/v1/task-runs/{run_id}/events", request.model_dump()))

    def finalize_run(self, run_id: str, request: RunFinalize) -> ExperienceView:
        return ExperienceView.model_validate(self._post(f"/v1/task-runs/{run_id}/finalize", request.model_dump()))

    def search(self, **query: Any) -> SearchResponse:
        response = self._client.get("/v1/experiences/search", params=query)
        response.raise_for_status()
        return SearchResponse.model_validate(response.json())

    def get_experience(self, experience_id: str) -> ExperienceDetail:
        response = self._client.get(f"/v1/experiences/{experience_id}")
        response.raise_for_status()
        return ExperienceDetail.model_validate(response.json())

    def submit_feedback(self, experience_id: str, request: FeedbackCreate) -> ExperienceView:
        return ExperienceView.model_validate(self._post(f"/v1/experiences/{experience_id}/feedback", request.model_dump()))

    def lifecycle(self, experience_id: str, request: LifecycleRequest) -> ExperienceView:
        return ExperienceView.model_validate(self._post(f"/v1/experiences/{experience_id}/lifecycle", request.model_dump()))

    def apply_lifecycle(self, experience_id: str, request: LifecycleRequest) -> ExperienceView:
        """Apply an audited experience lifecycle action."""
        return self.lifecycle(experience_id, request)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(path, json=payload)
        response.raise_for_status()
        return response.json()
