"""FastAPI adapter. Routes translate HTTP requests into core application commands only."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from packages.engine.application.embeddings import configured_embedding_provider
from packages.engine.application.service import ConflictError, ExperienceService, NotFoundError, ValidationPolicyError
from packages.engine.domain.schemas import (
    ExperienceDetail,
    ExperienceSearch,
    ExperienceView,
    FeedbackCreate,
    HealthView,
    LifecycleRequest,
    RunCreate,
    RunEventCreate,
    RunEventView,
    RunFinalize,
    RunView,
    SearchResponse,
)
from packages.engine.infrastructure.db import get_session, initialize_database

VERSION = "0.1.0"


def _service(session: Session = Depends(get_session)) -> Generator[ExperienceService, None, None]:
    yield ExperienceService(session)


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, NotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))
    if isinstance(error, ConflictError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error))
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error))


def create_app() -> FastAPI:
    initialize_database()
    app = FastAPI(
        title="Experience Engine",
        version=VERSION,
        description="Model-agnostic capture, validation, scoped retrieval, and lifecycle management for agent experiences.",
    )

    @app.get("/health", response_model=HealthView, tags=["operations"])
    def health(session: Session = Depends(get_session)) -> HealthView:
        """Report readiness only after checking the database and embedder."""
        try:
            session.execute(text("SELECT 1"))
            embedding_provider = configured_embedding_provider().name
        except SQLAlchemyError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database is unavailable.") from error
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Embedding provider is unavailable.") from error
        return HealthView(
            version=VERSION,
            database="ready",
            embedding_provider=embedding_provider,
        )

    @app.post("/v1/task-runs", response_model=RunView, status_code=status.HTTP_201_CREATED, tags=["task-runs"])
    def create_run(request: RunCreate, service: ExperienceService = Depends(_service)) -> RunView:
        try:
            return service.create_run(request)
        except (ConflictError, ValidationPolicyError, NotFoundError) as error:
            raise _http_error(error) from error

    @app.post("/v1/task-runs/{run_id}/events", response_model=RunEventView, status_code=status.HTTP_201_CREATED, tags=["task-runs"])
    def append_event(run_id: str, request: RunEventCreate, service: ExperienceService = Depends(_service)) -> RunEventView:
        try:
            return service.append_event(run_id, request)
        except (ConflictError, ValidationPolicyError, NotFoundError) as error:
            raise _http_error(error) from error

    @app.post("/v1/task-runs/{run_id}/finalize", response_model=ExperienceView, tags=["task-runs"])
    def finalize_run(run_id: str, request: RunFinalize, service: ExperienceService = Depends(_service)) -> ExperienceView:
        try:
            return service.finalize_run(run_id, request)
        except (ConflictError, ValidationPolicyError, NotFoundError) as error:
            raise _http_error(error) from error

    @app.get("/v1/experiences/search", response_model=SearchResponse, tags=["experiences"])
    def search_experiences(
        query: str = Query(min_length=1),
        domain: str | None = None,
        task_type: str | None = None,
        environment_fingerprint: str | None = None,
        preconditions: str | None = None,
        limit: int = Query(default=3, ge=1, le=20),
        run_id: str | None = None,
        service: ExperienceService = Depends(_service),
    ) -> SearchResponse:
        """Search compatible active advice; returned experiences are advisory, not instructions."""
        try:
            return service.search(
                ExperienceSearch(
                    query=query,
                    domain=domain,
                    task_type=task_type,
                    environment_fingerprint=environment_fingerprint,
                    preconditions=preconditions,
                    limit=limit,
                    run_id=run_id,
                )
            )
        except (ConflictError, ValidationPolicyError, NotFoundError) as error:
            raise _http_error(error) from error

    @app.get("/v1/experiences/{experience_id}", response_model=ExperienceDetail, tags=["experiences"])
    def get_experience(experience_id: str, service: ExperienceService = Depends(_service)) -> ExperienceDetail:
        try:
            return service.get_experience(experience_id)
        except NotFoundError as error:
            raise _http_error(error) from error

    @app.post("/v1/experiences/{experience_id}/feedback", response_model=ExperienceView, tags=["experiences"])
    def submit_feedback(
        experience_id: str,
        request: FeedbackCreate,
        service: ExperienceService = Depends(_service),
    ) -> ExperienceView:
        try:
            return service.submit_feedback(experience_id, request)
        except (ValidationPolicyError, NotFoundError) as error:
            raise _http_error(error) from error

    @app.post("/v1/experiences/{experience_id}/lifecycle", response_model=ExperienceView, tags=["experiences"])
    def lifecycle_action(
        experience_id: str,
        request: LifecycleRequest,
        service: ExperienceService = Depends(_service),
    ) -> ExperienceView:
        try:
            return service.apply_lifecycle(experience_id, request)
        except (ValidationPolicyError, NotFoundError) as error:
            raise _http_error(error) from error

    return app


app = create_app()
