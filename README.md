# Experience Engine

Experience Engine is a model-agnostic service for capturing agent runs, validating evidence-backed lessons, retrieving compatible experiences, and maintaining feedback and lifecycle history. It exposes one shared application service through REST, a Python SDK, and MCP tools. Stored experiences are advisory context, not privileged instructions.

## Quick start (Windows PowerShell)

1. Create/activate a Python 3.12+ virtual environment and install the project with its development extras: `python -m pip install -e ".[dev]"`.
2. Start the API: `python -m uvicorn apps.api.main:app --reload`.
3. In another terminal, run the controlled reference agent: `python -m apps.demo_agent.main`.
4. Open `http://127.0.0.1:8000/docs` to inspect the REST contract.

By default the service uses a local SQLite database at `experience_engine.db`. To use the included PostgreSQL + pgvector container, start Docker Compose and set `EXPERIENCE_ENGINE_DATABASE_URL` to `postgresql+psycopg://experience_engine:experience_engine@localhost:5432/experience_engine`. The initial MVP stores embeddings as JSON for exact, portable baseline ranking; the current Compose image makes a PostgreSQL/pgvector deployment available for a future vector-column migration.

## Run a reproducible demo and benchmark

The reference agent talks to the REST API only through `ExperienceClient`. It searches before acting, records a normalized event, finalizes the run, and prints the resulting provenance.

Run the controlled three-condition benchmark with:

`python -m benchmark.run_experiment`

The report is written to `benchmark/results/latest.json` (ignored by Git). It compares a no-retrieval baseline, unfiltered vector-only retrieval, and the full engine with scope filtering and ranking. The 24 scenarios are synthetic development cases for one Compose troubleshooting task family. The benchmark reports retrieval relevance, not agent task success: it does not run an LLM or execute tasks. It is not a held-out evaluation or evidence of general improvement. See each report's limitations before interpreting its metrics.

## MCP adapter

Run `python -m apps.mcp_server.main` as an MCP stdio server. Its tools call the same `ExperienceService` used by REST; `search_experiences`, `record_task_run`, `append_task_event`, `finalize_task_run`, `submit_experience_feedback`, and `get_experience` do not implement a separate policy.

## Tests

Run `python -m pytest -q`. Tests cover the core service and REST contract; benchmark and adapter tests are included as those paths are exercised. Use isolated test databases rather than the default local database.

## Core safeguards and behavior

- Finalization requires an event and a structured outcome; successful runs create active experiences, while failed runs remain review candidates.
- Event sequences are contiguous and unique per run; a finalized run is immutable and finalization retries must match the original outcome.
- Obvious credential forms and sensitive context keys are redacted before persistence.
- Search excludes non-active records and hard-filters declared scope incompatibilities. The returned rank uses fixed, visible weights (semantic 0.45, scope 0.20, confidence 0.15, helpfulness 0.10, recency 0.10).
- Feedback can be tied to a retrieval that actually returned that experience. Lifecycle transitions retain provenance and require an active replacement for supersession.
- The offline hashing embedder is deterministic and convenient for tests; it is not a production semantic model. Embedding providers remain replaceable.

To enable the optional local semantic provider, install the `semantic` extra, set `EXPERIENCE_ENGINE_EMBEDDING_PROVIDER=sentence-transformers`, and optionally select a model with `EXPERIENCE_ENGINE_EMBEDDING_MODEL` (default `all-MiniLM-L6-v2`). Existing records must be explicitly re-embedded before switching models; search intentionally rejects mixed embedding spaces.
