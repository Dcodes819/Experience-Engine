"""Reproducible, model-free retrieval benchmark for the Compose scenario family."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from packages.engine.application.embeddings import HashingEmbeddingProvider, cosine_similarity
from packages.engine.application.service import ExperienceService
from packages.engine.domain.schemas import ExperienceSearch, RunCreate, RunEventCreate, RunFinalize
from packages.engine.infrastructure.db import Base, Experience

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "benchmark" / "scenarios" / "compose_troubleshooting.json"
LESSONS = {
    "database-connectivity": {
        "task": "Connect an API container to PostgreSQL in a Compose network.",
        "event": "Replace localhost with the database service DNS name db.",
        "outcome": "Database hostname corrected; API connected successfully.",
        "action": "Use the Compose service name db instead of localhost for the database host.",
    },
    "port-conflict": {
        "task": "Resolve a host port conflict when starting the Compose API.",
        "event": "The requested host port was occupied; remap the host side of the published port.",
        "outcome": "The API started on an unused host port.",
        "action": "Keep the container port and choose an unused host port in the Compose port mapping.",
    },
    "environment-configuration": {
        "task": "Provide required database configuration to an API container.",
        "event": "The database connection variable was absent; load it from the service environment configuration.",
        "outcome": "The API received its required database configuration.",
        "action": "Check env_file and environment entries; do not place secret values in source control.",
    },
}


def seed_experiences(session: Session) -> dict[str, Experience]:
    service = ExperienceService(session)
    experiences: dict[str, Experience] = {}
    for task_type, lesson in LESSONS.items():
        run = service.create_run(
            RunCreate(
                task_text=lesson["task"],
                agent_id="benchmark-seed",
                context={
                    "domain": "container-configuration",
                    "task_type": task_type,
                    "environment_fingerprint": "docker-compose-v2",
                },
                idempotency_key=f"benchmark-{task_type}",
            )
        )
        service.append_event(
            run.id,
            RunEventCreate(
                sequence=1,
                event_type="tool",
                tool_name="compose-diagnostics",
                output_summary=lesson["event"],
            ),
        )
        experience = service.finalize_run(
            run.id,
            RunFinalize(success=True, final_output_summary=lesson["outcome"], outcome_code=f"verified-{task_type}"),
        )
        experiences[task_type] = session.get(Experience, experience.id)
    return experiences


def evaluate_condition(
    scenarios: list[dict[str, Any]],
    condition: str,
    service: ExperienceService,
    experiences: dict[str, Experience],
) -> dict[str, float | int]:
    provider = HashingEmbeddingProvider()
    reciprocal_ranks: list[float] = []
    precision_at_3: list[float] = []
    harmful_at_1: list[float] = []
    successes: list[float] = []

    for scenario in scenarios:
        expected = scenario["expected"]
        if condition == "baseline":
            results: list[Experience] = []
        elif condition == "vector-only":
            query_vector = provider.embed(scenario["query"])
            results = sorted(
                experiences.values(),
                key=lambda item: cosine_similarity(query_vector, item.embedding or []),
                reverse=True,
            )[:3]
        else:
            response = service.search(
                ExperienceSearch(
                    query=scenario["query"],
                    domain="container-configuration",
                    task_type=scenario["task_type"],
                    environment_fingerprint=scenario["environment_fingerprint"],
                    limit=3,
                )
            )
            results = [experiences[scenario_type] for scenario_type in [
                next((key for key, item in experiences.items() if item.id == result.id), "")
                for result in response.results
            ]]

        result_types = [
            next((key for key, experience in experiences.items() if experience.id == result.id), "")
            for result in results
        ]
        rank = next((index for index, label in enumerate(result_types, start=1) if label == expected), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        precision_at_3.append(1 / 3 if expected in result_types else 0.0)
        harmful_at_1.append(float(bool(result_types) and result_types[0] != expected))
        successes.append(float(bool(result_types) and result_types[0] == expected))

    return {
        "scenario_count": len(scenarios),
        "top1_relevance_rate": round(statistics.mean(successes), 4),
        "precision_at_3": round(statistics.mean(precision_at_3), 4),
        "mrr": round(statistics.mean(reciprocal_ranks), 4),
        "harmful_advice_rate_at_1": round(statistics.mean(harmful_at_1), 4),
    }


def run_benchmark(scenarios_path: Path = SCENARIOS) -> dict[str, Any]:
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        experiences = seed_experiences(session)
        service = ExperienceService(session)
        metrics = {
            condition: evaluate_condition(scenarios, condition, service, experiences)
            for condition in ("baseline", "vector-only", "full-engine")
        }
    engine.dispose()
    return {
        "benchmark": "compose-troubleshooting-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario_file": str(scenarios_path.relative_to(ROOT)) if scenarios_path.is_relative_to(ROOT) else str(scenarios_path),
        "embedding_provider": HashingEmbeddingProvider.name,
        "metrics": metrics,
        "limitations": [
            "Controlled retrieval comparison, not an LLM agent trial; top1 relevance is not task success.",
            "One selected task family with synthetic development scenarios.",
            "No held-out split or confidence intervals; results are descriptive only.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline, vector-only, and full-engine retrieval.")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS, help="JSON scenario set.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark" / "results" / "latest.json",
        help="Where to write the JSON metrics report.",
    )
    args = parser.parse_args()
    report = run_benchmark(args.scenarios)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
