"""Controlled reference agent that consumes the engine through its Python SDK."""

from __future__ import annotations

import argparse
import os
import uuid

from packages.engine.domain.schemas import RunCreate, RunEventCreate, RunFinalize
from packages.engine.sdk import ExperienceClient

DOMAIN = "container-configuration"
TASK_TYPE = "database-connectivity"
ENVIRONMENT = "docker-compose-v2"


def run_demo(base_url: str) -> None:
    task = "Configure a containerized API to connect to PostgreSQL on its Compose network."
    context = {
        "domain": DOMAIN,
        "task_type": TASK_TYPE,
        "environment_fingerprint": ENVIRONMENT,
    }
    with ExperienceClient(base_url=base_url) as client:
        print("1. Search for bounded, compatible advice before acting")
        search = client.search(
            query=task,
            domain=DOMAIN,
            task_type=TASK_TYPE,
            environment_fingerprint=ENVIRONMENT,
            limit=3,
        )
        if not search.results:
            print("   No lesson is seeded yet; record one verified example through the SDK for this first-run demo.")
            seed = client.create_run(
                RunCreate(
                    task_text="Connect an API container to PostgreSQL in a Compose network.",
                    agent_id="experience-engine-demo-seed",
                    context=context,
                    idempotency_key="demo-verified-compose-database-hostname-v1",
                )
            )
            client.append_event(
                seed.id,
                RunEventCreate(
                    sequence=1,
                    event_type="configuration",
                    tool_name="compose-config-check",
                    input_summary="Inspect the API database host setting",
                    output_summary="Changed the database hostname from localhost to the Compose service name db.",
                ),
            )
            seeded_experience = client.finalize_run(
                seed.id,
                RunFinalize(
                    success=True,
                    final_output_summary="The API connected to PostgreSQL using the Compose service hostname db.",
                    outcome_code="database-connected",
                ),
            )
            print(f"   Verified experience recorded: {seeded_experience.id}")
            search = client.search(
                query=task,
                domain=DOMAIN,
                task_type=TASK_TYPE,
                environment_fingerprint=ENVIRONMENT,
                limit=3,
            )
        if search.results:
            for index, item in enumerate(search.results, start=1):
                print(f"   {index}. {item.lesson}")
                print(f"      action: {item.recommended_action}")
                print(f"      source: run {item.source_run_id}; evidence {', '.join(item.evidence_event_ids)}")
                if item.rank:
                    print(f"      rank: {item.rank.final_score:.3f} (semantic={item.rank.semantic_similarity:.3f})")
        else:
            print("   No compatible experience found; proceed without stored advice.")

        advice = "Use the Compose database service name 'db' as the hostname, not localhost."
        run = client.create_run(
            RunCreate(
                task_text=task,
                agent_id="experience-engine-reference-agent",
                context=context,
                idempotency_key=f"demo-{uuid.uuid4()}",
            )
        )
        client.append_event(
            run.id,
            RunEventCreate(
                sequence=1,
                event_type="configuration",
                tool_name="compose-config-check",
                input_summary="Inspect the API database host setting",
                output_summary=advice if search.results else "Changed localhost to the Compose service hostname db.",
            ),
        )
        experience = client.finalize_run(
            run.id,
            RunFinalize(
                success=True,
                final_output_summary="The API connected to PostgreSQL through the Compose network.",
                outcome_code="database-connected",
            ),
        )
        print("2. Record the run through the same SDK; the service validates and extracts it")
        print(f"   run: {run.id}")
        print(f"   experience: {experience.id} ({experience.status.value})")
        print(f"   evidence: {', '.join(experience.evidence_event_ids)}")
        print("Advice is demonstration context only; it does not override the current task or policy.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Experience Engine controlled reference agent.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("EXPERIENCE_ENGINE_URL", "http://127.0.0.1:8000"),
        help="Base URL of the Experience Engine REST API.",
    )
    args = parser.parse_args()
    run_demo(args.base_url)


if __name__ == "__main__":
    main()
