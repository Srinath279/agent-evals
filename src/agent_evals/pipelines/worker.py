"""Temporal worker. Local dev:

    temporal server start-dev          # terminal 1
    python -m agent_evals.pipelines.worker   # terminal 2

    # then start a run:
    temporal workflow start --type EvalRunWorkflow --task-queue agent-evals \
      --workflow-id support-agent-eval-$(date +%s) \
      --input '"configs/support_agent.yaml"' --input '3' --input '"runs/temporal-cache"'

Production: point TEMPORAL_ADDRESS at the real cluster; workers run on Cloud Run.
"""

from __future__ import annotations

import asyncio
import os

from temporalio.client import Client
from temporalio.worker import Worker

from agent_evals.pipelines.activities import list_case_ids, score_case, score_online_trace
from agent_evals.pipelines.workflows import EvalRunWorkflow, TraceScoreWorkflow

TASK_QUEUE = "agent-evals"


async def main() -> None:
    client = await Client.connect(os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[EvalRunWorkflow, TraceScoreWorkflow],
        activities=[list_case_ids, score_case, score_online_trace],
    )
    print(f"agent-evals worker listening on task queue '{TASK_QUEUE}'")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
