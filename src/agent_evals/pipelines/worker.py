"""Temporal worker (opt-in — pipeline.engine: temporal). Local dev:

    temporal server start-dev                                   # terminal 1
    evals worker --config configs/support_agent.yaml           # terminal 2
    # (or: python -m agent_evals.pipelines.worker configs/support_agent.yaml)

    # then start a run against the same config:
    evals run --config configs/support_agent.yaml              # dispatches to the cluster

Production: point pipeline.temporal.address at the real cluster; workers run on
Cloud Run. Connection settings come from the config's pipeline.temporal block,
so one flag (pipeline.engine) turns the durable path on or off.
"""

from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from agent_evals.core.config import TemporalConfig, load_config
from agent_evals.pipelines import require_temporal


async def _run_worker(tc: TemporalConfig) -> None:
    # Imports are deferred until after require_temporal(): importing this module
    # (e.g. via `evals worker` argparse) must not need temporalio installed.
    from temporalio.client import Client
    from temporalio.worker import Worker

    from agent_evals.pipelines.activities import list_case_ids, score_case, score_online_trace
    from agent_evals.pipelines.workflows import EvalRunWorkflow, TraceScoreWorkflow

    client = await Client.connect(tc.address, namespace=tc.namespace)
    worker = Worker(
        client,
        task_queue=tc.task_queue,
        workflows=[EvalRunWorkflow, TraceScoreWorkflow],
        activities=[list_case_ids, score_case, score_online_trace],
        # activities are sync (they call the library + agent task_fns) —
        # temporalio requires an executor for sync activities
        activity_executor=ThreadPoolExecutor(max_workers=tc.max_concurrent_activities),
    )
    print(
        f"agent-evals worker listening on task queue '{tc.task_queue}' "
        f"@ {tc.address} (ns={tc.namespace})"
    )
    await worker.run()


def run_worker(config_path: Optional[str] = None) -> None:
    """Start a worker using the config's pipeline.temporal settings. Refuses to
    start unless pipeline.engine == 'temporal' (the flag is the single source of
    truth) and the 'temporal' extra is installed."""
    require_temporal()
    if config_path is None:
        tc = TemporalConfig()  # bare defaults for `python -m ...worker` with no arg
    else:
        cfg = load_config(config_path)
        if cfg.pipeline.engine != "temporal":
            raise SystemExit(
                f"pipeline.engine is '{cfg.pipeline.engine}' in {config_path}; "
                "set it to 'temporal' to run a worker (or run in-process with "
                "`evals run`)."
            )
        tc = cfg.pipeline.temporal
    asyncio.run(_run_worker(tc))


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for `python -m agent_evals.pipelines.worker [config_path]`."""
    args = sys.argv[1:] if argv is None else argv
    run_worker(args[0] if args else None)


if __name__ == "__main__":
    main()
