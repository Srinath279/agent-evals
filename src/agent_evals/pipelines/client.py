"""Client-side Temporal dispatch (note 08). Connects to the cluster, runs the
durable EvalRunWorkflow (which fans out scoring), then aggregates the returned
case results with the SAME aggregate_run() the local engine uses — so both
engines gate, compare to baseline, and emit identical artifacts. Kept out of
cli.py so the CLI imports temporalio only when pipeline.engine == 'temporal'.

Requires a worker listening on the same task queue (see worker.py).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

from agent_evals.core.config import AgentConfig
from agent_evals.core.schemas import Score
from agent_evals.pipelines import require_temporal
from agent_evals.runner import CaseRunResult, RunResult, aggregate_run


async def _dispatch(cfg: AgentConfig, config_path: str, k: int) -> dict:
    from temporalio.client import Client

    from agent_evals.pipelines.workflows import EvalRunWorkflow

    tc = cfg.pipeline.temporal
    client = await Client.connect(tc.address, namespace=tc.namespace)
    workflow_id = f"{cfg.agent}-eval-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    return await client.execute_workflow(
        EvalRunWorkflow.run,
        args=[config_path, k, tc.cache_dir],
        id=workflow_id,
        task_queue=tc.task_queue,
    )


def _to_case_result(d: dict) -> CaseRunResult:
    return CaseRunResult(
        case_id=d["case_id"],
        repeat_index=d["repeat_index"],
        trace_id=d.get("trace_id", ""),
        scores=[Score(**s) for s in d.get("scores", [])],
        passed=d["passed"],
        failures=d.get("failures", []),
    )


def submit_eval_run(
    cfg: AgentConfig,
    config_path: str,
    k: Optional[int] = None,
    *,
    out_dir: str | Path = "runs",
    baselines_dir: Optional[str | Path] = None,
    post_scores: bool = False,
) -> RunResult:
    """Run the eval on the Temporal cluster and aggregate the result locally
    into a full RunResult (gated, baseline-compared, artifacts written) — the
    same object `run_offline` returns. Raises a friendly error if the 'temporal'
    extra isn't installed."""
    import asyncio

    require_temporal()
    k = k or cfg.repeats
    summary = asyncio.run(_dispatch(cfg, config_path, k))

    case_results = [_to_case_result(d) for d in summary["case_results"]]
    run_id = summary["run_id"]
    run_dir = Path(out_dir) / f"{cfg.agent}-{run_id}"
    return aggregate_run(
        cfg,
        case_results,
        run_id=run_id,
        run_dir=run_dir,
        k=summary.get("k", k),
        n_cases=summary.get("n_cases", len({cr.case_id for cr in case_results})),
        mode="experiment",
        # manifest stamps the configured judge; runtime fallback/budget wrappers
        # don't change these names, and the client avoids instantiating a judge
        # (no API keys needed just to aggregate)
        judge_provider=cfg.judge.provider,
        judge_model=cfg.judge.model,
        baselines_dir=baselines_dir,
        post_scores=post_scores,
    )
