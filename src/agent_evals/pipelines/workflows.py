"""EvalRunWorkflow — the offline experiment runner as a durable workflow
(master plan §9, note 08). Same evaluator library as `evals run`; Temporal
only supplies durability, retries, and scheduling.

Determinism discipline: no I/O, clocks, or randomness here — activities only.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from agent_evals.pipelines.activities import list_case_ids, score_case

CHUNK_SIZE = 50  # fan out in chunks to keep workflow histories small (rule 5)


@workflow.defn
class EvalRunWorkflow:
    @workflow.run
    async def run(self, config_path: str, k: int, cache_dir: str) -> dict:
        run_id = workflow.info().workflow_id

        case_ids: list[str] = await workflow.execute_activity(
            list_case_ids,
            args=[config_path],
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        executions = [(case_id, r) for case_id in case_ids for r in range(k)]
        results: list[dict] = []
        for start in range(0, len(executions), CHUNK_SIZE):
            chunk = executions[start : start + CHUNK_SIZE]
            handles = [
                workflow.execute_activity(
                    score_case,
                    args=[config_path, case_id, repeat, run_id, cache_dir],
                    start_to_close_timeout=timedelta(minutes=10),
                    # aggressive retries: judge 429s/outages; the score cache
                    # makes retried executions idempotent
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=5),
                        backoff_coefficient=2.0,
                        maximum_attempts=5,
                    ),
                )
                for case_id, repeat in chunk
            ]
            for handle in handles:
                results.append(await handle)

        pass_rate = sum(r["passed"] for r in results) / len(results) if results else 0.0
        by_case: dict[str, list[bool]] = {}
        for r in results:
            by_case.setdefault(r["case_id"], []).append(r["passed"])
        pass_k = sum(all(v) for v in by_case.values()) / len(by_case) if by_case else 0.0

        return {
            "run_id": run_id,
            "n_cases": len(case_ids),
            "k": k,
            "pass_rate": pass_rate,
            "pass_k_rate": pass_k,
            "failing": [r for r in results if not r["passed"]],
        }
