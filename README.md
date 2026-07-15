# agent-evals

Reusable eval harness for AI agents — **Phase 0 vertical slice** of the
[master plan](https://github.com/Srinath279/agent-eval-notes) (note 00):
canonical schemas → trace adapter → PII redaction → 3 evaluators →
pass^k offline runner with an idempotent score cache, runnable locally with
zero external dependencies and as a Temporal workflow.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# end-to-end demo: fake support agent + 8-ticket golden set + mock judge
evals run --config configs/support_agent.yaml

pytest          # the harness's own test suite ("evals for the evals")
```

`evals run` executes every golden case k times (pass^k), scores each trace
with all configured evaluators, enforces the score thresholds as a gate
(exit code 1 on failure — wire it straight into CI), and writes
`runs/<agent>-<run_id>/` containing `report.md`, `manifest.json`,
`scores.jsonl`, and the score cache.

## Onboarding a real agent

1. Copy `configs/support_agent.yaml`; point `task_fn` at your agent's invoke
   function (contract: `task_fn(case_input) -> Trace`).
2. Golden set: a local JSONL of `Case` rows, or set `langfuse_dataset` and
   install the `langfuse` extra (env: `LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST`).
   `--post-scores` writes results back to Langfuse.
3. Real judge: set `judge.provider: anthropic` + a pinned model and install
   the `anthropic` extra (`ANTHROPIC_API_KEY`). Vertex/OpenAI land in Phase 2.
4. New metrics: subclass `BaseEvaluator`, `@register` it, add a row to
   [metrics.md](metrics.md) in the same PR.

## Layout

```
src/agent_evals/
├── core/            schemas, config, evaluator base+registry, multi-provider
│                    judge, score cache (idempotency layer), PII redaction,
│                    trace adapters, Langfuse client
├── evaluators/      label_match (deterministic) · goal_success (LLM judge)
│                    · tool_selection (trajectory)
├── runner.py        pass^k offline runner + gate + run artifacts
├── report.py        markdown run report
├── cli.py           `evals run` / `evals list-evaluators` — never needs Temporal
├── pipelines/       the ONLY package importing temporalio:
│                    EvalRunWorkflow, activities, worker
└── examples/        deterministic demo support agent
configs/             one YAML per agent (the reusability contract)
datasets/            demo golden set (JSONL)
metrics.md           enforced score registry
tests/               fixture traces, mock judge — no network, no spend
```

## Temporal mode (optional)

```bash
pip install -e ".[temporal]"
temporal server start-dev                    # terminal 1
python -m agent_evals.pipelines.worker       # terminal 2
```

See `pipelines/worker.py` for the `temporal workflow start` command. Rules
honored (note 08): orchestration only in `pipelines/`, IDs-not-payloads
through workflow history, activities-only I/O, chunked fan-out, and the score
cache makes at-least-once activity retries idempotent.

## Design invariants

- **Evaluators are pure**: no I/O except judge calls via the injected
  `JudgeClient` — the same code runs offline, online, and in activities.
- **Judge scores are versioned measurements**: provider + model +
  `rubric_version` stamped on every score and baked into the cache key.
- **PII is redacted** (regex baseline; Cloud DLP later) before any payload
  reaches a judge.
- **The gate is the API**: CI/CD consumes the exit code and `report.md`.

## Roadmap (from the master plan)

Phase 1: full deterministic set, real golden data, Cloud Run workers ·
Phase 2: Vertex/OpenAI judges, calibration vs annotations, bootstrap-CI
gating, trace-replay mode · Phase 3: online `TraceScoreWorkflow`, budgets,
feedback ingestion, BigQuery export · Phase 4+: onboarding kit, batch
judging, simulated users, red-team, chaos.
