# agent-evals

Reusable eval harness for AI agents, implementing the
[master plan](https://github.com/Srinath279/agent-eval-notes) (note 00):
canonical schemas → trace adapters → PII redaction → 13 registered
evaluators → pass^k offline runner with idempotent score caching,
baseline/bootstrap-CI regression gating, trace-replay mode, judge
calibration, online sampling + cheap-tier scoring, red-team suites
(pass-100% gates), chaos injection, and Temporal workflows — runnable
locally with zero external dependencies.

```
evals run              offline experiment (pass^k, gate, artifacts)
evals run --mode replay --traces prod.jsonl    re-score stored traces
evals promote-baseline                         register the current baseline
evals calibrate        judge-vs-human agreement (kappa/pearson)
evals seed-dataset     redacted golden-set upload to Langfuse
evals list-evaluators
```

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
2. Golden set: a local JSONL of `Case` rows, or set `dataset` and install the
   extra for your `trace_store` — `langfuse` (env:
   `LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST`, the default) or `langsmith`
   (env: `LANGSMITH_API_KEY`; pair with `trace_adapter: langsmith-generic`).
   `--post-scores` writes results back to the configured store.
3. Real judge: set `judge.provider: anthropic` + a pinned model and install
   the `anthropic` extra (`ANTHROPIC_API_KEY`). Vertex/OpenAI land in Phase 2.
4. New metrics: subclass `BaseEvaluator`, `@register` it, add a row to
   [metrics.md](metrics.md) in the same PR.

## Layout

```
src/agent_evals/
├── core/            schemas, config, evaluator base+registry, multi-provider
│                    judge, score cache (idempotency layer), PII redaction,
│                    trace adapters, trace stores (Langfuse, LangSmith)
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

## Switching platforms

The observability platform is a config field, not a dependency:

```yaml
trace_store: langsmith          # was: langfuse (the default)
trace_adapter: langsmith-generic
dataset: my-golden-v1
```

Both stores implement the same `TraceStore` contract (`core/store.py`):
datasets, score write-back, prompt/rubric management, trace fetching, and
annotation queues. Adding platform N = one `TraceStore` subclass + one
`TraceAdapter` for its raw trace shape. Legacy configs using
`langfuse_dataset` / `rubric_from_langfuse` keep working.

## Design invariants

- **Evaluators are pure**: no I/O except judge calls via the injected
  `JudgeClient` — the same code runs offline, online, and in activities.
- **Judge scores are versioned measurements**: provider + model +
  `rubric_version` stamped on every score and baked into the cache key.
- **PII is redacted** (regex baseline; Cloud DLP later) before any payload
  reaches a judge.
- **The gate is the API**: CI/CD consumes the exit code and `report.md`.

## Implemented phases (code-complete; see the notes repo for the plan)

- **Phase 0–1**: schemas, `langfuse-generic` adapter, PII redaction, full
  deterministic + trajectory evaluator set, pass^k runner, gate, artifacts.
- **Phase 2**: multi-provider judges (Anthropic, **Vertex**, **OpenAI**) with
  **fallback** and truthful score stamping; **daily budget enforcement**
  (`BudgetExceededError` kill switch); **baseline registry +
  bootstrap-CI regression gating**; **trace-replay mode**; `evals calibrate`
  (Cohen's kappa / Pearson vs human labels).
- **Phase 3**: online sampling (`should_score`: rate + 100% of tool-error /
  negative-feedback / latency / cost outliers), cheap-tier `score_online`,
  `TraceScoreWorkflow`, annotation-queue push (best-effort with graceful
  fallback), `post_user_feedback`, BigQuery export.
- **Phase 4/5 (code parts)**: onboarding kit (`configs/_template.yaml` +
  `assert_adapter_conformance`), failure clustering in reports, **red-team
  suite** (`gate_mode: all`, `forbidden_content`), **chaos injection**
  (`ChaosInjector`) + `recovery_after_error`.

## Remaining wiring (needs your infra, not code)

dev/staging/prod Langfuse projects + Secret Manager keys · seed the real
golden set (`evals seed-dataset`) · Temporal workers on Cloud Run against
the real cluster + Schedules · Pub/Sub starter for `TraceScoreWorkflow` ·
BigQuery dataset + Looker dashboards · CI wiring (`evals run --baselines`
exit code as the PR gate). Later code phases: batch judging APIs,
simulated users (τ-bench style), drift detection, canary/shadow workflows.
