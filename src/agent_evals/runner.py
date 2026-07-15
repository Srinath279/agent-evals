"""Offline experiment runner with pass^k (master plan §8, §9).

Pure library code — no Temporal imports. cli.py calls run_offline
directly; the Temporal EvalRunWorkflow calls the same building blocks
(load_cases / run_single_case / score_trace) from activities.
"""

from __future__ import annotations

import importlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import agent_evals.evaluators  # noqa: F401  (registers built-ins)
from agent_evals import __version__
from agent_evals.core.cache import ScoreCache, score_cache_key
from agent_evals.core.config import AgentConfig
from agent_evals.core.evaluator import (
    BaseEvaluator,
    create_evaluator,
    evaluator_requires_judge,
    get_evaluator_class,
)
from agent_evals.core.judge import BaseJudge, make_judge
from agent_evals.core.schemas import Case, Score, Trace
from agent_evals.report import render_report


def load_cases(cfg: AgentConfig) -> list[Case]:
    if cfg.local_dataset:
        path = cfg.resolve_path(cfg.local_dataset)
        with open(path) as f:
            return [Case(**json.loads(line)) for line in f if line.strip()]
    if cfg.langfuse_dataset:
        from agent_evals.core.langfuse_client import LangfuseClient

        return LangfuseClient().load_dataset(cfg.langfuse_dataset)
    raise ValueError("Config must set local_dataset or langfuse_dataset")


def resolve_task_fn(spec: str) -> Callable:
    """Resolve 'package.module:function' (or dotted fallback) to a callable.
    Contract: task_fn(case_input) -> Trace (or a dict coercible to one)."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        module_name, _, attr = spec.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def build_evaluators(cfg: AgentConfig, judge: Optional[BaseJudge] = None) -> tuple[list[BaseEvaluator], Optional[BaseJudge]]:
    needs_judge = any(evaluator_requires_judge(s.name) for s in cfg.evaluators)
    if needs_judge and judge is None:
        judge = make_judge(cfg.judge)
    return [create_evaluator(spec, judge=judge) for spec in cfg.evaluators], judge


def score_trace(
    trace: Trace,
    case: Optional[Case],
    evaluators: list[BaseEvaluator],
    cache: Optional[ScoreCache] = None,
    repeat_index: int = 0,
) -> list[Score]:
    """Score one trace with every evaluator, going through the score cache
    so retries are idempotent (never re-pay a judge call). Shared by the
    offline runner and the online/Temporal paths."""
    scores = []
    for ev in evaluators:
        key = score_cache_key(
            trace.trace_id,
            ev.name,
            ev.rubric_version,
            ev.judge.provider if ev.judge else "none",
            ev.judge.model if ev.judge else "none",
        )
        score = cache.get(key) if cache else None
        if score is None:
            score = ev.evaluate(trace, case)
            score.trace_id = trace.trace_id
            score.case_id = case.case_id if case else None
            score.repeat_index = repeat_index
            if trace.metadata.get("source_trace_id"):
                # so Langfuse write-back attaches to the agent's real trace
                score.metadata.setdefault("source_trace_id", trace.metadata["source_trace_id"])
            if cache:
                cache.put(key, score)
        scores.append(score)
    return scores


@dataclass
class CaseRunResult:
    case_id: str
    repeat_index: int
    trace_id: str
    scores: list[Score]
    passed: bool
    failures: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    run_id: str
    agent: str
    k: int
    n_cases: int
    case_results: list[CaseRunResult]
    metric_means: dict[str, float]
    pass_rate: float       # fraction of (case, repeat) executions meeting all thresholds
    pass_k_rate: float     # fraction of cases meeting all thresholds on EVERY repeat
    gate_passed: bool
    gate_failures: list[str]
    out_dir: Optional[str] = None


def _check_thresholds(scores: list[Score], thresholds: dict[str, float]) -> list[str]:
    by_name = {s.name: s for s in scores}
    failures = []
    for metric, threshold in thresholds.items():
        score = by_name.get(metric)
        if score is None:
            failures.append(f"{metric}: no score produced (required >= {threshold})")
        elif score.value < threshold:
            failures.append(f"{metric}: {score.value:.3f} < {threshold} — {score.comment}")
    return failures


def run_single_case(
    case: Case,
    task_fn: Callable,
    evaluators: list[BaseEvaluator],
    cfg: AgentConfig,
    cache: Optional[ScoreCache],
    repeat_index: int,
    run_id: str,
) -> CaseRunResult:
    result = task_fn(case.input)
    trace = result if isinstance(result, Trace) else Trace(**result)
    # Deterministic ID per (run, case, repeat): score-cache hits must survive
    # retries even when the agent mints a random trace ID (note 09 §1). The
    # agent's own Langfuse trace ID is preserved for score write-back.
    source_trace_id = trace.trace_id
    trace.trace_id = f"{run_id}/{case.case_id}/r{repeat_index}"
    if source_trace_id:
        trace.metadata["source_trace_id"] = source_trace_id
    trace.agent = trace.agent or cfg.agent

    scores = score_trace(trace, case, evaluators, cache=cache, repeat_index=repeat_index)
    failures = _check_thresholds(scores, cfg.score_thresholds)
    return CaseRunResult(
        case_id=case.case_id,
        repeat_index=repeat_index,
        trace_id=trace.trace_id,
        scores=scores,
        passed=not failures,
        failures=failures,
    )


def run_offline(
    cfg: AgentConfig,
    k: Optional[int] = None,
    out_dir: str | Path = "runs",
    judge: Optional[BaseJudge] = None,
    cache: Optional[ScoreCache] = None,
    post_to_langfuse: bool = False,
) -> RunResult:
    k = k or cfg.repeats
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = Path(out_dir) / f"{cfg.agent}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.task_fn is None:
        raise ValueError("Offline experiment mode requires task_fn (trace-replay mode lands in Phase 2)")
    task_fn = resolve_task_fn(cfg.task_fn)
    evaluators, judge = build_evaluators(cfg, judge=judge)
    cache = cache or ScoreCache(run_dir / "scores.sqlite3")
    cases = load_cases(cfg)

    case_results: list[CaseRunResult] = []
    for case in cases:
        for r in range(k):
            case_results.append(
                run_single_case(case, task_fn, evaluators, cfg, cache, repeat_index=r, run_id=run_id)
            )

    all_scores = [s for cr in case_results for s in cr.scores]
    metric_means: dict[str, float] = {}
    for name in sorted({s.name for s in all_scores}):
        values = [s.value for s in all_scores if s.name == name]
        metric_means[name] = sum(values) / len(values)

    pass_rate = sum(cr.passed for cr in case_results) / len(case_results) if case_results else 0.0
    by_case: dict[str, list[bool]] = {}
    for cr in case_results:
        by_case.setdefault(cr.case_id, []).append(cr.passed)
    pass_k_rate = (
        sum(all(passes) for passes in by_case.values()) / len(by_case) if by_case else 0.0
    )

    gate_failures = []
    for metric, threshold in cfg.score_thresholds.items():
        mean = metric_means.get(metric)
        if mean is None:
            gate_failures.append(f"{metric}: never scored (required >= {threshold})")
        elif mean < threshold:
            gate_failures.append(f"{metric}: mean {mean:.3f} < {threshold}")

    result = RunResult(
        run_id=run_id,
        agent=cfg.agent,
        k=k,
        n_cases=len(cases),
        case_results=case_results,
        metric_means=metric_means,
        pass_rate=pass_rate,
        pass_k_rate=pass_k_rate,
        gate_passed=not gate_failures,
        gate_failures=gate_failures,
        out_dir=str(run_dir),
    )

    _write_artifacts(result, cfg, judge, run_dir, all_scores)

    if post_to_langfuse:
        from agent_evals.core.langfuse_client import LangfuseClient

        client = LangfuseClient()
        for score in all_scores:
            client.post_score(score)
        client.flush()

    return result


def _write_artifacts(
    result: RunResult,
    cfg: AgentConfig,
    judge: Optional[BaseJudge],
    run_dir: Path,
    all_scores: list[Score],
) -> None:
    with open(run_dir / "scores.jsonl", "w") as f:
        for score in all_scores:
            f.write(score.model_dump_json() + "\n")

    # run manifest: everything needed to reproduce/attribute this run (§8)
    manifest = {
        "run_id": result.run_id,
        "harness_version": __version__,
        "agent": cfg.agent,
        "k": result.k,
        "dataset": cfg.langfuse_dataset or cfg.local_dataset,
        "judge_provider": judge.provider if judge else None,
        "judge_model": judge.model if judge else None,
        "rubric_versions": {
            spec.name: rv
            for spec in cfg.evaluators
            if (rv := getattr(get_evaluator_class(spec.name), "rubric_version", None)) not in (None, "n/a")
        },
        "config": cfg.model_dump(exclude={"config_dir"}),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    with open(run_dir / "report.md", "w") as f:
        f.write(render_report(result, cfg))
