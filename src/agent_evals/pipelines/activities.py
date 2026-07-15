"""Temporal activities — all I/O lives here (note 08, rule 4).

Activities receive IDs and paths, never payloads (rule 2): each call
loads the config/dataset itself and returns a compact summary dict, so
workflow histories stay small. The score cache (a shared file per run)
keeps at-least-once execution idempotent (rule 3).
"""

from __future__ import annotations

from pathlib import Path

from temporalio import activity

from agent_evals.core.cache import ScoreCache
from agent_evals.core.config import load_config
from agent_evals.runner import build_evaluators, load_cases, resolve_task_fn, run_single_case


@activity.defn
def list_case_ids(config_path: str) -> list[str]:
    cfg = load_config(config_path)
    return [case.case_id for case in load_cases(cfg)]


@activity.defn
def score_case(config_path: str, case_id: str, repeat_index: int, run_id: str, cache_dir: str) -> dict:
    cfg = load_config(config_path)
    case = next(c for c in load_cases(cfg) if c.case_id == case_id)
    task_fn = resolve_task_fn(cfg.task_fn)
    evaluators, _ = build_evaluators(cfg)
    cache = ScoreCache(Path(cache_dir) / f"{run_id}.sqlite3")
    try:
        result = run_single_case(
            case, task_fn, evaluators, cfg, cache, repeat_index=repeat_index, run_id=run_id
        )
    finally:
        cache.close()
    return {
        "case_id": result.case_id,
        "repeat_index": result.repeat_index,
        "passed": result.passed,
        "failures": result.failures,
        "scores": {s.name: s.value for s in result.scores},
    }
