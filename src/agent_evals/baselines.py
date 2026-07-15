"""Baseline-run registry (master plan §8): an explicit record of the
current baseline per agent, promoted deliberately — never implicitly.
Runs compare their per-execution metric values against it with bootstrap
CIs; the gate fails on significant regression, not point noise."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from agent_evals.core.schemas import Score


def _registry_file(baselines_dir: str | Path, agent: str) -> Path:
    return Path(baselines_dir) / f"{agent}.json"


def load_baseline(baselines_dir: str | Path, agent: str) -> Optional[dict]:
    path = _registry_file(baselines_dir, agent)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def promote_baseline(baselines_dir: str | Path, agent: str, run_dir: str | Path) -> dict:
    """Promote a completed run (its scores.jsonl) to be the agent's baseline."""
    run_dir = Path(run_dir)
    metrics: dict[str, list[float]] = {}
    with open(run_dir / "scores.jsonl") as f:
        for line in f:
            score = Score(**json.loads(line))
            metrics.setdefault(score.name, []).append(score.value)

    manifest = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    baseline = {
        "agent": agent,
        "run_id": manifest.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "judge_provider": manifest.get("judge_provider"),
        "judge_model": manifest.get("judge_model"),
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "metrics": metrics,
    }
    path = _registry_file(baselines_dir, agent)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    return baseline


def compare_to_baseline(
    baseline: dict,
    current_values: dict[str, list[float]],
    metrics: list[str],
) -> dict[str, dict]:
    """Bootstrap-CI comparison per metric. regression=True when the CI for
    (current - baseline) lies entirely below zero."""
    from agent_evals.core.stats import bootstrap_mean_diff_ci, mean

    comparison: dict[str, dict] = {}
    for metric in metrics:
        cur = current_values.get(metric)
        base = baseline.get("metrics", {}).get(metric)
        if not cur or not base:
            continue
        lo, hi = bootstrap_mean_diff_ci(cur, base)
        comparison[metric] = {
            "mean_diff": mean(cur) - mean(base),
            "ci": [lo, hi],
            "regression": hi < 0,
            "baseline_run": baseline.get("run_id"),
        }
    return comparison
