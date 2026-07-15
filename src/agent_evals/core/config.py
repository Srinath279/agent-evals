"""Per-agent YAML config — the reusability contract: onboarding a new
agent must cost only a config file, a trace adapter, and optionally a
few custom evaluators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator


class JudgeConfig(BaseModel):
    provider: str = "mock"  # mock | anthropic | vertex | openai
    model: str = "mock-judge"
    fallback: Optional["JudgeConfig"] = None
    daily_budget_usd: Optional[float] = None      # hard cap, enforced per (day, provider, model)
    est_cost_per_call_usd: float = 0.01           # budget accounting unit until real token costing
    budget_db: Optional[str] = None               # sqlite path; default runs/judge_budget.sqlite3
    # vertex only (fall back to GCP env/ADC when unset)
    project_id: Optional[str] = None
    region: Optional[str] = None


class EvaluatorSpec(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    agent: str
    trace_adapter: str = "langfuse-generic"
    task_fn: Optional[str] = None  # "package.module:function"
    langfuse_dataset: Optional[str] = None
    local_dataset: Optional[str] = None  # JSONL of Cases; path relative to config file
    trace_filter: dict[str, Any] = Field(default_factory=dict)
    online_sample_rate: float = 0.0
    repeats: int = 1  # k for pass^k
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    evaluators: list[EvaluatorSpec] = Field(default_factory=list)
    # cheap-tier subset for online scoring; empty = non-judge evaluators only
    online_evaluators: list[EvaluatorSpec] = Field(default_factory=list)
    score_thresholds: dict[str, float] = Field(default_factory=dict)
    # "mean": gate on metric means (default). "all": every execution must pass
    # every threshold — the red-team / pass-100% gate (master plan §9).
    gate_mode: str = "mean"
    # online sampling: traces above these are always scored (suspicious)
    outlier_latency_ms: Optional[float] = None
    outlier_cost_usd: Optional[float] = None
    # raw traces JSONL for replay mode (adapter-shaped lines)
    replay_traces: Optional[str] = None

    # set by load_config so relative paths resolve against the config file
    config_dir: Optional[str] = None

    @field_validator("evaluators", "online_evaluators", mode="before")
    @classmethod
    def _normalize_evaluators(cls, v: list[Union[str, dict]]) -> list[dict]:
        specs = []
        for item in v or []:
            if isinstance(item, str):
                specs.append({"name": item})
            elif isinstance(item, dict):
                name = item.pop("name")
                specs.append({"name": name, "params": item.pop("params", item)})
            else:
                specs.append(item)
        return specs

    def resolve_path(self, p: str) -> Path:
        path = Path(p)
        if not path.is_absolute() and self.config_dir:
            path = Path(self.config_dir) / path
        return path.resolve()


def load_config(path: str | Path) -> AgentConfig:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = AgentConfig(**raw)
    cfg.config_dir = str(path.parent)
    return cfg
