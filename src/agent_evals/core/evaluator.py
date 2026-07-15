"""Evaluator base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from agent_evals.core.config import EvaluatorSpec
from agent_evals.core.judge import BaseJudge
from agent_evals.core.schemas import Case, Score, Trace

_REGISTRY: dict[str, type["BaseEvaluator"]] = {}


def register(cls: type["BaseEvaluator"]) -> type["BaseEvaluator"]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name`")
    _REGISTRY[cls.name] = cls
    return cls


def available_evaluators() -> list[str]:
    return sorted(_REGISTRY)


def get_evaluator_class(name: str) -> Optional[type["BaseEvaluator"]]:
    return _REGISTRY.get(name)


def evaluator_requires_judge(name: str) -> bool:
    cls = _REGISTRY.get(name)
    return bool(cls and cls.requires_judge)


def create_evaluator(spec: EvaluatorSpec, judge: Optional[BaseJudge] = None) -> "BaseEvaluator":
    if spec.name not in _REGISTRY:
        raise KeyError(
            f"Unknown evaluator '{spec.name}'. Available: {available_evaluators()}"
        )
    ev = _REGISTRY[spec.name](**spec.params)
    if ev.requires_judge:
        if judge is None:
            raise ValueError(f"Evaluator '{spec.name}' requires a judge but none was configured")
        ev.judge = judge
    return ev


class BaseEvaluator(ABC):
    """One metric. `evaluate` must be pure: no I/O other than judge calls
    through self.judge, so the same evaluator runs offline, online, and
    inside Temporal activities."""

    name: str = ""
    level: str = "trace"  # output | trace | session
    requires_judge: bool = False
    # needs a golden Case to grade (fail-closed without one) — excluded from
    # the default online tier, where production traces have no case
    requires_case: bool = False
    # judge evaluators must bump this whenever their rubric text changes;
    # it is part of the score-cache key and stamped on every score
    rubric_version: str = "n/a"

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.judge: Optional[BaseJudge] = None

    @abstractmethod
    def evaluate(self, trace: Trace, case: Optional[Case] = None) -> Score: ...

    def _judge_stamp(self) -> dict[str, str]:
        assert self.judge is not None
        return {
            "judge_provider": self.judge.provider,
            "judge_model": self.judge.model,
            "rubric_version": self.rubric_version,
        }
