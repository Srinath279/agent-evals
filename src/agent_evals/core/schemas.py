"""Canonical data models. Every adapter maps raw traces into these;
every evaluator consumes only these. This is the reusability contract."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None

    @property
    def failed(self) -> bool:
        return self.error is not None


class Trace(BaseModel):
    """One agent execution, normalized. Produced by a trace adapter
    (from Langfuse observations) or directly by a task_fn."""

    trace_id: str
    agent: str = ""
    input: Any = None
    output: Any = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    steps: int = 0
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Case(BaseModel):
    """One golden-dataset item (a Langfuse dataset item, or a JSONL row
    in local mode)."""

    case_id: str
    input: Any
    expected_output: Any = None
    expected_labels: dict[str, Any] = Field(default_factory=dict)
    expected_tools: list[str] = Field(default_factory=list)
    reference: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Score(BaseModel):
    """One metric result. Posted to Langfuse as a Score; judge scores
    carry judge_provider/judge_model/rubric_version in metadata."""

    name: str
    value: float
    comment: str = ""
    level: str = "trace"  # output | trace | session
    trace_id: Optional[str] = None
    case_id: Optional[str] = None
    repeat_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
