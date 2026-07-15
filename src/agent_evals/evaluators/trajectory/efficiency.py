"""Trajectory efficiency evaluators — deterministic, no judge.

- redundant_calls: identical (tool, arguments) invocations are the classic
  confused-agent signature. Value = unique signatures / total calls.
- tool_success_rate: fraction of tool calls that did not error (higher
  better, so it gates like every other metric).
- steps_efficiency: 1.0 when steps <= max_steps, degrading proportionally
  past it. max_steps from params or the case's metadata.
"""

from __future__ import annotations

import json
from typing import Optional

from agent_evals.core.evaluator import BaseEvaluator, register
from agent_evals.core.schemas import Case, Score, Trace


@register
class RedundantCallsEvaluator(BaseEvaluator):
    name = "redundant_calls"
    level = "trace"

    def evaluate(self, trace: Trace, case: Optional[Case] = None) -> Score:
        if not trace.tool_calls:
            return Score(name=self.name, value=1.0, level=self.level, comment="no tool calls")
        signatures = [
            (t.name, json.dumps(t.arguments, sort_keys=True, default=str))
            for t in trace.tool_calls
        ]
        unique = set(signatures)
        value = len(unique) / len(signatures)
        duplicates = sorted({name for name, args in signatures
                             if signatures.count((name, args)) > 1})
        comment = "no redundant calls" if value == 1.0 else f"repeated identical calls: {duplicates}"
        return Score(name=self.name, value=value, level=self.level, comment=comment)


@register
class ToolSuccessRateEvaluator(BaseEvaluator):
    name = "tool_success_rate"
    level = "trace"

    def evaluate(self, trace: Trace, case: Optional[Case] = None) -> Score:
        if not trace.tool_calls:
            return Score(name=self.name, value=1.0, level=self.level, comment="no tool calls")
        failed = [t.name for t in trace.tool_calls if t.failed]
        value = 1 - len(failed) / len(trace.tool_calls)
        comment = "all tool calls succeeded" if not failed else f"failed calls: {failed}"
        return Score(name=self.name, value=value, level=self.level, comment=comment)


@register
class StepsEfficiencyEvaluator(BaseEvaluator):
    name = "steps_efficiency"
    level = "trace"

    def evaluate(self, trace: Trace, case: Optional[Case] = None) -> Score:
        max_steps = self.params.get("max_steps") or (case.metadata.get("max_steps") if case else None)
        if not max_steps:
            return Score(name=self.name, value=1.0, level=self.level,
                         comment="no max_steps configured — vacuously efficient")
        if trace.steps <= max_steps:
            return Score(name=self.name, value=1.0, level=self.level,
                         comment=f"{trace.steps} steps (budget {max_steps})")
        return Score(name=self.name, value=max_steps / trace.steps, level=self.level,
                     comment=f"{trace.steps} steps exceeds budget {max_steps}")
