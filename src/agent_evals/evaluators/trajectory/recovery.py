"""recovery_after_error — deterministic resilience metric (Strands-style,
pairs with chaos injection). For each failed tool call: did a later call
to the same tool succeed? Value = recovered failures / total failures;
vacuously 1.0 when nothing failed."""

from __future__ import annotations

from typing import Optional

from agent_evals.core.evaluator import BaseEvaluator, register
from agent_evals.core.schemas import Case, Score, Trace


@register
class RecoveryAfterErrorEvaluator(BaseEvaluator):
    name = "recovery_after_error"
    level = "trace"

    def evaluate(self, trace: Trace, case: Optional[Case] = None) -> Score:
        calls = trace.tool_calls
        failures = [(i, t.name) for i, t in enumerate(calls) if t.failed]
        if not failures:
            return Score(name=self.name, value=1.0, level=self.level,
                         comment="no tool failures to recover from")

        unrecovered = []
        recovered = 0
        for i, name in failures:
            if any(t.name == name and not t.failed for t in calls[i + 1:]):
                recovered += 1
            else:
                unrecovered.append(name)

        value = recovered / len(failures)
        comment = (f"recovered {recovered}/{len(failures)} tool failures"
                   + (f"; unrecovered: {sorted(set(unrecovered))}" if unrecovered else ""))
        return Score(name=self.name, value=value, level=self.level, comment=comment)
