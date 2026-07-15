"""tool_selection — trajectory-level. Did the agent call the tools the
case expected, without spurious extras? Value = F1 between the expected
tool set and the tools actually called."""

from __future__ import annotations

from typing import Optional

from agent_evals.core.evaluator import BaseEvaluator, register
from agent_evals.core.schemas import Case, Score, Trace


@register
class ToolSelectionEvaluator(BaseEvaluator):
    name = "tool_selection"
    level = "trace"

    def evaluate(self, trace: Trace, case: Optional[Case] = None) -> Score:
        if case is None or not case.expected_tools:
            return Score(name=self.name, value=1.0, level=self.level,
                         comment="no expected tools on case — vacuously correct")

        expected = set(case.expected_tools)
        actual = {t.name for t in trace.tool_calls}

        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        hit = len(expected & actual)

        precision = hit / len(actual) if actual else 0.0
        recall = hit / len(expected)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        parts = []
        if missing:
            parts.append(f"missing: {missing}")
        if unexpected:
            parts.append(f"unexpected: {unexpected}")
        comment = "; ".join(parts) if parts else "expected tools called exactly"
        return Score(name=self.name, value=f1, level=self.level, comment=comment)
