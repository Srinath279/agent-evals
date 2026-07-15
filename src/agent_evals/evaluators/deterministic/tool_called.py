"""tool_called — deterministic. The listed tools (param `tools`, falling
back to the case's expected_tools) must each have been called AND
succeeded. A call that errored does not count — 'the agent tried' is not
'the ticket got updated'."""

from __future__ import annotations

from typing import Optional

from agent_evals.core.evaluator import BaseEvaluator, register
from agent_evals.core.schemas import Case, Score, Trace


@register
class ToolCalledEvaluator(BaseEvaluator):
    name = "tool_called"
    level = "trace"

    def evaluate(self, trace: Trace, case: Optional[Case] = None) -> Score:
        tools = self.params.get("tools") or (case.expected_tools if case else [])
        if not tools:
            return Score(name=self.name, value=1.0, level=self.level,
                         comment="no tools required — vacuously correct")

        succeeded = {t.name for t in trace.tool_calls if not t.failed}
        missing = [t for t in tools if t not in succeeded]
        value = (len(tools) - len(missing)) / len(tools)
        comment = "all required tools called successfully" if not missing else \
            f"not called successfully: {missing}"
        return Score(name=self.name, value=value, level=self.level, comment=comment)
