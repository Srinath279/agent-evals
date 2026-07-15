"""goal_success — LLM-judge, session level. Did the agent accomplish the
user's goal, consistent with the tool evidence? Payload is PII-redacted
before it reaches the judge (note 05 §2). Rubric changes MUST bump
rubric_version — it is part of the score-cache key."""

from __future__ import annotations

import json
from typing import Any, Optional

from agent_evals.core.evaluator import BaseEvaluator, register
from agent_evals.core.redaction import redact_text
from agent_evals.core.schemas import Case, Score, Trace

RUBRIC = """\
Evaluate whether the support agent accomplished the customer's goal.

You are given:
- TICKET: the customer's request
- EXPECTED: what a correct resolution looks like (may be absent)
- AGENT_OUTPUT: the agent's final output
- TOOL_CALLS: the tools the agent actually invoked, with results/errors

Scoring guide:
- 1.0  goal fully accomplished; output consistent with tool evidence
- 0.7  goal mostly accomplished; minor omissions, nothing incorrect
- 0.4  partial progress; significant gaps or an unnecessary detour
- 0.0  goal not accomplished, or the agent claimed an action (e.g. "ticket
       updated", "refund issued") that no successful tool call supports —
       fabricated success is always 0.0
"""

RUBRIC_VERSION = "goal_success/v1"


def _fmt(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


@register
class GoalSuccessEvaluator(BaseEvaluator):
    name = "goal_success"
    level = "session"
    requires_judge = True
    rubric_version = RUBRIC_VERSION

    def evaluate(self, trace: Trace, case: Optional[Case] = None) -> Score:
        tools = [
            {"name": t.name, "arguments": t.arguments, "result": t.result, "error": t.error}
            for t in trace.tool_calls
        ]
        expected = case.expected_output if case else None
        payload = redact_text(
            f"TICKET:\n{_fmt(trace.input)}\n\n"
            f"EXPECTED:\n{_fmt(expected) if expected is not None else '(not provided)'}\n\n"
            f"AGENT_OUTPUT:\n{_fmt(trace.output)}\n\n"
            f"TOOL_CALLS:\n{_fmt(tools)}"
        )
        verdict = self.judge.verdict(RUBRIC, payload)
        return Score(
            name=self.name,
            value=verdict.score,
            level=self.level,
            comment=verdict.reasoning,
            metadata=self._judge_stamp(),
        )
