"""exact_match — deterministic. Compare the agent output (or one field of
it) against the case's expected_output. Strings are normalized
(strip+lower) unless normalize: false."""

from __future__ import annotations

from typing import Any, Optional

from agent_evals.core.evaluator import BaseEvaluator, register
from agent_evals.core.schemas import Case, Score, Trace


@register
class ExactMatchEvaluator(BaseEvaluator):
    name = "exact_match"
    level = "output"

    def _norm(self, value: Any) -> Any:
        if isinstance(value, str) and self.params.get("normalize", True):
            return value.strip().lower()
        return value

    def evaluate(self, trace: Trace, case: Optional[Case] = None) -> Score:
        if case is None or case.expected_output is None:
            return Score(name=self.name, value=0.0, level=self.level,
                         comment="no expected_output on case — cannot grade")

        field = self.params.get("field")
        actual, expected = trace.output, case.expected_output
        if field:
            actual = actual.get(field) if isinstance(actual, dict) else None
            expected = expected.get(field) if isinstance(expected, dict) else expected

        if self._norm(actual) == self._norm(expected):
            return Score(name=self.name, value=1.0, level=self.level, comment="exact match")
        return Score(name=self.name, value=0.0, level=self.level,
                     comment=f"expected {expected!r}, got {actual!r}")
