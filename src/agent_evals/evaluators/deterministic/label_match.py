"""label_match — deterministic. For classification-style outputs (ticket
triage: category, priority, routing) compare agent output fields against
the case's expected labels. Value = fraction of fields that match."""

from __future__ import annotations

import json
from typing import Any, Optional

from agent_evals.core.evaluator import BaseEvaluator, register
from agent_evals.core.schemas import Case, Score, Trace


def _output_as_dict(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


@register
class LabelMatchEvaluator(BaseEvaluator):
    name = "label_match"
    level = "output"

    def evaluate(self, trace: Trace, case: Optional[Case] = None) -> Score:
        if case is None or not case.expected_labels:
            return Score(name=self.name, value=0.0, level=self.level,
                         comment="no expected labels on case — cannot grade")

        fields = self.params.get("fields") or list(case.expected_labels)
        output = _output_as_dict(trace.output)

        mismatches = []
        matched = 0
        for field in fields:
            expected = case.expected_labels.get(field)
            actual = output.get(field)
            if actual == expected:
                matched += 1
            else:
                mismatches.append(f"{field}: expected {expected!r}, got {actual!r}")

        value = matched / len(fields)
        comment = "all labels match" if not mismatches else "; ".join(mismatches)
        return Score(name=self.name, value=value, level=self.level, comment=comment)
