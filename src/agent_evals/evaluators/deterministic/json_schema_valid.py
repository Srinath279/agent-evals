"""json_schema_valid — deterministic. The agent output must parse as JSON
(if a string) and validate against the configured JSON schema. Structured
outputs that drift break downstream consumers silently — this catches it."""

from __future__ import annotations

import json
from typing import Optional

import jsonschema

from agent_evals.core.evaluator import BaseEvaluator, register
from agent_evals.core.schemas import Case, Score, Trace


@register
class JsonSchemaValidEvaluator(BaseEvaluator):
    name = "json_schema_valid"
    level = "output"

    def evaluate(self, trace: Trace, case: Optional[Case] = None) -> Score:
        schema = self.params.get("schema")
        if not schema:
            return Score(name=self.name, value=0.0, level=self.level,
                         comment="misconfigured: 'schema' param is required")

        output = trace.output
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except (json.JSONDecodeError, ValueError) as e:
                return Score(name=self.name, value=0.0, level=self.level,
                             comment=f"output is not valid JSON: {e}")

        try:
            jsonschema.validate(output, schema)
        except jsonschema.ValidationError as e:
            return Score(name=self.name, value=0.0, level=self.level,
                         comment=f"schema violation at {list(e.absolute_path)}: {e.message}")
        return Score(name=self.name, value=1.0, level=self.level, comment="schema valid")
