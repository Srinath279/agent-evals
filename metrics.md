# Score Registry

The enforced metric registry (master plan §5). Every score name used by any
agent must be defined here with its exact semantics — "faithfulness" must mean
the same thing on every agent or cross-agent dashboards are fiction. Changes
go through PR review (CODEOWNERS).

| name | scale | direction | level | type | owner | definition |
|---|---|---|---|---|---|---|
| `goal_success` | 0–1 | higher better | session | LLM judge | platform | Did the agent accomplish the user's goal, consistent with tool evidence? Fabricated success (claimed action with no successful tool call) is always 0.0. Rubric: `goal_success/v1`. |
| `label_match` | 0–1 | higher better | output | deterministic | platform | Fraction of configured classification fields (e.g. category, priority) exactly matching the case's expected labels. |
| `tool_selection` | 0–1 | higher better | trace | deterministic (trajectory) | platform | F1 between the case's expected tool set and the tools actually called. Vacuously 1.0 when the case specifies no expected tools. |
| `exact_match` | 0/1 | higher better | output | deterministic | platform | Output (or configured `field`) equals expected_output; strings normalized (strip+lower) unless `normalize: false`. |
| `json_schema_valid` | 0/1 | higher better | output | deterministic | platform | Output parses as JSON (if string) and validates against the configured JSON schema. Misconfiguration scores 0 (fail-closed). |
| `tool_called` | 0–1 | higher better | trace | deterministic | platform | Fraction of required tools (param `tools` or case expected_tools) that were called AND succeeded — errored calls don't count. |
| `latency_threshold` | 0/1 | higher better | trace | deterministic | platform | trace latency_ms <= `threshold_ms`. Missing measurement scores 0 (fail-closed). |
| `cost_threshold` | 0/1 | higher better | trace | deterministic | platform | trace cost_usd <= `max_usd`. Missing measurement scores 0 (fail-closed). |
| `redundant_calls` | 0–1 | higher better | trace | deterministic (trajectory) | platform | Unique (tool, arguments) signatures / total tool calls. 1.0 = no identical repeated invocations. |
| `tool_success_rate` | 0–1 | higher better | trace | deterministic (trajectory) | platform | Fraction of tool calls that did not error. Vacuously 1.0 with no tool calls. |
| `steps_efficiency` | 0–1 | higher better | trace | deterministic (trajectory) | platform | 1.0 when steps <= `max_steps` (param or case metadata), else max_steps/steps. Vacuously 1.0 unconfigured. |

## Rules

- Judge-based scores are **judge-specific measurements**: `judge_provider`,
  `judge_model`, and `rubric_version` are stamped in every score's metadata.
  Never aggregate across judges/rubrics in one dashboard series without a
  marked epoch.
- Rubric text changes REQUIRE bumping the evaluator's `rubric_version`
  (it is part of the score-cache key) and re-running the golden set.
- New metrics: add the row here in the same PR that adds the evaluator.
