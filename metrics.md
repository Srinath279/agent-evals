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

## Rules

- Judge-based scores are **judge-specific measurements**: `judge_provider`,
  `judge_model`, and `rubric_version` are stamped in every score's metadata.
  Never aggregate across judges/rubrics in one dashboard series without a
  marked epoch.
- Rubric text changes REQUIRE bumping the evaluator's `rubric_version`
  (it is part of the score-cache key) and re-running the golden set.
- New metrics: add the row here in the same PR that adds the evaluator.
