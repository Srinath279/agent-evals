"""agent-evals: reusable eval harness for AI agents.

Phase 0 vertical slice (see agent-eval-notes/00-master-plan.md):
schemas -> trace adapter -> PII redaction -> 3 evaluators
(label_match, goal_success, tool_selection) -> pass^k offline runner.

Design rules honored here:
- A pluggable trace store (Langfuse by default; LangSmith supported) is
  the system of record (scores/datasets), but the harness runs fully
  local without one.
- Evaluators/adapters/judge/redaction are pure library code; Temporal
  imports live only in agent_evals.pipelines.
- The score cache is the idempotency layer for judge calls.
"""

__version__ = "0.4.0"

from agent_evals.core.schemas import Case, Score, ToolCall, Trace  # noqa: F401
