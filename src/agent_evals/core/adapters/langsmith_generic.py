"""Generic LangSmith adapter: maps the run + child_runs shape (as produced
by LangSmithClient.fetch_trace_raw) into the canonical Trace.

Heuristics (mirroring langfuse-generic, asserted by the conformance test):
- run_type "llm"   -> LLM step: accumulate token usage
- run_type "tool"  -> ToolCall (error taken from the run's error field)
- every child run counts as a step
"""

from __future__ import annotations

from typing import Any

from agent_evals.core.adapters.base import TraceAdapter, register_adapter
from agent_evals.core.schemas import ToolCall, Trace


def _duration_ms(run: dict[str, Any]) -> float | None:
    start, end = run.get("start_time"), run.get("end_time")
    if start is None or end is None:
        return None
    try:
        return (end - start).total_seconds() * 1000
    except TypeError:
        return None  # string timestamps from a raw JSONL replay file


@register_adapter
class LangSmithGenericAdapter(TraceAdapter):
    name = "langsmith-generic"

    def to_trace(self, raw: dict[str, Any]) -> Trace:
        run = raw.get("run", raw)
        children = raw.get("child_runs", [])

        tool_calls: list[ToolCall] = []
        tokens_in = 0
        tokens_out = 0

        for child in children:
            run_type = (child.get("run_type") or "").lower()
            if run_type == "llm":
                tokens_in += child.get("prompt_tokens") or 0
                tokens_out += child.get("completion_tokens") or 0
            elif run_type == "tool":
                arguments = child.get("inputs") or {}
                if not isinstance(arguments, dict):
                    arguments = {"input": arguments}
                tool_calls.append(
                    ToolCall(
                        name=child.get("name") or "unknown_tool",
                        arguments=arguments,
                        result=child.get("outputs"),
                        error=child.get("error") or None,
                        duration_ms=_duration_ms(child),
                    )
                )

        metadata = dict((run.get("extra") or {}).get("metadata") or {})
        return Trace(
            trace_id=str(run.get("id") or run.get("trace_id") or ""),
            agent=metadata.get("agent", run.get("name", "")),
            input=run.get("inputs"),
            output=run.get("outputs"),
            tool_calls=tool_calls,
            steps=len(children),
            latency_ms=_duration_ms(run),
            cost_usd=run.get("total_cost"),
            tokens_in=(tokens_in or run.get("prompt_tokens")) or None,
            tokens_out=(tokens_out or run.get("completion_tokens")) or None,
            metadata=metadata,
        )
