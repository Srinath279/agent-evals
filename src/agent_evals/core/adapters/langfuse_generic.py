"""Generic Langfuse adapter: works off the Langfuse API's trace +
observations shape without assuming an agent framework.

Heuristics (documented so the conformance test can assert them):
- observation type GENERATION  -> LLM step: accumulate token usage
- observation type TOOL, or SPAN whose metadata has tool=true
                               -> ToolCall (error taken from level=ERROR)
- every observation counts as a step
Framework-specific adapters (langgraph, ...) refine these mappings.
"""

from __future__ import annotations

from typing import Any

from agent_evals.core.adapters.base import TraceAdapter, register_adapter
from agent_evals.core.schemas import ToolCall, Trace


def _duration_ms(obs: dict[str, Any]) -> float | None:
    return obs.get("latency_ms") or obs.get("latency")


@register_adapter
class LangfuseGenericAdapter(TraceAdapter):
    name = "langfuse-generic"

    def to_trace(self, raw: dict[str, Any]) -> Trace:
        trace = raw.get("trace", raw)
        observations = raw.get("observations", [])

        tool_calls: list[ToolCall] = []
        tokens_in = 0
        tokens_out = 0
        cost = 0.0

        for obs in observations:
            obs_type = (obs.get("type") or "").upper()
            metadata = obs.get("metadata") or {}
            if obs_type == "GENERATION":
                usage = obs.get("usage") or {}
                tokens_in += usage.get("input", 0) or 0
                tokens_out += usage.get("output", 0) or 0
                cost += obs.get("calculatedTotalCost", 0) or 0
            elif obs_type == "TOOL" or (obs_type == "SPAN" and metadata.get("tool")):
                error = None
                if (obs.get("level") or "").upper() == "ERROR":
                    error = obs.get("statusMessage") or "tool error"
                arguments = obs.get("input") or {}
                if not isinstance(arguments, dict):
                    arguments = {"input": arguments}
                tool_calls.append(
                    ToolCall(
                        name=obs.get("name") or "unknown_tool",
                        arguments=arguments,
                        result=obs.get("output"),
                        error=error,
                        duration_ms=_duration_ms(obs),
                    )
                )

        return Trace(
            trace_id=trace.get("id") or trace.get("trace_id") or "",
            agent=(trace.get("metadata") or {}).get("agent", trace.get("name", "")),
            input=trace.get("input"),
            output=trace.get("output"),
            tool_calls=tool_calls,
            steps=len(observations),
            latency_ms=trace.get("latency_ms") or trace.get("latency"),
            cost_usd=cost or trace.get("totalCost"),
            tokens_in=tokens_in or None,
            tokens_out=tokens_out or None,
            metadata=trace.get("metadata") or {},
        )
