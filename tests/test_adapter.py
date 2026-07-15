from agent_evals.core.adapters import get_adapter


def test_langfuse_generic_adapter_maps_canonical_trace():
    raw = {
        "trace": {
            "id": "lf-trace-1",
            "name": "support-agent",
            "input": {"subject": "help"},
            "output": {"category": "general"},
            "latency": 1234.0,
            "metadata": {"agent": "support-agent", "env": "prod"},
        },
        "observations": [
            {"type": "GENERATION", "name": "plan", "usage": {"input": 100, "output": 40},
             "calculatedTotalCost": 0.002},
            {"type": "TOOL", "name": "lookup_customer", "input": {"email": "a@b.com"},
             "output": {"id": 7}, "latency_ms": 55.0},
            {"type": "SPAN", "name": "update_ticket", "metadata": {"tool": True},
             "input": "raw-arg", "level": "ERROR", "statusMessage": "timeout"},
            {"type": "SPAN", "name": "internal-step"},
        ],
    }
    trace = get_adapter("langfuse-generic").to_trace(raw)

    assert trace.trace_id == "lf-trace-1"
    assert trace.agent == "support-agent"
    assert trace.steps == 4
    assert trace.tokens_in == 100 and trace.tokens_out == 40
    assert trace.cost_usd == 0.002
    assert trace.latency_ms == 1234.0

    assert [t.name for t in trace.tool_calls] == ["lookup_customer", "update_ticket"]
    assert trace.tool_calls[0].error is None
    assert trace.tool_calls[1].error == "timeout"
    assert trace.tool_calls[1].arguments == {"input": "raw-arg"}  # non-dict input wrapped
