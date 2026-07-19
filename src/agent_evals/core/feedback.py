"""User-feedback ingestion (note 05 §8): the app posts thumbs as trace-store
scores with the trace_id propagated from the agent response to the UI.
This is the whole wire — call it from your API's feedback endpoint."""

from __future__ import annotations

from typing import Optional

from agent_evals.core.schemas import Score


def post_user_feedback(trace_id: str, value: float, comment: str = "",
                       user_id: Optional[str] = None,
                       trace_store: str = "langfuse") -> Score:
    """value convention: 1.0 thumbs-up, 0.0 thumbs-down (registry: user_feedback)."""
    from agent_evals.core.store import get_store

    score = Score(
        name="user_feedback",
        value=value,
        comment=comment,
        level="session",
        trace_id=trace_id,
        metadata={"user_id": user_id} if user_id else {},
    )
    store = get_store(trace_store)
    store.post_score(score)
    store.flush()
    return score
