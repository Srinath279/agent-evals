"""User-feedback ingestion (note 05 §8): the app posts thumbs as Langfuse
scores with the trace_id propagated from the agent response to the UI.
This is the whole wire — call it from your API's feedback endpoint."""

from __future__ import annotations

from typing import Optional

from agent_evals.core.schemas import Score


def post_user_feedback(trace_id: str, value: float, comment: str = "",
                       user_id: Optional[str] = None) -> Score:
    """value convention: 1.0 thumbs-up, 0.0 thumbs-down (registry: user_feedback)."""
    from agent_evals.core.langfuse_client import LangfuseClient

    score = Score(
        name="user_feedback",
        value=value,
        comment=comment,
        level="session",
        trace_id=trace_id,
        metadata={"user_id": user_id} if user_id else {},
    )
    client = LangfuseClient()
    client.post_score(score)
    client.flush()
    return score
