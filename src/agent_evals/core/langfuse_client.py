"""Thin Langfuse wrapper — the only module that talks to Langfuse.

Lazy-imported so the harness runs fully local (JSONL datasets, no score
posting) without the SDK or credentials. Credentials come from the
standard env vars (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST).
"""

from __future__ import annotations

from agent_evals.core.schemas import Case, Score


class LangfuseClient:
    def __init__(self) -> None:
        try:
            from langfuse import Langfuse
        except ImportError as e:
            raise ImportError(
                "Langfuse features require the 'langfuse' extra: pip install 'agent-evals[langfuse]'"
            ) from e
        self._lf = Langfuse()

    def load_dataset(self, name: str) -> list[Case]:
        dataset = self._lf.get_dataset(name)
        cases = []
        for item in dataset.items:
            meta = dict(item.metadata or {})
            cases.append(
                Case(
                    case_id=str(item.id),
                    input=item.input,
                    expected_output=item.expected_output,
                    expected_labels=meta.get("expected_labels", {}),
                    expected_tools=meta.get("expected_tools", []),
                    metadata=meta,
                )
            )
        return cases

    def post_score(self, score: Score) -> None:
        payload = dict(
            # the agent's real Langfuse trace, not the runner's deterministic ID
            trace_id=score.metadata.get("source_trace_id") or score.trace_id,
            name=score.name,
            value=score.value,
            comment=score.comment or None,
            metadata=score.metadata or None,
        )
        # SDK v3 renamed score() -> create_score()
        create = getattr(self._lf, "create_score", None) or getattr(self._lf, "score")
        create(**payload)

    def seed_dataset(self, name: str, cases: list[Case]) -> int:
        """Create/extend a Langfuse dataset from cases. Callers are
        responsible for redacting PII first (the CLI does)."""
        try:
            self._lf.create_dataset(name=name)
        except Exception:
            pass  # dataset already exists
        for case in cases:
            self._lf.create_dataset_item(
                dataset_name=name,
                input=case.input,
                expected_output=case.expected_output,
                metadata={
                    "expected_labels": case.expected_labels,
                    "expected_tools": case.expected_tools,
                    "source_case_id": case.case_id,
                    **case.metadata,
                },
            )
        return len(cases)

    def fetch_trace_raw(self, trace_id: str) -> dict:
        """Fetch a trace + its observations in the adapter-shaped raw dict
        (see adapters.langfuse_generic). Used by the online pipeline."""
        fetch = getattr(self._lf, "fetch_trace", None) or getattr(self._lf, "get_trace")
        trace = fetch(trace_id)
        data = getattr(trace, "data", trace)
        as_dict = data.dict() if hasattr(data, "dict") else dict(data)
        observations = as_dict.pop("observations", None) or []
        observations = [o.dict() if hasattr(o, "dict") else dict(o) for o in observations]
        return {"trace": as_dict, "observations": observations}

    def enqueue_annotation(self, trace_id: str, queue_name: str, reason: str) -> bool:
        """Push a trace into an annotation queue (below-threshold traces →
        human review → golden-set candidates). Best-effort: queue APIs vary
        by Langfuse version/tier, so failures degrade to a tagged comment
        score instead of dropping the signal."""
        try:
            api = self._lf.api  # low-level generated client
            queues = api.annotation_queues.list_queues()
            queue = next(q for q in queues.data if q.name == queue_name)
            api.annotation_queues.create_queue_item(
                queue_id=queue.id, object_id=trace_id, object_type="TRACE"
            )
            return True
        except Exception:
            self.post_score(Score(
                name="needs_annotation", value=1.0, trace_id=trace_id,
                comment=f"[{queue_name}] {reason}", level="session",
            ))
            return False

    def flush(self) -> None:
        self._lf.flush()
