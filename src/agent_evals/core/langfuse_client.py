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
            trace_id=score.trace_id,
            name=score.name,
            value=score.value,
            comment=score.comment or None,
            metadata=score.metadata or None,
        )
        # SDK v3 renamed score() -> create_score()
        create = getattr(self._lf, "create_score", None) or getattr(self._lf, "score")
        create(**payload)

    def flush(self) -> None:
        self._lf.flush()
